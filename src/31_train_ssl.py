
from pathlib import Path
import argparse
import csv
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftir_batch_dataset import FTIRBatchDataset
from ftir_file_batch_sampler import FTIRFileBatchSampler
from ftir_model import MaskedFTIRTransformer


# ==================================================
# 1. Configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--train-limit", type=int, default=128)
parser.add_argument("--val-limit", type=int, default=64)

parser.add_argument(
    "--train-indices-file",
    type=str,
    default="train_indices.npy"
)

parser.add_argument(
    "--val-indices-file",
    type=str,
    default="val_indices.npy"
)

parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--checkpoint-dir", type=str, required=True)
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MASK_RATIO = 0.20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

checkpoint_dir = Path(args.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Epochs:", args.epochs)
print("Batch size:", args.batch_size)
print("Checkpoint directory:", checkpoint_dir)


# ==================================================
# 2. Load group-aware train/validation splits
# ==================================================

train_indices = np.load(
    ARTIFACTS / args.train_indices_file
)

val_indices = np.load(
    ARTIFACTS / args.val_indices_file
)

if args.train_limit > 0:
    train_indices = train_indices[:args.train_limit]

if args.val_limit > 0:
    val_indices = val_indices[:args.val_limit]

assert len(train_indices) > 0
assert len(val_indices) > 0

# Keep train and validation samples separate
assert not np.intersect1d(
    train_indices, val_indices
).size, "Train/validation overlap detected"


# ==================================================
# 3. File-aware datasets and batch samplers
# ==================================================

train_dataset = FTIRBatchDataset(train_indices)
val_dataset = FTIRBatchDataset(val_indices)

train_sampler = FTIRFileBatchSampler(
    indices=train_indices,
    batch_size=args.batch_size,
    shuffle=True,
    seed=SEED,
)

val_sampler = FTIRFileBatchSampler(
    indices=val_indices,
    batch_size=args.batch_size,
    shuffle=False,
    seed=SEED,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

print("Training spectra:", len(train_dataset))
print("Validation spectra:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ==================================================
# 4. Model and optimizer
# ==================================================

model = MaskedFTIRTransformer().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
)

best_val_loss = float("inf")
start_epoch = 0

last_checkpoint = checkpoint_dir / "ssl_last.pt"
best_checkpoint = checkpoint_dir / "ssl_best.pt"
history_file = checkpoint_dir / "ssl_history.csv"


# ==================================================
# 5. Resume training
# ==================================================

if args.resume:
    if not last_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {last_checkpoint}"
        )

    checkpoint = torch.load(
        last_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint["best_val_loss"])

    # Restore PyTorch RNG state
    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    if DEVICE.type == "cuda":
        cuda_rng_state = checkpoint.get("cuda_rng_state")

        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                cuda_rng_state.cpu()
            )

    print("Resuming from epoch:", start_epoch + 1)


# ==================================================
# 6. SSL masking and loss
# ==================================================

def make_ssl_mask(valid_mask, generator=None):
    random_values = torch.rand(
        valid_mask.shape,
        generator=generator,
        device="cpu",
    ).to(valid_mask.device)

    return (random_values < MASK_RATIO) & valid_mask


def masked_mse(predictions, targets, ssl_mask):
    squared_error = (predictions - targets).square()
    return squared_error[ssl_mask].mean()


# ==================================================
# 7. Training and validation
# ==================================================

def run_epoch(loader, training, epoch):

    if training:
        model.train()

        # Change sample and batch order every epoch
        train_sampler.set_epoch(epoch)
    else:
        model.eval()

    total_squared_error = 0.0
    total_masked_channels = 0

    start = time.perf_counter()

    for batch_number, batch in enumerate(loader):

        spectra = batch["spectrum"].to(
            DEVICE,
            non_blocking=True,
        )

        valid_mask = batch["valid_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        if training:
            ssl_mask = make_ssl_mask(valid_mask)
        else:
            # Reproducible validation masks
            val_generator = torch.Generator(
                device="cpu"
            )

            val_generator.manual_seed(
                SEED + batch_number
            )

            ssl_mask = make_ssl_mask(
                valid_mask,
                generator=val_generator,
            )

        masked_count = int(ssl_mask.sum().item())

        if masked_count == 0:
            continue

        masked_spectra = spectra.masked_fill(
            ssl_mask,
            0.0,
        )

        with torch.set_grad_enabled(training):

            predictions = model(
                masked_spectra,
                valid_mask,
                ssl_mask,
            )

            loss = masked_mse(
                predictions,
                spectra,
                ssl_mask,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite SSL loss detected"
                )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_squared_error += (
            loss.item() * masked_count
        )

        total_masked_channels += masked_count

    if total_masked_channels == 0:
        raise RuntimeError(
            "No valid masked channels were found"
        )

    elapsed = time.perf_counter() - start

    average_loss = (
        total_squared_error / total_masked_channels
    )

    return average_loss, elapsed


# ==================================================
# 8. Checkpoint saving
# ==================================================

def save_checkpoint(epoch, validation_loss):

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,

        "torch_rng_state": torch.get_rng_state(),

        "cuda_rng_state": (
            torch.cuda.get_rng_state()
            if DEVICE.type == "cuda"
            else torch.empty(0, dtype=torch.uint8)
        ),

        "config": {
            "mask_ratio": MASK_RATIO,
            "seed": SEED,
            "batch_size": args.batch_size,
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
            "train_indices_file": args.train_indices_file,
            "val_indices_file": args.val_indices_file,
            "batching": "file_aware",
        },
    }

    # Save last checkpoint
    temp_path = checkpoint_dir / "ssl_last.tmp"

    torch.save(state, temp_path)
    temp_path.replace(last_checkpoint)

    # Save best model
    if validation_loss < previous_best:
        best_temp_path = checkpoint_dir / "ssl_best.tmp"

        torch.save(
            model.state_dict(),
            best_temp_path,
        )

        best_temp_path.replace(best_checkpoint)


# ==================================================
# 9. Save training history
# ==================================================

def append_history(
    epoch,
    train_loss,
    val_loss,
    train_time,
    val_time,
):

    new_file = not history_file.exists()

    with open(
        history_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "epoch",
                "train_masked_mse",
                "val_masked_mse",
                "train_seconds",
                "val_seconds",
            ])

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            train_time,
            val_time,
        ])


# ==================================================
# 10. Main training loop
# ==================================================

for epoch in range(start_epoch, args.epochs):

    print(
        f"\n=== EPOCH {epoch + 1}/{args.epochs} ===",
        flush=True,
    )

    train_loss, train_time = run_epoch(
        train_loader,
        training=True,
        epoch=epoch,
    )

    val_loss, val_time = run_epoch(
        val_loader,
        training=False,
        epoch=epoch,
    )

    previous_best = best_val_loss
    best_val_loss = min(
        best_val_loss,
        val_loss,
    )

    save_checkpoint(epoch, val_loss)

    append_history(
        epoch,
        train_loss,
        val_loss,
        train_time,
        val_time,
    )

    print(
        f"Train Masked MSE: {train_loss:.6f}"
    )

    print(
        f"Val Masked MSE:   {val_loss:.6f}"
    )

    print(
        f"Best Val MSE:     {best_val_loss:.6f}"
    )

    print(
        f"Train time: {train_time:.2f} seconds"
    )

    print(
        f"Val time:   {val_time:.2f} seconds"
    )

    print(
        "Checkpoint saved.",
        flush=True,
    )

print("\nSSL training completed.")
