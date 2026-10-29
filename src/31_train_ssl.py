
from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftir_dataset import FTIRDataset
from ftir_model import MaskedFTIRTransformer



ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--train-limit", type=int, default=128)
parser.add_argument("--val-limit", type=int, default=64)
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
print("Checkpoint directory:", checkpoint_dir)


train_indices = np.load(ARTIFACTS / "train_indices.npy")
val_indices = np.load(ARTIFACTS / "val_indices.npy")

if args.train_limit > 0:
    train_indices = train_indices[:args.train_limit]

if args.val_limit > 0:
    val_indices = val_indices[:args.val_limit]

assert len(train_indices) > 0
assert len(val_indices) > 0

train_dataset = FTIRDataset(train_indices)
val_dataset = FTIRDataset(val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

print("Training spectra:", len(train_dataset))
print("Validation spectra:", len(val_dataset))


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



if args.resume:
    if not last_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {last_checkpoint}"
        )

    checkpoint = torch.load(
        last_checkpoint,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    # Restore random states for reproducibility
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if DEVICE.type == "cuda" and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(
            checkpoint["cuda_rng_state"].cpu()
        )

    print("Resuming from epoch:", start_epoch + 1)


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



def run_epoch(loader, training, epoch):
    if training:
        model.train()
    else:
        model.eval()

    total_squared_error = 0.0
    total_masked_channels = 0

    for batch_number, batch in enumerate(loader):
        spectra = batch["spectrum"].to(
            DEVICE, non_blocking=True
        )
        valid_mask = batch["valid_mask"].to(
            DEVICE, non_blocking=True
        )

        if training:
            ssl_mask = make_ssl_mask(valid_mask)
        else:
            # Fixed masking for each validation batch
            val_generator = torch.Generator(device="cpu")
            val_generator.manual_seed(SEED + batch_number)
            ssl_mask = make_ssl_mask(
                valid_mask,
                generator=val_generator,
            )

        masked_count = int(ssl_mask.sum().item())

        if masked_count == 0:
            continue

        masked_spectra = spectra.masked_fill(ssl_mask, 0.0)

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

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite SSL loss detected")

        total_squared_error += loss.item() * masked_count
        total_masked_channels += masked_count

    if total_masked_channels == 0:
        raise RuntimeError("No valid masked channels were found")

    return total_squared_error / total_masked_channels



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
        },
    }

    temp_path = checkpoint_dir / "ssl_last.tmp"
    torch.save(state, temp_path)
    temp_path.replace(last_checkpoint)

    if validation_loss < previous_best:
        torch.save(
            model.state_dict(),
            best_checkpoint,
        )


def append_history(epoch, train_loss, val_loss):
    new_file = not history_file.exists()

    with open(
        history_file, "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow(
                ["epoch", "train_masked_mse", "val_masked_mse"]
            )

        writer.writerow(
            [epoch + 1, train_loss, val_loss]
        )



try:
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")

        train_loss = run_epoch(
            train_loader,
            training=True,
            epoch=epoch,
        )

        val_loss = run_epoch(
            val_loader,
            training=False,
            epoch=epoch,
        )

        previous_best = best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_checkpoint(epoch, val_loss)
        append_history(epoch, train_loss, val_loss)

        print(f"Train Masked MSE: {train_loss:.6f}")
        print(f"Val Masked MSE:   {val_loss:.6f}")
        print(f"Best Val MSE:     {best_val_loss:.6f}")
        print("Checkpoint saved.")

finally:
    train_dataset.close()
    val_dataset.close()

print("\nSSL training completed.")
