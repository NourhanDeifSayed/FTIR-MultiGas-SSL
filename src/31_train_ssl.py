from pathlib import Path
import argparse
import csv
import hashlib
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftir_batch_dataset import FTIRBatchDataset
from ftir_file_batch_sampler import FTIRFileBatchSampler
from ftir_model import MaskedFTIRTransformer

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MASK_RATIO = 0.20
SEED = 42
CHECKPOINT_VERSION = 2

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--train-limit", type=int, default=128)
parser.add_argument("--val-limit", type=int, default=64)
parser.add_argument("--train-indices-file", type=str, default="train_indices.npy")
parser.add_argument("--val-indices-file", type=str, default="val_indices.npy")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--checkpoint-dir", type=str, required=True)
parser.add_argument("--save-every", type=int, default=200)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
    parser.error("epochs, batch-size and save-every must be positive")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

checkpoint_dir = Path(args.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
last_checkpoint = checkpoint_dir / "ssl_last.pt"
best_checkpoint = checkpoint_dir / "ssl_best.pt"
history_file = checkpoint_dir / "ssl_history.csv"

if not args.resume and last_checkpoint.exists():
    raise FileExistsError(
        f"Checkpoint already exists: {last_checkpoint}. Use --resume or a new directory."
    )

train_indices = np.asarray(np.load(ARTIFACTS / args.train_indices_file), dtype=np.int64)
val_indices = np.asarray(np.load(ARTIFACTS / args.val_indices_file), dtype=np.int64)
if args.train_limit > 0:
    train_indices = train_indices[:args.train_limit]
if args.val_limit > 0:
    val_indices = val_indices[:args.val_limit]

if len(train_indices) == 0 or len(val_indices) == 0:
    raise ValueError("Training and validation splits must both be nonempty")
if np.intersect1d(train_indices, val_indices).size:
    raise ValueError("Train/validation sample overlap detected")

def index_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

config = {
    "checkpoint_version": CHECKPOINT_VERSION,
    "batching": "file_aware",
    "seed": SEED,
    "mask_ratio": MASK_RATIO,
    "batch_size": args.batch_size,
    "train_indices_file": args.train_indices_file,
    "val_indices_file": args.val_indices_file,
    "train_count": len(train_indices),
    "val_count": len(val_indices),
    "train_digest": index_digest(train_indices),
    "val_digest": index_digest(val_indices),
}

train_dataset = FTIRBatchDataset(train_indices)
val_dataset = FTIRBatchDataset(val_indices)
train_sampler = FTIRFileBatchSampler(
    indices=train_indices, batch_size=args.batch_size, shuffle=True, seed=SEED
)
val_sampler = FTIRFileBatchSampler(
    indices=val_indices, batch_size=args.batch_size, shuffle=False, seed=SEED
)
val_batches = list(val_sampler)
val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_batches,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    generator=torch.Generator().manual_seed(SEED + 1000),
)
model = MaskedFTIRTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

best_val_loss = float("inf")
start_epoch = 0
start_batch = 0
running_squared_error = 0.0
running_masked_channels = 0
history = []

if args.resume:
    if not last_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {last_checkpoint}")
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("config") != config:
        raise ValueError("Checkpoint configuration or selected indices do not match")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_loss = float(checkpoint["best_val_loss"])
    start_epoch = int(checkpoint["epoch"])
    start_batch = int(checkpoint["next_batch"])
    running_squared_error = float(checkpoint["running_squared_error"])
    running_masked_channels = int(checkpoint["running_masked_channels"])
    history = list(checkpoint["history"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if DEVICE.type == "cuda":
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

print("Device:", DEVICE)
print("Epochs:", args.epochs)
print("Batch size:", args.batch_size)
print("Checkpoint directory:", checkpoint_dir)
print("Save every batches:", args.save_every)
print("Training spectra:", len(train_indices))
print("Validation spectra:", len(val_indices))
print("Validation batches:", len(val_batches))
if args.resume:
    print(f"Resuming at epoch {start_epoch + 1}, next batch {start_batch + 1}")


def save_atomic(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def save_progress(epoch, next_batch, squared_error, masked_channels):
    state = {
        "epoch": epoch,
        "next_batch": next_batch,
        "running_squared_error": squared_error,
        "running_masked_channels": masked_channels,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state() if DEVICE.type == "cuda" else torch.empty(0, dtype=torch.uint8)
        ),
        "config": config,
        "history": history,
    }
    save_atomic(state, last_checkpoint)


def write_history():
    temporary = history_file.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_masked_mse", "val_masked_mse", "train_seconds", "val_seconds"])
        writer.writerows(history)
    temporary.replace(history_file)


def validation_epoch():
    model.eval()
    squared_error_sum = 0.0
    masked_channel_sum = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch_number, batch in enumerate(val_loader):
            spectra = batch["spectrum"].to(DEVICE, non_blocking=True)
            valid_mask = batch["valid_mask"].to(DEVICE, non_blocking=True)
            generator = torch.Generator(device="cpu").manual_seed(SEED + batch_number)
            ssl_mask = (
                torch.rand(valid_mask.shape, generator=generator, device="cpu").to(DEVICE) < MASK_RATIO
            ) & valid_mask
            masked_count = int(ssl_mask.sum().item())
            if masked_count == 0:
                continue
            predictions = model(spectra.masked_fill(ssl_mask, 0.0), valid_mask, ssl_mask)
            loss = (predictions[ssl_mask] - spectra[ssl_mask]).square().mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite validation loss")
            squared_error_sum += float(loss.item()) * masked_count
            masked_channel_sum += masked_count
    if masked_channel_sum == 0:
        raise RuntimeError("No valid masked channels during validation")
    return squared_error_sum / masked_channel_sum, time.perf_counter() - start


for epoch in range(start_epoch, args.epochs):
    train_sampler.set_epoch(epoch)
    all_batches = list(train_sampler)
    batch_offset = start_batch if epoch == start_epoch else 0
    if not 0 <= batch_offset <= len(all_batches):
        raise ValueError("Checkpoint batch position is invalid")

    squared_error_sum = running_squared_error if epoch == start_epoch else 0.0
    masked_channel_sum = running_masked_channels if epoch == start_epoch else 0
    model.train()
    train_start = time.perf_counter()
    print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===", flush=True)
    print(f"Training batches: {len(all_batches)} | Starting batch: {batch_offset + 1}", flush=True)

    remaining_loader = DataLoader(
        train_dataset,
        batch_sampler=all_batches[batch_offset:],
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        generator=torch.Generator().manual_seed(SEED + epoch + 2000),
    )

    for batch_number, batch in enumerate(remaining_loader, start=batch_offset):
        spectra = batch["spectrum"].to(DEVICE, non_blocking=True)
        valid_mask = batch["valid_mask"].to(DEVICE, non_blocking=True)
        ssl_mask = (torch.rand(valid_mask.shape, device="cpu").to(DEVICE) < MASK_RATIO) & valid_mask
        masked_count = int(ssl_mask.sum().item())
        if masked_count:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(spectra.masked_fill(ssl_mask, 0.0), valid_mask, ssl_mask)
            loss = (predictions[ssl_mask] - spectra[ssl_mask]).square().mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            squared_error_sum += float(loss.item()) * masked_count
            masked_channel_sum += masked_count

        completed_batches = batch_number + 1
        if completed_batches % args.save_every == 0:
            save_progress(epoch, completed_batches, squared_error_sum, masked_channel_sum)
            print(f"Saved progress: epoch {epoch + 1}, batch {completed_batches}/{len(all_batches)}", flush=True)

    if masked_channel_sum == 0:
        raise RuntimeError("No valid masked channels during training")

    train_time = time.perf_counter() - train_start
    train_loss = squared_error_sum / masked_channel_sum
    save_progress(epoch, len(all_batches), squared_error_sum, masked_channel_sum)
    val_loss, val_time = validation_epoch()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_atomic(model.state_dict(), best_checkpoint)

    history.append([epoch + 1, train_loss, val_loss, train_time, val_time])
    save_progress(epoch + 1, 0, 0.0, 0)
    write_history()

    print(f"Train Masked MSE: {train_loss:.6f}")
    print(f"Val Masked MSE:   {val_loss:.6f}")
    print(f"Best Val MSE:     {best_val_loss:.6f}")
    print(f"Train time: {train_time:.2f} seconds")
    print(f"Val time:   {val_time:.2f} seconds")
    print("Checkpoint saved.", flush=True)

    start_batch = 0
    running_squared_error = 0.0
    running_masked_channels = 0

print("\nSSL training completed.")
