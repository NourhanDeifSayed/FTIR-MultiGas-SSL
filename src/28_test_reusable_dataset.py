
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftir_dataset import FTIRDataset

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

train_indices = np.load(
    ART / "train_indices.npy"
)

val_indices = np.load(
    ART / "val_indices.npy"
)

train_dataset = FTIRDataset(train_indices[:16])
val_dataset = FTIRDataset(val_indices[:8])

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print("=== REUSABLE DATASET TEST ===")

print("Train batch shape:",
      tuple(train_batch["spectrum"].shape))

print("Validation batch shape:",
      tuple(val_batch["spectrum"].shape))

print("Train and validation overlap:",
      len(
          set(train_batch["global_idx"].tolist())
          & set(val_batch["global_idx"].tolist())
      ))

for name, batch in [
    ("Train", train_batch),
    ("Validation", val_batch)
]:
    spectra = batch["spectrum"]
    mask = batch["valid_mask"]

    print(f"\n{name}:")
    print("All values finite:",
          bool(torch.isfinite(spectra).all()))
    print("Invalid points zero:",
          bool(torch.all(spectra[~mask] == 0)))

    assert spectra.shape == (8, 257)
    assert mask.shape == (8, 257)
    assert torch.isfinite(spectra).all()
    assert torch.all(spectra[~mask] == 0)

assert not (
    set(train_batch["global_idx"].tolist())
    & set(val_batch["global_idx"].tolist())
)

train_dataset.close()
val_dataset.close()

print("\nAll reusable dataset checks passed!")
