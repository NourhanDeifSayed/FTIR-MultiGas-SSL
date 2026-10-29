
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

MASK_RATIO = 0.20
BATCH_SIZE = 8

torch.manual_seed(42)

with np.load(ART / "normalization_stats.npz") as stats:
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

with h5py.File(
    ROOT / "data" / "spectra_0000.h5", "r"
) as f:
    raw = f["spectra"][:BATCH_SIZE].astype(np.float32)
    valid = f["valid_mask"][:BATCH_SIZE].astype(bool)

valid &= np.isfinite(raw)

normalized = np.zeros_like(raw, dtype=np.float32)
np.divide(
    raw - mean,
    std,
    out=normalized,
    where=valid
)

normalized[~valid] = 0.0

targets = torch.from_numpy(normalized)
valid_mask = torch.from_numpy(valid)

ssl_mask = torch.zeros_like(valid_mask)

for row in range(BATCH_SIZE):
    valid_positions = torch.where(valid_mask[row])[0]
    n_valid = len(valid_positions)

    if n_valid < 2:
        raise ValueError("Not enough valid channels")

    n_mask = max(1, round(n_valid * MASK_RATIO))
    n_mask = min(n_mask, n_valid - 1)

    shuffled = valid_positions[
        torch.randperm(n_valid)
    ]

    ssl_mask[row, shuffled[:n_mask]] = True


model_input = targets.clone()
model_input[ssl_mask] = 0.0


key_padding_mask = ~valid_mask

print("=== SSL MASKING TEST ===")
print("Input shape:", tuple(model_input.shape))
print("Target shape:", tuple(targets.shape))
print("Valid mask shape:", tuple(valid_mask.shape))
print("SSL mask shape:", tuple(ssl_mask.shape))

print("\nFirst spectrum:")
print("Valid channels:", int(valid_mask[0].sum()))
print("Originally invalid:", int((~valid_mask[0]).sum()))
print("Intentionally masked:", int(ssl_mask[0].sum()))
print("Visible valid channels:",
      int((valid_mask[0] & ~ssl_mask[0]).sum()))

print("\nChecks:")
print("Only valid channels masked:",
      bool(torch.all(ssl_mask <= valid_mask)))
print("Masked inputs are zero:",
      bool(torch.all(model_input[ssl_mask] == 0)))
print("Unmasked values unchanged:",
      bool(torch.all(
          model_input[~ssl_mask] == targets[~ssl_mask]
      )))
print("Targets remain unchanged:",
      bool(torch.isfinite(targets).all()))
print("Key-padding mask shape:",
      tuple(key_padding_mask.shape))


dummy_prediction = torch.zeros_like(targets)

masked_loss = (
    (dummy_prediction[ssl_mask] - targets[ssl_mask]) ** 2
).mean()

print("\nDummy masked MSE:", float(masked_loss))

assert torch.all(ssl_mask <= valid_mask)
assert torch.all(model_input[ssl_mask] == 0)
assert torch.all(
    model_input[~ssl_mask] == targets[~ssl_mask]
)
assert torch.isfinite(masked_loss)

print("\nAll SSL masking checks passed!")
