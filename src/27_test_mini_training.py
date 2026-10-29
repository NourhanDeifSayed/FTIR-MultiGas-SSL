
from pathlib import Path

import h5py
import numpy as np
import torch

from ftir_model import MaskedFTIRTransformer

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

torch.manual_seed(42)

MASK_RATIO = 0.20
TRAIN_STEPS = 10


with np.load(ART / "normalization_stats.npz") as stats:
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)


rows = [0, 3, 6, 9]

with h5py.File(
    ROOT / "data" / "spectra_0000.h5", "r"
) as f:
    raw = f["spectra"][rows].astype(np.float32)
    valid = f["valid_mask"][rows].astype(bool)
    groups = f["group_id"][rows]

assert len(np.unique(groups)) == 4

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

for row in range(len(rows)):
    positions = torch.where(valid_mask[row])[0]

    n_valid = len(positions)
    n_mask = max(1, round(n_valid * MASK_RATIO))
    n_mask = min(n_mask, n_valid - 1)

    chosen = positions[torch.randperm(n_valid)[:n_mask]]
    ssl_mask[row, chosen] = True

model_input = targets.clone()
model_input[ssl_mask] = 0.0


model = MaskedFTIRTransformer()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

def calculate_loss():
    prediction = model(
        model_input,
        valid_mask,
        ssl_mask
    )

    return (
        (prediction[ssl_mask] - targets[ssl_mask]) ** 2
    ).mean()


model.eval()

with torch.no_grad():
    initial_loss = float(calculate_loss().item())

print("=== MINI SSL TRAINING TEST ===")
print("Spectra:", len(rows))
print("Different group IDs:", groups.tolist())
print("Masked positions:", int(ssl_mask.sum()))
print("Training steps:", TRAIN_STEPS)
print("Initial evaluation MSE:", round(initial_loss, 6))



model.train()

for step in range(1, TRAIN_STEPS + 1):

    optimizer.zero_grad()

    loss = calculate_loss()

    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss")

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    print(
        f"Step {step:02d}/{TRAIN_STEPS} | "
        f"Training MSE: {loss.item():.6f}"
    )


model.eval()

with torch.no_grad():
    final_loss = float(calculate_loss().item())

print("\n=== MINI TRAINING SUMMARY ===")
print("Initial evaluation MSE:", round(initial_loss, 6))
print("Final evaluation MSE:", round(final_loss, 6))

improvement = initial_loss - final_loss

print("Loss improvement:", round(improvement, 6))
print("Loss decreased:", final_loss < initial_loss)

assert np.isfinite(final_loss)
assert np.isfinite(improvement)

print("\nMini training completed successfully!")
