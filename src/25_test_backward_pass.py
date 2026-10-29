
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

torch.manual_seed(42)

BATCH_SIZE = 8
SEQ_LEN = 257
D_MODEL = 128
MASK_RATIO = 0.20


class MaskedFTIRTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(2, D_MODEL)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, SEQ_LEN, D_MODEL)
        )
        nn.init.normal_(
            self.position_embedding, std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
            enable_nested_tensor=False
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, 1)
        )

    def forward(self, spectra, valid_mask, ssl_mask):
        features = torch.stack(
            [spectra, ssl_mask.float()],
            dim=-1
        )

        tokens = (
            self.input_projection(features)
            + self.position_embedding
        )

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=~valid_mask
        )

        return self.reconstruction_head(
            encoded
        ).squeeze(-1)

with np.load(ART / "normalization_stats.npz") as stats:
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

with h5py.File(
    ROOT / "data" / "spectra_0000.h5", "r"
) as f:
    raw = f["spectra"][:BATCH_SIZE].astype(np.float32)
    valid = f["valid_mask"][:BATCH_SIZE].astype(bool)

valid &= np.isfinite(raw)

normalized = np.zeros_like(raw)

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

    n_mask = max(1, round(n_valid * MASK_RATIO))
    n_mask = min(n_mask, n_valid - 1)

    shuffled = valid_positions[
        torch.randperm(n_valid)
    ]

    ssl_mask[row, shuffled[:n_mask]] = True

model_input = targets.clone()
model_input[ssl_mask] = 0.0

model = MaskedFTIRTransformer()
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

weights_before = (
    model.input_projection.weight.detach().clone()
)



optimizer.zero_grad()

prediction = model(
    model_input,
    valid_mask,
    ssl_mask
)

loss = (
    (prediction[ssl_mask] - targets[ssl_mask]) ** 2
).mean()

loss_before = float(loss.item())


loss.backward()

gradient = model.input_projection.weight.grad

assert gradient is not None

grad_norm = float(gradient.norm().item())

all_gradients_finite = all(
    p.grad is None or torch.isfinite(p.grad).all().item()
    for p in model.parameters()
)


optimizer.step()

weights_after = (
    model.input_projection.weight.detach().clone()
)

weights_changed = not torch.equal(
    weights_before,
    weights_after
)



print("=== BACKWARD PASS TEST ===")

print("Batch size:", BATCH_SIZE)
print("Initial masked MSE:", loss_before)

print("\nGradients:")
print("Input projection gradient norm:", grad_norm)
print("All gradients finite:", all_gradients_finite)

print("\nOptimizer:")
print("Optimizer:", "AdamW")
print("Learning rate:", 3e-4)
print("Weights changed after step:", weights_changed)

assert np.isfinite(loss_before)
assert np.isfinite(grad_norm)
assert grad_norm > 0
assert all_gradients_finite
assert weights_changed

print("\nAll backward pass checks passed!")
