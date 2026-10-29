
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
NUM_HEADS = 4
NUM_LAYERS = 4
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
            nhead=NUM_HEADS,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
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

        tokens = self.input_projection(features)
        tokens = tokens + self.position_embedding

      
        padding_mask = ~valid_mask

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask
        )

        reconstruction = self.reconstruction_head(
            encoded
        ).squeeze(-1)

        return reconstruction


def create_ssl_mask(valid_mask, ratio=MASK_RATIO):

    ssl_mask = torch.zeros_like(valid_mask)

    for row in range(valid_mask.shape[0]):

        positions = torch.where(valid_mask[row])[0]
        n_valid = len(positions)

        if n_valid < 2:
            raise ValueError(
                "Spectrum has too few valid channels"
            )

        n_mask = max(1, round(n_valid * ratio))
        n_mask = min(n_mask, n_valid - 1)

        random_positions = positions[
            torch.randperm(n_valid)
        ]

        ssl_mask[row, random_positions[:n_mask]] = True

    return ssl_mask




with np.load(ART / "normalization_stats.npz") as f:
    mean = f["mean"].astype(np.float32)
    std = f["std"].astype(np.float32)

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


ssl_mask = create_ssl_mask(valid_mask)

model_input = targets.clone()
model_input[ssl_mask] = 0.0



model = MaskedFTIRTransformer()

model.eval()

with torch.no_grad():

    prediction = model(
        model_input,
        valid_mask,
        ssl_mask
    )

    loss = (
        (prediction[ssl_mask] - targets[ssl_mask]) ** 2
    ).mean()

print("=== TRANSFORMER FORWARD TEST ===")
print("Input shape:", tuple(model_input.shape))
print("Prediction shape:", tuple(prediction.shape))
print("Target shape:", tuple(targets.shape))

print("\nModel configuration:")
print("Sequence length:", SEQ_LEN)
print("Embedding dimension:", D_MODEL)
print("Attention heads:", NUM_HEADS)
print("Transformer layers:", NUM_LAYERS)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)

print("\nMasking:")
print("Valid channels in first spectrum:",
      int(valid_mask[0].sum()))
print("SSL masked channels in first spectrum:",
      int(ssl_mask[0].sum()))

print("\nUntrained model:")
print("Masked reconstruction MSE:", float(loss))

assert prediction.shape == (BATCH_SIZE, SEQ_LEN)
assert torch.isfinite(prediction).all()
assert torch.isfinite(loss)
assert torch.all(ssl_mask <= valid_mask)

print("\nAll Transformer forward checks passed!")
