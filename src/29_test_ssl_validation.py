
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ftir_dataset import FTIRDataset
from ftir_model import MaskedFTIRTransformer

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

torch.manual_seed(42)

MASK_RATIO = 0.20
TRAIN_STEPS = 6
BATCH_SIZE = 8

device = torch.device("cpu")



train_indices = np.load(ART / "train_indices.npy")
val_indices = np.load(ART / "val_indices.npy")

train_dataset = FTIRDataset(train_indices[:BATCH_SIZE])
val_dataset = FTIRDataset(val_indices[:BATCH_SIZE])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

assert not (
    set(train_batch["global_idx"].tolist())
    & set(val_batch["global_idx"].tolist())
)



def make_ssl_mask(valid_mask, ratio=MASK_RATIO, seed=42):

    generator = torch.Generator()
    generator.manual_seed(seed)

    ssl_mask = torch.zeros_like(valid_mask)

    for row in range(valid_mask.shape[0]):
        positions = torch.where(valid_mask[row])[0]
        n_valid = len(positions)

        if n_valid < 2:
            raise ValueError("Too few valid channels")

        n_mask = max(1, round(n_valid * ratio))
        n_mask = min(n_mask, n_valid - 1)

        permutation = torch.randperm(
            n_valid,
            generator=generator
        )

        chosen = positions[permutation[:n_mask]]
        ssl_mask[row, chosen] = True

    return ssl_mask


def prepare_batch(batch, seed):

    targets = batch["spectrum"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    ssl_mask = make_ssl_mask(
        valid_mask, seed=seed
    ).to(device)

    model_input = targets.clone()
    model_input[ssl_mask] = 0.0

    return model_input, targets, valid_mask, ssl_mask


# Fixed masks, so before/after comparisons are fair.
train_data = prepare_batch(train_batch, seed=42)
val_data = prepare_batch(val_batch, seed=123)



model = MaskedFTIRTransformer().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


def calculate_loss(data):
    inputs, targets, valid_mask, ssl_mask = data

    prediction = model(
        inputs,
        valid_mask,
        ssl_mask
    )

    return (
        (prediction[ssl_mask] - targets[ssl_mask]) ** 2
    ).mean()


def evaluate(data):
    model.eval()

    with torch.no_grad():
        return float(calculate_loss(data).item())


initial_train = evaluate(train_data)
initial_val = evaluate(val_data)

print("=== SSL TRAIN + VALIDATION TEST ===")
print("Training spectra:", BATCH_SIZE)
print("Validation spectra:", BATCH_SIZE)
print("Training steps:", TRAIN_STEPS)

print("\nBefore training:")
print("Train MSE:", round(initial_train, 6))
print("Validation MSE:", round(initial_val, 6))


for step in range(1, TRAIN_STEPS + 1):

    model.train()
    optimizer.zero_grad()

    train_loss = calculate_loss(train_data)

    if not torch.isfinite(train_loss):
        raise ValueError("Non-finite training loss")

    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    print(
        f"Step {step:02d}/{TRAIN_STEPS} | "
        f"Training MSE: {train_loss.item():.6f}"
    )



final_train = evaluate(train_data)
final_val = evaluate(val_data)

print("\n=== SSL VALIDATION SUMMARY ===")

print("Train before:", round(initial_train, 6))
print("Train after:", round(final_train, 6))
print("Train improved:", final_train < initial_train)

print()

print("Validation before:", round(initial_val, 6))
print("Validation after:", round(final_val, 6))
print("Validation improved:", final_val < initial_val)

assert np.isfinite(final_train)
assert np.isfinite(final_val)

train_dataset.close()
val_dataset.close()

print("\nMini SSL validation test completed!")
