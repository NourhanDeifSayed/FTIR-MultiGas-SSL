
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "spectra_0000.h5"
STATS_FILE = PROJECT_DIR / "artifacts" / "normalization_stats.npz"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with np.load(STATS_FILE) as stats:
    mean = stats["mean"]
    std = stats["std"]
    counts = stats["counts"]

def normalize_spectrum(spectrum, valid_mask):
    """
    Normalize valid spectral channels using
    training-only per-channel statistics.

    Invalid channels are assigned zero,
    but remain marked as invalid.
    """
    valid = valid_mask.astype(bool)

    normalized = (spectrum - mean) / std

    normalized[~valid] = 0.0

    return normalized.astype(np.float32), valid

with h5py.File(DATA_FILE, "r") as f:
    spectrum = f["spectra"][0]
    valid_mask = f["valid_mask"][0]
    wavenumbers = f["wavenumbers"][:]

normalized, valid = normalize_spectrum(
    spectrum, valid_mask
)

print("=== NORMALIZATION CHECK ===")
print("Original shape:", spectrum.shape)
print("Normalized shape:", normalized.shape)
print("Valid channels:", valid.sum())
print("Invalid channels:", (~valid).sum())

print("\nFirst 10 original values:")
print(spectrum[:10])

print("\nFirst 10 normalized values:")
print(normalized[:10])

print("\nFirst 10 validity flags:")
print(valid_mask[:10])

assert normalized.shape == (257,)
assert np.all(normalized[~valid] == 0)
assert np.all(np.isfinite(normalized))

print("\nAll checks passed!")

original_plot = spectrum.copy()
normalized_plot = normalized.copy()

original_plot[~valid] = np.nan
normalized_plot[~valid] = np.nan

fig, axes = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True
)

axes[0].plot(wavenumbers, original_plot)
axes[0].set_title("Original FTIR Spectrum")
axes[0].set_ylabel("Original Signal")
axes[0].grid(alpha=0.3)

axes[1].plot(wavenumbers, normalized_plot)
axes[1].set_title("Normalized FTIR Spectrum")
axes[1].set_xlabel("Wavenumber (cm^-1)")
axes[1].set_ylabel("Z-score")
axes[1].grid(alpha=0.3)

plt.tight_layout()

output_path = OUTPUT_DIR / "normalization_comparison.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nPlot saved to:", output_path)
