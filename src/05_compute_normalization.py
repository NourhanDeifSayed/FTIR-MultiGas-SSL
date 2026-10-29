
from pathlib import Path
import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

split_codes = np.load(
    ARTIFACTS_DIR / "split_codes.npy"
)

counts = np.zeros(257, dtype=np.int64)
sums = np.zeros(257, dtype=np.float64)
squared_sums = np.zeros(257, dtype=np.float64)

files = sorted(DATA_DIR.glob("spectra_*.h5"))
assert len(files) == 100
assert len(split_codes) == 3_000_000

for file_idx, file_path in enumerate(files):

    print(f"[{file_idx + 1}/100] {file_path.name}")

    start_global = file_idx * 30000
    end_global = start_global + 30000

    file_splits = split_codes[start_global:end_global]

    with h5py.File(file_path, "r") as f:

        for start in range(0, 30000, 1024):
            end = min(start + 1024, 30000)

            train_rows = file_splits[start:end] == 0

            if not np.any(train_rows):
                continue

            spectra = f["spectra"][start:end][train_rows]
            valid_mask = (
                f["valid_mask"][start:end][train_rows] == 1
            )

            valid = valid_mask & np.isfinite(spectra)

            safe_spectra = np.where(valid, spectra, 0.0)

            counts += valid.sum(axis=0)
            sums += safe_spectra.sum(axis=0, dtype=np.float64)
            squared_sums += (
                safe_spectra.astype(np.float64) ** 2
            ).sum(axis=0)

assert np.all(counts > 0), (
    "Some channels have no valid training measurements."
)

mean = sums / counts

variance = squared_sums / counts - mean ** 2
variance = np.maximum(variance, 0.0)

std = np.sqrt(variance)

std_safe = np.maximum(std, 1e-6)

np.savez(
    ARTIFACTS_DIR / "normalization_stats.npz",
    mean=mean.astype(np.float32),
    std=std_safe.astype(np.float32),
    counts=counts
)

print("\n=== NORMALIZATION SUMMARY ===")
print("Channels:", len(mean))
print("Minimum mean:", mean.min())
print("Maximum mean:", mean.max())
print("Minimum std:", std.min())
print("Maximum std:", std.max())
print("Minimum valid count per channel:", counts.min())
print("Maximum valid count per channel:", counts.max())

print("\nFirst 10 channel means:")
print(mean[:10])

print("\nFirst 10 channel stds:")
print(std_safe[:10])

print("\nSaved: artifacts/normalization_stats.npz")
