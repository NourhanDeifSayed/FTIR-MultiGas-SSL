
from pathlib import Path

import h5py
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("spectra_*.h5"))

assert len(files) == 100, "Expected 100 HDF5 files"

SEED = 42
rng = np.random.default_rng(SEED)

all_groups = []

for file_path in files:
    with h5py.File(file_path, "r") as f:
        groups = f["group_id"][:]
        all_groups.append(groups)

    print(f"Loaded groups from {file_path.name}")

all_groups = np.concatenate(all_groups)

unique_groups = np.unique(all_groups)

print("\nTotal spectra:", len(all_groups))
print("Unique groups:", len(unique_groups))

assert len(all_groups) == 3_000_000
assert len(unique_groups) == 1_000_000


shuffled_groups = rng.permutation(unique_groups)

n_groups = len(shuffled_groups)

n_train = int(0.80 * n_groups)
n_val = int(0.10 * n_groups)

train_groups = np.sort(shuffled_groups[:n_train])
val_groups = np.sort(
    shuffled_groups[n_train:n_train + n_val]
)
test_groups = np.sort(
    shuffled_groups[n_train + n_val:]
)

group_positions = np.searchsorted(
    unique_groups, all_groups
)

group_split_codes = np.full(
    len(unique_groups), 2, dtype=np.uint8
)

train_positions = np.searchsorted(
    unique_groups, train_groups
)
val_positions = np.searchsorted(
    unique_groups, val_groups
)

group_split_codes[train_positions] = 0
group_split_codes[val_positions] = 1

split_codes = group_split_codes[group_positions]



train_indices = np.flatnonzero(split_codes == 0)
val_indices = np.flatnonzero(split_codes == 1)
test_indices = np.flatnonzero(split_codes == 2)


assert len(np.intersect1d(train_groups, val_groups)) == 0
assert len(np.intersect1d(train_groups, test_groups)) == 0
assert len(np.intersect1d(val_groups, test_groups)) == 0

assert (
    len(train_indices)
    + len(val_indices)
    + len(test_indices)
    == 3_000_000
)

for group_position in range(len(unique_groups)):
    pass  


np.save(OUTPUT_DIR / "train_indices.npy", train_indices)
np.save(OUTPUT_DIR / "val_indices.npy", val_indices)
np.save(OUTPUT_DIR / "test_indices.npy", test_indices)
np.save(OUTPUT_DIR / "split_codes.npy", split_codes)

np.savez(
    OUTPUT_DIR / "group_split_ids.npz",
    train_groups=train_groups,
    val_groups=val_groups,
    test_groups=test_groups,
)


print("\n=== GROUP-AWARE SPLIT SUMMARY ===")

print(
    "Train:",
    len(train_groups), "groups,",
    len(train_indices), "spectra"
)

print(
    "Validation:",
    len(val_groups), "groups,",
    len(val_indices), "spectra"
)

print(
    "Test:",
    len(test_groups), "groups,",
    len(test_indices), "spectra"
)

print("\nGroup leakage: NONE")
print("Random seed:", SEED)
print("Saved outputs to:", OUTPUT_DIR)
