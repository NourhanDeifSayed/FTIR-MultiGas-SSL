
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

SEED = 42
rng = np.random.default_rng(SEED)

with np.load(ARTIFACTS / "group_split_ids.npz") as saved:
    train_groups = saved["train_groups"]

train_indices = np.load(
    ARTIFACTS / "train_indices.npy"
)

n_selected = int(len(train_groups) * 0.10)

selected_groups = np.sort(
    rng.choice(
        train_groups,
        size=n_selected,
        replace=False
    )
)

selected_indices = []

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

for file_idx, path in enumerate(files):
    with h5py.File(path, "r") as f:
        groups = f["group_id"][:]

    selected_rows = np.flatnonzero(
        np.isin(groups, selected_groups)
    )

    global_indices = file_idx * 30000 + selected_rows

    selected_indices.append(global_indices)

selected_indices = np.concatenate(selected_indices)
selected_indices.sort()

assert len(selected_groups) == 80_000
assert len(selected_indices) == 240_000

assert np.all(
    np.isin(selected_indices, train_indices)
)

# Save the 10% labeled subset.
np.save(
    ARTIFACTS / "downstream_train_indices.npy",
    selected_indices
)

np.save(
    ARTIFACTS / "downstream_train_groups.npy",
    selected_groups
)

with np.load(
    ARTIFACTS / "regression_eligibility_audit.npz"
) as audit:
    eligible = audit["regression_eligible"]

regression_indices = selected_indices[
    eligible[selected_indices]
]

np.save(
    ARTIFACTS / "downstream_regression_indices.npy",
    regression_indices
)

print("\n=== LABELED SUBSET SUMMARY ===")
print("Selected groups:", len(selected_groups))
print("Selected spectra:", len(selected_indices))
print("Percentage of training data:",
      round(100 * len(selected_indices) / len(train_indices), 2))
print("Classification samples:", len(selected_indices))
print("Regression-eligible samples:", len(regression_indices))

print("\nSaved:")
print("downstream_train_indices.npy")
print("downstream_train_groups.npy")
print("downstream_regression_indices.npy")
