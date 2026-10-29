
from pathlib import Path
import json

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

with open(ARTIFACTS_DIR / "component_id_mapping.json") as f:
    canonical_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with open(ARTIFACTS_DIR / "active_class_mapping.json") as f:
    class_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

split_codes = np.load(
    ARTIFACTS_DIR / "split_codes.npy"
)

C = len(class_mapping)

max_id = max(canonical_mapping)
id_to_class = np.full(max_id + 1, -1, dtype=np.int32)

for original_id, canonical_id in canonical_mapping.items():
    id_to_class[original_id] = class_mapping[canonical_id]

train_entry_counts = np.zeros(C, dtype=np.int64)
val_entry_counts = np.zeros(C, dtype=np.int64)
test_entry_counts = np.zeros(C, dtype=np.int64)

duplicate_spectra_first_file = 0

files = sorted(DATA_DIR.glob("spectra_*.h5"))

for file_idx, file_path in enumerate(files):

    with h5py.File(file_path, "r") as f:
        ids = f["component_ids"][:]
        offsets = f["component_offsets"][:]

    classes = id_to_class[ids]

    if np.any(classes < 0):
        raise ValueError("Unmapped component ID found")

    row_ids = np.repeat(
        np.arange(30000),
        np.diff(offsets)
    )

    global_start = file_idx * 30000
    file_split_codes = split_codes[
        global_start:global_start + 30000
    ]

    entry_splits = file_split_codes[row_ids]

    for split_code, counter in [
        (0, train_entry_counts),
        (1, val_entry_counts),
        (2, test_entry_counts),
    ]:
        selected = classes[entry_splits == split_code]
        counter += np.bincount(
            selected, minlength=C
        )

    if file_idx == 0:
        for row in range(30000):
            row_classes = classes[
                offsets[row]:offsets[row + 1]
            ]
            if len(np.unique(row_classes)) < len(row_classes):
                duplicate_spectra_first_file += 1

    print(f"[{file_idx + 1}/100] {file_path.name}")

print("\n=== CLASS DISTRIBUTION SUMMARY ===")
print("Total classes:", C)

print(
    "Classes present in training:",
    np.count_nonzero(train_entry_counts)
)

print(
    "Classes absent from training:",
    np.count_nonzero(train_entry_counts == 0)
)

print(
    "Classes with fewer than 10 training entries:",
    np.count_nonzero(
        (train_entry_counts > 0)
        & (train_entry_counts < 10)
    )
)

print(
    "Classes with fewer than 100 training entries:",
    np.count_nonzero(
        (train_entry_counts > 0)
        & (train_entry_counts < 100)
    )
)

print(
    "Classes present in test but absent from train:",
    np.count_nonzero(
        (train_entry_counts == 0)
        & (test_entry_counts > 0)
    )
)

print(
    "Spectra with duplicate canonical IDs "
    "in first file:",
    duplicate_spectra_first_file
)

np.savez(
    ARTIFACTS_DIR / "class_distribution_by_split.npz",
    train=train_entry_counts,
    val=val_entry_counts,
    test=test_entry_counts,
)

print(
    "\nSaved: artifacts/class_distribution_by_split.npz"
)
