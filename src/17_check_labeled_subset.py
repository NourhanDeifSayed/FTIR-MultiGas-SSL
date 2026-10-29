
from pathlib import Path
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

with open(ARTIFACTS / "component_id_mapping.json",
          encoding="utf-8") as f:
    canonical_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with open(ARTIFACTS / "active_class_mapping.json",
          encoding="utf-8") as f:
    class_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

C = len(class_mapping)

lookup = np.full(
    max(canonical_mapping) + 1, -1, dtype=np.int32
)

for original_id, canonical_id in canonical_mapping.items():
    lookup[original_id] = class_mapping[canonical_id]

selected_indices = np.load(
    ARTIFACTS / "downstream_train_indices.npy"
)

regression_indices = np.load(
    ARTIFACTS / "downstream_regression_indices.npy"
)

classification_counts = np.zeros(C, dtype=np.int64)
regression_counts = np.zeros(C, dtype=np.int64)

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

for file_idx, path in enumerate(files):

    global_start = file_idx * 30000
    global_end = global_start + 30000

    selected_local = (
        selected_indices[
            (selected_indices >= global_start)
            & (selected_indices < global_end)
        ] - global_start
    )

    regression_local = (
        regression_indices[
            (regression_indices >= global_start)
            & (regression_indices < global_end)
        ] - global_start
    )

    with h5py.File(path, "r") as f:
        offsets = f["component_offsets"][:]
        original_ids = f["component_ids"][:]

    classes = lookup[original_ids]

    if np.any(classes < 0):
        raise ValueError(f"Unmapped component in {path.name}")

    row_ids = np.repeat(
        np.arange(30000, dtype=np.int32),
        np.diff(offsets)
    )

    unique_pairs = np.unique(
        row_ids.astype(np.int64) * C + classes
    )

    unique_rows = unique_pairs // C
    unique_classes = (unique_pairs % C).astype(np.int32)

    selected_mask = np.zeros(30000, dtype=bool)
    selected_mask[selected_local] = True

    regression_mask = np.zeros(30000, dtype=bool)
    regression_mask[regression_local] = True

    classification_counts += np.bincount(
        unique_classes[selected_mask[unique_rows]],
        minlength=C
    )

    regression_counts += np.bincount(
        unique_classes[regression_mask[unique_rows]],
        minlength=C
    )

    print(f"[{file_idx + 1}/100] {path.name}")

present = classification_counts > 0

print("\n=== LABELED SUBSET COVERAGE ===")
print("Total classes:", C)
print("Classes covered:", np.count_nonzero(present))
print("Classes missing:", np.count_nonzero(~present))
print(
    "Classes with 1-9 labeled spectra:",
    np.count_nonzero(
        (classification_counts > 0)
        & (classification_counts < 10)
    )
)
print(
    "Classes with 10-99 labeled spectra:",
    np.count_nonzero(
        (classification_counts >= 10)
        & (classification_counts < 100)
    )
)
print(
    "Classes covered in regression subset:",
    np.count_nonzero(regression_counts)
)

print("\n10 rarest classes in labeled subset:")
for idx in np.argsort(classification_counts)[:10]:
    print(
        f"Class {idx}: "
        f"{classification_counts[idx]} spectra"
    )

np.savez(
    ARTIFACTS / "labeled_subset_coverage.npz",
    classification_counts=classification_counts,
    regression_counts=regression_counts
)

print("\nSaved: artifacts/labeled_subset_coverage.npz")
