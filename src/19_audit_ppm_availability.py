
from pathlib import Path
import json
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

with open(ARTIFACTS / "component_id_mapping.json",
          encoding="utf-8") as f:
    canonical = {int(k): int(v)
                 for k, v in json.load(f).items()}

with open(ARTIFACTS / "active_class_mapping.json",
          encoding="utf-8") as f:
    class_map = {int(k): int(v)
                 for k, v in json.load(f).items()}

C = len(class_map)

lookup = np.full(max(canonical) + 1, -1, dtype=np.int32)
for original_id, canonical_id in canonical.items():
    lookup[original_id] = class_map[canonical_id]

split_codes = np.load(ARTIFACTS / "split_codes.npy")
selected = np.load(
    ARTIFACTS / "downstream_train_indices.npy"
)

# Boolean lookup for the selected 10% sample indices.
selected_mask = np.zeros(len(split_codes), dtype=bool)
selected_mask[selected] = True

train_total = np.zeros(C, dtype=np.int64)
train_known = np.zeros(C, dtype=np.int64)
subset_total = np.zeros(C, dtype=np.int64)
subset_known = np.zeros(C, dtype=np.int64)

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

for file_idx, path in enumerate(files):
    with h5py.File(path, "r") as f:
        offsets = f["component_offsets"][:]
        ids = f["component_ids"][:]
        ppm = f["component_ppm"][:]

    classes = lookup[ids]
    assert np.all(classes >= 0)

    rows = np.repeat(
        np.arange(30000, dtype=np.int32),
        np.diff(offsets)
    )

    start = file_idx * 30000
    global_rows = start + rows

    is_train = split_codes[global_rows] == 0
    is_subset = selected_mask[global_rows]
    has_ppm = np.isfinite(ppm) & (ppm >= 0)

    train_total += np.bincount(
        classes[is_train], minlength=C
    )
    train_known += np.bincount(
        classes[is_train & has_ppm], minlength=C
    )
    subset_total += np.bincount(
        classes[is_subset], minlength=C
    )
    subset_known += np.bincount(
        classes[is_subset & has_ppm], minlength=C
    )

    print(f"[{file_idx + 1}/100] {path.name}")

print("\n=== PPM AVAILABILITY SUMMARY ===")
print("Total classes:", C)
print("Classes present in full training:",
      np.count_nonzero(train_total))
print("Classes with any ppm in full training:",
      np.count_nonzero(train_known))
print("Classes with no ppm in full training:",
      np.count_nonzero(
          (train_total > 0) & (train_known == 0)
      ))
print("Classes with any ppm in labeled 10%:",
      np.count_nonzero(subset_known))
print("Classes with ppm in full train "
      "but none in 10%:",
      np.count_nonzero(
          (train_known > 0) & (subset_known == 0)
      ))

print("\nPPM entries in full training:",
      int(train_known.sum()))
print("PPM entries in 10% subset:",
      int(subset_known.sum()))

np.savez(
    ARTIFACTS / "ppm_availability.npz",
    train_total=train_total,
    train_known=train_known,
    subset_total=subset_total,
    subset_known=subset_known
)

print("\nSaved: artifacts/ppm_availability.npz")
