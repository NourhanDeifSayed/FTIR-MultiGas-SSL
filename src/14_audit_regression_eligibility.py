
from pathlib import Path
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

with open(
    ARTIFACTS / "component_id_mapping.json",
    encoding="utf-8"
) as f:
    mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

split_codes = np.load(
    ARTIFACTS / "split_codes.npy"
)

lookup = np.full(max(mapping) + 1, -1, dtype=np.int32)

for original_id, canonical_id in mapping.items():
    lookup[original_id] = canonical_id

missing_ppm = np.zeros(3_000_000, dtype=bool)
duplicate_canonical = np.zeros(3_000_000, dtype=bool)

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

for file_index, path in enumerate(files):

    with h5py.File(path, "r") as f:
        offsets = f["component_offsets"][:]
        ids = f["component_ids"][:]
        ppm = f["component_ppm"][:]

    n_rows = len(offsets) - 1
    lengths = np.diff(offsets)

    rows = np.repeat(np.arange(n_rows), lengths)

    bad_ppm_rows = np.bincount(
        rows[~np.isfinite(ppm)],
        minlength=n_rows
    ) > 0

    canonical_ids = lookup[ids]

    if np.any(canonical_ids < 0):
        raise ValueError(f"Unknown component ID in {path.name}")

    order = np.lexsort((canonical_ids, rows))
    sorted_rows = rows[order]
    sorted_ids = canonical_ids[order]

    same_pair = (
        (sorted_rows[1:] == sorted_rows[:-1])
        & (sorted_ids[1:] == sorted_ids[:-1])
    )

    duplicate_rows = np.zeros(n_rows, dtype=bool)
    duplicate_rows[sorted_rows[1:][same_pair]] = True

    start = file_index * 30000
    end = start + n_rows

    missing_ppm[start:end] = bad_ppm_rows
    duplicate_canonical[start:end] = duplicate_rows

    print(f"[{file_index + 1}/100] {path.name}")


regression_eligible = (
    ~missing_ppm & ~duplicate_canonical
)

np.savez_compressed(
    ARTIFACTS / "regression_eligibility_audit.npz",
    missing_ppm=missing_ppm,
    duplicate_canonical=duplicate_canonical,
    regression_eligible=regression_eligible
)

print("\n=== REGRESSION ELIGIBILITY SUMMARY ===")
print("Total spectra:", len(regression_eligible))
print("Spectra with missing ppm:", missing_ppm.sum())
print("Spectra with canonical duplicates:",
      duplicate_canonical.sum())
print("Eligible spectra:", regression_eligible.sum())
print("Eligible percentage:",
      round(100 * regression_eligible.mean(), 2))

for code, name in [(0, "Train"), (1, "Validation"), (2, "Test")]:
    selected = split_codes == code
    print(
        f"{name}: "
        f"{regression_eligible[selected].sum()} / "
        f"{selected.sum()} eligible"
    )

print("\nSaved: artifacts/regression_eligibility_audit.npz")
