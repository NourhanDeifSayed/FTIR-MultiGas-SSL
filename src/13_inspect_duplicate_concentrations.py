
from pathlib import Path
from collections import defaultdict, Counter
import json

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

with open(
    PROJECT_DIR / "artifacts" / "component_id_mapping.json",
    encoding="utf-8"
) as f:
    mapping = {int(k): int(v) for k, v in json.load(f).items()}

with h5py.File(
    PROJECT_DIR / "data" / "spectra_0000.h5", "r"
) as f:
    offsets = f["component_offsets"][:]
    ids = f["component_ids"][:]
    ppm = f["component_ppm"][:]
    groups = f["group_id"][:]

duplicate_spectra = 0
equal_ppm = 0
different_ppm = 0
missing_ppm = 0
pair_counts = Counter()

examples = []

for row in range(len(groups)):
    start = int(offsets[row])
    end = int(offsets[row + 1])

    occurrences = defaultdict(list)

    for j in range(start, end):
        original_id = int(ids[j])
        canonical_id = mapping[original_id]

        occurrences[canonical_id].append(
            (original_id, float(ppm[j]))
        )

    repeated = {
        cid: values
        for cid, values in occurrences.items()
        if len(values) > 1
    }

    if not repeated:
        continue

    duplicate_spectra += 1

    for canonical_id, values in repeated.items():
        original_ids = tuple(sorted(v[0] for v in values))
        pair_counts[original_ids] += 1

        concentrations = np.array(
            [v[1] for v in values],
            dtype=np.float64
        )

        if not np.all(np.isfinite(concentrations)):
            status = "missing_ppm"
            missing_ppm += 1

        elif np.allclose(
            concentrations,
            concentrations[0],
            rtol=1e-5,
            atol=1e-5
        ):
            status = "equal_ppm"
            equal_ppm += 1

        else:
            status = "different_ppm"
            different_ppm += 1

        if len(examples) < 12:
            examples.append({
                "spectrum": row,
                "group_id": int(groups[row]),
                "canonical_id": canonical_id,
                "original_ids": original_ids,
                "ppm_values": concentrations.tolist(),
                "status": status
            })

print("=== DUPLICATE CONCENTRATION SUMMARY ===")
print("Spectra with duplicates:", duplicate_spectra)
print("Duplicate occurrences with equal ppm:", equal_ppm)
print("Duplicate occurrences with different ppm:", different_ppm)
print("Duplicate occurrences with missing ppm:", missing_ppm)

print("\nMost common original ID combinations:")
for ids_pair, count in pair_counts.most_common(10):
    print(ids_pair, "->", count)

print("\nExamples:")
for example in examples:
    print(example)
