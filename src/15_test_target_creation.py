
from pathlib import Path
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

with open(
    ARTIFACTS / "component_id_mapping.json",
    encoding="utf-8"
) as f:
    canonical_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with open(
    ARTIFACTS / "active_class_mapping.json",
    encoding="utf-8"
) as f:
    class_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with np.load(
    ARTIFACTS / "regression_eligibility_audit.npz"
) as audit:
    regression_eligible = audit[
        "regression_eligible"
    ].copy()

C = len(class_mapping)
print("Number of classes:", C)


def create_targets(original_ids, ppm_values, eligible):
    """
    Create classification and concentration targets
    for one spectrum.

    Regression is generated only for eligible spectra.
    """

    classification = np.zeros(C, dtype=np.float32)

    regression = np.full(C, np.nan, dtype=np.float32)

    if eligible:
        regression[:] = 0.0

    seen_classes = set()

    for original_id, ppm in zip(original_ids, ppm_values):

        original_id = int(original_id)

        canonical_id = canonical_mapping[original_id]
        class_index = class_mapping[canonical_id]

        classification[class_index] = 1.0

        if eligible:
            if class_index in seen_classes:
                raise ValueError(
                    "Duplicate canonical class in eligible sample"
                )

            if not np.isfinite(ppm) or ppm < 0:
                raise ValueError("Invalid ppm target")

            regression[class_index] = float(ppm)

        seen_classes.add(class_index)

    return classification, regression


with h5py.File(
    ROOT / "data" / "spectra_0000.h5", "r"
) as f:

    offsets = f["component_offsets"][:]

    for row in [0, 1, 2, 3, 4, 90]:

        start = int(offsets[row])
        end = int(offsets[row + 1])

        original_ids = f["component_ids"][start:end]
        ppm_values = f["component_ppm"][start:end]

        eligible = bool(regression_eligible[row])

        detection, concentration = create_targets(
            original_ids,
            ppm_values,
            eligible
        )

        active_indices = np.flatnonzero(detection)

        print(f"\n=== SPECTRUM {row} ===")
        print("Original IDs:", original_ids.tolist())
        print("Detection shape:", detection.shape)
        print("Present classes:", active_indices.tolist())
        print("Number of detected classes:", len(active_indices))
        print("Regression eligible:", eligible)

        if eligible:
            print(
                "Nonzero concentrations:",
                concentration[active_indices].tolist()
            )
        else:
            print("Concentration target: unavailable")

        assert detection.shape == (C,)
        assert concentration.shape == (C,)
        assert int(detection.sum()) == len(active_indices)

print("\nAll target creation checks passed!")
