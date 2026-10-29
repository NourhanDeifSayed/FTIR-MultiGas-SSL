
from pathlib import Path
import json
from collections import Counter

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

with open(
    ARTIFACTS_DIR / "component_id_mapping.json",
    "r",
    encoding="utf-8"
) as f:
    mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

files = sorted(DATA_DIR.glob("spectra_*.h5"))
assert len(files) == 100

original_counts = Counter()
canonical_counts = Counter()

for index, file_path in enumerate(files):

    with h5py.File(file_path, "r") as f:
        ids = f["component_ids"][:]

    unique_ids, counts = np.unique(
        ids, return_counts=True
    )

    for component_id, count in zip(
        unique_ids.tolist(),
        counts.tolist()
    ):
        component_id = int(component_id)

        if component_id not in mapping:
            raise ValueError(
                f"Unknown component ID: {component_id}"
            )

        canonical_id = mapping[component_id]

        original_counts[component_id] += count
        canonical_counts[canonical_id] += count

    print(f"[{index + 1}/100] {file_path.name}")

active_ids = sorted(canonical_counts)

active_class_mapping = {
    canonical_id: class_index
    for class_index, canonical_id
    in enumerate(active_ids)
}

with open(
    ARTIFACTS_DIR / "active_class_mapping.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            str(k): int(v)
            for k, v in active_class_mapping.items()
        },
        f,
        indent=2
    )


with open(
    ARTIFACTS_DIR / "component_frequency.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            str(k): int(v)
            for k, v in canonical_counts.items()
        },
        f,
        indent=2
    )

print("\n=== ACTIVE COMPONENT SUMMARY ===")

print("Unique original IDs:", len(original_counts))
print("Unique canonical classes:", len(canonical_counts))
print("Total component entries:", sum(original_counts.values()))
print(
    "Reference classes never observed:",
    len(set(mapping.values()) - set(active_ids))
)

print("\nTop 10 most frequent canonical IDs:")
for component_id, count in canonical_counts.most_common(10):
    print(f"Canonical ID {component_id}: {count} entries")

print("\nSaved: artifacts/active_class_mapping.json")
print("Saved: artifacts/component_frequency.json")
