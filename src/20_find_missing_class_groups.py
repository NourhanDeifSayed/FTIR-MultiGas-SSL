
from pathlib import Path
from collections import defaultdict
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

with np.load(ARTIFACTS / "labeled_subset_coverage.npz") as f:
    counts = f["classification_counts"]

missing_classes = np.flatnonzero(counts == 0)

with open(ARTIFACTS / "component_id_mapping.json",
          encoding="utf-8") as f:
    canonical = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with open(ARTIFACTS / "active_class_mapping.json",
          encoding="utf-8") as f:
    class_map = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

with open(ROOT / "components.json",
          encoding="utf-8") as f:
    components = {
        int(x["component_id"]): x
        for x in json.load(f)
    }

index_to_id = {v: k for k, v in class_map.items()}

lookup = np.full(max(canonical) + 1, -1, dtype=np.int32)
for original_id, canonical_id in canonical.items():
    lookup[original_id] = class_map[canonical_id]

with np.load(ARTIFACTS / "group_split_ids.npz") as f:
    train_groups = f["train_groups"]

selected_groups = np.load(
    ARTIFACTS / "downstream_train_groups.npy"
)

train_group_set = set(map(int, train_groups))
selected_group_set = set(map(int, selected_groups))
missing_set = set(map(int, missing_classes))

candidate_groups = defaultdict(set)

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

for file_idx, path in enumerate(files):

    with h5py.File(path, "r") as f:
        offsets = f["component_offsets"][:]
        ids = f["component_ids"][:]
        groups = f["group_id"][:]

    mapped_classes = lookup[ids]

    target_entries = np.flatnonzero(
        np.isin(mapped_classes, missing_classes)
    )

    if len(target_entries):
        rows = np.searchsorted(
            offsets[1:],
            target_entries,
            side="right"
        )

        for class_idx, group_id in zip(
            mapped_classes[target_entries],
            groups[rows]
        ):
            class_idx = int(class_idx)
            group_id = int(group_id)

            if group_id in train_group_set:
                candidate_groups[class_idx].add(group_id)

    print(f"[{file_idx + 1}/100] {path.name}")

print("\n=== MISSING CLASS GROUP COVERAGE ===")

report = {}

for class_idx in missing_classes:
    class_idx = int(class_idx)
    canonical_id = index_to_id[class_idx]
    name = components[canonical_id]["name"]

    candidates = sorted(candidate_groups[class_idx])
    already_selected = sum(
        g in selected_group_set for g in candidates
    )

    report[str(class_idx)] = {
        "name": name,
        "canonical_id": int(canonical_id),
        "training_group_count": len(candidates),
        "candidate_group_ids": candidates
    }

    print(
        f"Class {class_idx} | {name} | "
        f"Train groups: {len(candidates)} | "
        f"Already selected: {already_selected}"
    )

output_path = ARTIFACTS / "missing_class_group_candidates.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nMissing classes:", len(missing_set))
print(
    "Missing classes with no training candidates:",
    sum(len(v) == 0 for v in candidate_groups.values())
    if len(candidate_groups) == len(missing_set)
    else sum(
        len(candidate_groups[c]) == 0
        for c in missing_set
    )
)
print("Saved:", output_path)
