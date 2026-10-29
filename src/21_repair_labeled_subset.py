
from pathlib import Path
from collections import Counter, defaultdict
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
DATA = ROOT / "data"

old_groups = np.load(ART / "downstream_train_groups.npy")
old_groups_set = set(map(int, old_groups))

with np.load(ART / "group_split_ids.npz") as f:
    train_groups = set(map(int, f["train_groups"]))

with np.load(ART / "labeled_subset_coverage.npz") as f:
    counts = f["classification_counts"]

missing_classes = set(map(int, np.flatnonzero(counts == 0)))

with open(
    ART / "missing_class_group_candidates.json",
    encoding="utf-8"
) as f:
    candidate_report = json.load(f)

with open(
    ART / "component_id_mapping.json",
    encoding="utf-8"
) as f:
    canonical = {
        int(k): int(v) for k, v in json.load(f).items()
    }

with open(
    ART / "active_class_mapping.json",
    encoding="utf-8"
) as f:
    class_map = {
        int(k): int(v) for k, v in json.load(f).items()
    }

C = len(class_map)

lookup = np.full(max(canonical) + 1, -1, dtype=np.int32)

for original_id, canonical_id in canonical.items():
    lookup[original_id] = class_map[canonical_id]

candidate_groups = set()

for record in candidate_report.values():
    candidate_groups.update(
        map(int, record["candidate_group_ids"])
    )

assert candidate_groups <= train_groups
assert old_groups_set <= train_groups

relevant_groups = np.array(
    sorted(old_groups_set | candidate_groups),
    dtype=np.int64
)

group_classes = defaultdict(set)

files = sorted(DATA.glob("spectra_*.h5"))
assert len(files) == 100

print("Reading group labels...")

for file_idx, path in enumerate(files):

    with h5py.File(path, "r") as f:
        groups = f["group_id"][:]
        offsets = f["component_offsets"][:]
        ids = f["component_ids"][:]

    classes = lookup[ids]
    assert np.all(classes >= 0)

    entry_groups = np.repeat(
        groups, np.diff(offsets)
    )

    relevant = np.isin(
        entry_groups, relevant_groups
    )

    pair_codes = np.unique(
        entry_groups[relevant].astype(np.int64) * C
        + classes[relevant]
    )

    for pair_code in pair_codes:
        group_id = int(pair_code // C)
        class_idx = int(pair_code % C)
        group_classes[group_id].add(class_idx)

    print(f"[{file_idx + 1}/100] {path.name}")

group_class_counts = Counter()

for group_id in old_groups_set:
    group_class_counts.update(group_classes[group_id])

print("\n=== BEFORE REPAIR ===")
print("Selected groups:", len(old_groups_set))
print("Missing classes:", len(missing_classes))


new_groups = set(old_groups_set)
uncovered = set(missing_classes)
added = []

while uncovered:

    best_group = None
    best_coverage = set()

    for group_id in sorted(candidate_groups - new_groups):
        coverage = group_classes[group_id] & uncovered

        if len(coverage) > len(best_coverage):
            best_group = group_id
            best_coverage = coverage

    if best_group is None:
        raise RuntimeError(
            f"Could not cover these classes: {uncovered}"
        )

    new_groups.add(best_group)
    added.append(best_group)

    group_class_counts.update(
        group_classes[best_group]
    )

    uncovered -= best_coverage

    print(
        f"Added group {best_group} | "
        f"New classes covered: {len(best_coverage)}"
    )


removed = []

for group_id in sorted(old_groups_set):

    if len(removed) == len(added):
        break

    present_classes = group_classes[group_id]

    safe_to_remove = all(
        group_class_counts[class_idx] > 1
        for class_idx in present_classes
    )

    if safe_to_remove:
        new_groups.remove(group_id)
        removed.append(group_id)

        for class_idx in present_classes:
            group_class_counts[class_idx] -= 1

if len(removed) != len(added):
    raise RuntimeError(
        "Not enough safe groups to remove"
    )

assert len(new_groups) == len(old_groups_set) == 80_000
assert new_groups <= train_groups
assert np.all([
    group_class_counts[class_idx] > 0
    for class_idx in range(C)
])


new_groups_array = np.array(
    sorted(new_groups), dtype=np.int64
)

new_indices_parts = []

for file_idx, path in enumerate(files):
    with h5py.File(path, "r") as f:
        groups = f["group_id"][:]

    local_rows = np.flatnonzero(
        np.isin(groups, new_groups_array)
    )

    global_rows = file_idx * 30000 + local_rows
    new_indices_parts.append(global_rows)

new_indices = np.sort(
    np.concatenate(new_indices_parts)
)

assert len(new_indices) == 240_000

split_codes = np.load(ART / "split_codes.npy")
assert np.all(split_codes[new_indices] == 0)

with np.load(
    ART / "regression_eligibility_audit.npz"
) as f:
    eligible = f["regression_eligible"]

new_regression_indices = new_indices[
    eligible[new_indices]
]

np.save(
    ART / "downstream_train_groups_v2.npy",
    new_groups_array
)

np.save(
    ART / "downstream_train_indices_v2.npy",
    new_indices
)

np.save(
    ART / "downstream_regression_indices_v2.npy",
    new_regression_indices
)

print("\n=== REPAIRED SUBSET SUMMARY ===")
print("Groups added:", len(added))
print("Groups removed:", len(removed))
print("Final groups:", len(new_groups))
print("Final spectra:", len(new_indices))
print("Classes covered:", sum(
    group_class_counts[c] > 0 for c in range(C)
))
print("Missing classes:", sum(
    group_class_counts[c] == 0 for c in range(C)
))
print(
    "Regression-eligible spectra:",
    len(new_regression_indices)
)

print("\nSaved three new files ending in _v2.npy")
print("Original selection files were not changed.")
