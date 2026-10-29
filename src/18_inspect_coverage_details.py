
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

with np.load(
    ARTIFACTS / "labeled_subset_coverage.npz"
) as data:
    classification_counts = data["classification_counts"]
    regression_counts = data["regression_counts"]

with open(
    ARTIFACTS / "active_class_mapping.json",
    encoding="utf-8"
) as f:
    class_mapping = {
        int(k): int(v)
        for k, v in json.load(f).items()
    }

index_to_id = {
    class_index: canonical_id
    for canonical_id, class_index in class_mapping.items()
}

with open(
    ROOT / "components.json",
    encoding="utf-8"
) as f:
    components = json.load(f)

components_by_id = {
    int(item["component_id"]): item
    for item in components
}

def class_name(class_index):
    canonical_id = index_to_id[int(class_index)]
    record = components_by_id[canonical_id]

    return record.get("name", "Unknown")


missing = np.flatnonzero(classification_counts == 0)

print("=== MISSING CLASSIFICATION CLASSES ===")
print("Total:", len(missing))

for idx in missing:
    print(
        f"Class {idx} | "
        f"Canonical ID {index_to_id[int(idx)]} | "
        f"{class_name(idx)}"
    )


regression_present = np.flatnonzero(
    regression_counts > 0
)

print("\n=== REGRESSION COVERAGE ===")
print("Classes with regression examples:",
      len(regression_present))

print("\nTop 20 regression classes:")

top_regression = sorted(
    regression_present,
    key=lambda i: regression_counts[i],
    reverse=True
)[:20]

for idx in top_regression:
    print(
        f"Class {idx} | "
        f"{class_name(idx)} | "
        f"{regression_counts[idx]} spectra"
    )

print("\n=== COVERAGE COMPARISON ===")

print(
    "Classes with classification examples:",
    np.count_nonzero(classification_counts)
)

print(
    "Classes with regression examples:",
    np.count_nonzero(regression_counts)
)

print(
    "Classes with classification but no regression examples:",
    np.count_nonzero(
        (classification_counts > 0)
        & (regression_counts == 0)
    )
)
