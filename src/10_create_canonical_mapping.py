
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
COMPONENTS_FILE = PROJECT_DIR / "components.json"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
    components = json.load(f)

components_by_id = {
    int(item["component_id"]): item
    for item in components
}

assert len(components_by_id) == len(components)

approved_pairs = [
    (7, 86),    # Acetylene
    (9, 73),    # Ammonia
    (15, 64),   # Carbon dioxide
    (16, 102),  # Carbon disulfide
    (17, 67),   # Carbon monoxide
    (19, 80),   # Carbonyl sulfide
    (26, 87),   # Ethane
    (31, 93),   # Ethylene
    (36, 84),   # Hydrogen cyanide
    (37, 90),   # Hydrogen sulfide
    (48, 70),   # Nitric oxide
    (49, 72),   # Nitrogen dioxide
    (59, 71),   # Sulfur dioxide
    (62, 63),   # Water
]


mapping = {
    component_id: component_id
    for component_id in components_by_id
}

for first_id, second_id in approved_pairs:
    first = components_by_id[first_id]
    second = components_by_id[second_id]

    assert first["name"].strip().casefold() == (
        second["name"].strip().casefold()
    )

    assert first.get("formula") == second.get("formula")

    assert first.get("cas") == second.get("cas")

    canonical_id = min(first_id, second_id)

    mapping[first_id] = canonical_id
    mapping[second_id] = canonical_id

mapping_path = ARTIFACTS_DIR / "component_id_mapping.json"

with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(
        {str(k): int(v) for k, v in sorted(mapping.items())},
        f,
        indent=2
    )

canonical_ids = sorted(set(mapping.values()))

canonical_to_class = {
    canonical_id: index
    for index, canonical_id in enumerate(canonical_ids)
}

class_path = ARTIFACTS_DIR / "canonical_class_indices.json"

with open(class_path, "w", encoding="utf-8") as f:
    json.dump(
        {str(k): v for k, v in canonical_to_class.items()},
        f,
        indent=2
    )

print("=== CANONICAL MAPPING SUMMARY ===")
print("Original component IDs:", len(components_by_id))
print("Approved merged pairs:", len(approved_pairs))
print("Canonical classes in reference:", len(canonical_ids))

print("\nMapping examples:")
for component_id in [15, 64, 9, 73, 62, 63]:
    canonical_id = mapping[component_id]
    class_index = canonical_to_class[canonical_id]

    print(
        f"ID {component_id} -> "
        f"Canonical ID {canonical_id} -> "
        f"Class Index {class_index}"
    )

print("\nSaved:", mapping_path)
print("Saved:", class_path)
