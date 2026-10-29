
from pathlib import Path
from collections import defaultdict
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
COMPONENTS_FILE = PROJECT_DIR / "components.json"

with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
    components = json.load(f)

name_groups = defaultdict(list)

for item in components:
    name = item.get("name", "").strip().casefold()
    if name:
        name_groups[name].append(item)

duplicate_groups = {
    name: records
    for name, records in name_groups.items()
    if len(records) > 1
}

print("=== ALL DUPLICATE CHEMICAL NAMES ===")
print("Duplicate name groups:", len(duplicate_groups))

for name, records in sorted(duplicate_groups.items()):
    print(f"\nChemical name: {name}")

    for record in records:
        print(
            " ID:", record["component_id"],
            "| Formula:", record.get("formula"),
            "| CAS:", record.get("cas", "Not provided")
        )

output_path = (
    PROJECT_DIR / "artifacts" / "duplicate_components_review.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        duplicate_groups,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nReview file saved:", output_path)
