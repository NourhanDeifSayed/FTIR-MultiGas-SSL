
from pathlib import Path
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent
COMPONENTS_FILE = PROJECT_DIR / "components.json"

with open(COMPONENTS_FILE, "r", encoding="utf-8") as f:
    components = json.load(f)

print("=== COMPONENTS.JSON STRUCTURE ===")
print("Total component records:", len(components))

components_by_id = {
    int(item["component_id"]): item
    for item in components
}

print("Unique component IDs:", len(components_by_id))

if len(components_by_id) != len(components):
    raise ValueError(
        "Duplicate component_id records found in components.json"
    )

print("\n=== FIRST SPECTRUM COMPONENTS ===")

target_ids = [41, 51, 46, 15, 62]

for component_id in target_ids:

    component = components_by_id.get(component_id)

    print(f"\nComponent ID: {component_id}")

    if component is None:
        print("Not found in components.json")
        continue

    print("Name:", component.get("name"))
    print("Formula:", component.get("formula"))
    print("CAS:", component.get("cas", "Not provided"))

print("\n=== DUPLICATE CHEMICAL NAMES ===")

names_to_ids = {}

for component in components:
    name = component.get("name")

    if name:
        names_to_ids.setdefault(
            name.strip().lower(), []
        ).append(component["component_id"])

duplicate_names = {
    name: ids
    for name, ids in names_to_ids.items()
    if len(ids) > 1
}

print(
    "Number of names appearing under multiple IDs:",
    len(duplicate_names)
)

print("\nFirst 10 duplicate-name examples:")

for name, ids in list(duplicate_names.items())[:10]:
    print(name, "->", ids)
