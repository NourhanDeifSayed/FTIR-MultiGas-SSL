
from pathlib import Path
import json

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

files = sorted(DATA_DIR.glob("spectra_*.h5"))

print(f"Found {len(files)} HDF5 files")

if len(files) != 100:
    raise ValueError("Expected exactly 100 HDF5 files")

required_keys = {
    "spectra",
    "valid_mask",
    "wavenumbers",
    "group_id",
    "component_offsets",
    "component_ids",
    "component_ppm",
    "optical_depth_scale",
}

reference_wavenumbers = None
all_group_ids = set()
total_spectra = 0
total_valid_points = 0
total_points = 0
errors = []
file_reports = []

for index, file_path in enumerate(files):

    print(
        f"[{index + 1}/100] Checking {file_path.name}"
    )

    try:
        with h5py.File(file_path, "r") as f:

            missing = required_keys - set(f.keys())
            if missing:
                raise ValueError(
                    f"Missing datasets: {missing}"
                )

            n_samples = f["spectra"].shape[0]
            n_channels = f["spectra"].shape[1]

            if (n_samples, n_channels) != (30000, 257):
                raise ValueError("Unexpected spectra shape")

            if f["valid_mask"].shape != (30000, 257):
                raise ValueError("Invalid mask shape")

            if f["group_id"].shape != (30000,):
                raise ValueError("Invalid group_id shape")

            wavenumbers = f["wavenumbers"][:]

            if wavenumbers.shape != (257,):
                raise ValueError("Invalid wavenumber shape")

            if reference_wavenumbers is None:
                reference_wavenumbers = wavenumbers
            elif not np.array_equal(
                reference_wavenumbers, wavenumbers
            ):
                raise ValueError(
                    "Wavenumbers differ between files"
                )

            offsets = f["component_offsets"][:]
            component_count = len(f["component_ids"])

            if len(offsets) != 30001:
                raise ValueError("Invalid offsets length")

            if offsets[0] != 0:
                raise ValueError("Offsets must start at zero")

            if offsets[-1] != component_count:
                raise ValueError(
                    "Offsets do not match component IDs"
                )

            if np.any(np.diff(offsets) < 0):
                raise ValueError(
                    "Component offsets are not sorted"
                )

            if len(f["component_ppm"]) != component_count:
                raise ValueError("ppm length mismatch")

            if len(f["optical_depth_scale"]) != component_count:
                raise ValueError(
                    "optical_depth_scale length mismatch"
                )

            groups = f["group_id"][:]
            unique_groups, counts = np.unique(
                groups, return_counts=True
            )

            if not np.all(counts == 3):
                raise ValueError(
                    "Some groups do not contain 3 spectra "
                    "within this file"
                )

            duplicate_groups = (
                set(unique_groups.tolist()) & all_group_ids
            )

            if duplicate_groups:
                raise ValueError(
                    "Group IDs repeated across files"
                )

            all_group_ids.update(
                unique_groups.tolist()
            )

            file_valid = 0

        
            for start in range(0, n_samples, 1024):
                stop = min(start + 1024, n_samples)

                mask = f["valid_mask"][start:stop]

                if np.any((mask != 0) & (mask != 1)):
                    raise ValueError(
                        "Invalid valid_mask values"
                    )

                file_valid += int(mask.sum())

            total_spectra += n_samples
            total_valid_points += file_valid
            total_points += n_samples * n_channels

            file_reports.append({
                "file": file_path.name,
                "spectra": n_samples,
                "unique_groups": len(unique_groups),
                "valid_percentage": round(
                    100 * file_valid /
                    (n_samples * n_channels), 2
                ),
            })

    except Exception as e:
        errors.append({
            "file": file_path.name,
            "error": str(e),
        })
        print("ERROR:", e)

report = {
    "total_files": len(files),
    "total_spectra": total_spectra,
    "total_unique_groups": len(all_group_ids),
    "valid_percentage": (
        round(
            100 * total_valid_points / total_points,
            2
        ) if total_points else None
    ),
    "errors": errors,
    "files": file_reports,
}

report_path = (
    OUTPUT_DIR / "dataset_validation_report.json"
)

with open(report_path, "w", encoding="utf-8") as out:
    json.dump(report, out, indent=2)

print("\n=== FINAL SUMMARY ===")
print("Total files:", len(files))
print("Validated spectra:", total_spectra)
print("Unique groups:", len(all_group_ids))
print("Valid percentage:", report["valid_percentage"])
print("Errors:", len(errors))
print("Report saved:", report_path)
