
from pathlib import Path
import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_DIR / "data" / "spectra_0000.h5"

with h5py.File(FILE_PATH, "r") as f:
    offsets = f["component_offsets"][:]
    component_ids = f["component_ids"][:]
    component_ppm = f["component_ppm"][:]
    optical_scales = f["optical_depth_scale"][:]
    group_ids = f["group_id"][:]

    print("=== FIRST 5 SPECTRA LABELS ===")

    for i in range(5):
        start = int(offsets[i])
        end = int(offsets[i + 1])

        ids = component_ids[start:end]
        ppm = component_ppm[start:end]
        scales = optical_scales[start:end]

        print(f"\nSpectrum {i}")
        print("Group ID:", group_ids[i])
        print("Number of components:", len(ids))

        for j in range(len(ids)):
            print(
                f"  Component ID: {ids[j]} | "
                f"ppm: {ppm[j]} | "
                f"optical_depth_scale: {scales[j]}"
            )

        complete_ppm = bool(np.all(np.isfinite(ppm)))

        print("Regression eligible:", complete_ppm)

    print("\n=== FILE-LEVEL SUMMARY ===")
    print("Unique component IDs:",
          len(np.unique(component_ids)))
    print("Total component entries:", len(component_ids))
    print("Missing ppm entries:",
          int(np.isnan(component_ppm).sum()))
