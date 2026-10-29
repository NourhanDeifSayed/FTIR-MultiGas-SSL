from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_DIR / "data" / "spectra_0000.h5"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with h5py.File(FILE_PATH, "r") as f:

    spectra = f["spectra"][:]
    valid_mask = f["valid_mask"][:].astype(bool)
    wavenumbers = f["wavenumbers"][:]
    group_ids = f["group_id"][:]
    offsets = f["component_offsets"][:]

    print("=== DATA QUALITY CHECKS ===")

    total_points = spectra.size
    valid_points = valid_mask.sum()
    invalid_points = total_points - valid_points

    print("Number of spectra:", len(spectra))
    print("Total spectral points:", total_points)
    print("Valid points:", valid_points)
    print("Invalid points:", invalid_points)
    print("Valid percentage:",
          round(100 * valid_points / total_points, 2), "%")

    unique_mask_values = np.unique(f["valid_mask"][:])
    print("Unique mask values:", unique_mask_values)

    valid_values = spectra[valid_mask]

    print("\n=== VALID SPECTRAL VALUES ===")
    print("NaN count:", np.isnan(valid_values).sum())
    print("Infinity count:", np.isinf(valid_values).sum())

    finite_values = valid_values[np.isfinite(valid_values)]

    if finite_values.size > 0:
        print("Minimum:", finite_values.min())
        print("Maximum:", finite_values.max())
        print("Mean:", finite_values.mean())
        print("Std:", finite_values.std())

    print("\n=== VALID CHANNELS PER SPECTRUM ===")
    valid_per_spectrum = valid_mask.sum(axis=1)

    print("Minimum:", valid_per_spectrum.min())
    print("Maximum:", valid_per_spectrum.max())
    print("Average:", round(valid_per_spectrum.mean(), 2))
    print("Fully invalid spectra:",
          np.sum(valid_per_spectrum == 0))

    print("\n=== SUBSTANCES PER SPECTRUM ===")
    components_per_spectrum = np.diff(offsets)

    print("Minimum:", components_per_spectrum.min())
    print("Maximum:", components_per_spectrum.max())
    print("Average:",
          round(components_per_spectrum.mean(), 2))
    print("Spectra with no substances:",
          np.sum(components_per_spectrum == 0))

    print("\n=== GROUP INFORMATION ===")
    unique_groups, counts = np.unique(
        group_ids, return_counts=True
    )

    print("Unique groups:", len(unique_groups))
    print("Minimum group size:", counts.min())
    print("Maximum group size:", counts.max())

  
    fig, axes = plt.subplots(
        5, 1, figsize=(12, 12), sharex=True
    )

    for i, ax in enumerate(axes):
        plot_values = spectra[i].copy()
        plot_values[~valid_mask[i]] = np.nan

        ax.plot(wavenumbers, plot_values, linewidth=1)
        ax.set_ylabel("Signal")
        ax.set_title(
            f"Spectrum {i} | "
            f"Valid channels: {valid_per_spectrum[i]}/257"
        )
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Wavenumber (cm^-1)")
    fig.suptitle("First 5 FTIR Spectra — Valid Points Only")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "first_5_spectra.png"
    plt.savefig(output_path, dpi=200)
    output_path = OUTPUT_DIR / "first_5_spectra.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nPlot saved to:", output_path)
