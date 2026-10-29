
from pathlib import Path
import h5py

file_path = Path("data/spectra_0000.h5")

if not file_path.exists():
    raise FileNotFoundError(
        f"File not found: {file_path.resolve()}"
    )

with h5py.File(file_path, "r") as f:

    print("=== HDF5 DATASETS ===")

    for key in f.keys():
        dataset = f[key]
        print(
            f"{key}: "
            f"shape={dataset.shape}, "
            f"dtype={dataset.dtype}"
        )

    print("\n=== FIRST 5 SPECTRA ===")

    spectra = f["spectra"][:5]
    valid_mask = f["valid_mask"][:5]
    wavenumbers = f["wavenumbers"][:]

    print("Spectra shape:", spectra.shape)
    print("Valid mask shape:", valid_mask.shape)
    print("Wavenumbers shape:", wavenumbers.shape)

    print("\nFirst 10 spectral values:")
    print(spectra[0][:10])

    print("\nFirst 10 validity flags:")
    print(valid_mask[0][:10])

    print("\nWavenumber range:")
    print(wavenumbers.min(), "to", wavenumbers.max())
