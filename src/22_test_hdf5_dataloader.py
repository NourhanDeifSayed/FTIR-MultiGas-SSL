
from pathlib import Path
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

SPECTRA_PER_FILE = 30000


class FTIRDataset(Dataset):

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

        stats = np.load(
            ARTIFACTS / "normalization_stats.npz"
        )

        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

        assert self.mean.shape == (257,)
        assert self.std.shape == (257,)
        assert np.all(self.std > 0)

        # Cache file handles for efficient repeated reads.
        self.files = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        global_idx = int(self.indices[position])

        file_idx = global_idx // SPECTRA_PER_FILE
        row_idx = global_idx % SPECTRA_PER_FILE

        if file_idx not in self.files:

            path = (
                DATA / f"spectra_{file_idx:04d}.h5"
            )

            self.files[file_idx] = h5py.File(
                path, "r"
            )

        f = self.files[file_idx]

        spectrum = f["spectra"][row_idx].astype(
            np.float32
        )

        valid_mask = f["valid_mask"][row_idx].astype(
            bool
        )

        # Exclude any unexpected non-finite points.
        valid_mask &= np.isfinite(spectrum)

        # Normalize valid points only.
        normalized = np.zeros(
            257, dtype=np.float32
        )

        normalized[valid_mask] = (
            spectrum[valid_mask] - self.mean[valid_mask]
        ) / self.std[valid_mask]

        return {
            "spectrum": torch.from_numpy(normalized),
            "valid_mask": torch.from_numpy(valid_mask),
            "global_idx": global_idx
        }

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()


if __name__ == "__main__":

    train_indices = np.load(
        ARTIFACTS / "train_indices.npy"
    )

    # Check the first 64 training spectra only.
    dataset = FTIRDataset(train_indices[:64])

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(loader))

    spectra = batch["spectrum"]
    masks = batch["valid_mask"]

    print("=== DATALOADER TEST ===")
    print("Dataset length:", len(dataset))
    print("Spectrum batch shape:", tuple(spectra.shape))
    print("Mask batch shape:", tuple(masks.shape))
    print("Spectrum dtype:", spectra.dtype)
    print("Mask dtype:", masks.dtype)

    print(
        "Valid points in first spectrum:",
        int(masks[0].sum())
    )

    print(
        "Invalid points in first spectrum:",
        int((~masks[0]).sum())
    )

    print(
        "All values finite:",
        bool(torch.isfinite(spectra).all())
    )

    print(
        "Invalid points are zero:",
        bool(torch.all(spectra[~masks] == 0))
    )

    assert spectra.shape == (8, 257)
    assert masks.shape == (8, 257)
    assert torch.isfinite(spectra).all()
    assert torch.all(spectra[~masks] == 0)

    dataset.close()

    print("\nAll DataLoader checks passed!")
