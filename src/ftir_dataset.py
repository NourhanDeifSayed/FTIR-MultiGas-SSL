
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

SPECTRA_PER_FILE = 30000


class FTIRDataset(Dataset):
    def __init__(self, indices):

        self.indices = np.asarray(
            indices, dtype=np.int64
        )

        with np.load(
            ARTIFACTS / "normalization_stats.npz"
        ) as stats:
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        assert self.mean.shape == (257,)
        assert self.std.shape == (257,)
        assert np.all(self.std > 0)

        # Keep only one HDF5 file open at a time.
        self.file = None
        self.current_file_idx = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        global_idx = int(self.indices[position])

        file_idx = global_idx // SPECTRA_PER_FILE
        row_idx = global_idx % SPECTRA_PER_FILE

        # Open a new HDF5 file when needed.
        if file_idx != self.current_file_idx:
            self.close()

            file_path = (
                DATA / f"spectra_{file_idx:04d}.h5"
            )

            self.file = h5py.File(file_path, "r")
            self.current_file_idx = file_idx

        spectrum = self.file["spectra"][row_idx].astype(
            np.float32
        )

        valid = self.file["valid_mask"][row_idx].astype(
            bool
        )

        valid &= np.isfinite(spectrum)

        normalized = np.zeros(257, dtype=np.float32)

        normalized[valid] = (
            spectrum[valid] - self.mean[valid]
        ) / self.std[valid]

        return {
            "spectrum": torch.from_numpy(normalized),
            "valid_mask": torch.from_numpy(valid),
            "global_idx": global_idx,
        }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

        self.current_file_idx = None
