
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

SPECTRA_PER_FILE = 30000


class FTIRBatchDataset(Dataset):

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

        with np.load(ARTIFACTS / "normalization_stats.npz") as stats:
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        assert self.mean.shape == (257,)
        assert self.std.shape == (257,)
        assert np.all(self.std > 0)

    def __len__(self):
        return len(self.indices)

    def __getitems__(self, positions):
        positions = np.asarray(positions, dtype=np.int64)
        global_indices = self.indices[positions]

        file_ids = global_indices // SPECTRA_PER_FILE
        row_ids = global_indices % SPECTRA_PER_FILE

        batch_size = len(positions)

        spectra = np.empty((batch_size, 257), dtype=np.float32)
        valid_masks = np.empty((batch_size, 257), dtype=bool)

        # Group requested samples by HDF5 file
        for file_id in np.unique(file_ids):

            batch_positions = np.flatnonzero(file_ids == file_id)
            rows = row_ids[batch_positions]

            # h5py requires indices in increasing order
            order = np.argsort(rows)
            sorted_rows = rows[order]

            # Handle duplicate row requests safely
            unique_rows, inverse = np.unique(
                sorted_rows, return_inverse=True
            )

            file_path = DATA / f"spectra_{int(file_id):04d}.h5"

            with h5py.File(file_path, "r") as f:
                raw = f["spectra"][unique_rows].astype(np.float32)
                valid = f["valid_mask"][unique_rows].astype(bool)

            raw = raw[inverse]
            valid = valid[inverse]

            # Restore original batch order
            original_positions = batch_positions[order]

            spectra[original_positions] = raw
            valid_masks[original_positions] = valid

        valid_masks &= np.isfinite(spectra)

        normalized = np.zeros_like(spectra, dtype=np.float32)

        z_scores = (
            spectra - self.mean[None, :]
        ) / self.std[None, :]

        normalized[valid_masks] = z_scores[valid_masks]

        return [
            {
                "spectrum": torch.from_numpy(normalized[i]),
                "valid_mask": torch.from_numpy(valid_masks[i]),
                "global_idx": int(global_indices[i]),
            }
            for i in range(batch_size)
        ]

    def __getitem__(self, position):
        return self.__getitems__([position])[0]
