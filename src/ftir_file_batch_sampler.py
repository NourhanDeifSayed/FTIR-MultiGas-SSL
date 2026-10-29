
import numpy as np
from torch.utils.data import Sampler


class FTIRFileBatchSampler(Sampler):

    def __init__(
        self,
        indices,
        batch_size=32,
        shuffle=True,
        seed=42,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Dataset positions grouped by HDF5 file
        file_ids = self.indices // 30000

        self.file_positions = {
            int(file_id): np.flatnonzero(file_ids == file_id)
            for file_id in np.unique(file_ids)
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        all_batches = []

        for positions in self.file_positions.values():
            positions = positions.copy()

            if self.shuffle:
                rng.shuffle(positions)

            for start in range(0, len(positions), self.batch_size):
                batch = positions[start:start + self.batch_size]
                all_batches.append(batch.tolist())

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return sum(
            (len(positions) + self.batch_size - 1)
            // self.batch_size
            for positions in self.file_positions.values()
        )
