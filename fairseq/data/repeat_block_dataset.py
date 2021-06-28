import numpy as np
import torch

from fairseq.data import FairseqDataset, plasma_utils


class RepeatBlockDataset(FairseqDataset):
    def __init__(self, dataset, sizes, block_size, eos):
        super().__init__()
        self.dataset = dataset
        self.eos = eos

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        self._repeat_times = self._get_repeat_times(sizes, block_size)
        self._sizes = sizes + (sizes + 1) * self._repeat_times

        self._repeat_times = plasma_utils.PlasmaArray(self._repeat_times)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)

    def _get_repeat_times(self, sizes, block_size):
        res_sizes = (block_size - sizes).clip(0)
        repeat_times = res_sizes // (sizes + 1)
        return repeat_times

    @property
    def repeat_times(self):
        return self._repeat_times.array

    @property
    def sizes(self):
        return self._sizes.array

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def __getitem__(self, index):
        item = self.dataset[index]
        repeat_time = self.repeat_times[index]
        if repeat_time > 0:
            repeat_item = torch.cat([item.new([self.eos]), item]).repeat(repeat_time)
            item = torch.cat([item, repeat_item])
        return item

    def __len__(self):
        return len(self.sizes)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )
