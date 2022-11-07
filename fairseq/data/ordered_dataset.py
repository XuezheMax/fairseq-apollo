# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
from . import data_utils, FairseqDataset

    
class OrderedDataset(FairseqDataset):
    def __init__(self, dataset, pad, world_size, bsz, rank, tgt_len):
        self.dataset = dataset
        
        assert len(self.dataset) == world_size

        source, item, past_target = dataset[rank]
        logger.warning("{} {} {}".format(world_size, rank, len(item)))
        assert len(source) == len(item)
        reminder = len(item) % bsz
        add  = bsz - reminder if reminder > 0 else 0
        item = torch.cat([item, item.new([pad] * add)])
        assert len(item) % bsz == 0
        source = torch.cat([source, source.new([pad] * add)])
        
        source = source.view(bsz, -1)
        data = item.view(bsz, -1)

        # data = data.split(tgt_len, dim=1)
        # data = torch.cat([bb.reshape(-1) for bb in data])

        reminder = data.size(1) % tgt_len
        if reminder != 0:
            add = tgt_len - reminder
            data = torch.cat([data, data.new_full((bsz, add), pad)], dim=1)
            source = torch.cat([source, source.new_full((bsz, add), pad)], dim=1)
        assert data.size(1) % tgt_len == 0

        n_tokens_per_seg = data.size(1)
        n_steps = n_tokens_per_seg // tgt_len
        data = torch.chunk(data, n_steps, dim=1)
        data = torch.cat([bb.reshape(-1) for bb in data]).view(-1, tgt_len)

        source = torch.chunk(source, n_steps, dim=1)
        source = torch.cat([bb.reshape(-1) for bb in source]).view(-1, tgt_len)
        assert data.size(0) == bsz * n_steps
        assert source.size(0) == bsz * n_steps

        self.dataset = data
        self.source = source
        self.tgt_len = tgt_len

    @property
    def sizes(self):
        return np.zeros(len(self.dataset)) * self.tgt_len
    
    def __len__(self):
        return self.dataset.size(0)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        source = self.source[index]
        return source, item, item
