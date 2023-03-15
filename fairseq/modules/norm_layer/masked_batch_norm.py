# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
from ._functions import MaskedBatchNorm as batch_norm_with_mask
from ._functions import MaskedSyncBatchNorm as sync_batch_norm_with_mask


class MaskedBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, momentum=0.1, process_group=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum

        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # TODO: avoid running stats to be destroyed by overflows
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))

        self.reset_parameters()

        self.process_group = process_group

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x, padding_mask=None):
        if self.momentum is None:
            exponential_average_factor = 1.0 / np.sqrt(float(self.num_batches_tracked))
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked.add_(1)

        need_sync = self.training
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group

            if process_group is None:
                world_size = 0
            else:
                world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            # B x L
            inverse_mask = 1.0 - padding_mask.to(x)
            # B x D x L
            x = x * inverse_mask.unsqueeze(1)

        weight = self.weight + 1.0 if self.affine else None
        if need_sync:
            out = sync_batch_norm_with_mask.apply(x, weight, self.bias, padding_mask,
                                                  self.running_mean, self.running_var,
                                                  self.eps, exponential_average_factor,
                                                  process_group, world_size)
        elif self.training:
            out = batch_norm_with_mask.apply(x, weight, self.bias, padding_mask,
                                             self.running_mean, self.running_var,
                                             self.eps, exponential_average_factor)
        else:
            mean, var = self.running_mean, self.running_var
            invstd = torch.rsqrt(var + self.eps)
            out = torch.batch_norm_elemt(x, weight, self.bias, mean, invstd, self.eps)

        out = out.permute(2, 0, 1)
        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}, momentum={momentum}'.format(**self.__dict__)
