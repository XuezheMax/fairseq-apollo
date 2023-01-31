# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from ._functions import MaskedSyncBatchNorm as sync_batch_norm_with_mask


class MaskedBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.01, momentum_decay=0.99999,  eps=1e-5, affine=True, process_group=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.momentum_decay = momentum_decay

        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))

        self.reset_parameters()

        self.process_group = process_group

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def _compute_mean_var(self, x, nums, momentum):
        sum_x = torch.sum(x, dim=(0, 1))
        ssum_x = torch.sum(torch.square(x), dim=(0, 1))

        mean = sum_x / nums
        var = ssum_x / nums - torch.square(mean)

        with torch.no_grad():
            self.running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
            self.running_var.mul_(1.0 - momentum).add_(var, alpha=momentum * nums / (nums - 1)) # unbias var estimator for running var

        return mean, var

    def _batch_norm_with_padding(self, x, padding_mask, momentum):
        if self.training:
            # zero out paddings
            # L x B
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).type_as(x)
            nums = inverse_mask.sum()
            # L x B x D
            x = x * inverse_mask.unsqueeze(2)
            mean, var = self._compute_mean_var(x, nums, momentum)
        else:
            mean, var = self.running_mean, self.running_var

        inv_std = torch.rsqrt(var + self.eps)

        if self.affine:
            weight = self.weight + 1.0
            out = (x - mean) * (weight * inv_std) + self.bias
        else:
            out = (x - mean) * inv_std
        return out

    def forward(self, x, padding_mask=None):
        if self.momentum is None:
            exponential_average_factor = 1.0 / np.sqrt(float(self.num_batches_tracked))
        else:
            exponential_average_factor = self.momentum * (self.momentum_decay ** float(self.num_batches_tracked))

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

        if padding_mask is None:
            x = x.permute(1, 2, 0)
            if not need_sync:
                out = F.batch_norm(x, self.running_mean, self.running_var, self.weight + 1.0,
                                   self.bias, self.training, exponential_average_factor, self.eps)
            else:
                out = sync_batch_norm.apply(x, self.weight + 1.0, self.bias,
                                            self.running_mean, self.running_var,
                                            self.eps, exponential_average_factor,
                                            process_group, world_size)
            out = out.permute(2, 0, 1)
        else:
            if not need_sync:
                out = self._batch_norm_with_padding(x, padding_mask, exponential_average_factor)
            else:
                x = x.permute(1, 2, 0)
                out = sync_batch_norm_with_mask.apply(x, self.weight + 1.0, self.bias,
                                                      padding_mask,
                                                      self.running_mean, self.running_var,
                                                      self.eps, exponential_average_factor,
                                                      process_group, world_size)
                out = out.permute(2, 0, 1)

        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}, momentum={momentum}(momentum_decay)'.format(**self.__dict__)
