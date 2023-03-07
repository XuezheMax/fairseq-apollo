# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))

        self.reset_parameters()

        self.process_group = process_group

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def _compute_mean_var(self, x, nums, total):
        ratio = (total / nums).type_as(x)
        mean = torch.mean(x, dim=(0, 1)) * ratio
        var = torch.mean(torch.square(x), dim=(0, 1)) * ratio - torch.square(mean)
        return mean, var

    def _batch_norm_with_padding(self, x, padding_mask, momentum):
        if self.training:
            # zero out paddings
            # L x B
            total = padding_mask.numel()
            nums = total - padding_mask.sum()
            # L x B x D
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).type_as(x)
            x = x * inverse_mask.unsqueeze(2)
            mean, var = self._compute_mean_var(x, nums, total)
            with torch.no_grad():
                self.running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
                self.running_var.mul_(1.0 - momentum).add_(var, alpha=momentum * nums / (nums - 1))  # unbias var estimator for running var
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

        weight = self.weight + 1.0 if self.affine else None
        if not need_sync:
            if padding_mask is None:
                x = x.permute(1, 2, 0)
                out = F.batch_norm(x, self.running_mean, self.running_var, weight, self.bias,
                                   self.training, exponential_average_factor, self.eps)
                out = out.permute(2, 0, 1)
            else:
                out = self._batch_norm_with_padding(x, padding_mask, exponential_average_factor)
        else:
            x_float = x.permute(1, 2, 0).float()
            weight = weight.float()
            bias = self.bias.float()
            running_mean = self.running_mean.float()
            running_var = self.running_var.float()
            out = sync_batch_norm_with_mask.apply(x_float, weight, bias, padding_mask,
                                                  running_mean, running_var,
                                                  self.eps, exponential_average_factor,
                                                  process_group, world_size)
            out = out.type_as(x).permute(2, 0, 1)
            running_mean = running_mean.type_as(self.running_mean)
            running_var = running_var.type_as(self.running_var)
            if self.running_mean.data_ptr != running_mean.data_ptr:
                with torch.no_grad():
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)

        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}, momentum={momentum}'.format(**self.__dict__)
