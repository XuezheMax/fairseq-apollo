# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class DualNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, export=False):
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

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _compute_mean_var(self, x, nums):
        sum_x = torch.sum(x, dim=(0, 1))
        ssum_x = torch.sum(torch.square(x), dim=(0, 1))

        mean = sum_x / nums
        var = ssum_x / nums - torch.square(mean)

        with torch.no_grad():
            self.num_batches_tracked = self.num_batches_tracked + 1
            self.running_mean.mul_(1.0 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1.0 - self.momentum).add_(var, alpha=self.momentum * nums / (nums - 1)) # unbias var estimator for running var

        return mean, var

    def _forward_with_padding(self, x, padding_mask):
        if self.training:
            # zero out paddings
            # L x B
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).type_as(x)
            nums = inverse_mask.sum()
            # L x B x D
            x = x * inverse_mask.unsqueeze(2)
            mean, var = self._compute_mean_var(x, nums)
        else:
            mean, var = self.running_mean, self.running_var

        inv_std = torch.rsqrt(var + self.eps)

        if self.affine:
            out = (x - mean) * (self.weight * inv_std) + self.bias
        else:
            out = (x - mean) * inv_std
        return out

    def forward(self, x, padding_mask=None):
        # L x B x D
        # x = F.normalize(x, dim=-1, eps=self.eps)

        if padding_mask is None:
            out = F.batch_norm(x.permute(1, 2, 0), self.running_mean, self.running_var,
                               self.weight, self.bias, self.training, self.momentum, self.eps)
            return out.permute(2, 0, 1)
        else:
            return self._forward_with_padding(x, padding_mask)

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)


class DualNorm2(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, export=False):
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
        self.register_buffer('running_var', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _compute_mean_var(self, x, nums):
        sum_x = torch.sum(x, dim=(0, 1))
        ssum_x = torch.sum(torch.square(x), dim=(0, 1))

        mean = sum_x / nums
        var = ssum_x / nums - torch.square(mean)

        running_mean = self.running_mean.mul(1.0 - self.momentum).add(mean, alpha=self.momentum)
        running_var = self.running_var.mul(1.0 - self.momentum).add(var, alpha=self.momentum)

        with torch.no_grad():
            self.num_batches_tracked = self.num_batches_tracked + 1
            self.running_mean.copy_(running_mean.data)
            self.running_var.copy_(running_var.data)

        return running_mean, running_var

    def forward(self, x, padding_mask=None):
        # L x B x D
        # x = F.normalize(x, dim=-1, eps=self.eps)

        if self.training:
            if padding_mask is not None:
                # zero out paddings
                # L x B
                inverse_mask = 1.0 - padding_mask.transpose(0, 1).type_as(x)
                nums = inverse_mask.sum()
                # L x B x D
                x = x * inverse_mask.unsqueeze(2)
            else:
                nums = x.size(0) * x.size(1)

            mean, var = self._compute_mean_var(x, nums)
        else:
            mean, var = self.running_mean, self.running_var

        bias_correction = 1 - (1 - self.momentum) ** self.num_batches_tracked
        mean = mean / bias_correction
        inv_std = torch.rsqrt(var / bias_correction + self.eps)

        if self.affine:
            out = (x - mean) * (self.weight * inv_std) + self.bias
        else:
            out = (x - mean) * inv_std
        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
