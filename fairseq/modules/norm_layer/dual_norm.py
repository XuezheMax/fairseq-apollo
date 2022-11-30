# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from fairseq.modules.norm_layer.layer_norm import LayerNorm


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
            self.running_var.mul_(1.0 - self.momentum).add_(var, alpha=self.momentum)

        return mean, var

    def forward(self, x, padding_mask=None):
        # L x B x D
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)

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

        # bias_correction = 1 - (1 - self.momentum) ** self.num_batches_tracked
        inv_std = torch.rsqrt(var + self.eps)

        if self.affine:
            out = (x - mean) * (self.weight * inv_std) + self.bias
        else:
            out = (x - mean) * inv_std
        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
