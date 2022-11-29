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

        self.feature_norm = LayerNorm(num_features, eps=eps, elementwise_affine=False, export=export)
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_square', torch.zeros(num_features))
        self.register_buffer('steps', torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _update_mean_square(self, x, nums):
        with torch.no_grad():
            # D
            mean_square = torch.sum(torch.square(x) / nums, dim=(0, 1))
            self.steps.copy_(self.steps + 1.0)
            self.running_square.mul_(1.0 - self.momentum).add_(mean_square, alpha=self.momentum)

    def forward(self, x, padding_mask=None):
        # L x B x D
        out = self.feature_norm(x)
        if self.training:
            if padding_mask is not None:
                # zero out paddings
                # L x B
                inverse_mask = 1.0 - padding_mask.transpose(0, 1).type_as(x)
                nums = inverse_mask.sum()
                # L x B x D
                out = out * inverse_mask.unsqueeze(2)
            else:
                nums = x.size(0) * x.size(1)
            self._update_mean_square(out, nums)

        bias_correction = 1 - (1 - self.momentum) ** self.steps
        inv_square = torch.rsqrt(self.running_square / bias_correction + self.eps)

        if self.affine:
            out = out * (self.weight * inv_square) + self.bias
        else:
            out = out * inv_square
        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
