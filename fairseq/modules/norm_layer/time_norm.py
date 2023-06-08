# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor
import torch.nn as nn


class TimeLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TimeLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.num_features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(
        self, x,
        padding_mask=None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude keys that are pads, of shape `(batch, src_len)`,
            where padding elements are indicated by 1s.
        """
        if padding_mask is None:
            # 1 x B x D
            var, mean = torch.var_mean(x.float(), dim=0, keepdim=True, unbiased=False)
        else:
            # L x B
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).to(x)
            # L x B x D
            x = x * inverse_mask.unsqueeze(2)

            slen = x.size(0)
            # B x 1
            count = slen - padding_mask.sum(dim=1, keepdim=True)
            # 1 x B x D
            var, mean = torch.var_mean(x.float(), dim=0, keepdim=True, unbiased=False)
            square_mean = var + torch.square(mean)
            # adjust by ratio
            # B x 1
            ratio = slen / count
            # 1 x B x D
            mean = mean * ratio
            var = square_mean * ratio - torch.square(mean)

        # 1 x B x D
        mean = mean.to(x)
        invstd = torch.rsqrt(var + self.eps).to(x)

        # L x B x D
        if self.affine:
            weight = torch.sigmoid(self.weight) * 2
            out = (x - mean) * (weight * invstd) + self.bias
        else:
            out = (x - mean) * invstd

        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
