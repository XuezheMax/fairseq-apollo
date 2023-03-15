# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union, List
import torch
import torch.nn as nn


class TimeAwareLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, causal: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TimeAwareLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.causal = causal
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

    def _forward_noncausal(self, x, padding_mask):
        if padding_mask is None:
            # 1 x B x 1
            var, mean = torch.var_mean(x, dim=(0, 2), keepdim=True, unbiased=False)
        else:
            slen, bsz, num_feats = x.size()
            total = slen * num_feats
            # B x 1
            count = total - padding_mask.sum(dim=1, keepdim=True) * num_feats
            # 1 x B x 1
            var, mean = torch.var_mean(x, dim=(0, 2), keepdim=True, unbiased=False)
            square_mean = var + torch.square(mean)
            # adjust by ratio
            count = count.to(mean)
            ratio = total / count
            mean = mean * ratio
            var = square_mean * ratio - torch.square(mean)

        # 1 x B x 1
        invstd = torch.rsqrt(var + self.eps)
        # L x B x D
        out = (x - mean) * invstd
        return out

    def _forward_causal(self, x, padding_mask):
        raise NotImplementedError

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            # L x B
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).to(x)
            # L x B x D
            x = x * inverse_mask.unsqueeze(2)

        if self.causal:
            out = self._forward_causal(x.float(), padding_mask)
        else:
            out = self._forward_noncausal(x.float(), padding_mask)

        # L x B x D
        out = out.to(x)
        if self.weight is not None:
            weight = self.weight + 1.0
            out = out * weight + self.bias

        return out

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}, causal={causal}'.format(**self.__dict__)
