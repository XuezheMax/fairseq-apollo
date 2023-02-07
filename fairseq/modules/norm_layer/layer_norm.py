# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union, List
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex.normalization import FusedLayerNorm
    from apex.normalization import FusedRMSNorm

    has_fusednorm = True

except ImportError:
    has_fusednorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    # if torch.jit.is_scripting():
    #     export = True
    if not export and torch.cuda.is_available() and has_fusednorm and not elementwise_affine:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return FairseqLayerNorm(normalized_shape, eps, elementwise_affine)


def RMSNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fusednorm:
        return FusedRMSNorm(normalized_shape, eps, elementwise_affine)
    else:
        raise NotImplementedError('No RMSNorm implementation installed')


_shape_t = Union[int, List[int], torch.Size]


class FairseqLayerNorm(nn.Module):

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FairseqLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
