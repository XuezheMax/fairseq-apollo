# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn

from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class TimeLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, causal: bool = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TimeLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.causal = causal
        if causal:
            self.pseudo_embs = nn.Parameter(torch.empty(2, self.num_features, **factory_kwargs))
        else:
            self.register_parameter('pseudo_embs', None)

        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.num_features, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.causal:
            std = 1.0 / math.sqrt(self.num_features)
            nn.init.normal_(self.pseudo_embs, mean=0, std=std)

        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def _forward_noncausal(self, x, padding_mask):
        if padding_mask is None:
            # 1 x B x D
            var, mean = torch.var_mean(x.float(), dim=0, keepdim=True, unbiased=False)
        else:
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

        return mean, invstd

    def _forward_causal(self, x, padding_mask, incremental_state):
        slen = x.size(0)
        bsz = x.size(1)
        x_float = x.float()

        prev_sum = None
        prev_ssum = None
        pidx = None
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_sum' in saved_state:
                prev_sum = saved_state['prev_sum']
                assert 'prev_ssum' in saved_state and 'prev_num_steps' in saved_state
                prev_ssum = saved_state['prev_ssum']
                pidx = saved_state['prev_num_steps']
        else:
            saved_state = None

        if prev_sum is None:
            # D
            pseudo_embs = self.pseudo_embs.float()
            prev_sum = pseudo_embs.sum(dim=0)
            prev_ssum = torch.square(pseudo_embs).sum(dim=0)
            # B x 1
            pidx = torch.full((bsz, 1), 3).to(x_float)

        # 1 x B x 1 + L x 1 x 1 -> L x B x 1
        positions = pidx.unsqueeze(0) + torch.arange(0, slen).to(x_float).view(slen, 1, 1)
        # L x B x D
        curr_sum = torch.cumsum(x_float, dim=0) + prev_sum
        mean = curr_sum / positions
        curr_ssum = torch.cumsum(torch.square(x_float), dim=0) + prev_ssum
        square_mean = curr_ssum / positions
        var = square_mean - torch.square(mean)

        if incremental_state is not None:
            if padding_mask is not None:
                lengths = slen - padding_mask.sum(dim=-1, keepdim=True)
            else:
                lengths = slen
            # B x D
            saved_state['prev_sum'] = curr_sum[-1]
            saved_state['prev_ssum'] = curr_ssum[-1]
            saved_state['prev_num_steps'] = pidx + lengths
            self._set_input_buffer(incremental_state, saved_state)

        # L x B x D
        mean = mean.to(x)
        invstd = torch.rsqrt(var + self.eps).to(x)

        return mean, invstd

    def forward(
        self, x,
        padding_mask=None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude keys that are pads, of shape `(batch, src_len)`,
            where padding elements are indicated by 1s.
        """
        if padding_mask is not None:
            # L x B
            inverse_mask = 1.0 - padding_mask.transpose(0, 1).to(x)
            # L x B x D
            x = x * inverse_mask.unsqueeze(2)

        assert self.causal or incremental_state is None, 'Non-causal TimeNorm does not support incremental state'
        if self.causal:
            mean, invstd = self._forward_causal(x, padding_mask, incremental_state)
        else:
            mean, invstd = self._forward_noncausal(x, padding_mask)

        # L x B x D
        if self.affine:
            weight = self.weight + 1.0
            out = (x - mean) * (weight * invstd) + self.bias
        else:
            out = (x - mean) * invstd

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "norm_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "norm_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, affine={affine}, causal={causal}'.format(**self.__dict__)
