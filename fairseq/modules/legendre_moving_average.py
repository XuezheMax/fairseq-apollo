# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class LMALayer(nn.Module):
    """Legendre Moving Average Layer.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        kernel_size=-1,
        bidirectional=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.bidirectional = bidirectional

        self.beta = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        nn.init.normal_(self.beta, mean=0.0, std=1.0)

    def make_positions(self, x, mask):
        seq_len, bsz, embed_dim = x.size()
        # N x B
        if mask is not None:
            return torch.cumsum(mask, dim=0)
        else:
            mask = torch.ones(seq_len, 1).type_as(x)
            return torch.cumsum(mask, dim=0)

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        # N x B x D
        residual = x * self.beta

        # N x B
        mask = 1.0 - padding_mask.float().transpose(0, 1) if padding_mask is not None else None
        # N x B x D
        fwd = torch.cumsum(x, dim=0)
        fwdp = self.make_positions(x, mask)
        out = fwd / fwdp.unsqueeze(-1)

        if self.bidirectional:
            bwd = torch.cumsum(x.flip(0), dim=0)
            if mask is not None:
                mask = mask.flip(0)
            bwdp = self.make_positions(x, mask)
            out = out + (bwd / bwdp.unsqueeze(-1)).flip(0)

        out = out + residual
        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "lma_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "lma_state", buffer)

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat(
                [prev_padding_mask.float(), padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - prev_padding_mask.size(1)), device=prev_padding_mask.device)
            new_padding_mask = torch.cat([prev_padding_mask.float(), filler.float()], dim=1)
        elif padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - padding_mask.size(1)), device=padding_mask.device)
            new_padding_mask = torch.cat([filler.float(), padding_mask.float()], dim=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask
