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
from fairseq.modules.fairseq_dropout import FairseqDropout


@with_incremental_state
class EMALayer(nn.Module):
    """Exponential Moving Average Gated Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        activation='tanh',
        bidirectional=False,
        bias=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        assert activation in ['tanh', 'sin']
        self.activation = utils.get_activation_fn(activation=activation)
        self.bidirectional = bidirectional

        num_directions = 2 if self.bidirectional else 1
        self.input_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim * num_directions, embed_dim, bias=bias)
        self.weight = nn.Parameter(torch.Tensor(embed_dim * num_directions))
        self._kernel = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        if self.input_proj.weight is not None:
            nn.init.constant_(self.input_proj.bias, 0.0)

        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.out_proj.weight is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.constant_(self.weight, -1.)

    def compute_kernel(self, length: int):
        # D*c x 1
        gamma = torch.sigmoid(self.weight).unsqueeze(1)
        # D*c x N
        vander = torch.arange(length).to(gamma).unsqueeze(0) * torch.log(1 - gamma)
        self._kernel = torch.exp(vander) * gamma
        return self._kernel

    def kernel(self, length: int):
        if self.training:
            return self.compute_kernel(length)
        elif self._kernel is None:
            return self.compute_kernel(length)
        elif self._kernel.size(-1) < length:
            return self.compute_kernel(length)
        else:
            return self._kernel[..., :length]

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

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # N x B x D -> B x N x D
        u = F.silu(self.input_proj(x.transpose(0, 1)))
        if padding_mask is not None:
            u = u.masked_fill(padding_mask.unsqueeze(2), 0.)

        # B x N x D -> B x D x N
        u = u.transpose(1, 2)

        if self.bidirectional:
            # B x 2*D x N
            u = torch.cat([u, u.flip(-1)], dim=1)

        # D x N
        k = self.kernel(seq_len)

        u_f = torch.fft.rfft(u, n=2 * seq_len)
        k_f = torch.fft.rfft(k, n=2 * seq_len)

        # B x D x N
        out = torch.fft.irfft(u_f * k_f, n=2 * seq_len)[..., :seq_len]

        if self.bidirectional:
            out1, out2 = torch.split(out, [embed_dim, embed_dim], dim=1)
            # B x 2*D x N
            out = torch.cat([out1, out2.flip(-1)], dim=1)

        # B x D x N -> N x B x D
        out = out.permute(2, 0, 1)
        out = self.activation(self.out_proj(out))
        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

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
