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
class EMALayer(nn.Module):
    """Exponential Moving Average Layer.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        bidirectional=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.zdim = zdim
        self.bidirectional = bidirectional

        kernel_dim = 2 * zdim if self.bidirectional else zdim
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim))
        self.beta = nn.Parameter(torch.Tensor(zdim))
        self.proj = nn.Linear(embed_dim, zdim)
        self._kernel = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=-1.0, std=1.0)
        nn.init.normal_(self.beta, mean=0.0, std=1.0)

        # proj
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)

    def compute_kernel(self, length: int):
        # D x 1
        # gamma = torch.sigmoid(self.alpha).unsqueeze(1)
        gamma = torch.sigmoid(self.alpha).unsqueeze(1)
        # D x N
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

        # N x B x D
        x = self.proj(x)
        residual = x * self.beta

        # N x B x D -> B x D x N
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        # D x N
        k = self.kernel(seq_len)
        k_f = torch.fft.rfft(k, n=2 * seq_len)
        k_f2 = None
        if self.bidirectional:
            k_f, k_f2 = torch.split(k_f, [self.zdim, self.zdim], dim=0)

        x_f = torch.fft.rfft(x, n=2 * seq_len)
        # B x D x N
        out = torch.fft.irfft(x_f * k_f, n=2 * seq_len)[..., :seq_len]

        if self.bidirectional:
            # B x D x N
            x_f2 = torch.fft.rfft(x.flip(-1), n=2 * seq_len)
            out2 = torch.fft.irfft(x_f2 * k_f2, n=2 * seq_len)[..., :seq_len]
            out = out + out2.flip(-1)

        # B x D x N -> N x B x D
        out = out.permute(2, 0, 1) + residual
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
