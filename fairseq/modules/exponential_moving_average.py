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
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, 1, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta
            nn.init.uniform_(self.delta, a=0.001, b=0.1)
            # alpha & beta
            A = torch.tril(torch.ones(self.ndim, self.ndim)) - torch.eye(self.ndim) / 2
            B = torch.ones(self.ndim, 1)
            w, V = torch.linalg.eig(A)
            w = w.real
            V_inv = V.conj().real
            self.alpha.normal_(mean=0.0, std=0.02).add_(w.unsqueeze(1))
            self.beta.normal_(mean=0.0, std=0.02).add_(torch.mm(V_inv, B))
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def calc_params(self):
        # D x 1 x 1
        delta = torch.exp(self.delta)
        # D x N x 1
        p = delta / (1.0 + 0.5 * delta * self.alpha)
        q = 1.0 - p * self.alpha
        return p, q

    def compute_kernel(self, length: int):
        # D x N x 1
        p, q = self.calc_params()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        self._kernel = torch.einsum('dnl,dn->dl', kernel, self.gamma)
        return self._kernel

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self.compute_kernel(kernel_size)
        elif self._kernel is None:
            return self.compute_kernel(kernel_size)
        elif self._kernel.size(-1) < kernel_size:
            return self.compute_kernel(kernel_size)
        else:
            return self._kernel[..., :kernel_size]

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

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        # D x L
        k = self.kernel(seq_len)
        fft_len = seq_len
        s = 0
        kernel_size = k.size(1)
        if self.bidirectional:
            k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
            # D x 2*L-1
            k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
            x = F.pad(x, (kernel_size - 1, 0))
            fft_len = fft_len + kernel_size - 1
            s = 2 * kernel_size - 2

        k_f = torch.fft.rfft(k, n=2 * fft_len)
        x_f = torch.fft.rfft(x, n=2 * fft_len)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]

        # B x D x L -> L x B x D
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
