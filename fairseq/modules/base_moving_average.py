# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq.incremental_decoding_utils import with_incremental_state
from .fused_ops.fftconv import fftconv, bidirectional_fftconv
from .fused_ops.ema_hidden import ema_hidden


@with_incremental_state
class BaseMovingLayer(nn.Module):
    """Base Class for Moving Layers
    """

    def __init__(self, bidirectional=False, truncation=None, shift=True):
        super().__init__()
        self._parameters_no_weight_decay = OrderedDict()
        self.complex = False
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.shift = shift
        assert self.bidirectional or (self.truncation is None or self.truncation < 1), \
            'one directional moving average should not have positive trunction: {}'.format(truncation)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def register_parameters_no_weight_decay(self, name, param):
        self._parameters_no_weight_decay[name] = param

    def parameters_no_weight_decay(self):
        for name, param in self._parameters_no_weight_decay.items():
            yield name, param

    def _calc_coeffs(self):
        raise NotImplementedError

    def _compute_kernel(self, length: int, hx: Tensor):
        raise NotImplementedError

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int, hx: Optional[Tensor]):
        kernel_size = length if self.truncation is None or self.truncation < 1 else min(self.truncation, length)
        return self._compute_kernel(kernel_size, hx)

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q, gamma = self.coeffs()
        # D x N x L+1
        vander = torch.arange(length + 1).to(q).view(1, 1, length + 1) * torch.log(q)
        vander = torch.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * gamma.unsqueeze(-1)
            ox = torch.einsum('bdn,dnl->bdl', hx, k)
            if self.complex:
                ox = ox.real
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = p * vander if p is not None else vander
        k = torch.einsum('dnl,dn->dl', kernel, gamma)
        if self.complex:
            k = k.real

        k_f = torch.fft.rfft(k, n=2 * length, norm="forward")
        x_f = torch.fft.rfft(x, n=2 * length)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * length, norm="forward")[..., 0:length]
        if ox is not None:
            out = out + ox

        h = torch.einsum('bdl,dnl->bdn', x.to(kernel), torch.flip(kernel, dims=[2]))
        if hh is not None:
            h = h + hh
        # B x D x L, B x D x N
        return out, h

    def one_step(self, x, hx=None):
        p, q, gamma = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = p.squeeze(-1) * x if p is not None else x
        if hx is not None:
            h = h + q.squeeze(-1) * hx
        else:
            h = h.to(q)
        # B x D
        out = torch.einsum('bdn,dn->bd', h, gamma)
        if self.complex:
            out = out.real
        # B x D x 1, B x D x N
        return out.unsqueeze(-1), h

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

        bsz, embed_dim, seq_len = x.size()
        assert embed_dim == self.embed_dim

        # B x D x L
        residual = x * self.omega

        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).to(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            # out, h = self.step(x.float(), seq_len, hx=h)
            # out = out.to(x)
            # D x N x 1
            p, q, _ = self.coeffs()
            k, b = self.kernel(seq_len, hx=h)
            out = fftconv(x, k)
            if b is not None:
                out = out + b
            h = ema_hidden(x, p, q, h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
        else:
            # D x L
            k, b = self.kernel(seq_len, None)
            assert b is None
            # B x D x L
            if self.bidirectional:
                out = bidirectional_fftconv(x, k, self.shift)
            else:
                out = fftconv(x, k)

        out = out + residual
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
