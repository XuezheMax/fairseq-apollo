import math
from typing import Dict, Optional, Tuple

import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq.incremental_decoding_utils import with_incremental_state

logger = logging.getLogger(__name__)

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


@with_incremental_state
class S4D(nn.Module):
    def __init__(
        self,
        embed_dim,
        ndim=16,
        bidirectional=False,
        disc='bilinear'
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.dt_max = 0.1
        self.dt_min = 0.001
        self.disc = disc
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.log_dt = nn.Parameter(Tensor(kernel_dim, 1))
        self.log_A_real = nn.Parameter(Tensor(kernel_dim, ndim))
        self.A_imaginary = nn.Parameter(Tensor(kernel_dim, ndim))
        self.B = nn.Parameter(Tensor(kernel_dim, ndim, 2))
        self.C = nn.Parameter(Tensor(kernel_dim, ndim, 2))
        self.D = nn.Parameter(Tensor(embed_dim))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        log_dt = torch.randn_like(self.log_dt) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        with torch.no_grad():
            self.log_dt.copy_(log_dt)

        nn.init.constant_(self.log_A_real, math.log(0.5))
        # N
        imaginary = torch.arange(self.ndim)
        imaginary = 1.0 / math.pi * self.ndim * (self.ndim / (1 + 2 * imaginary) - 1)
        with torch.no_grad():
            self.A_imaginary.copy_(imaginary)

        # N x 2
        B = _c2r(torch.ones(self.ndim, dtype=torch.cfloat))
        C = _c2r(torch.randn(self.ndim, dtype=torch.cfloat))
        with torch.no_grad():
            self.B.copy_(B)
            self.C.copy_(C)

        nn.init.normal_(self.D, mean=0.0, std=1.0)

    def calc_A(self):
        # D x N
        A_real = -torch.exp(self.log_A_real)
        A = A_real + 1j * self.A_imaginary
        return A

    def compute_kernel(self, length: int):
        # D x 1
        dt = torch.exp(self.log_dt)
        # D x N
        A = self.calc_A()
        # D x N
        dtA = A * dt
        # D x N
        C = (_r2c(self.B) * _r2c(self.C))

        if self.disc == 'bilinear':
            # D x N
            p = C * dt * (1. - dtA / 2).reciprocal()
            log_q = torch.log((1. + dtA / 2) / (1. - dtA / 2))
        elif self.disc == 'zoh':
            # D x N
            p = C * (torch.exp(dtA) - 1.) / A
            log_q = dtA
        else:
            raise ValueError("Unknown discretization method: {}".format(self.disc))

        # D x N x L
        vander = torch.arange(length).to(dt).view(1, 1, length) * log_q.unsqueeze(-1)
        # D x N x L
        kernel = torch.exp(vander)

        # D x L
        self._kernel = torch.einsum('dnl,dn->dl', kernel, p * self.scale).real
        return self._kernel

    def kernel(self, length: int):
        return self.compute_kernel(length)

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        raise NotImplementedError

    def one_step(self, x, hx=None):
        raise NotImplementedError

    def forward(
        self,
        x,
        padding_mask: Optional[torch.Tensor] = None,
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

        # L x B x D
        residual = x * self.D

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional S4D does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            out = F.silu(out + residual)
        else:
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

            k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
            x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
            # B x D x L
            out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
            out = out.type_as(x)
            # B x D x L -> L x B x D
            out = F.silu(out.permute(2, 0, 1) + residual)

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

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, disc={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.disc)
