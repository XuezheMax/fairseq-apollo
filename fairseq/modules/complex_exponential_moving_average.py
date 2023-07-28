# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn

from .base_moving_average import BaseMovingLayer
from .fused_ops.ema_filter import ema_filter

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class MultiHeadComplexEMA(BaseMovingLayer):
    """Complex Exponential Moving Average Layer.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
        shift=True,
    ):
        super().__init__(bidirectional, truncation, shift)
        self.complex = True
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.theta = nn.Parameter(torch.Tensor(kernel_dim, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim, 2))
        self.omega = nn.Parameter(torch.Tensor(embed_dim, 1))
        self._kernel = None
        self._coeffs = None

        self.register_parameters_no_weight_decay('alpha', self.alpha)
        self.register_parameters_no_weight_decay('delta', self.delta)
        self.register_parameters_no_weight_decay('theta', self.theta)

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            # theta
            freqs = math.log(self.embed_dim) / self.embed_dim
            freqs = torch.exp(torch.arange(1, self.embed_dim + 1) * -freqs)
            freqs = freqs.to(self.theta).view(self.embed_dim, 1, 1)
            if self.bidirectional:
                freqs = freqs.repeat(2, 1, 1)
            freqs = torch.log(freqs / (1.0 - freqs))
            self.theta.copy_(freqs)
            # gamma
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            self.gamma[:, :, 1] = 0.
            # omega
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x 1 x 1
        theta = torch.sigmoid(self.theta.float()) * (2 * math.pi / self.ndim)
        # 1 x N
        wavelets = torch.arange(1, self.ndim + 1, dtype=theta.dtype, device=theta.device).view(1, self.ndim)
        # D x N x 1
        theta = wavelets.unsqueeze(2) * theta

        # D x N x 1
        alpha = torch.sigmoid(self.alpha.float())
        delta = torch.sigmoid(self.delta.float())
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta)
        # D x N
        gamma = _r2c(self.gamma.float()) * self.scale
        return p, q, gamma

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q, gamma = self._calc_coeffs()
        if p.is_cuda:
            return ema_filter(p, q, gamma, length)

        # D x N x L
        vander = torch.arange(length, dtype=p.dtype, device=p.device).view(1, 1, length) * torch.log(q)
        # D x N x L
        kernel = p * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, gamma).real

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}, shift={}'.format(self.embed_dim, self.ndim, self.bidirectional,
                                                                                   self.truncation, self.shift)
