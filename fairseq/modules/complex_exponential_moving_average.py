# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_moving_average import BaseMovingLayer
from fairseq.incremental_decoding_utils import with_incremental_state

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


@with_incremental_state
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
    ):
        super().__init__()
        self.complex = True
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.theta = nn.Parameter(torch.Tensor(kernel_dim, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim, 2))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None

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
            nn.init.normal_(self.theta, mean=0.0, std=1.0)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            self.gamma[:, :, 1] = 0.
            # omega
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.alpha)
        delta = torch.sigmoid(self.delta)
        q = 1.0 - p * delta
        # D x 1 x 1
        theta = F.softplus(self.theta, beta=-1.0)
        # 1 x N
        wavelets = torch.arange(1, self.ndim + 1).to(p).view(1, self.ndim)
        # D x N x 1
        theta = torch.exp(wavelets.unsqueeze(2) * theta)
        # D x N x 1
        c = torch.cos(theta) + 1j * torch.sin(theta)
        # coeffs
        p = p * self.beta
        q = q * c
        # D x N
        gamma = _r2c(self.gamma) * self.scale
        return p, q, gamma

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q, gamma = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        # D x N x L
        kernel = p * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, gamma).real

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.truncation)
