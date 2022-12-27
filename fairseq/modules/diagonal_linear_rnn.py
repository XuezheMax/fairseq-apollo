# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_moving_average import BaseMovingLayer

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class DiagonalLinearRNN(BaseMovingLayer):
    """Diagonal Linear RNN Layer.
    See "https://arxiv.org/abs/2212.00768" for more details.
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
        self.scale = 1.0 / self.ndim

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.theta = nn.Parameter(torch.Tensor(kernel_dim, 1, 1))
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
            # alpha
            nn.init.normal_(self.alpha, mean=0.0, std=0.1)
            # theta
            nn.init.normal_(self.theta, mean=0.0, std=1.0)
            # gamma
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            self.gamma[:, :, 1] = 0.
            # omega
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x 1 x 1
        theta = torch.sigmoid(self.theta) * (2 * math.pi / self.ndim)
        # 1 x N
        wavelets = torch.arange(1, self.ndim + 1).to(theta).view(1, self.ndim)
        # D x N x 1
        theta = wavelets.unsqueeze(2) * theta
        # D x N x 1
        c = torch.cos(theta) + 1j * torch.sin(theta)

        # D x N x 1
        alpha = 0.5 * (1.0 - torch.erf(self.alpha - math.sqrt(2.0)))
        # coeffs
        q = alpha * c
        # D x N
        gamma = _r2c(self.gamma) * self.scale
        return None, q, gamma

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        _, q, gamma = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(q).view(1, 1, length) * torch.log(q)
        # D x N x L
        kernel = torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, gamma).real

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}, shift={}'.format(self.embed_dim, self.ndim, self.bidirectional,
                                                                                   self.truncation, self.shift)
