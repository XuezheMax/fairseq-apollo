# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRelativePositionalBias(nn.Module):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.rel_pos_bias, mean=0, std=0.02)

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        b = self.rel_pos_bias[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        return t

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions, base=None):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = 10000 if base is None else base
        self.register_buffer("freqs", self._precompute_freqs())
        self.freqs_cis: Optional[torch.Tensor] = None

    def _precompute_freqs(self):
        freqs = [self.base ** (j / self.embed_dim) for j in range(0, self.embed_dim, 2)]
        freqs = torch.tensor(freqs, dtype=torch.float32)
        freqs = 1.0 / freqs
        return freqs

    @torch.no_grad()
    def _precompute_until(self, max_positions: int):
        assert self.max_positions <= max_positions
        self.max_positions = max_positions
        # C
        t = torch.arange(max_positions, dtype=torch.float, device=self.freqs.device)
        # C x D/2
        freqs = torch.outer(t, self.freqs.float())
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def get_freqs_cis(self, start: int, end: int) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_until(self.max_positions)
        if end > self.freqs_cis.shape[0]:
            warnings.warn('Extending rotary range from {} to {}'.format(self.max_positions, end))
            self.freqs_cis = self._precompute_until(end)
        return self.freqs_cis[start:end]  # type: ignore

    def rotary(self, x, sidx):
        seq_len = x.shape[2]
        freqs_cis = self.get_freqs_cis(sidx, sidx + seq_len)
        # B x N x C x D/2
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # B x N x C x D
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
        return x_out

    def forward(self, xq, xk, qidx=0):
        xq = self.rotary(xq, qidx)
        xk = self.rotary(xk, 0)
        return xq, xk

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}, base={:.1f}'.format(self.embed_dim, self.max_positions, self.base)
