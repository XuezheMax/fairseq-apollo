# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
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


class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, embed_dim, max_positions, base=None):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = max_positions / (math.pi * 2) if base is None else base
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_freqs(max_positions, embed_dim, self.base)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_sinusoid_freqs(max_positions: int, embedding_dim: int, base: float):
        half_dim = embedding_dim // 2
        freqs = math.log(base) / half_dim
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float) * -freqs)
        # C x D/2
        freqs = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * freqs.unsqueeze(0)
        # C x D
        freqs = freqs.repeat_interleave(2, dim=-1)
        return torch.sin(freqs), torch.cos(freqs)

    def rotate_half(self, x):
        B, N, C, D = x.shape
        x = x.reshape((B, N, C, D // 2, 2))
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        B, N, C, D, E = x.shape
        return x.reshape((B, N, C, D * E))

    def rotary(self, x, sidx):
        B, N, C, D = x.shape
        eidx = sidx + C
        if self.sine is None or eidx > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_freqs(eidx, D)
            self.max_positions = eidx
        # C x D
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[sidx:eidx]
        cos = self.cosine[sidx:eidx]
        return x * cos + self.rotate_half(x) * sin

    def forward(self, xq, xk, qidx=0):
        xq = self.rotary(xq, qidx)
        xk = self.rotary(xk, 0)
        return xq, xk

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}, base={:.1f}'.format(self.embed_dim, self.max_positions, self.base)
