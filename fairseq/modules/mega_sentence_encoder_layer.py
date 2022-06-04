# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import math

import torch
import torch.nn as nn

from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention
from fairseq.modules.sequence_norm import SequenceNorm


class MegaSentenceEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 2,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        truncation=None,
        max_positions: int = 1024,
        activation='silu',
        attention_activation='softmax',
        norm_type: str = 'scalenorm',
        export: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.net = self.build_layer(embedding_dim, hidden_dim, z_dim, n_dim,
                                    dropout, attention_dropout, hidden_dropout,
                                    activation, attention_activation,
                                    chunk_size, truncation, max_positions)

        self.norm = SequenceNorm(norm_type, embedding_dim, export=export)

    def build_layer(self, embedding_dim, hidden_dim, z_dim, n_dim,
                    dropout, attention_dropout, hidden_dropout,
                    activation, attention_activation,
                    chunk_size, truncation, max_positions):
        return MovingAverageGatedAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            ndim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            truncation=truncation,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):

        seq_len = x.size(0)
        if self.chunk_size > 0:
            assert seq_len % self.chunk_size == 0, 'the input sequence length {} cannot be divided by chunk size {}'.format(seq_len, self.chunk_size)
        x, attn = self.net(x, x_padding_mask)
        x = self.norm(x)
        return x, attn
