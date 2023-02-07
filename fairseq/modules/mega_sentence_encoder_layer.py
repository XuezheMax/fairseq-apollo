# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn

from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention
from fairseq.modules.normalized_feedforward_network import NormalizedFeedForwardNetwork


class MegaSentenceEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        moving_layer='ema',
        truncation: int = None,
        norm_type='layernorm',
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias = 'simple',
        max_positions: int = 1024,
        activation='silu',
        attention_activation: str = 'softmax',
        layer_scale=None,
        init_mode='gaussian',
        export: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.mega_layer = MovingAverageGatedAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            ndim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            moving_layer=moving_layer,
            truncation=truncation,
            norm_affine=norm_affine,
            norm_eps=norm_eps,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=True,
            init_mode=init_mode,
        )

        if ffn_hidden_dim is not None and ffn_hidden_dim > 0:
            self.nffn = NormalizedFeedForwardNetwork(
                embed_dim=embedding_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                hidden_dropout=hidden_dropout,
                activation=activation,
                norm_type=norm_type,
                norm_affine=norm_affine,
                norm_eps=norm_eps,
                layer_scale=layer_scale,
                init_mode=init_mode,
                export=export,
            )
        else:
            self.nffn = None

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):

        seq_len = x.size(0)
        if self.chunk_size > 0:
            assert seq_len % self.chunk_size == 0, 'the input sequence length {} cannot be divided by chunk size {}'.format(seq_len, self.chunk_size)
        x, attn = self.mega_layer(x, x_padding_mask)

        if self.nffn is not None:
            x = self.nffn(x)

        return x, attn
