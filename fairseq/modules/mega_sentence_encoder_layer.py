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
        efficient_attn: bool = False,
        chunk_size: int = -1,
        moving_layer='cema',
        moving_act='rmsnorm',
        truncation: int = None,
        norm_type='layernorm',
        norm_num_groups=None,
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias = 'rotary',
        max_positions: int = 1024,
        attention_activation: str = 'softmax',
        layer_scale=None,
        init_mode='bert',
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
            efficient_attn=efficient_attn,
            chunk_size=chunk_size,
            moving_layer=moving_layer,
            moving_act=moving_act,
            truncation=truncation,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
            norm_eps=norm_eps,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
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

        seq_len = x.size(1)
        if self.chunk_size > 0:
            info = 'the input sequence length {} cannot be divided by chunk size {}'.format(seq_len, self.chunk_size)
            assert seq_len < self.chunk_size or seq_len % self.chunk_size == 0, info
        y, attn = self.mega_layer(x, x_padding_mask)

        if self.nffn is not None:
            y = self.nffn(y, residual=x)

        return y, attn
