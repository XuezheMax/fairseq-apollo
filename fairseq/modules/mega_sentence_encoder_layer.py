# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    GatedStructuredStateAttention,
    MovingAverageGatedAttention,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class MegaSentenceEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        z_dim: int = 128,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        truncation=None,
        max_positions: int = 1024,
        activation='tanh',
        export: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.net = self.build_layer(embedding_dim, hidden_dim, z_dim,
                                    dropout, attention_dropout, hidden_dropout,
                                    truncation, max_positions, activation)

    def build_layer(self, embedding_dim, hidden_dim, z_dim,
                    dropout, attention_dropout, hidden_dropout,
                    truncation, max_positions, activation):
        return MovingAverageGatedAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            truncation=truncation,
            max_positions=max_positions,
            activation=activation,
            bidirectional=True
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):

        x, attn = self.net(x, x_padding_mask)
        x = self.dropout_module(x)

        return x, attn
