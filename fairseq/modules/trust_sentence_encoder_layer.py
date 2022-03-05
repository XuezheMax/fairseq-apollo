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
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout


class TrustSentenceEncoderLayer(nn.Module):
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
        max_positions: int = 1024,
        activation='tanh',
        export: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.net = self.build_gated_attention_unit(embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions, activation)

    def build_gated_attention_unit(self, embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions, activation):
        return GatedStructuredStateAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_positions=max_positions,
            activation=activation
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):

        x, attn = self.net(x, x_padding_mask)
        x = self.dropout_module(x)

        return x, attn
