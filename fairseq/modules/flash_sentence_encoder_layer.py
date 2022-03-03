# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    LayerDropModuleList,
    GatedAttentionUnit,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout


class FlashSentenceEncoderLayer(nn.Module):
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
        hidden_dropout: float = 0.1,
        max_positions: int = 1024,
        export: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.gau = self.build_gated_attention_unit(embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions)
        self.layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_gated_attention_unit(self, embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions):
        return GatedAttentionUnit(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_positions=max_positions
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.layer_norm(x)

        x, attn = self.gau(x, x_padding_mask)

        x = self.dropout_module(x)
        x = residual + x

        return x, attn
