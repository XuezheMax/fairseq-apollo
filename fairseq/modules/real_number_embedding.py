# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.scale_norm import ScaleNorm


class RealNumberEmbedding(nn.Module):
    def __init__(self, embedding_dim, norm_type=None, export=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        self.bias = Parameter(torch.Tensor(embedding_dim))
        if norm_type is None:
            self.embed_norm = None
        elif norm_type == 'layernorm':
            self.embed_norm = LayerNorm(embedding_dim, export=export)
        elif norm_type == 'scalenorm':
            self.embed_norm = ScaleNorm(dim=-1)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, x):
        if self.embed_norm is None:
            weight = self.weight
        else:
            weight = self.embed_norm(self.weight)
        return x.unsqueeze(-1) * weight + self.bias
