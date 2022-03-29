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
        self.weight_norm = self.build_weight_norm(embedding_dim, norm_type, export)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(1.0 / self.embedding_dim)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def build_weight_norm(self, embedding_dim, norm_type, export):
        if norm_type == None:
            return None
        elif norm_type == 'layernorm':
            return LayerNorm(embedding_dim, export=export)
        elif norm_type == 'scalenorm':
            return ScaleNorm(dim=0)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def forward(self, x):
        weight = self.weight
        if self.weight_norm is not None:
            weight = self.weight_norm(weight)
        return x.unsqueeze(-1) * weight + self.bias
