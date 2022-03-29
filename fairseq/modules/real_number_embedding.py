# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
from fairseq.modules.layer_norm import LayerNorm


class RealNumberEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        self.bias = Parameter(torch.Tensor(embedding_dim))
        self.weight_norm = LayerNorm(embedding_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.weight, mean=0.0, std=std)
        nn.init.normal_(self.bias, mean=0.0, std=std)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight_norm(self.weight) + self.bias
