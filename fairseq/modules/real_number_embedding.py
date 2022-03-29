# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import Parameter

from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.scale_norm import ScaleNorm


class RealNumberEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        self.bias = Parameter(torch.Tensor(embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = self.embedding_dim ** -0.5
        nn.init.normal_(self.weight, mean=0, std=std)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return torch.addcmul(self.bias, x.unsqueeze(-1), self.weight)
        # return x.unsqueeze(-1) * self.weight + self.bias
