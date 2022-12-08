# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import Parameter


class RealNumberEmbedding(nn.Module):
    def __init__(self, embedding_dim, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.Tensor(embedding_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(embedding_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.weight
        if self.bias is not None:
            emb = emb + self.bias
        return emb
