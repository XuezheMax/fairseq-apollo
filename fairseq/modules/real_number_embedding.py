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
            self._embed_norm = None
        elif norm_type == 'layernorm':
            self._embed_norm = LayerNorm(embedding_dim, export=export)
        elif norm_type == 'scalenorm':
            self._embed_norm = ScaleNorm(dim=-1)
        elif norm_type == 'batchnorm':
            self._embed_norm = nn.BatchNorm1d(embedding_dim)
        elif norm_type == 'syncbatchnorm':
            self._embed_norm = nn.SyncBatchNorm(embedding_dim)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def embed_norm(self, emb):
        if self._embed_norm is None:
            return emb
        elif isinstance(self._embed_norm, nn.modules.batchnorm._BatchNorm):
            emb = emb.transpose(-2, -1)
            emb = self._embed_norm(emb)
            return emb.transpose(-2, -1)
        else:
            return self._embed_norm(emb)

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.weight + self.bias
        emb = self.embed_norm(emb)
        return emb
