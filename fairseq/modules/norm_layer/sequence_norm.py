# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from fairseq.modules.norm_layer.layer_norm import LayerNorm
from fairseq.modules.norm_layer.scale_norm import ScaleNorm
from fairseq.modules.norm_layer.root_mean_square_norm import RMSNorm
from fairseq.modules.norm_layer.dual_norm import MaskedBatchNorm, DualNorm2


class SequenceNorm(nn.Module):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == 'maskedbatchnorm':
            self.norm = MaskedBatchNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'dualnorm2':
            self.norm = DualNorm2(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'layernorm':
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=affine, export=export)
        elif norm_type == 'scalenorm':
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def normalize(self, x, padding_mask=None):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            return x.permute(2, 0, 1)
        elif isinstance(self.norm, MaskedBatchNorm) or isinstance(self.norm, DualNorm2):
            return self.norm(x, padding_mask)
        else:
            return self.norm(x)

    def forward(self, x, padding_mask=None):
        return self.normalize(x, padding_mask)
