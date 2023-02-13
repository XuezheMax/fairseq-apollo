# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F

from .norm_layer.layer_norm import RMSNorm, LayerNorm


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 embed_norm: Optional[str] = None, init_std: Optional[float] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2.,
                 scale_grad_by_freq: bool = False, sparse: bool = False,
                 _weight: Optional[Tensor] = None, device=None, dtype=None) -> None:

        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                         max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse, _weight=_weight, device=device, dtype=dtype)

        self.init_std = init_std

        if embed_norm is None:
            self.norm = None
        elif embed_norm == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, elementwise_affine=False)
        elif embed_norm == 'layernorm':
            self.norm = LayerNorm(embed_norm, elementwise_affine=False)
        else:
            raise ValueError('unknown norm type: {}'.format(norm_type))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.embedding_dim ** -0.5 if self.init_std is None else self.init_std
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self._fill_padding_idx_with_zero()

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        if self.norm is not None:
            emb = self.norm(emb)
        return emb

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.init_std is not None:
            s += ', init_std={init_std}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
