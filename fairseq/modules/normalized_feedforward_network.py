# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.norm_layer.layer_norm import LayerNorm, RMSNorm


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        norm_type='layernorm',
        norm_affine=True,
        norm_eps=1e-5,
        layer_scale=None,
        init_mode='gaussian',
        export=False,
    ):
        super().__init__()

        assert activation in ['silu', 'gelu']
        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = 'swiglu' if activation == 'silu' else 'geglu'
        # self.activation = utils.get_activation_fn(activation)
        self.init_mode = init_mode
        self.layer_scale = layer_scale

        self.dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = FairseqDropout(hidden_dropout, module_name=self.__class__.__name__)

        if norm_type == 'layernorm':
            self.norm = LayerNorm(embed_dim, elementwise_affine=norm_affine, eps=norm_eps, export=export)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embed_dim, elementwise_affine=norm_affine, eps=norm_eps, export=export)
        else:
            raise ValueError('unknown norm type: {}'.format(norm_type))

        self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim * 2, bias=True)
        self.fc2 = nn.Linear(ffn_hidden_dim, embed_dim, bias=True)
        if layer_scale is None:
            self.register_parameter('layerscale_weight', None)
        else:
            assert layer_scale > 0., 'Layer scale init value should be positive.'
            self.layerscale_weight = nn.Parameter(torch.Tensor(embed_dim))

        assert init_mode in ['gaussian', 'xavier']
        self.reset_parameters(init_mode)

    def reset_parameters(self, mode):
        # weights
        if mode == 'gaussian':
            std = 0.02
            nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        elif mode == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        else:
            raise ValueError('Unknown init mode: {}'.format(mode))
        # bias
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        # layer scale weight
        if self.layerscale_weight is not None:
            nn.init.constant_(self.layerscale_weight, self.layer_scale)

    def forward(self, x):
        residual = x
        # layernorm
        x = self.norm(x)
        # fc1
        x = F.glu(self.fc1(x), dim=-1)
        x = self.hidden_dropout(x)
        # fc2
        x = self.fc2(x)
        x = self.dropout(x)
        # residual
        if self.layerscale_weight is None:
            out = x + residual
        else:
            out = x * self.layerscale_weight + residual

        return out

    def extra_repr(self) -> str:
        return 'edim={}, hdim={}, act={}, init={}, layerscale={}'.format(self.embedding_dim, self.hidden_dim, self.act_fn,
                                                                         self.init_mode, self.layer_scale)
