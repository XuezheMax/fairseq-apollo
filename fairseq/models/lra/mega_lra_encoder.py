# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    TimeLayerNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class MegaLRAEncoder(nn.Module):

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        attention_activation: str = 'softmax',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        moving_layer: str = 'ema',
        layerdrop: float = 0.0,
        truncation: int = None,
        norm_type='layernorm',
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 256,
        embed_max_norm: bool = False,
        sen_rep_type: str = 'cls',
        layer_scale: bool = False,
        init_mode='bert',
        export: bool = False,
        traceable: bool = False,
    ):

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.embedding_dim, self.vocab_size, self.padding_idx, embed_max_norm)
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        ls_weights = [0.1 * (0.5 ** i) for i in range(self.num_layers)] if layer_scale else [None,] * self.num_layers
        self.layers.extend([
            MegaSentenceEncoderLayer(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                z_dim=z_dim,
                n_dim=n_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                chunk_size=chunk_size,
                moving_layer=moving_layer,
                truncation=truncation,
                norm_type=norm_type,
                norm_affine=norm_affine,
                norm_eps=norm_eps,
                rel_pos_bias=rel_pos_bias,
                max_positions=max_seq_len,
                attention_activation=attention_activation,
                layer_scale=ls_weights[i],
                init_mode=init_mode,
                export=export
            )
            for i in range(self.num_layers)
        ])

        self.final_norm = TimeLayerNorm(embedding_dim, affine=norm_affine, eps=norm_eps, causal=False)
        self.final_proj = nn.Linear(embedding_dim, embedding_dim)

        self.reset_parameters(init_mode)

    def reset_parameters(self, mode):
        # weights
        if mode == 'bert':
            std = 0.02
            nn.init.normal_(self.final_proj.weight, mean=0.0, std=std)
        elif mode == 'he':
            a = math.sqrt(5.0)
            nn.init.kaiming_normal_(self.final_proj.weight, a=a)
        elif mode == 'xavier':
            nn.init.xavier_uniform_(self.final_proj.weight)
        else:
            raise ValueError('Unknown init mode: {}'.format(mode))
        # bias
        nn.init.constant_(self.final_proj.bias, 0.0)

    def build_embedding(self, embedding_type, embedding_dim, vocab_size, padding_idx, embed_max_norm):
        if embedding_type == 'sparse':
            max_norm = math.sqrt(embedding_dim) if embed_max_norm else None
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx, max_norm=max_norm)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def forward(
        self,
        tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        last_state_only: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        bsz, seq_len = tokens.size()
        if self.chunk_size > 0 and seq_len > self.chunk_size and seq_len % self.chunk_size != 0:
            assert self.embedding_type == 'sparse', 'for image the sequence length {} must be divided by chunk size {}'.format(seq_len, self.chunk_size)

            num_paddings = math.ceil(seq_len / self.chunk_size) * self.chunk_size - seq_len
            tokens = F.pad(tokens, (0, num_paddings), value=self.padding_idx)

        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)

        x = self.embedding_dropout(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            # B x N
            inverse_mask = 1.0 - padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        x = self.final_norm(x, padding_mask)
        x = F.silu(self.final_proj(x) + x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
