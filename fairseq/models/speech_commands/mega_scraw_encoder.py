# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    TimeNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class MegaSCRawEncoder(nn.Module):
    """
    Implementation for a Bi-directional FLASH based Sentence Encoder used
    in masked pre-trained language models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        num_encoder_layers: int = 6,
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
        moving_layer: str = 'cema',
        layerdrop: float = 0.0,
        truncation: int = None,
        norm_type='layernorm',
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 16000,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
        layer_scale: bool = False,
        init_mode='gaussian',
        export: bool = False,
    ):

        super().__init__()
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        self.embed_tokens = RealNumberEmbedding(embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        ls_weights = [0.1 * (0.5 ** i) for i in range(self.num_layers)] if layer_scale else [None, ] * self.num_layers
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
                rel_pos_bias=rel_pos_bias,
                max_positions=max_seq_len,
                attention_activation=attention_activation,
                norm_type=norm_type,
                norm_affine=norm_affine,
                norm_eps=norm_eps,
                layer_scale=ls_weights[i],
                init_mode=init_mode,
                export=export
            )
            for i in range(self.num_layers)
        ])

        self.final_norm = TimeNorm(embedding_dim, affine=norm_affine, eps=norm_eps, causal=False)
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

    def forward(
        self,
        tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        last_state_only: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        bsz, seq_len = tokens.size()
        assert self.chunk_size <= 0 or seq_len % self.chunk_size == 0, 'sequence length {} must be divided by chunk size {}'.format(seq_len, self.chunk_size)

        padding_mask = None
        # B x T -> B x T x D
        x = self.embed_tokens(tokens)
        x = self.embedding_dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        x = self.final_norm(x)
        x = F.silu(self.final_proj(x))

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
