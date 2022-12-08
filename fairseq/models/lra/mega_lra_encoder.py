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
    MaskedBatchNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class MegaLRAEncoder(nn.Module):
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
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        activation: str = 'silu',
        attention_activation: str = 'softmax',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        moving_layer: str = 'ema',
        normalize_embedding: bool = False,
        feature_dropout: bool = False,
        layerdrop: float = 0.0,
        truncation: int = None,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 256,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.embedding_dim,
                                                 self.vocab_size, self.padding_idx,
                                                 not normalize_embedding)

        self.embed_norm = MaskedBatchNorm(embedding_dim) if normalize_embedding else None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_mega_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
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
                max_positions=self.max_seq_len,
                activation=activation,
                attention_activation=attention_activation,
                feature_dropout=feature_dropout,
                export=export
            )
            for _ in range(self.num_layers)
        ])

        self.final_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.final_proj.weight)
        self.final_norm = MaskedBatchNorm(embedding_dim)

    def build_embedding(self, embedding_type, embedding_dim, vocab_size, padding_idx, bias):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim, bias=bias)
            return embed_tokens

    def build_mega_sentence_encoder_layer(
        self,
        embedding_dim,
        hidden_dim,
        ffn_hidden_dim,
        z_dim,
        n_dim,
        dropout,
        attention_dropout,
        hidden_dropout,
        chunk_size,
        moving_layer,
        truncation,
        rel_pos_bias,
        max_positions,
        activation,
        attention_activation,
        feature_dropout,
        export,
    ):
        return MegaSentenceEncoderLayer(
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
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            feature_dropout=feature_dropout,
            export=export
        )

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

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if self.embed_norm is not None:
            x = self.embed_norm(x, padding_mask)
        x = self.embedding_dropout(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            # B x N
            inverse_mask = 1.0 - padding_mask.type_as(x)
        else:
            inverse_mask = None

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        x = self.final_proj(x)
        x = self.final_norm(x, padding_mask)

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
