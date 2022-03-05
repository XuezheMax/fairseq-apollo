# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn

from fairseq.modules import (
    ScaleNorm,
    LayerDropModuleList,
    PositionalEmbedding,
    FlashSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class FlashLRAEncoder(nn.Module):
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
        z_dim: int = 128,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        use_position_embeddings: bool = False,
        offset_positions_by_padding: bool = True,
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.use_position_embeddings = use_position_embeddings
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim, self.padding_idx)
        self.embed_scale = embed_scale

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.use_position_embeddings and not self.learned_pos_embedding:
            if self.embed_scale is None:
                self.embed_scale = math.sqrt(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_flash_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                max_positions=self.max_seq_len,
                export=export
            )
            for _ in range(self.num_layers)
        ])

        self.layer_norm = ScaleNorm(dim=-1)

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = nn.Linear(1, embedding_dim, bias=True)
            nn.init.xavier_normal_(embed_tokens.weight)
            nn.init.normal_(embed_tokens.bias, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens

    def build_flash_sentence_encoder_layer(
        self,
        embedding_dim,
        hidden_dim,
        z_dim,
        dropout,
        attention_dropout,
        hidden_dropout,
        max_positions,
        export,
    ):
        return FlashSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_positions=max_positions,
            export=export
        )

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            padding_mask = None
            tokens = (tokens - 0.5) / 0.5
            # B x T -> B x T x 1 -> B x T x D
            x = self.embed_tokens(tokens.unsqueeze(2))

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        x = self.layer_norm(x)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.mean(dim=0)
        else:
            sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
