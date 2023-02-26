# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    FairseqDropout,
    MaskedBatchNorm,
    MegaEncoderLayer,
    MegaDecoderLayer
)
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("mega")
class MegaModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'],
                            help='activation function for attention mechanism')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--hidden-dropout', type=float, metavar='D',
                            help='dropout probability for hidden vectors in Mega.')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N',
                            help='encoder hidden dimension for Mega')
        parser.add_argument('--encoder-z-dim', type=int, metavar='N',
                            help='encoder z dimension for Mega')
        parser.add_argument('--encoder-n-dim', type=int, metavar='N',
                            help='encoder n dimension for Mega')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-chunk-size', type=int, metavar='N',
                            help='chunk size of Mega encoder.')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')

        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-dim', type=int, metavar='N',
                            help='decoder hidden dimension for Mega')
        parser.add_argument('--decoder-z-dim', type=int, metavar='N',
                            help='decoder z dimension for Mega')
        parser.add_argument('--decoder-n-dim', type=int, metavar='N',
                            help='decoder n dimension for Mega')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-chunk-size', type=int, metavar='N',
                            help='chunk size of Mega decoder.')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--representation-dim', type=int, metavar='N', default=None,
                            help='the dimension of pre-logict representation')

        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings (requires shared dictionary and embed dim)')

        parser.add_argument('--embedding-max-norm', action='store_true', default=False, help='max norm of embeddings')
        parser.add_argument('--rel-pos-bias', choices=['simple', 'rotary'], default='simple')
        parser.add_argument('--moving-layer', choices=['ema', 'cema'], default='cema')
        parser.add_argument('--truncation-length', type=int, metavar='N', default=0,
                            help='truncation length of moving average layer.')
        parser.add_argument('--norm-type', choices=['layernorm', 'rmsnorm'], default='layernorm')
        parser.add_argument('--norm-eps', type=float, default=1e-5, help='normalization eps')
        parser.add_argument('--no-affine-norm', action='store_true', default=False,
                            help='no affine parameters in normalization layers.')
        parser.add_argument('--layer-scale', default=False, action='store_true', help='use layer scale')
        parser.add_argument('--init-mode', choices=['bert', 'xavier', 'he'], default='bert')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        assert args.encoder_embed_dim == args.decoder_embed_dim, 'Mega requires --encoder-embed-dim to match --decoder-embed-dim'

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")

            encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.embedding_max_norm, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim, args.embedding_max_norm, args.encoder_embed_path)
            decoder_embed_tokens = cls.build_embedding(args, tgt_dict, args.decoder_embed_dim, args.embedding_max_norm, args.decoder_embed_path)

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, embed_max_norm, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        max_norm = 2.0 if embed_max_norm else None
        emb = Embedding(num_embeddings, embed_dim, padding_idx, max_norm)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MegaEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MegaDecoder(args, tgt_dict, embed_tokens)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(self, net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
                             log_probs: bool, sample: Optional[Dict[str, Tensor]] = None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class MegaEncoder(FairseqEncoder):
    """
    Mega encoder consisting of *args.encoder_layers* layers. Each layer is a :class:`MegaEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.embedding_dropout = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.max_source_positions = args.max_source_positions
        self.chunk_size = args.encoder_chunk_size
        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList([])
        depth = args.encoder_layers
        lsw = [0.1 * (0.5 ** i) for i in range(depth)] if args.layer_scale else [None, ] * depth
        self.layers.extend([self.build_encoder_layer(args, lsw[i]) for i in range(depth)])
        self.num_layers = len(self.layers)

        norm_affine = not args.no_affine_norm
        self.final_norm = MaskedBatchNorm(embed_dim, affine=norm_affine, eps=args.norm_eps)

    def build_encoder_layer(self, args, layer_scale):
        return MegaEncoderLayer(args, layer_scale=layer_scale)

    def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        bsz, seq_len = src_tokens.size()
        if 0 < self.chunk_size < seq_len and seq_len % self.chunk_size != 0:
            num_paddings = math.ceil(seq_len / self.chunk_size) * self.chunk_size - seq_len
            src_tokens = F.pad(src_tokens, (0, num_paddings), value=self.padding_idx)
        else:
            num_paddings = 0

        x = encoder_embedding = self.embed_tokens(src_tokens)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embedding_dropout(x)

        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            # B x T
            inverse_mask = 1.0 - encoder_padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        x = self.final_norm(x, encoder_padding_mask)
        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        # remove padding tokens for chunk
        if num_paddings > 0:
            x = x[:seq_len]
            encoder_embedding = encoder_embedding[:, :seq_len]
            if encoder_padding_mask is not None:
                encoder_padding_mask = encoder_padding_mask[:, :seq_len]

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class MegaDecoder(FairseqIncrementalDecoder):
    """
    Mega decoder consisting of *args.decoder_layers* layers. Each layer is a :class:`MegaDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embedding_dropout = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.max_target_positions = args.max_target_positions
        self.chunk_size = args.decoder_chunk_size
        self.attention_activation = args.attention_activation_fn

        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList([])
        depth = args.decoder_layers
        lsw = [0.1 * (0.5 ** i) for i in range(depth)] if args.layer_scale else [None, ] * depth
        self.layers.extend([self.build_decoder_layer(args, lsw[i]) for i in range(depth)])
        self.num_layers = len(self.layers)

        norm_affine = not args.no_affine_norm
        self.final_norm = MaskedBatchNorm(embed_dim, affine=norm_affine, eps=args.norm_eps)

        if args.representation_dim is not None:
            self.num_features = args.representation_dim
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', Linear(embed_dim, args.representation_dim, bias=True, init_mode=args.init_mode)),
                ('act', nn.GELU()),
                ('dropout', FairseqDropout(args.dropout, module_name=self.__class__.__name__))
            ]))
        else:
            self.num_features = self.embed_dim
            self.pre_logits = nn.Identity()

        if self.share_input_output_embed:
            assert args.representation_dim is None or args.representation_dim == self.embed_dim
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(self.num_features, len(dictionary), bias=False)
            std = 1.0 / math.sqrt(self.num_features)
            nn.init.normal_(self.output_projection.weight, mean=0, std=std)

    def build_decoder_layer(self, args, layer_scale):
        return MegaDecoderLayer(args, layer_scale=layer_scale)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seq_len = prev_output_tokens.size()
        if 0 < self.chunk_size < seq_len and seq_len % self.chunk_size != 0:
            num_paddings = math.ceil(seq_len / self.chunk_size) * self.chunk_size - seq_len
            prev_output_tokens = F.pad(prev_output_tokens, (0, num_paddings), value=self.padding_idx)
        else:
            num_paddings = 0

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        x = self.embedding_dropout(x)

        # account for padding while computing the representation
        if decoder_padding_mask is not None:
            # B x T
            inverse_mask = 1.0 - decoder_padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                attn_mask = self.buffered_future_mask(x)
            else:
                attn_mask = None

            x, layer_attn, _ = layer(x, encoder_out.encoder_out, encoder_out.encoder_padding_mask, incremental_state,
                                     attn_mask=attn_mask, decoder_padding_mask=decoder_padding_mask,
                                     need_attn=bool(idx == alignment_layer))
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        x = self.final_norm(x, decoder_padding_mask)
        x = self.pre_logits(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        # remove padding tokens for chunk
        if num_paddings > 0:
            x = x[:seq_len]
            attn = attn[:, :seq_len]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if 0 < self.chunk_size < dim:
            assert dim % self.chunk_size == 0
            dim = self.chunk_size
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            if self.attention_activation == 'softmax':
                self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros(dim, dim)), 1)
            elif self.attention_activation in ['relu2', 'laplace']:
                self._future_mask = torch.tril(torch.ones(dim, dim), 0)
            else:
                raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


def Linear(in_features, out_features, bias=True, init_mode='xavier'):
    m = nn.Linear(in_features, out_features, bias)
    if init_mode == 'bert':
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif init_mode == 'he':
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5.0))
    elif init_mode == 'xavier':
        nn.init.xavier_uniform_(m.weight)
    else:
        raise ValueError('Unknown init mode: {}'.format(init_mode))
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx, max_norm):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)
    std = 1.0 / math.sqrt(embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture("mega", "mega")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_hidden_dim = getattr(args, "encoder_hidden_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_hidden_dim)
    args.encoder_z_dim = getattr(args, 'encoder_z_dim', 128)
    args.encoder_n_dim = getattr(args, 'encoder_n_dim', 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_chunk_size = getattr(args, 'encoder_chunk_size', -1)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", args.encoder_hidden_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', args.encoder_z_dim)
    args.decoder_n_dim = getattr(args, 'decoder_n_dim', args.encoder_n_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_chunk_size = getattr(args, 'decoder_chunk_size', args.encoder_chunk_size)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.hidden_dropout = getattr(args, "hidden_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.rel_pos_bias = getattr(args, 'rel_pos_bias', 'simple')
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.attention_activation_fn = getattr(args, 'attention_activation_fn', 'softmax')
    args.moving_layer = getattr(args, 'moving_layer', 'cema')
    args.truncation_length = getattr(args, 'truncation_length', 0)
    args.norm_type = getattr(args, 'norm_type', 'layernorm')
    args.no_affine_norm = getattr(args, 'no_affine_norm', False)
    args.layer_scale = getattr(args, 'layer_scale', False)


@register_model_architecture("mega", "mega_wmt_en_de")
def mega_wmt_en_de(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.hidden_dropout = getattr(args, "hidden_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    base_architecture(args)


@register_model_architecture("mega", "mega_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_hidden_dim = getattr(args, "encoder_hidden_dim", 2048)
    args.encoder_z_dim = getattr(args, 'encoder_z_dim', 256)
    args.dropout = getattr(args, "dropout", 0.3)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.hidden_dropout = getattr(args, "hidden_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    base_architecture(args)
