# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, LinearMultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor


class MoonEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, index):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.embed_dim = args.encoder_embed_dim
        self.proj_len = args.encoder_projected_length
        position_encodings = self._get_sinusoidal_positional_embedding()
        self.register_buffer('position_encodings', position_encodings)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.normalize_before = args.encoder_normalize_before

        self.self_attn = self.build_self_attention(self.embed_dim, self.proj_len, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def _get_sinusoidal_positional_embedding(self):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(self.proj_len, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(self.proj_len, -1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(self.proj_len, 1)], dim=1)
        return emb

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, proj_length, args):
        return LinearMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            proj_length,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x,
                              key_padding_mask=encoder_padding_mask,
                              position_encodings=self.position_encodings)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class MoonDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, index, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.embed_dim = args.decoder_embed_dim
        self.encoder_proj_len = args.encoder_projected_length
        self.decoder_proj_len = args.decoder_projected_length
        encoder_position_encodings = self._get_sinusoidal_positional_embedding(self.encoder_proj_len)
        self.register_buffer('encoder_position_encodings', encoder_position_encodings)
        decoder_position_encodings = self._get_sinusoidal_positional_embedding(self.decoder_proj_len)
        self.register_buffer('decoder_position_encodings', decoder_position_encodings)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        self.self_attn = self.build_self_attention(self.embed_dim, self.decoder_proj_len, args,
                                                   add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, self.encoder_proj_len, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True
        self.onnx_trace = False

    def _get_sinusoidal_positional_embedding(self, length):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(length, -1)
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
        return emb

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, proj_length, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, proj_length, args):
        return LinearMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            proj_length,
            kvdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            position_encodings=self.encoder_position_encodings,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
