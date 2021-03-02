# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

@with_incremental_state
class LunarMultiheadAttention(nn.Module):
    """Lunar Multi-headed attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        pembed_dim,
        num_heads,
        num_pheads,
        dropout=0.0,
        attention_dropout=0.0,
        bias=True,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pembed_dim = pembed_dim
        self.num_pheads = num_pheads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.attention_dropout_module = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        self.phead_dim = pembed_dim // num_pheads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        assert (self.phead_dim * num_pheads == self.pembed_dim), "projected embed_dim must be divisible by num_pheads"
        self.scaling = self.head_dim ** -0.5
        self.pscaling = self.phead_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.ck_proj = quant_noise(nn.Linear(embed_dim, pembed_dim, bias=bias), q_noise, qn_block_size)
        self.cv_proj = quant_noise(nn.Linear(embed_dim, pembed_dim, bias=bias), q_noise, qn_block_size)
        self.pck_proj = quant_noise(nn.Linear(pembed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.pcv_proj = quant_noise(nn.Linear(pembed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
        raise NotImplementedError('onnx for linear attention not implemented')

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True
        raise NotImplementedError('TPU for linear attention not implemented')

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.ck_proj.weight, gain=gain)
        if self.ck_proj.bias is not None:
            nn.init.constant_(self.ck_proj.bias, 0.)
        nn.init.xavier_uniform_(self.cv_proj.weight, gain=gain)
        if self.cv_proj.bias is not None:
            nn.init.constant_(self.cv_proj.bias, 0.)
        nn.init.xavier_uniform_(self.pck_proj.weight, gain=gain)
        if self.pck_proj.bias is not None:
            nn.init.constant_(self.pck_proj.bias, 0.)
        nn.init.xavier_uniform_(self.pcv_proj.weight, gain=gain)
        if self.pcv_proj.bias is not None:
            nn.init.constant_(self.pcv_proj.bias, 0.)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def _compute_pcontext_singlehead(self, pquery, context, context_padding_mask):
        # N x B x D -> B x D x N
        k = self.ck_proj(context).permute(1, 2, 0)
        # N x B x D -> B x N x D
        v = self.cv_proj(context).transpose(0, 1)

        # L x B x D -> B x L x D
        pq = pquery.transpose(0, 1) * self.pscaling
        # B x L x N
        pqc = pq.matmul(k)
        if context_padding_mask is not None:
            pqc = pqc.masked_fill(context_padding_mask.unsqueeze(1).to(torch.bool), float("-inf"))
        pqc = F.softmax(pqc, dim=-1)
        pqc = self.attention_dropout_module(pqc)
        # B x L x D -> L x B x D
        pc = torch.bmm(pqc, v).transpose(0, 1)
        return pc

    def _compute_pcontext_multiheads(self, pquery, context, context_padding_mask):
        # N x B x D
        k = self.ck_proj(context)
        v = self.cv_proj(context)
        len, bsz, dim = k.size()
        # N x B x D -> N x B x H x K
        k = k.view(len, bsz, self.num_pheads, self.phead_dim)
        v = v.view(len, bsz, self.num_pheads, self.phead_dim)
        # N x B x H x K -> B x H x K x N
        k = k.permute(1, 2, 3, 0)
        # N x B x H x K -> B x H x N x K
        v = v.permute(1, 2, 0, 3)

        plen = pquery.size(0)
        # L x B x D -> L x B x H x K
        pq = pquery.view(plen, -1, self.num_pheads, self.phead_dim)
        # L x B x H x K -> B x H x L x K
        pq = pq.permute(1, 2, 0, 3) * self.pscaling
        # B x H x L x N
        pqc = pq.matmul(k)
        if context_padding_mask is not None:
            pqc = pqc.masked_fill(context_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
        pqc = F.softmax(pqc, dim=-1)
        pqc = self.attention_dropout_module(pqc)
        # B x H x L x K
        pc = torch.matmul(pqc, v)
        # B x H x L x K -> L x B x H x K
        pc = pc.permute(2, 0, 1, 3).contiguous()
        pc = pc.view(plen, bsz, dim)
        return pc

    def compute_pcontext(self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        static_context: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, None]:

        if context is None:
            return context
        else:
            if self.num_pheads == 1:
                return self._compute_pcontext_singlehead(pquery, context, context_padding_mask)
            else:
                return self._compute_pcontext_multiheads(pquery, context, context_padding_mask)

    def forward(
        self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_context: bool = False,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            context_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if attn_mask is not None:
            raise NotImplementedError('Causal attention has not implemented.')

        assert not self.self_attention or incremental_state is None, 'linear self attention has not supported incremental processing yet'

        if self.self_attention:
            context = query

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_context:
                    assert self.encoder_decoder_attention and not self.self_attention
                    context = None
        else:
            saved_state = None

        # L x B x D
        pcontext = self.compute_pcontext(query, pquery, context, context_padding_mask,
                                         incremental_state, static_context, attn_mask)
        key_padding_mask = None

        if pcontext is None:
            assert context is None
            k = v = None
        else:
            k = self.pck_proj(pcontext)
            v = self.pcv_proj(pcontext)

        q = query * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            # pcontext are stored with shape (bsz, proj_len, model_dim)
            if "prev_pcontext" in saved_state:
                # TODO save prev_pcontext for causal attention
                _prev_pcontext = saved_state["prev_pcontext"]
                assert _prev_pcontext is not None
                prev_pcontext = _prev_pcontext.transpose(0, 1)
                if static_context:
                    pcontext = prev_pcontext
                else:
                    raise RuntimeError('pcontext error')
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None and pcontext is not None
            key_padding_mask = LunarMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_context,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            saved_state["prev_pcontext"] = pcontext.transpose(0, 1)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None and v is not None and pcontext is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = LunarMultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, pcontext, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


@with_incremental_state
class LunarCausalAttention(nn.Module):
    """Lunar Causal attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        pembed_dim,
        num_heads,
        num_pheads,
        dropout=0.0,
        attention_dropout=0.0,
        bias=True,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pembed_dim = pembed_dim
        self.num_pheads = num_pheads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.attention_dropout_module = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        self.phead_dim = pembed_dim // num_pheads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        assert (self.phead_dim * num_pheads == self.pembed_dim), "projected embed_dim must be divisible by num_pheads"
        self.scaling = self.head_dim ** -0.5
        self.pscaling = self.phead_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.ck_proj = quant_noise(nn.Linear(embed_dim, pembed_dim, bias=bias), q_noise, qn_block_size)
        self.cv_proj = quant_noise(nn.Linear(embed_dim, pembed_dim, bias=bias), q_noise, qn_block_size)
        self.pck_proj = quant_noise(nn.Linear(pembed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.pcv_proj = quant_noise(nn.Linear(pembed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
        raise NotImplementedError('onnx for linear attention not implemented')

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True
        raise NotImplementedError('TPU for linear attention not implemented')

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.ck_proj.weight, gain=gain)
        if self.ck_proj.bias is not None:
            nn.init.constant_(self.ck_proj.bias, 0.)
        nn.init.xavier_uniform_(self.cv_proj.weight, gain=gain)
        if self.cv_proj.bias is not None:
            nn.init.constant_(self.cv_proj.bias, 0.)
        nn.init.xavier_uniform_(self.pck_proj.weight, gain=gain)
        if self.pck_proj.bias is not None:
            nn.init.constant_(self.pck_proj.bias, 0.)
        nn.init.xavier_uniform_(self.pcv_proj.weight, gain=gain)
        if self.pcv_proj.bias is not None:
            nn.init.constant_(self.pcv_proj.bias, 0.)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def _compute_pcontext_singlehead(self, pquery, context, context_padding_mask):
        # N x B x D -> B x D x N
        k = self.ck_proj(context).permute(1, 2, 0)
        # N x B x D -> B x N x D
        v = self.cv_proj(context).transpose(0, 1)

        # L x B x D -> B x L x D
        pq = pquery.transpose(0, 1) * self.pscaling
        # B x L x N
        pqc = pq.matmul(k)
        if context_padding_mask is not None:
            pqc = pqc.masked_fill(context_padding_mask.unsqueeze(1).to(torch.bool), float("-inf"))
        pqc = F.softmax(pqc, dim=-1)
        pqc = self.attention_dropout_module(pqc)
        # B x L x D -> L x B x D
        pc = torch.bmm(pqc, v).transpose(0, 1)
        return pc

    def _compute_pcontext_multiheads(self, pquery, context, context_padding_mask):
        # N x B x D
        k = self.ck_proj(context)
        v = self.cv_proj(context)
        len, bsz, dim = k.size()
        # N x B x D -> N x B x H x K
        k = k.view(len, bsz, self.num_pheads, self.phead_dim)
        v = v.view(len, bsz, self.num_pheads, self.phead_dim)
        # N x B x H x K -> B x H x K x N
        k = k.permute(1, 2, 3, 0)
        # N x B x H x K -> B x H x N x K
        v = v.permute(1, 2, 0, 3)

        plen = pquery.size(0)
        # L x B x D -> L x B x H x K
        pq = pquery.view(plen, -1, self.num_pheads, self.phead_dim)
        # L x B x H x K -> B x H x L x K
        pq = pq.permute(1, 2, 0, 3) * self.pscaling
        # B x H x L x N
        pqc = pq.matmul(k)
        if context_padding_mask is not None:
            pqc = pqc.masked_fill(context_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
        pqc = F.softmax(pqc, dim=-1)
        pqc = self.attention_dropout_module(pqc)
        # B x H x L x K
        pc = torch.matmul(pqc, v)
        # B x H x L x K -> L x B x H x K
        pc = pc.permute(2, 0, 1, 3).contiguous()
        pc = pc.view(plen, bsz, dim)
        return pc

    def compute_pcontext(self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        static_context: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, None]:

        if context is None:
            return context
        else:
            if self.num_pheads == 1:
                return self._compute_pcontext_singlehead(pquery, context, context_padding_mask)
            else:
                return self._compute_pcontext_multiheads(pquery, context, context_padding_mask)

    def forward(
        self,
        query,
        pquery,
        context: Optional[Tensor],
        context_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_context: bool = False,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            context_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if attn_mask is not None:
            raise NotImplementedError('Causal attention has not implemented.')

        assert not self.self_attention or incremental_state is None, 'linear self attention has not supported incremental processing yet'

        if self.self_attention:
            context = query

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_context:
                    assert self.encoder_decoder_attention and not self.self_attention
                    context = None
        else:
            saved_state = None

        # L x B x D
        pcontext = self.compute_pcontext(query, pquery, context, context_padding_mask,
                                         incremental_state, static_context, attn_mask)
        key_padding_mask = None

        if pcontext is None:
            assert context is None
            k = v = None
        else:
            k = self.pck_proj(pcontext)
            v = self.pcv_proj(pcontext)

        q = query * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_context:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            # pcontext are stored with shape (bsz, proj_len, model_dim)
            if "prev_pcontext" in saved_state:
                # TODO save prev_pcontext for causal attention
                _prev_pcontext = saved_state["prev_pcontext"]
                assert _prev_pcontext is not None
                prev_pcontext = _prev_pcontext.transpose(0, 1)
                if static_context:
                    pcontext = prev_pcontext
                else:
                    raise RuntimeError('pcontext error')
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None and pcontext is not None
            key_padding_mask = LunarMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_context,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            saved_state["prev_pcontext"] = pcontext.transpose(0, 1)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None and v is not None and pcontext is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = LunarMultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, pcontext, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
