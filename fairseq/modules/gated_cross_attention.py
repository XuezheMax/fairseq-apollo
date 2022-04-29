# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout


@with_incremental_state
class GatedCrossAttention(nn.Module):
    """Gated Structured State Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='tanh',
        attention_activation='softmax',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.zdim = zdim
        assert activation in ['tanh', 'sin']
        self.activation = utils.get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = FairseqDropout(hidden_dropout, module_name=self.__class__.__name__)

        self.k_proj = nn.Linear(embed_dim, zdim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim + zdim)
        self.h_proj = nn.Linear(embed_dim, embed_dim)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.k_proj.bias, 0.0)

        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def relu2_attention(self, q, k, key_padding_mask, before_attn_fn):
        bsz, slen, _ = k.size()
        if key_padding_mask is not None:
            # B x L1
            inverse_mask = 1.0 - key_padding_mask.type_as(q)
            # B x 1 x 1
            lengths = inverse_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = slen
            inverse_mask = None

        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2))
        qk = qk / lengths

        if inverse_mask is not None:
            qk = qk * inverse_mask.unsqueeze(1)

        if before_attn_fn:
            return qk

        attn_weights = utils.relu2(qk)
        return attn_weights

    def softmax_attention(self, q, k, key_padding_mask, before_attn_fn):
        q = q * self.scaling
        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2))

        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = utils.softmax(qk, dim=-1, onnx_trace=self.onnx_trace)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            static_kv (bool, optional): static key and value pair.
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                assert static_kv
                key = value = None
        else:
            saved_state = None

        # L2 x B x (2*D+S)
        base = self.q_proj(query)
        u, rq = torch.split(base, [self.embed_dim, self.embed_dim + self.zdim], dim=-1)

        # L2 x B x D
        u = torch.sigmoid(u)

        # L2 x B x (D+S)
        rq = F.silu(rq)
        r, q = torch.split(rq, [self.embed_dim, self.zdim], dim=-1)
        q = q * self.gamma[0] + self.beta[0]

        if key is None:
            assert value is None
            k = v = None
        else:
            # L1 x B x S
            k = F.silu(self.k_proj(key))
            k = k * self.gamma[1] + self.beta[1]
            v = F.silu(self.v_proj(key))

        # N x B x S -> B x N x S
        q = q.transpose(0, 1)
        if k is not None:
            k = k.transpose(0, 1)
        if v is not None:
            v = v.transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                k = prev_key
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                v = prev_value
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                key_padding_mask = prev_key_padding_mask

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        ctx_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == ctx_len

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, key_padding_mask, before_attn_fn)
        elif self.attention_activation == 'relu2':
            attn_weights = self.relu2_attention(q, k, key_padding_mask, before_attn_fn)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if before_attn_fn:
            return attn_weights, v

        kernel = self.attention_dropout(attn_weights)
        # B x L1 x D -> L1 x B x D
        h = torch.bmm(kernel, v).transpose(0, 1)
        h = self.hidden_dropout(h)
        # L1 x B x D
        h = self.activation(self.h_proj(h * r))
        out = torch.addcmul(query, u, h - query)

        if need_weights:
            return out, attn_weights
        else:
            return out, None

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

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
                    if input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, attn_act={}'.format(self.embed_dim, self.zdim, self.attention_activation)