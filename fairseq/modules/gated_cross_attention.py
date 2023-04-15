# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.relative_positional_bias import SimpleRelativePositionalBias, RotaryRelativePositionalBias
from fairseq.modules.norm_layer.layer_norm import LayerNorm, RMSNorm


@with_incremental_state
class GatedCrossAttention(nn.Module):
    """Gated Structured State Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        ndim=2,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_activation='softmax',
        norm_type='layernorm',
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias=None,
        max_positions=1024,
        export=False,
        init_mode='bert',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.zdim = zdim
        self.ndim = ndim
        self.attention_activation = attention_activation
        self.init_mode = init_mode

        self.dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = FairseqDropout(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)

        if norm_type == 'layernorm':
            self.norm = LayerNorm(embed_dim, elementwise_affine=norm_affine, eps=norm_eps, export=export)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embed_dim, elementwise_affine=norm_affine, eps=norm_eps, export=export)
        else:
            raise ValueError('unknown norm type: {}'.format(norm_type))

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, zdim, bias=True)
        self.qru_proj = nn.Linear(embed_dim, 2 * embed_dim + zdim, bias=True)
        self.h_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        if rel_pos_bias is None:
            self.rel_pos_bias = None
        elif rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        assert init_mode in ['bert', 'xavier', 'he']
        self.reset_parameters(init_mode)

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self, mode):
        # weights
        if mode == 'bert':
            std = 0.02
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.qru_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        elif mode == 'he':
            a = math.sqrt(5.0)
            nn.init.kaiming_normal_(self.v_proj.weight, a=a)
            nn.init.kaiming_normal_(self.k_proj.weight, a=a)
            nn.init.kaiming_normal_(self.qru_proj.weight, a=a)
            nn.init.kaiming_normal_(self.h_proj.weight, a=a)
        elif mode == 'xavier':
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.qru_proj.weight)
            nn.init.xavier_uniform_(self.h_proj.weight)
        else:
            raise ValueError('Unknown init mode: {}'.format(mode))

        # bias
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.qru_proj.bias, 0.0)
        nn.init.constant_(self.h_proj.bias, 0.0)
        # gamma & beta
        nn.init.constant_(self.gamma, 0.0)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # B x L1
            inverse_mask = 1.0 - key_padding_mask.to(q)
            # B
            lengths = slen - key_padding_mask.sum(dim=-1)
            # B x 1 x 1
            len_scale = torch.rsqrt(lengths).to(q).view(bsz, 1, 1)
        else:
            len_scale = 1.0 / math.sqrt(clen)
            inverse_mask = None

        if self.rel_pos_bias is None:
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2)) * len_scale
        elif isinstance(self.rel_pos_bias, SimpleRelativePositionalBias):
            # L x L1
            bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
            if pidx is not None:
                assert q.size(1) == 1
                # L1
                bias = bias[pidx]
            else:
                # L2 x L1
                bias = bias[:slen]
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2)) * len_scale + bias
        elif isinstance(self.rel_pos_bias, RotaryRelativePositionalBias):
            if pidx is not None:
                assert q.size(1) == 1
                qidx = pidx
            else:
                qidx = 0
            # B x 1 x L x D
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            q, k = self.rel_pos_bias(q, k, qidx=qidx)
            # B x L x D
            q = q.squeeze(1)
            k = k.squeeze(1)
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2)) * len_scale
        else:
            raise ValueError('unknown relative position bias')

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = utils.relu2(qk).to(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = utils.laplace(qk).to(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1

        if self.rel_pos_bias is None:
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2))
        elif isinstance(self.rel_pos_bias, SimpleRelativePositionalBias):
            # L x L1
            bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
            if pidx is not None:
                assert q.size(1) == 1
                # L1
                bias = bias[pidx]
            else:
                # L2 x L1
                bias = bias[:slen]
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2)) + bias
        elif isinstance(self.rel_pos_bias, RotaryRelativePositionalBias):
            if pidx is not None:
                assert q.size(1) == 1
                qidx = pidx
            else:
                qidx = 0
            # B x 1 x L x D
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            q, k = self.rel_pos_bias(q, k, qidx=qidx)
            # B x L x D
            q = q.squeeze(1)
            k = k.squeeze(1)
            # B x L2 x L1
            qk = torch.bmm(q, k.transpose(1, 2))
        else:
            raise ValueError('unknown relative position bias')

        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = utils.softmax(qk, dim=-1, onnx_trace=self.onnx_trace).to(qk)
        return attn_weights

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                queries that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s.
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
            pidx = 0
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                assert static_kv
                key = value = None
        else:
            pidx = None
            saved_state = None

        nq = self.norm(query)

        gamma = self.gamma + 1.0
        # L2 x B x (2*D+S)
        base = self.qru_proj(nq)
        q, u, r = torch.split(base, [self.zdim, self.embed_dim, self.embed_dim], dim=-1)
        # L2 x B x S
        q = F.normalize(q, p=2, dim=-1, eps=1e-5)
        q = q * gamma[0] + self.beta[0]
        # L2 x B x D
        u = torch.sigmoid(u)
        r = F.silu(r)

        if key is None:
            assert value is None
            k = v = None
        else:
            # L1 x B x S
            k = self.k_proj(key)
            k = F.normalize(k, p=2, dim=-1, eps=1e-5)
            k = k * gamma[1] + self.beta[1]
            # L1 x B x D
            v = F.silu(self.v_proj(value))

        # L2 x B x S -> B x L2 x S
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
            if "prev_num_steps" in saved_state:
                _prev_num_steps = saved_state["prev_num_steps"]
                pidx = _prev_num_steps + 1

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = key_padding_mask
            saved_state["prev_num_steps"] = pidx
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
            attn_weights = self.softmax_attention(q, k, key_padding_mask, pidx, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, key_padding_mask, pidx, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        kernel = self.attention_dropout(attn_weights)
        # B x L2 x D -> L2 x B x D
        attn = torch.bmm(kernel, v).transpose(0, 1)
        attn = self.hidden_dropout(attn * r)
        # L2 x B x D
        h = F.silu(self.h_proj(attn))
        h = self.dropout(h)
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
                if input_buffer_k is not None and isinstance(input_buffer_k, Tensor):
                    if input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, ndim={}, attn_act={}, init={}'.format(self.embed_dim, self.zdim, self.ndim,
                                                                        self.attention_activation, self.init_mode)
