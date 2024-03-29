# Author: Xuezhe Ma (Max)
#
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
from fairseq.modules.relative_positional_bias import SimpleRelativePositionalBias, RotaryEmbedding
from fairseq.modules.norm_layer.layer_norm import RMSNorm
from fairseq.modules.norm_layer.sequence_norm import SequenceNorm
from fairseq.modules.norm_layer.timestep_norm import TimestepNorm
from fairseq.modules.exponential_moving_average import MultiHeadEMA
from fairseq.modules.complex_exponential_moving_average import MultiHeadComplexEMA
from fairseq.modules.efficient_attention import EfficientAttention
from fairseq.modules.attention_softmax import AttentionSoftmax


@with_incremental_state
class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.
    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        efficient_attn=False,
        attention_activation='softmax',
        bidirectional=False,
        chunk_size=-1,
        moving_layer='ema',
        moving_act='rmsnorm',
        truncation=None,
        norm_num_groups=None,
        norm_affine=True,
        norm_eps=1e-5,
        rel_pos_bias='simple',
        max_positions=1024,
        init_mode='bert',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.attention_activation = attention_activation
        self.init_mode = init_mode

        self.dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = FairseqDropout(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        if efficient_attn:
            assert attention_activation == 'softmax'
            self.attention_dropout = None
            if bidirectional:
                self.efficient_attn = None
                self.attn_softmax = AttentionSoftmax(dropout=attention_dropout, use_causal_mask=False)
            else:
                self.efficient_attn = EfficientAttention(dropout=attention_dropout, use_causal_mask=True, scale=1.0)
                self.attn_softmax = None
        else:
            self.efficient_attn = None
            self.attn_softmax = None
            self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.bidirectional = bidirectional

        if bidirectional:
            self.norm = SequenceNorm(embed_dim, num_groups=norm_num_groups, eps=norm_eps)
        else:
            self.norm = TimestepNorm(embed_dim, num_groups=norm_num_groups, eps=norm_eps)

        if moving_layer == 'ema':
            self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        elif moving_layer == 'cema':
            self.move = MultiHeadComplexEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        else:
            raise ValueError("Unknown moving type: {}".format(moving_layer))

        assert moving_act in ['rmsnorm', 'silu']
        if moving_act == 'rmsnorm':
            self.move_act = RMSNorm(embed_dim, elementwise_affine=norm_affine, eps=norm_eps)
        else:
            self.move_act = nn.SiLU()

        self.v_proj = nn.Linear(embed_dim, hdim, bias=True)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim, bias=True)
        self.h_proj = nn.Linear(hdim, embed_dim, bias=False)
        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryEmbedding(zdim, max_positions)
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
            nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        elif mode == 'he':
            a = math.sqrt(5.0)
            nn.init.kaiming_normal_(self.v_proj.weight, a=a)
            nn.init.kaiming_normal_(self.mx_proj.weight, a=a)
            nn.init.kaiming_normal_(self.h_proj.weight, a=a)
        elif mode == 'xavier':
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.mx_proj.weight)
            nn.init.xavier_uniform_(self.h_proj.weight)
        else:
            raise ValueError('Unknown init mode: {}'.format(mode))
        # bias
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.mx_proj.bias, 0.0)
        # gamma & beta
        nn.init.constant_(self.gamma, 0.0)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(1)
        if padding_mask is not None:
            # B*K x 1
            lengths = slen - padding_mask.sum(dim=-1, keepdim=True)
            # B*K x 1 x 1
            len_scale = torch.rsqrt(lengths.clamp(min=1.0)).to(q).unsqueeze(-1)
            # B*K x C
            inverse_mask = 1.0 - padding_mask.to(q)
        else:
            len_scale = 1.0 / math.sqrt(slen)
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            len_scale = torch.rsqrt(attn_mask.float().sum(dim=-1, keepdim=True)).to(q)

        if isinstance(self.rel_pos_bias, SimpleRelativePositionalBias):
            # C x C
            bias = self.rel_pos_bias(slen)
            if slen != q.size(1):
                assert q.size(1) == 1
                # 1 x C
                bias = bias[-1:]
            # B*K x C x C
            qk = torch.bmm(q, k.transpose(1, 2)) * len_scale + bias
        elif isinstance(self.rel_pos_bias, RotaryEmbedding):
            if slen != q.size(1):
                assert q.size(1) == 1
                qidx = slen - 1
            else:
                qidx = 0
            q, k = self.rel_pos_bias(q, k, qidx=qidx)
            # B*K x C x C
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

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        attn_weights = self.attention_dropout(attn_weights)
        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(1)
        if isinstance(self.rel_pos_bias, SimpleRelativePositionalBias):
            # C x C
            bias = self.rel_pos_bias(slen)
            if slen != q.size(1):
                assert q.size(1) == 1
                # 1 x C
                bias = bias[-1:]
            # B*K x C x C
            qk = torch.bmm(q, k.transpose(1, 2)) + bias
        elif isinstance(self.rel_pos_bias, RotaryEmbedding):
            if slen != q.size(1):
                assert q.size(1) == 1
                qidx = slen - 1
            else:
                qidx = 0
            q, k = self.rel_pos_bias(q, k, qidx=qidx)
            # B*K x C x C
            qk = torch.bmm(q, k.transpose(1, 2))
        else:
            raise ValueError('unknown relative position bias')

        if attn_mask is not None:
            assert self.attn_softmax is None
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        if self.attn_softmax is None:
            attn_weights = utils.softmax(qk, dim=-1, onnx_trace=self.onnx_trace).to(qk)
            attn_weights = self.attention_dropout(attn_weights)
        else:
            attn_weights = self.attn_softmax(qk)
        return attn_weights

    def efficient_softmax_attention(self, q, k, v):
        assert isinstance(self.rel_pos_bias, RotaryEmbedding)
        slen = k.size(1)
        if slen != q.size(1):
            assert q.size(1) == 1
            qidx = slen - 1
        else:
            qidx = 0
        q, k = self.rel_pos_bias(q, k, qidx=qidx)
        return self.efficient_attn(q, k, v)

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_attn_fn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        bsz, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # B x L x D
        residual = x

        if self.bidirectional:
            # B x L x D -> B x D x L
            x = x.transpose(1, 2)
            x = self.norm(x, padding_mask)
            # B x L x E
            v = F.silu(self.v_proj(x.transpose(1, 2)))
        else:
            # B x L x D
            x = self.norm(x, padding_mask=padding_mask, incremental_state=incremental_state)
            # B x L x E
            v = F.silu(self.v_proj(x))
            # B x L x D -> B x D x L
            x = x.transpose(1, 2)

        # B x D x L
        mx = self.move(x, padding_mask, incremental_state)
        # B x D x L -> B x L x D
        mx = mx.transpose(1, 2)
        mx = self.hidden_dropout(self.move_act(mx))

        # B x L x D -> B x L x (D+S+E+D)
        base = self.mx_proj(mx)
        u, z, r, hx = torch.split(base, [self.embed_dim, self.zdim, self.hdim, self.embed_dim], dim=-1)
        # B x L x D
        u = torch.sigmoid(u)
        # B x L x S
        z = F.normalize(z, p=2, dim=-1, eps=1e-5)
        # B x L x S -> B x L x 1 x S -> B x L x 2 x S
        z = z.unsqueeze(2) * (self.gamma + 1.0) + self.beta
        # B x L x 2 x S -> B x L x S
        q, k = torch.unbind(z, dim=2)
        # B x L x E
        r = F.silu(r)

        if saved_state is not None:
            # assert self.chunk_size < 0 or q.size(1) <= self.chunk_size
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
            prev_padding_mask: Optional[Tensor] = None
            if "prev_padding_mask" in saved_state:
                prev_padding_mask = saved_state["prev_padding_mask"]
            padding_mask = MovingAverageGatedAttention._append_prev_padding_mask(
                padding_mask=padding_mask,
                prev_padding_mask=prev_padding_mask,
                batch_size=bsz,
                seq_len=k.size(1),
            )

            if self.chunk_size < 0:
                saved_state["prev_key"] = k
                saved_state["prev_value"] = v
                saved_state["prev_key_padding_mask"] = padding_mask
            else:
                curr_len = k.size(1) % self.chunk_size
                if curr_len == 0:
                    if "prev_key" in saved_state:
                        del saved_state["prev_key"]
                        del saved_state["prev_value"]
                        del saved_state["prev_key_padding_mask"]
                else:
                    saved_state["prev_key"] = k
                    saved_state["prev_value"] = v
                    saved_state["prev_key_padding_mask"] = padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        if 1 < self.chunk_size < seq_len:
            # B x L x S -> B*K x C x S
            nc = seq_len // self.chunk_size
            q = q.reshape(bsz * nc, self.chunk_size, self.zdim)

        ctx_len = k.size(1)
        if 1 < self.chunk_size < ctx_len:
            # B x L x S -> B*K x C x S
            nc = ctx_len // self.chunk_size
            k = k.reshape(bsz * nc, self.chunk_size, self.zdim)
            v = v.reshape(bsz * nc, self.chunk_size, self.hdim)
            if padding_mask is not None:
                # B x L -> B*K x C
                padding_mask = padding_mask.view(bsz * nc, self.chunk_size)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        if self.efficient_attn is not None:
            # B*K x C x E -> B x L x E
            attn = self.efficient_softmax_attention(q, k, v).view(bsz, seq_len, self.hdim)
            attn_weights = None
        else:
            if self.attention_activation == 'softmax':
                attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
            else:
                attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)
            # B*K x C x E -> B x L x E
            attn = torch.bmm(attn_weights, v).view(bsz, seq_len, self.hdim)

        # B x L x E
        attn = self.hidden_dropout(attn * r)
        # B x L x E -> B x L x D
        h = F.silu(hx + self.h_proj(attn))
        h = self.dropout(h)
        # B x L x D
        out = torch.addcmul(residual, u, h - residual)

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
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat([prev_padding_mask, padding_mask], dim=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - prev_padding_mask.size(1)), device=prev_padding_mask.device)
            new_padding_mask = torch.cat([prev_padding_mask, filler.bool()], dim=1)
        elif padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - padding_mask.size(1)), device=padding_mask.device)
            new_padding_mask = torch.cat([filler.bool(), padding_mask], dim=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, ndim={}, chunk={}, attn_act={}, bidir={}, init={}'.format(self.embed_dim, self.zdim, self.hdim, self.ndim,
                                                                                                     self.chunk_size, self.attention_activation,
                                                                                                     self.bidirectional, self.init_mode)
