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
from fairseq.modules.relative_positional_bias import RelativePositionalBias
from fairseq.modules.exponential_moving_average import EMALayer
from fairseq.modules.legendre_moving_average import LMALayer


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
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='tanh',
        peephole=False,
        bidirectional=False,
        max_positions=1024,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        assert activation in ['tanh', 'sin']
        self.activation = utils.get_activation_fn(activation=activation)

        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = FairseqDropout(hidden_dropout, module_name=self.__class__.__name__)

        self.move = EMALayer(embed_dim, zdim, bidirectional=bidirectional)

        self.z_proj = nn.Linear(zdim, zdim, bias=True)
        self.proj = nn.Linear(embed_dim, 2 * hdim + embed_dim, bias=True)
        self.hw_proj = nn.Linear(hdim, embed_dim, bias=True)
        self.hu_proj = nn.Linear(zdim, embed_dim, bias=True)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        self.rel_pos_bias = RelativePositionalBias(max_positions)

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.z_proj.weight, mean=0.0, std=std)
        if self.z_proj.weight is not None:
            nn.init.constant_(self.z_proj.bias, 0.0)

        nn.init.normal_(self.proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.proj.bias, 0.0)

        nn.init.normal_(self.hw_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.hw_proj.bias, 0.0)
        nn.init.normal_(self.hu_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.hu_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
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
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # N x B x S
        mx = self.move(x, padding_mask, incremental_state)

        z = F.silu(self.z_proj(mx))
        # N x B x S -> N x B x 1 x S -> N x B x 2 x S
        z = z.unsqueeze(2) * self.gamma + self.beta
        # N x B x 2 x S -> N x B x S
        q, k = torch.unbind(z, dim=2)

        # N x B x D -> N x B x (2*E+D)
        base = self.proj(x)
        u, rv = torch.split(base, [self.embed_dim, 2 * self.hdim], dim=-1)

        # N x B x D
        u = torch.sigmoid(u)

        # N x B x (2*E+S)
        rv = F.silu(rv)
        r, v = torch.split(rv, [self.hdim, self.hdim], dim=-1)

        # N x B x S -> B x N x S
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        if saved_state is not None:
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

            saved_state["prev_key"] = k
            saved_state["prev_value"] = v
            saved_state["prev_key_padding_mask"] = padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        seq_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        if padding_mask is not None:
            # B x N
            inverse_mask = 1.0 - padding_mask.type_as(x)
            lengths = inverse_mask.sum(dim=1).view(bsz, 1, 1)
        else:
            lengths = seq_len
            inverse_mask = None

        # B x N x N
        qk = torch.bmm(q, k.transpose(1, 2))
        bias = self.rel_pos_bias(seq_len)
        qk = qk / lengths + bias
        if inverse_mask is not None:
            qk = qk * inverse_mask.unsqueeze(1)

        if attn_mask is not None:
            inverse_attn_mask = 1.0 - attn_mask.unsqueeze(0).type_as(x)
            if self.onnx_trace:
                inverse_attn_mask = inverse_attn_mask.repeat(qk.size(0), 1, 1)
            qk = qk * inverse_attn_mask

        if before_softmax:
            return qk, v

        # attn_weights = utils.softmax(qk, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = torch.square(F.relu(qk))
        kernel = self.attention_dropout(attn_weights)
        v = self.hidden_dropout(v)
        # B x N x E -> N x B x E
        h = torch.bmm(kernel, v).transpose(0, 1)
        # N x B x E -> N x B x D
        h = self.activation(self.hu_proj(mx) + self.hw_proj(h * r))
        # N x B x D
        out = torch.addcmul(x, u, h - x)

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

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat(
                [prev_padding_mask.float(), padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - prev_padding_mask.size(1)), device=prev_padding_mask.device)
            new_padding_mask = torch.cat([prev_padding_mask.float(), filler.float()], dim=1)
        elif padding_mask is not None:
            filler = torch.zeros((batch_size, seq_len - padding_mask.size(1)), device=padding_mask.device)
            new_padding_mask = torch.cat([filler.float(), padding_mask.float()], dim=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask
