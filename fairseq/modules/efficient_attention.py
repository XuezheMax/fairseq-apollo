from typing import Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx

import fairseq.mega2_extension.ops as mega2_ops


class AttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                scale: float = 1.0,
                dropout: float = 0.0,
                use_causal_mask: bool = True,
                training: bool = True) -> torch.Tensor:
        y, w = mega2_ops.attention_fwd(q, k, v, scale, dropout if training else 0.0, use_causal_mask)
        ctx.save_for_backward(q, k, v, w)
        # scale is not a torch.Tensor
        ctx.scale = scale
        # use_causal_mask is not a torch.Tensor
        ctx.use_causal_mask = use_causal_mask
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx, y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None,
               None]:
        q, k, v, w = ctx.saved_tensors
        scale = ctx.scale
        use_causal_mask = ctx.use_causal_mask
        q_grad, k_grad, v_grad = mega2_ops.attention_bwd(y_grad, q, k, v, w, scale, use_causal_mask)
        return q_grad, k_grad, v_grad, None, None, None, None


attention = AttentionFunc.apply


class EfficientAttention(nn.Module):

    def __init__(self,scale: float = 1.0, dropout: float = 0.0, use_causal_mask: bool = True) -> None:
        super().__init__()

        self._scale = scale
        self._dropout = dropout
        self._causal = use_causal_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        return attention(q, k, v, self._scale, self._dropout, self._causal, self.training)

    def extra_repr(self) -> str:
        return 'causal={}, dropout={}'.format(self._causal, self._dropout)
