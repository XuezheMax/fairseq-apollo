from typing import Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx

import fairseq.mega2_extension.ops as mega2_ops


class AttentionSoftmaxFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        causal_mask: bool = True,
        dropout: float = 0.0,
        training: bool = True
    ) -> torch.Tensor:
        y = mega2_ops.attention_softmax_fwd(
            x, causal_mask, dropout if training else 0.0
        )
        ctx.save_for_backward(y)
        ctx.causal_mask = causal_mask  # causal_mask is not a torch.Tensor

        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        y, = ctx.saved_tensors
        causal_mask = ctx.causal_mask
        x_grad = mega2_ops.attention_softmax_bwd(y_grad, y, causal_mask)
        return x_grad, None, None, None


attention_softmax = AttentionSoftmaxFunc.apply


class AttentionSoftmax(nn.Module):

    def __init__(self, causal_mask: bool = True, dropout: float = 0.0) -> None:
        super().__init__()

        self._causal_mask = causal_mask
        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attention_softmax(x, self._causal_mask, self._dropout, self.training)

    def extra_repr(self) -> str:
        return 'causal={}, dropout={}'.format(self._causal_mask, self._dropout)
