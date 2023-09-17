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
        inverted_dropout: bool = False,
        training: bool = True
    ) -> torch.Tensor:
        y = mega2_ops.attention_softmax_fwd(
            x, causal_mask, dropout if training else 0.0, inverted_dropout
        )
        ctx.save_for_backward(y)
        ctx.causal_mask = causal_mask  # causal_mask is not a torch.Tensor

        # scale is not a torch.Tensor
        if training and dropout > 0.0 and inverted_dropout:
            ctx.scale = 1.0 / (1.0 - dropout)
        else:
            ctx.scale = 1.0

        return y

    @staticmethod
    def backward(
            ctx: FunctionCtx, y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        y, = ctx.saved_tensors
        causal_mask = ctx.causal_mask
        scale = ctx.scale
        x_grad = mega2_ops.attention_softmax_bwd(y_grad, y, causal_mask, scale)
        return x_grad, None, None, None, None


attention_softmax = AttentionSoftmaxFunc.apply


class AttentionSoftmax(nn.Module):

    def __init__(self, causal_mask: bool = True, dropout: float = 0.0, inverted_dropout: bool = False) -> None:
        super().__init__()

        self._causal_mask = causal_mask
        self._dropout = dropout
        self._inverted_dropout = inverted_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attention_softmax(x, self._causal_mask, self._dropout,
                                 self._inverted_dropout, self.training)
