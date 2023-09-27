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
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        y = mega2_ops.attention_softmax_fwd(
            x, dropout if training else 0.0, use_causal_mask
        )
        ctx.save_for_backward(y)
        # use_causal_mask is not a torch.Tensor.
        ctx.use_causal_mask = use_causal_mask

        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        y, = ctx.saved_tensors
        use_causal_mask = ctx.use_causal_mask
        x_grad = mega2_ops.attention_softmax_bwd(y_grad, y, use_causal_mask)
        return x_grad, None, None, None


attention_softmax = AttentionSoftmaxFunc.apply


class AttentionSoftmax(nn.Module):

    def __init__(self, dropout: float = 0.0, use_causal_mask: bool = False) -> None:
        super().__init__()

        self._dropout = dropout
        self._causal = use_causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return attention_softmax(x, self._dropout, self._causal, self.training)

    def extra_repr(self) -> str:
        return 'causal={}, dropout={}'.format(self._causal, self._dropout)
