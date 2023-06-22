from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

import fairseq.mega2_extension.ops as mega2_ops


class SequenceNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        length_last: bool = False
    ) -> torch.Tensor:
        y, count, mean, rstd = mega2_ops.sequence_norm_fwd(x, gamma, beta, padding_mask, eps, length_last)
        ctx.save_for_backward(x, count, mean, rstd, gamma, padding_mask)
        ctx.length_last = length_last  # length_last is not a torch.Tensor
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x, count, mean, rstd, gamma, padding_mask = ctx.saved_tensors
        length_last = ctx.length_last
        x_grad, gamma_grad, beta_grad = mega2_ops.sequence_norm_bwd(y_grad, x, count, mean, rstd, gamma, padding_mask, length_last)
        return x_grad, gamma_grad, beta_grad, None, None, None


sequence_norm = SequenceNormFunc.apply


class SequenceNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, lenght_last: bool = False) -> None:
        super().__init__()

        self.num_features = num_features
        self.register_parameter("weight", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("bias", Parameter(torch.zeros(num_features), requires_grad=True))
        self.eps = eps
        self.length_last = lenght_last

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return sequence_norm(x, self.weight + 1.0, self.bias, padding_mask, self.eps, self.length_last)

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}, length_last={length_last}'.format(**self.__dict__)
