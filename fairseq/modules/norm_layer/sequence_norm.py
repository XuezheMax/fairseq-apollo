from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

import fairseq.mega2_extension.ops as mega2_ops


class SequenceNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx,
                x: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-5) -> torch.Tensor:
        y, count, mean, rstd = mega2_ops.sequence_norm_fwd(
            x, gamma, beta, padding_mask, eps)
        ctx.save_for_backward(x, count, mean, rstd, gamma, padding_mask)
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx, y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        x, count, mean, rstd, gamma, padding_mask = ctx.saved_tensors
        x_grad, gamma_grad, beta_grad = mega2_ops.sequence_norm_bwd(
            y_grad, x, count, mean, rstd, gamma, padding_mask)
        return x_grad, gamma_grad, beta_grad, None, None


sequence_norm = SequenceNormFunc.apply


class SequenceNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.num_features = num_features
        self.register_parameter("weight", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("bias", Parameter(torch.zeros(num_features), requires_grad=True))
        self.eps = eps

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # L x B x D -> B x L x D
        x = x.transpose(0, 1)
        weight = self.weight + 1.0
        out = sequence_norm(x, weight, self.bias, padding_mask, self.eps)
        # B x L x D -> L x B x D
        return out.transpose(0, 1)

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}'.format(**self.__dict__)
