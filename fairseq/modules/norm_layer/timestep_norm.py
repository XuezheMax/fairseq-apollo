from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

import fairseq.mega2_extension.ops as mega2_ops


class TimestepNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        prev_count: torch.Tensor,
        prev_mean: torch.Tensor,
        prev_var: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, count, mean, var, cummean, cumvar = mega2_ops.timestep_norm_fwd(
            x, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, eps)
        ctx.save_for_backward(x, count, cummean, cumvar, gamma, padding_mask)
        ctx.eps = eps  # eps is not a Tensor
        return y, count, mean, var

    @staticmethod
    def backward(
        ctx: FunctionCtx, y_grad: torch.Tensor, _, mean_grad: torch.Tensor,
        var_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        x, count, cummean, cumvar, gamma, padding_mask = ctx.saved_tensors
        eps = ctx.eps
        (x_grad, prev_mean_grad, prev_var_grad, gamma_grad,
         beta_grad) = mega2_ops.timestep_norm_bwd(y_grad, mean_grad, var_grad,
                                                  x, count, cummean, cumvar,
                                                  gamma, padding_mask, eps)
        return (x_grad, None, prev_mean_grad, prev_var_grad, gamma_grad,
                beta_grad, None, None)


timestep_norm = TimestepNormFunc.apply


class TimestepNorm(nn.Module):

    def __init__(self, num_features: int, prior_count: int = 2, eps: float = 1e-5) -> None:
        super().__init__()

        self.num_features = num_features
        self.register_buffer("prior_count", torch.tensor(prior_count, dtype=torch.int64))
        self.register_parameter("prior_mean", Parameter(torch.zeros(num_features, requires_grad=True)))
        self.register_parameter("prior_logv", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("weight", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("bias", Parameter(torch.zeros(num_features), requires_grad=True))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        prev_count: Optional[torch.Tensor] = None,
        prev_mean: Optional[torch.Tensor] = None,
        prev_var: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        if prev_count is None:
            prev_count = self.prior_count.expand(batch_size).contiguous()
        if prev_mean is None:
            prev_mean = self.prior_mean.expand(batch_size, -1).contiguous()
        if prev_var is None:
            prev_var = self.prior_logv.exp().expand(batch_size, -1).contiguous()

        return timestep_norm(x, prev_count, prev_mean, prev_var,
                             self.weight + 1.0, self.bias,
                             padding_mask, self.eps)

    def extra_repr(self) -> str:
        return 'num_features={num_features}, eps={eps}'.format(**self.__dict__)
