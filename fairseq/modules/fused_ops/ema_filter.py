from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter

import fairseq.mega2_extension.ops as mega2_ops


class EMAFilterFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, p: torch.Tensor, q: torch.Tensor,
                gamma: torch.Tensor, length: int) -> torch.Tensor:
        with torch.no_grad():
            log_q = q.log()
        kernel = mega2_ops.ema_filter_fwd(p, log_q, gamma, length)
        ctx.save_for_backward(p, log_q, gamma)
        return kernel

    @staticmethod
    def backward(
        ctx: FunctionCtx, kernel_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        p, log_q, gamma = ctx.saved_tensors
        p_grad, q_grad, gamma_grad = mega2_ops.ema_filter_bwd(
            kernel_grad, p, log_q, gamma)
        return p_grad, q_grad, gamma_grad, None


ema_filter = EMAFilterFunc.apply


class EMAFilter(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, p: torch.Tensor, q: torch.Tensor, gamma: torch.Tensor,
                length: int) -> torch.Tensor:
        if p.is_cuda:
            return ema_filter(p, q, gamma, length)

        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = p * torch.exp(vander)
        return torch.einsum('dnl,dn->dl', kernel, gamma).real
