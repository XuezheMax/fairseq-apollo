from typing import Optional, Tuple

import torch

from torch.autograd.function import FunctionCtx

import fairseq.mega2_extension.ops as mega2_ops


class EMAParametersFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, p: torch.Tensor, q: torch.Tensor,
                gamma: torch.Tensor, h: Optional[torch.Tensor],
                length: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            log_q = q.log()
        weight, bias, vander = mega2_ops.ema_parameters_fwd(p, log_q, gamma, h, length)
        ctx.save_for_backward(p, log_q, gamma, h, vander)
        return weight, bias

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        weight_grad: torch.Tensor,
        bias_grad: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        p, log_q, gamma, h, vander = ctx.saved_tensors
        p_grad, q_grad, gamma_grad, h_grad = mega2_ops.ema_parameters_bwd(
            weight_grad, bias_grad, p, log_q, gamma, h, vander)
        return p_grad, q_grad, gamma_grad, h_grad, None


ema_parameters = EMAParametersFunc.apply
