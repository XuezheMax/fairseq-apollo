from typing import Optional, Tuple
import torch
from torch.autograd.function import FunctionCtx

import fairseq.mega2_extension.ops as mega2_ops


class EMAHiddenFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, p: torch.Tensor,
                q: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            log_q = q.log()
        y, v = mega2_ops.ema_hidden_fwd(x, p, log_q, h)
        ctx.save_for_backward(x, p, log_q, h, v)
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        x, p, log_q, h, v = ctx.saved_tensors
        x_grad, p_grad, q_grad, h_grad = mega2_ops.ema_hidden_bwd(y_grad, x, p, log_q, h, v)
        return x_grad, p_grad, q_grad, h_grad


ema_hidden = EMAHiddenFunc.apply
