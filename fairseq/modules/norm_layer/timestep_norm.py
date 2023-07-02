from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from torch.autograd.function import FunctionCtx
from torch import Tensor
from torch.nn.parameter import Parameter

import fairseq.mega2_extension.ops as mega2_ops
from fairseq.incremental_decoding_utils import with_incremental_state


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
        num_groups: Optional[int] = None,
        eps: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_groups is None:
            y, count, mean, var, cummean, cumrstd = mega2_ops.timestep_norm_fwd(
                x, prev_count, prev_mean, prev_var, gamma, beta,padding_mask, eps
            )
            ctx.save_for_backward(x, prev_mean, count, cummean, cumrstd, gamma, padding_mask)
        else:
            y, count, mean, var, group_mean, cummean, cumrstd = mega2_ops.group_timestep_norm_fwd(
                x, prev_count, prev_mean, prev_var, gamma, beta, padding_mask, num_groups, eps
            )
            ctx.save_for_backward(x, prev_mean, count, group_mean, cummean, cumrstd, gamma, padding_mask)
        ctx.num_groups = num_groups  # num_groups is not a torch.Tensor
        return y, count, mean, var

    @staticmethod
    def backward(
        ctx: FunctionCtx, y_grad: torch.Tensor, _, mean_grad: torch.Tensor,
        var_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_groups = ctx.num_groups
        if num_groups is None:
            x, prev_mean, count, cummean, cumrstd, gamma, padding_mask = ctx.saved_tensors
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = mega2_ops.timestep_norm_bwd(
                y_grad, mean_grad, var_grad, x, prev_mean, count,
                cummean, cumrstd, gamma, padding_mask
            )
        else:
            x, prev_mean, count, group_mean, cummean, cumrstd, gamma, padding_mask = ctx.saved_tensors
            x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = mega2_ops.group_timestep_norm_bwd(
                y_grad, mean_grad, var_grad, x, prev_mean, count, group_mean,
                cummean, cumrstd, gamma, padding_mask, num_groups
            )
        return x_grad, None, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad, None, None, None


timestep_norm = TimestepNormFunc.apply


@with_incremental_state
class TimestepNorm(nn.Module):

    def __init__(self, num_features: int, num_groups: Optional[int] = None,
                 prior_count: int = 2, eps: float = 1e-5) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self._prior_count = prior_count

        if num_groups is None or num_groups == num_features:
            num_groups = num_features
        else:
            assert self.num_features % num_groups == 0

        self.register_buffer("prior_count", torch.tensor(prior_count, dtype=torch.int64))
        self.register_parameter("prior_mean", Parameter(torch.zeros(num_groups), requires_grad=True))
        self.register_parameter("prior_logv", Parameter(torch.zeros(num_groups), requires_grad=True))
        self.register_parameter("weight", Parameter(torch.zeros(num_features), requires_grad=True))
        self.register_parameter("bias", Parameter(torch.zeros(num_features), requires_grad=True))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # B x L x D
        batch_size = x.size(0)

        prev_mean = None
        prev_var = None
        prev_count = None
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_mean' in saved_state:
                prev_mean = saved_state['prev_mean']
                assert 'prev_var' in saved_state and 'prev_count' in saved_state
                prev_var = saved_state['prev_var']
                prev_count = saved_state['prev_count']
        else:
            saved_state = None

        if prev_count is None:
            prev_count = self.prior_count.expand(batch_size).contiguous()
        if prev_mean is None:
            prev_mean = self.prior_mean.expand(batch_size, -1).contiguous()
        if prev_var is None:
            prev_var = self.prior_logv.exp().expand(batch_size, -1).contiguous()

        out, prev_count, prev_mean, prev_var = timestep_norm(x, prev_count, prev_mean, prev_var,
                                                             self.weight + 1.0, self.bias,
                                                             padding_mask, self.num_groups, self.eps)

        if incremental_state is not None:
            saved_state['prev_mean'] = prev_mean
            saved_state['prev_var'] = prev_var
            saved_state['prev_count'] = prev_count
            self._set_input_buffer(incremental_state, saved_state)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "norm_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]):
        return self.set_incremental_state(incremental_state, "norm_state", buffer)

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def extra_repr(self) -> str:
        return 'num_features={num_features}, num_groups={num_groups}, prior_count={_prior_count}, eps={eps}'.format(**self.__dict__)
