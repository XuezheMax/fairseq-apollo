import torch
import torch.distributed as dist
from torch.autograd.function import Function


class MaskedBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, padding_mask, running_mean, running_var, eps, momentum):
        if not input.is_contiguous(memory_format=torch.channels_last):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        num_channels = input.shape[1]
        x = input
        if input.numel() <= 0:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            raise RuntimeError('empty input')
        elif padding_mask is None:
            var, mean = torch.var_mean(x.float(), dim=(0, 2), unbiased=False)
            nums = x.numel() // x.size(1)
            count = torch.full((1,), nums, dtype=mean.dtype, device=mean.device)
            bias_corr = nums / (nums - 1.0)
        else:
            total = padding_mask.numel()
            count = total - padding_mask.sum()
            var, mean = torch.var_mean(x.float(), dim=(0, 2), unbiased=False)
            square_mean = var + torch.square(mean)
            # adjust by ratio
            count = count.to(mean)
            ratio = total / count
            mean = mean * ratio
            var = square_mean * ratio - torch.square(mean)
            bias_corr = count / (count - 1.0)

        # update running stats
        running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1.0 - momentum).add_(var, alpha=momentum * bias_corr)  # unbias var estimator for running var

        invstd = torch.rsqrt(var + eps)
        self.save_for_backward(x, weight.float(), mean, invstd, count.to(torch.int32).view(1))
        # apply element-wise normalization
        out = torch.batch_norm_elemt(x, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_tensor
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class MaskedSyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, padding_mask, running_mean, running_var, eps, momentum, process_group, world_size):
        if not input.is_contiguous(memory_format=torch.channels_last):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        num_channels = input.shape[1]
        x = input
        if input.numel() <= 0:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(2 * num_channels + 1, dtype=torch.float32, device=input.device)
        elif padding_mask is None:
            mean, invstd = torch.batch_norm_stats(x.float(), eps)
            count = torch.full((1,), x.numel() // x.size(1), dtype=mean.dtype, device=mean.device)
            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            total = padding_mask.numel()
            count = total - padding_mask.sum()
            var, mean = torch.var_mean(x.float(), dim=(0, 2), unbiased=False)
            square_mean = var + torch.square(mean)
            # adjust by ratio
            count = count.to(mean)
            ratio = total / count
            mean = mean * ratio
            var = square_mean * ratio - torch.square(mean)
            invstd = torch.rsqrt(var + eps)
            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count.view(1)], dim=0)

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        if process_group._get_backend_name() == 'nccl':
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(1, combined_size * world_size, dtype=combined.dtype, device=combined.device)
            dist._all_gather_base(combined_flat, combined, process_group, async_op=False)
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            # world_size * (2C + 1)
            combined_list = [torch.empty_like(combined) for k in range(world_size)]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            x,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        self.save_for_backward(x, weight.float(), mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group

        # apply element-wise normalization
        out = torch.batch_norm_elemt(x, weight, bias, mean, invstd, eps)
        return out

    @staticmethod
    def backward(self, grad_output):
        if not grad_output.is_contiguous(memory_format=torch.channels_last):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_tensor
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None
