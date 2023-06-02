#pragma once

#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
                const torch::Tensor& gamma, const torch::Tensor& beta,
                const torch::Tensor& padding_mask, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCPUFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                   const torch::Tensor& prev_mean,
                   const torch::Tensor& prev_var, const torch::Tensor& gamma,
                   const torch::Tensor& beta, const torch::Tensor& padding_mask,
                   double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                    const torch::Tensor& prev_mean,
                    const torch::Tensor& prev_var, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const torch::Tensor& padding_mask, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                const torch::Tensor& var_grad, const torch::Tensor& X,
                const torch::Tensor& count, const torch::Tensor& cummean,
                const torch::Tensor& cumvar, const torch::Tensor& gamma,
                const torch::Tensor& padding_mask, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCPUBwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                   const torch::Tensor& var_grad, const torch::Tensor& X,
                   const torch::Tensor& count, const torch::Tensor& cummean,
                   const torch::Tensor& cumvar, const torch::Tensor& gamma,
                   const torch::Tensor& padding_mask, double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                    const torch::Tensor& var_grad, const torch::Tensor& X,
                    const torch::Tensor& count, const torch::Tensor& cummean,
                    const torch::Tensor& cumvar, const torch::Tensor& gamma,
                    const torch::Tensor& padding_mask, double eps);

void DefineTimestepNormOp(py::module& m);

}  // namespace ops
}  // namespace mega2