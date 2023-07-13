#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

torch::Tensor EMAFilterFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                           const torch::Tensor& gamma, int64_t L);

torch::Tensor EMAFilterCPUFwd(const torch::Tensor& p,
                              const torch::Tensor& log_q,
                              const torch::Tensor& gamma, int64_t L);

torch::Tensor EMAFilterCUDAFwd(const torch::Tensor& p,
                               const torch::Tensor& log_q,
                               const torch::Tensor& gamma, int64_t L);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterBwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterCPUBwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterCUDABwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma);

void DefineEMAFilterOp(py::module& m);

}  // namespace ops
}  // namespace mega2
