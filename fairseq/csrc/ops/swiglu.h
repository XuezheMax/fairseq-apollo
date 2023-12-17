#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, int64_t, int64_t> SwiGLUFwd(const torch::Tensor& x1,
                                                      const torch::Tensor& x2,
                                                      double dropout);

std::tuple<torch::Tensor, int64_t, int64_t> SwiGLUCUDAFwd(
    const torch::Tensor& x1, const torch::Tensor& x2, double dropout);

std::tuple<torch::Tensor, torch::Tensor> SwiGLUBwd(const torch::Tensor& y_grad,
                                                   const torch::Tensor& x1,
                                                   const torch::Tensor& x2,
                                                   double dropout, int64_t seed,
                                                   int64_t offset);

std::tuple<torch::Tensor, torch::Tensor> SwiGLUCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x1,
    const torch::Tensor& x2, double dropout, int64_t seed, int64_t offset);

void DefineSwiGLUOp(py::module& m);

}  // namespace ops
}  // namespace mega2
