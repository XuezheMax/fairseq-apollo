#pragma once

#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

torch::Tensor RFFT(const torch::Tensor& X, bool flip);
torch::Tensor RFFTCUDA(const torch::Tensor& X, bool flip);

// torch::Tensor FFTConvFwd(const torch::Tensor& X, const torch::Tensor&
// kernel); torch::Tensor FFTConvCUDAFwd(const torch::Tensor& X,
//                              const torch::Tensor& kernel);

std::tuple<torch::Tensor, torch::Tensor> FFTConvFwd(const torch::Tensor& X,
                                                    const torch::Tensor& K_f);
std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDAFwd(
    const torch::Tensor& X, const torch::Tensor& K_f);

std::tuple<torch::Tensor, torch::Tensor> FFTConvBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X_f,
    const torch::Tensor& K_f, const torch::Dtype& K_dtype);
std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X_f,
    const torch::Tensor& K_f, const torch::Dtype& K_dtype);

void DefineFFTConvOp(py::module& m);

}  // namespace ops
}  // namespace mega2
