#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                const torch::Tensor& beta,
                const c10::optional<torch::Tensor>& padding_mask, double eps,
                bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCPUFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                   const torch::Tensor& beta,
                   const c10::optional<torch::Tensor>& padding_mask, double eps,
                   bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last);

void DefineSequenceNormOp(py::module& m);

}  // namespace ops
}  // namespace mega2
