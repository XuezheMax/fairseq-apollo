#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

#include <tuple>

#include "utils.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, int64_t, int64_t>
FFNSwiGLUFwd(const torch::Tensor& x, const torch::Tensor& w1,
             const c10::optional<torch::Tensor>& b1, const torch::Tensor& w2,
             const c10::optional<torch::Tensor>& b2,
             const c10::optional<torch::Tensor>& w3,
             const c10::optional<torch::Tensor>& b3, double dropout);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, int64_t, int64_t>
FFNSwiGLUCUDAFwd(const torch::Tensor& x, const torch::Tensor& w1,
                 const c10::optional<torch::Tensor>& b1,
                 const torch::Tensor& w2,
                 const c10::optional<torch::Tensor>& b2,
                 const c10::optional<torch::Tensor>& w3,
                 const c10::optional<torch::Tensor>& b3, double dropout);

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
           torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FFNSwiGLUBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
             const torch::Tensor& w1, const c10::optional<torch::Tensor>& b1,
             const torch::Tensor& w2, const c10::optional<torch::Tensor>& b2,
             const c10::optional<torch::Tensor>& w3,
             const c10::optional<torch::Tensor>& b3, const torch::Tensor& h,
             double dropout, int64_t seed, int64_t offset, torch::Tensor& hw,
             c10::optional<torch::Tensor>& hv);

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
           torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FFNSwiGLUCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                 const torch::Tensor& w1,
                 const c10::optional<torch::Tensor>& b1,
                 const torch::Tensor& w2,
                 const c10::optional<torch::Tensor>& b2,
                 const c10::optional<torch::Tensor>& w3,
                 const c10::optional<torch::Tensor>& b3, const torch::Tensor& h,
                 double dropout, int64_t seed, int64_t offset,
                 torch::Tensor& hw, c10::optional<torch::Tensor>& hv);

void DefineFFNSwiGLUOp(py::module& m);

}  // namespace ops
}  // namespace mega2
