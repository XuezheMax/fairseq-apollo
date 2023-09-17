#pragma once

#include <torch/torch.h>

#include "utils.h"

namespace mega2 {
namespace ops {

torch::Tensor AttentionSoftmaxFwd(const torch::Tensor& x, bool causal_mask,
                                  double dropout, bool inverted_dropout);
torch::Tensor AttentionSoftmaxCUDAFwd(const torch::Tensor& x, bool causal_mask,
                                      double dropout, bool inverted_dropout);

torch::Tensor AttentionSoftmaxBwd(const torch::Tensor& y_grad,
                                  const torch::Tensor& y, bool causal_mask,
                                  double scale);
torch::Tensor AttentionSoftmaxCUDABwd(const torch::Tensor& y_grad,
                                      const torch::Tensor& y, bool causal_mask,
                                      double scale);

void DefineAttentionSoftmaxOp(py::module& m);

}  // namespace ops
}  // namespace mega2
