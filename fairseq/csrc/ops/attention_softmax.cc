#include "ops/attention_softmax.h"

namespace mega2 {
namespace ops {

torch::Tensor AttentionSoftmaxFwd(const torch::Tensor& x, bool causal_mask,
                                  double dropout, bool inverted_dropout) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  return AttentionSoftmaxCUDAFwd(x, causal_mask, dropout, inverted_dropout);
}

torch::Tensor AttentionSoftmaxBwd(const torch::Tensor& y_grad,
                                  const torch::Tensor& y, bool causal_mask,
                                  double scale) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(y.device().type() == torch::kCUDA);
  return AttentionSoftmaxCUDABwd(y_grad, y, causal_mask, scale);
}

void DefineAttentionSoftmaxOp(py::module& m) {
  m.def("attention_softmax_fwd", &AttentionSoftmaxFwd, "AttentionSoftmaxFwd")
      .def("attention_softmax_bwd", &AttentionSoftmaxBwd,
           "AttentionSoftmaxBwd");
}

}  // namespace ops
}  // namespace mega2
