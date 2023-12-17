#include "ops/ffn_swiglu.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, int64_t, int64_t>
FFNSwiGLUFwd(const torch::Tensor& x, const torch::Tensor& w1,
             const c10::optional<torch::Tensor>& b1, const torch::Tensor& w2,
             const c10::optional<torch::Tensor>& b2,
             const c10::optional<torch::Tensor>& w3,
             const c10::optional<torch::Tensor>& b3, double dropout) {
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(w1.device().type() == torch::kCUDA);
  if (b1.has_value()) {
    TORCH_CHECK(b1->device().type() == torch::kCUDA);
  }
  TORCH_CHECK(w2.device().type() == torch::kCUDA);
  if (b2.has_value()) {
    TORCH_CHECK(b2->device().type() == torch::kCUDA);
  }
  if (w3.has_value()) {
    TORCH_CHECK(w3->device().type() == torch::kCUDA);
  }
  if (b3.has_value()) {
    TORCH_CHECK(b3->device().type() == torch::kCUDA);
  }
  return FFNSwiGLUCUDAFwd(x, w1, b1, w2, b2, w3, b3, dropout);
}

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
           torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FFNSwiGLUBwd(const torch::Tensor& y_grad, const torch::Tensor& x,
             const torch::Tensor& w1, const c10::optional<torch::Tensor>& b1,
             const torch::Tensor& w2, const c10::optional<torch::Tensor>& b2,
             const c10::optional<torch::Tensor>& w3,
             const c10::optional<torch::Tensor>& b3, const torch::Tensor& h,
             double dropout, int64_t seed, int64_t offset, torch::Tensor& hw,
             c10::optional<torch::Tensor>& hv) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x.device().type() == torch::kCUDA);
  TORCH_CHECK(w1.device().type() == torch::kCUDA);
  TORCH_CHECK(w2.device().type() == torch::kCUDA);
  if (w3.has_value()) {
    TORCH_CHECK(w3->device().type() == torch::kCUDA);
  }
  return FFNSwiGLUCUDABwd(y_grad, x, w1, b1, w2, b2, w3, b3, h, dropout, seed,
                          offset, hw, hv);
}

void DefineFFNSwiGLUOp(py::module& m) {
  m.def("ffn_swiglu_fwd", &FFNSwiGLUFwd, "FFNSwiGLUFwd")
      .def("ffn_swiglu_bwd", &FFNSwiGLUBwd, "FFNSwiGLUBwd");
}

}  // namespace ops
}  // namespace mega2
