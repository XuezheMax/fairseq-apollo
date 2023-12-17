#include "ops/swiglu.h"

namespace mega2 {
namespace ops {

std::tuple<torch::Tensor, int64_t, int64_t> SwiGLUFwd(const torch::Tensor& x1,
                                                      const torch::Tensor& x2,
                                                      double dropout) {
  TORCH_CHECK(x1.device().type() == torch::kCUDA);
  TORCH_CHECK(x2.device().type() == torch::kCUDA);
  return SwiGLUCUDAFwd(x1, x2, dropout);
}

std::tuple<torch::Tensor, torch::Tensor> SwiGLUBwd(const torch::Tensor& y_grad,
                                                   const torch::Tensor& x1,
                                                   const torch::Tensor& x2,
                                                   double dropout, int64_t seed,
                                                   int64_t offset) {
  TORCH_CHECK(y_grad.device().type() == torch::kCUDA);
  TORCH_CHECK(x1.device().type() == torch::kCUDA);
  TORCH_CHECK(x2.device().type() == torch::kCUDA);
  return SwiGLUCUDABwd(y_grad, x1, x2, dropout, seed, offset);
}

void DefineSwiGLUOp(py::module& m) {
  m.def("swiglu_fwd", &SwiGLUFwd, "SwiGLUFwd")
      .def("swiglu_bwd", &SwiGLUBwd, "SwiGLUBwd");
}

}  // namespace ops
}  // namespace mega2
