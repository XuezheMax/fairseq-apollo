#include "ops/fftconv.h"

namespace mega2 {
namespace ops {

torch::Tensor RFFT(const torch::Tensor& X, bool flip) {
  return RFFTCUDA(X, flip);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvFwd(const torch::Tensor& X,
                                                    const torch::Tensor& K_f) {
  return FFTConvCUDAFwd(X, K_f);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X_f,
    const torch::Tensor& K_f, const torch::Dtype& K_dtype) {
  return FFTConvCUDABwd(Y_grad, X_f, K_f, K_dtype);
}

void DefineFFTConvOp(py::module& m) {
  m.def("rfft", &RFFT, "RFFT")
      .def("fftconv_fwd", &FFTConvFwd, "FFTConvFwd")
      .def(
          "fftconv_bwd",
          [](const torch::Tensor& Y_grad, const torch::Tensor& X_f,
             const torch::Tensor& K_f, const py::object& K_dtype) {
            return FFTConvBwd(
                Y_grad, X_f, K_f,
                torch::python::detail::py_object_to_dtype(K_dtype));
          },
          "FFTConvBwd");
}

}  // namespace ops
}  // namespace mega2
