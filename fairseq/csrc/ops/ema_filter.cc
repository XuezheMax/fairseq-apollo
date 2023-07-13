#include "ops/ema_filter.h"

#include <ATen/Parallel.h>
#include <c10/util/complex.h>

#include <cstring>
#include <vector>

namespace mega2 {
namespace ops {

namespace {

template <typename T>
void EMAFilterCPUFwdImpl(const torch::Tensor& p, const torch::Tensor& log_q,
                         const torch::Tensor& gamma, int64_t L,
                         torch::Tensor& kernel) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);

  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* kernel_data = kernel.data_ptr<T>();

  std::vector<c10::complex<T>> w(D * N);
  at::parallel_for(0, D * N, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      w[i] = p_data[i] * gamma_data[i];
    }
  });

  at::parallel_for(0, D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const c10::complex<T>* log_q_ptr = log_q_data + i * N;
      const c10::complex<T>* w_ptr = w.data() + i * N;
      T* kernel_ptr = kernel_data + i * L;
      for (int64_t j = 0; j < L; ++j) {
        T sum = T(0);
        for (int64_t k = 0; k < N; ++k) {
          const c10::complex<T> qw =
              c10_complex_math::exp(log_q_ptr[k] * static_cast<T>(j));
          sum += (w_ptr[k] * qw).real();
        }
        kernel_ptr[j] = sum;
      }
    }
  });
}

template <typename T>
std::tuple<T, c10::complex<T>, c10::complex<T>> RowwiseEMAFilterCPUBwd(
    int64_t L, const T* kernel_grad, T p, c10::complex<T> log_q,
    c10::complex<T> gamma) {
  const c10::complex<T> q = c10_complex_math::exp(log_q);
  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t i = 0; i < L; ++i) {
    const c10::complex<T> dk = kernel_grad[i];
    const c10::complex<T> qw1 =
        i == 0 ? c10::complex<T>(T(0))
               : c10_complex_math::exp(log_q * static_cast<T>(i - 1));
    const c10::complex<T> qw2 = i == 0 ? c10::complex<T>(T(1)) : qw1 * q;
    sum1 += dk * qw1 * static_cast<T>(i);
    sum2 += dk * qw2;
  }
  return std::make_tuple((sum2 * gamma).real(), std::conj(sum1 * p * gamma),
                         std::conj(sum2 * p));
}

template <typename T>
void EMAFilterCPUBwdImpl(const torch::Tensor& kernel_grad,
                         const torch::Tensor& p, const torch::Tensor& log_q,
                         const torch::Tensor& gamma, torch::Tensor& p_grad,
                         torch::Tensor& q_grad, torch::Tensor& gamma_grad) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);
  const int64_t L = kernel_grad.size(-1);

  const T* kernel_grad_data = kernel_grad.data_ptr<T>();
  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* p_grad_data = p_grad.data_ptr<T>();
  c10::complex<T>* q_grad_data = q_grad.data_ptr<c10::complex<T>>();
  c10::complex<T>* gamma_grad_data = gamma_grad.data_ptr<c10::complex<T>>();

  std::vector<c10::complex<T>> q_pow(D * N, c10::complex<T>(T(1)));

  at::parallel_for(0, D, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* kernel_grad_ptr = kernel_grad_data + i * L;
      const T* p_ptr = p_data + i * N;
      const c10::complex<T>* log_q_ptr = log_q_data + i * N;
      const c10::complex<T>* gamma_ptr = gamma_data + i * N;
      T* p_grad_ptr = p_grad_data + i * N;
      c10::complex<T>* q_grad_ptr = q_grad_data + i * N;
      c10::complex<T>* gamma_grad_ptr = gamma_grad_data + i * N;
      for (int64_t j = 0; j < N; ++j) {
        std::tie(p_grad_ptr[j], q_grad_ptr[j], gamma_grad_ptr[j]) =
            RowwiseEMAFilterCPUBwd(L, kernel_grad_ptr, p_ptr[j], log_q_ptr[j],
                                   gamma_ptr[j]);
      }
    }
  });
}

}  // namespace

torch::Tensor EMAFilterCPUFwd(const torch::Tensor& p,
                              const torch::Tensor& log_q,
                              const torch::Tensor& gamma, int64_t L) {
  const int64_t D = p.size(0);
  torch::Tensor kernel = torch::empty(
      {D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAFilterCPUFwd", [&]() {
    EMAFilterCPUFwdImpl<scalar_t>(*(p.expect_contiguous()),
                                  *(log_q.expect_contiguous()),
                                  *(gamma.expect_contiguous()), L, kernel);
  });

  return kernel;
}

torch::Tensor EMAFilterFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                           const torch::Tensor& gamma, int64_t L) {
  if (p.device().type() == torch::kCUDA) {
    return EMAFilterCUDAFwd(p, log_q, gamma, L);
  } else {
    return EMAFilterCPUFwd(p, log_q, gamma, L);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterCPUBwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma) {
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAFilterCPUFwd", [&]() {
    EMAFilterCPUBwdImpl<scalar_t>(
        *(kernel_grad.expect_contiguous()), *(p.expect_contiguous()),
        *(log_q.expect_contiguous()), *(gamma.expect_contiguous()), p_grad,
        q_grad, gamma_grad);
  });

  return std::make_tuple(p_grad, q_grad, gamma_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterBwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma) {
  if (p.device().type() == torch::kCUDA) {
    return EMAFilterCUDABwd(kernel_grad, p, log_q, gamma);
  } else {
    return EMAFilterCPUBwd(kernel_grad, p, log_q, gamma);
  }
}

void DefineEMAFilterOp(py::module& m) {
  m.def("ema_filter_fwd", &EMAFilterFwd, "EMAFilterFwd")
      .def("ema_filter_bwd", &EMAFilterBwd, "EMAFilterBwd");
}

}  // namespace ops
}  // namespace mega2
