#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>

#include <ATen/native/cuda/block_reduce.cuh>

#include "cuda_utils.cuh"
#include "ops/ema_filter.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T>
__global__ void EMAFitlerCUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ kernel) {
  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j >= L) {
    return;
  }
  const T* p_ptr = p + i * N;
  const c10::complex<T>* log_q_ptr = log_q + i * N;
  const c10::complex<T>* gamma_ptr = gamma + i * N;
  T* kernel_ptr = kernel + i * L;
  T sum = T(0);
  for (int64_t k = 0; k < N; ++k) {
    const c10::complex<T> qw =
        c10_complex_math::exp(log_q_ptr[k] * static_cast<T>(j));
    sum += (p_ptr[k] * qw * gamma_ptr[k]).real();
  }
  kernel_ptr[j] = sum;
}

template <typename T>
__global__ void RowwiseEMAFilterCUDABwdKernel(
    int64_t N, int64_t L, const T* __restrict__ kernel_grad,
    const T* __restrict__ p, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ dp,
    c10::complex<T>* __restrict__ dq, c10::complex<T>* __restrict__ dgamma) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const T* kernel_grad_ptr = kernel_grad + i * L;
  const T p_v = p[i * N + j];
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const T dk = kernel_grad_ptr[k];
    const c10::complex<T> qw1 =
        k == 0 ? c10::complex<T>(0)
               : c10_complex_math::exp(log_q_v * static_cast<T>(k - 1));
    const c10::complex<T> qw2 = k == 0 ? c10::complex<T>(1) : qw1 * q_v;
    sum1 += dk * qw1 * static_cast<T>(k);
    sum2 += dk * qw2;
  }

  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T>(sum2);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T>(sum2, sum2_shared_ptr);
  }
  if (threadIdx.x == 0) {
    dp[i * N + j] = (sum2 * gamma_v).real();
    dq[i * N + j] = std::conj(sum1 * p_v * gamma_v);
    dgamma[i * N + j] = std::conj(sum2 * p_v);
  }
}

template <typename T>
void EMAFilterCUDAFwdImpl(const torch::Tensor& p, const torch::Tensor& log_q,
                          const torch::Tensor& gamma, int64_t L,
                          torch::Tensor& kernel) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);

  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* kernel_data = kernel.data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t M = utils::DivUp(L, cuda_utils::kCUDANumThreads);
  EMAFitlerCUDAFwdKernel<T>
      <<<dim3(D, M), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          N, L, p_data, log_q_data, gamma_data, kernel_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void EMAFilterCUDABwdImpl(const torch::Tensor& kernel_grad,
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

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_threads = L < cuda_utils::kCUDABlockReduceNumThreads
                                  ? cuda_utils::kWarpSize
                                  : cuda_utils::kCUDABlockReduceNumThreads;
  RowwiseEMAFilterCUDABwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, kernel_grad_data, p_data, log_q_data, gamma_data, p_grad_data,
      q_grad_data, gamma_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor EMAFilterCUDAFwd(const torch::Tensor& p,
                               const torch::Tensor& log_q,
                               const torch::Tensor& gamma, int64_t L) {
  const int64_t D = p.size(0);
  torch::Tensor kernel = torch::empty(
      {D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAFilterCUDAFwd", [&]() {
    EMAFilterCUDAFwdImpl<scalar_t>(*(p.expect_contiguous()),
                                   *(log_q.expect_contiguous()),
                                   *(gamma.expect_contiguous()), L, kernel);
  });

  return kernel;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EMAFilterCUDABwd(
    const torch::Tensor& kernel_grad, const torch::Tensor& p,
    const torch::Tensor& log_q, const torch::Tensor& gamma) {
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAFilterCUDAFwd", [&]() {
    EMAFilterCUDABwdImpl<scalar_t>(
        *(kernel_grad.expect_contiguous()), *(p.expect_contiguous()),
        *(log_q.expect_contiguous()), *(gamma.expect_contiguous()), p_grad,
        q_grad, gamma_grad);
  });

  return std::make_tuple(p_grad, q_grad, gamma_grad);
}

}  // namespace ops
}  // namespace mega2
