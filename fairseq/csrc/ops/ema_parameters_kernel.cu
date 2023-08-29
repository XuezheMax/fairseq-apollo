#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <type_traits>

#include "cuda_utils.cuh"
#include "ops/ema_parameters.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T>
__global__ void EMAVandermondeCUDAFwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    c10::complex<T>* __restrict__ v) {
  const int64_t i = blockIdx.x;
  const int64_t k = blockIdx.y;
  const c10::complex<T> log_q_v = log_q[i * N + k];
  const c10::complex<T> gamma_v = gamma[i * N + k];

  for (int64_t j = threadIdx.x; j <= L; j += blockDim.x) {
    const c10::complex<T> qw =
        c10_complex_math::exp(log_q_v * static_cast<T>(j));
    v[(i * N + k) * (L + 1) + j] = qw * gamma_v;
  }
}

template <typename T>
__global__ void EMAWeightCUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ w) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T c_shared[cuda_utils::kWarpSize * 2];  // p * gamma
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* c_ptr = reinterpret_cast<c10::complex<T>*>(c_shared);

  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
    c_ptr[threadIdx.x] = p[i * N + threadIdx.x] * gamma[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum = T(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          c10_complex_math::exp(q_ptr[k] * static_cast<T>(j));
      sum += (c_ptr[k] * qw).real();
    }
    w[i * L + j] = sum;
  }
}

template <typename T>
__global__ void EMAWeightCUDAFwdKernel(int64_t N, int64_t L,
                                       const T* __restrict__ p,
                                       const c10::complex<T>* __restrict__ v,
                                       T* __restrict__ w) {
  __shared__ T p_shared[cuda_utils::kWarpSize];

  const int64_t i = blockIdx.x;
  if (threadIdx.x < N) {
    p_shared[threadIdx.x] = p[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum = T(0);
    for (int64_t k = 0; k < N; ++k) {
      sum += (p_shared[k] * v[(i * N + k) * (L + 1) + j]).real();
    }
    w[i * L + j] = sum;
  }
}

template <typename T>
__global__ void EMAParametersBatchSize1CUDAFwdKernel(
    int64_t N, int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ w,
    T* __restrict__ b) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T u_shared[cuda_utils::kWarpSize * 2];  // p * gamma
  __shared__ T v_shared[cuda_utils::kWarpSize * 2];  // q * gamma * h
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* u_ptr = reinterpret_cast<c10::complex<T>*>(u_shared);
  c10::complex<T>* v_ptr = reinterpret_cast<c10::complex<T>*>(v_shared);

  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    const c10::complex<T> q = log_q[i * N + threadIdx.x];
    const c10::complex<T> g = gamma[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = q;
    u_ptr[threadIdx.x] = p[i * N + threadIdx.x] * g;
    v_ptr[threadIdx.x] = c10_complex_math::exp(q) * g * h[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T sum1 = T(0);
    T sum2 = T(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          c10_complex_math::exp(q_ptr[k] * static_cast<T>(j));
      sum1 += (qw * u_ptr[k]).real();
      sum2 += (qw * v_ptr[k]).real();
    }
    w[i * L + j] = sum1;
    b[i * L + j] = sum2;
  }
}

template <typename T, int64_t B>
__global__ void EMABiasCUDAFwdKernel(int64_t D, int64_t N, int64_t L,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     T* __restrict__ b) {
  __shared__ T q_shared[cuda_utils::kWarpSize * 2];
  __shared__ T c_shared[B * cuda_utils::kWarpSize * 2];  // gamma * h
  c10::complex<T>* q_ptr = reinterpret_cast<c10::complex<T>*>(q_shared);
  c10::complex<T>* c_ptr = reinterpret_cast<c10::complex<T>*>(c_shared);

  const int64_t i = blockIdx.x;

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t batch = j / N;
    const int64_t n = j % N;
    if (batch == 0) {
      q_ptr[n] = log_q[i * N + n];
    }
    c_ptr[j] = gamma[i * N + n] * h[(batch * D + i) * N + n];
  }
  __syncthreads();

  T sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum[batch] = T(0);
    }
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T> qw =
          c10_complex_math::exp(q_ptr[k] * static_cast<T>(j + 1));
#pragma unroll
      for (int64_t batch = 0; batch < B; ++batch) {
        sum[batch] += (qw * c_ptr[batch * N + k]).real();
      }
    }
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      b[(batch * D + i) * L + j] = sum[batch];
    }
  }
}

template <typename T>
__global__ void EMAWeightCUDABwdKernel(
    int64_t N, int64_t L, const T* __restrict__ w_grad, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const T* w_grad_ptr = w_grad + i * L;
  const T p_v = p[i * N + j];
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const T dw = w_grad_ptr[k];
    const c10::complex<T> qw1 =
        k == 0 ? c10::complex<T>(0)
               : c10_complex_math::exp(log_q_v * static_cast<T>(k - 1));
    const c10::complex<T> qw2 = k == 0 ? c10::complex<T>(1) : qw1 * q_v;
    sum1 += dw * qw1 * static_cast<T>(k);
    sum2 += dw * qw2;
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T>(sum2);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T>(sum2, sum2_shared_ptr);
  }

  if (threadIdx.x == 0) {
    p_grad[i * N + j] = (sum2 * gamma_v).real();
    q_grad[i * N + j] = std::conj(sum1 * p_v * gamma_v);
    gamma_grad[i * N + j] = std::conj(sum2 * p_v);
  }
}

template <typename T>
__global__ void EMAParametersBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const T* __restrict__ w_grad,
    const T* __restrict__ b_grad, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ h, T* __restrict__ p_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad, c10::complex<T>* h_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum3_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum4_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);
  c10::complex<T>* sum3_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum3_shared);
  c10::complex<T>* sum4_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum4_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const T* w_grad_ptr = w_grad + i * L;
  const T* b_grad_ptr = b_grad + i * L;
  const T p_v = p[i * N + j];
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);
  const c10::complex<T> h_v = h[i * N + j];

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  c10::complex<T> sum3(T(0));
  c10::complex<T> sum4(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const T dw = w_grad_ptr[k];
    const T db = b_grad_ptr[k];
    const c10::complex<T> qw1 =
        k == 0 ? c10::complex<T>(0)
               : c10_complex_math::exp(log_q_v * static_cast<T>(k - 1));
    const c10::complex<T> qw2 = k == 0 ? c10::complex<T>(1) : qw1 * q_v;
    const c10::complex<T> qw3 = qw2 * q_v;
    sum1 += dw * qw1 * static_cast<T>(k);
    sum2 += dw * qw2;
    sum3 += db * qw2 * static_cast<T>(k + 1);
    sum4 += db * qw3;
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T>(sum2);
    sum3 = cuda_utils::WarpReduceComplexSum<T>(sum3);
    sum4 = cuda_utils::WarpReduceComplexSum<T>(sum4);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T>(sum2, sum2_shared_ptr);
    sum3 = cuda_utils::BlockReduceComplexSum<T>(sum3, sum3_shared_ptr);
    sum4 = cuda_utils::BlockReduceComplexSum<T>(sum4, sum4_shared_ptr);
  }

  if (threadIdx.x == 0) {
    p_grad[i * N + j] = (sum2 * gamma_v).real();
    q_grad[i * N + j] = std::conj((sum1 * p_v + sum3 * h_v) * gamma_v);
    gamma_grad[i * N + j] = std::conj(sum2 * p_v + sum4 * h_v);
    h_grad[i * N + j] = std::conj(sum4 * gamma_v);
  }
}

template <typename T, int64_t B>
__global__ void EMABiasCUDABwdKernel(int64_t D, int64_t N, int64_t L,
                                     const T* __restrict__ b_grad,
                                     const c10::complex<T>* __restrict__ log_q,
                                     const c10::complex<T>* __restrict__ gamma,
                                     const c10::complex<T>* __restrict__ h,
                                     c10::complex<T>* __restrict__ q_grad,
                                     c10::complex<T>* __restrict__ gamma_grad,
                                     c10::complex<T>* h_grad) {
  __shared__ T sum1_shared[B][cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[B][cuda_utils::kWarpSize * 2];

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;
  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T> sum1[B];
  c10::complex<T> sum2[B];
#pragma unroll
  for (int64_t batch = 0; batch < B; ++batch) {
    sum1[batch] = c10::complex<T>(0);
    sum2[batch] = c10::complex<T>(0);
  }

  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const c10::complex<T> qw1 =
        c10_complex_math::exp(log_q_v * static_cast<T>(k));
    const c10::complex<T> qw2 = qw1 * q_v;
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      const T db = b_grad[(batch * D + i) * L + k];
      sum1[batch] += db * qw1 * static_cast<T>(k + 1);
      sum2[batch] += db * qw2;
    }
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum1[batch] = cuda_utils::WarpReduceComplexSum<T>(sum1[batch]);
      sum2[batch] = cuda_utils::WarpReduceComplexSum<T>(sum2[batch]);
    }
  } else {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum1[batch] = cuda_utils::BlockReduceComplexSum<T>(
          sum1[batch], reinterpret_cast<c10::complex<T>*>(sum1_shared[batch]));
      sum2[batch] = cuda_utils::BlockReduceComplexSum<T>(
          sum2[batch], reinterpret_cast<c10::complex<T>*>(sum2_shared[batch]));
    }
  }

  if (threadIdx.x == 0) {
    c10::complex<T> dq(T(0));
    c10::complex<T> dgamma(T(0));
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      const c10::complex<T> h_v = h[(batch * D + i) * N + j];
      dq += sum1[batch] * h_v;
      dgamma += sum2[batch] * h_v;
      h_grad[(batch * D + i) * N + j] = std::conj(sum2[batch] * gamma_v);
    }
    q_grad[i * N + j] += std::conj(dq * gamma_v);
    gamma_grad[i * N + j] += std::conj(dgamma);
  }
}

template <typename T>
__global__ void EMAVandermondeCUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ gamma,
    const c10::complex<T>* __restrict__ v_grad,
    c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ gamma_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const int64_t j = blockIdx.y;

  const c10::complex<T> log_q_v = log_q[i * N + j];
  const c10::complex<T> gamma_v = gamma[i * N + j];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);
  const c10::complex<T>* v_grad_ptr = v_grad + (i * N + j) * L;

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t k = threadIdx.x; k < L; k += blockDim.x) {
    const c10::complex<T> dv = v_grad_ptr[k];
    const c10::complex<T> qw1 =
        c10_complex_math::exp(log_q_v * static_cast<T>(k));
    const c10::complex<T> qw2 = qw1 * q_v;
    sum1 += dv * qw1 * static_cast<T>(k + 1);
    sum2 += dv * qw2;
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T>(sum2);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T>(sum2, sum2_shared_ptr);
  }

  if (threadIdx.x == 0) {
    q_grad[i * N + j] += std::conj(sum1 * gamma_v);
    gamma_grad[i * N + j] += std::conj(sum2);
  }
}

#define DISPATCH_BATCH_CUDA_KERNEL(KernelFunc, T, B, dg, db,             \
                                   shared_memory_size, cuda_stream, ...) \
  do {                                                                   \
    switch (B) {                                                         \
      case 2: {                                                          \
        KernelFunc<T, 2>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 3: {                                                          \
        KernelFunc<T, 3>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 4: {                                                          \
        KernelFunc<T, 4>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 5: {                                                          \
        KernelFunc<T, 5>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 6: {                                                          \
        KernelFunc<T, 6>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 7: {                                                          \
        KernelFunc<T, 7>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 8: {                                                          \
        KernelFunc<T, 8>                                                 \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
    }                                                                    \
  } while (0)

template <typename T>
void EMAParametersCUDAFwdImpl(const torch::Tensor& p,
                              const torch::Tensor& log_q,
                              const torch::Tensor& gamma,
                              const torch::Tensor& h, int64_t L,
                              torch::Tensor& w, c10::optional<torch::Tensor>& b,
                              c10::optional<torch::Tensor>& v) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);
  TORCH_CHECK(N <= cuda_utils::kWarpSize);

  const T* p_data = p.data_ptr<T>();
  const c10::complex<T>* log_q_data = log_q.data_ptr<c10::complex<T>>();
  const c10::complex<T>* gamma_data = gamma.data_ptr<c10::complex<T>>();
  T* w_data = w.data_ptr<T>();

  const int64_t num_threads = L < cuda_utils::kCUDABlockReduceNumThreads
                                  ? cuda_utils::kCUDANumThreads
                                  : cuda_utils::kCUDABlockReduceNumThreads;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (!h.defined()) {
    EMAWeightCUDAFwdKernel<T><<<D, num_threads, 0, cuda_stream>>>(
        N, L, p_data, log_q_data, gamma_data, w_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const int64_t B = h.size(0);
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();

  if (B == 1) {
    b = c10::make_optional(torch::empty({B, D, L}, p.options()));
    T* b_data = b->data_ptr<T>();
    EMAParametersBatchSize1CUDAFwdKernel<T><<<D, num_threads, 0, cuda_stream>>>(
        N, L, p_data, log_q_data, gamma_data, h_data, w_data, b_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (B <= 8) {
    b = c10::make_optional(torch::empty({B, D, L}, p.options()));
    T* b_data = b->data_ptr<T>();
    EMAWeightCUDAFwdKernel<T><<<D, num_threads, 0, cuda_stream>>>(
        N, L, p_data, log_q_data, gamma_data, w_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    DISPATCH_BATCH_CUDA_KERNEL(EMABiasCUDAFwdKernel, T, B, D, num_threads, 0,
                               cuda_stream, D, N, L, log_q_data, gamma_data,
                               h_data, b_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  v = c10::make_optional(torch::empty({D, N, L + 1}, log_q.options()));
  c10::complex<T>* v_data = v->data_ptr<c10::complex<T>>();

  EMAVandermondeCUDAFwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, log_q_data, gamma_data, v_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  EMAWeightCUDAFwdKernel<T>
      <<<D, num_threads, 0, cuda_stream>>>(N, L, p_data, v_data, w_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  torch::Tensor y = torch::empty({B, D, L}, log_q.options());
  c10::complex<T>* y_data = y.data_ptr<c10::complex<T>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T> kAlpha(1);
  const c10::complex<T> kBeta(0);
  if constexpr (std::is_same<T, float>::value) {
    TORCH_CUDABLAS_CHECK(cublasCgemm3mStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, L, B, N,
        reinterpret_cast<const cuComplex*>(&kAlpha),
        reinterpret_cast<const cuComplex*>(v_data + 1), L + 1, N * (L + 1),
        reinterpret_cast<const cuComplex*>(h_data), D * N, N,
        reinterpret_cast<const cuComplex*>(&kBeta),
        reinterpret_cast<cuComplex*>(y_data), D * L, L, D));
  } else {
    TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, L, B, N,
        reinterpret_cast<const cuDoubleComplex*>(&kAlpha),
        reinterpret_cast<const cuDoubleComplex*>(v_data + 1), L + 1,
        N * (L + 1), reinterpret_cast<const cuDoubleComplex*>(h_data), D * N, N,
        reinterpret_cast<const cuDoubleComplex*>(&kBeta),
        reinterpret_cast<cuDoubleComplex*>(y_data), D * L, L, D));
  }
  b = c10::make_optional(torch::real(y));
  // b = c10::make_optional(torch::real(y).contiguous());
}

template <typename T>
void EMAParametersCUDABwdImpl(
    const torch::Tensor& w_grad, const torch::Tensor& b_grad,
    const torch::Tensor& p, const torch::Tensor& log_q,
    const torch::Tensor& gamma, const torch::Tensor& h, const torch::Tensor& v,
    torch::Tensor& p_grad, torch::Tensor& q_grad, torch::Tensor& gamma_grad,
    c10::optional<torch::Tensor>& h_grad) {
  const int64_t D = p.size(0);
  const int64_t N = p.size(1);
  const int64_t L = w_grad.size(-1);

  const T* w_grad_data = w_grad.data_ptr<T>();
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

  if (!h.defined()) {
    EMAWeightCUDABwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
        N, L, w_grad_data, p_data, log_q_data, gamma_data, p_grad_data,
        q_grad_data, gamma_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  TORCH_CHECK(b_grad.defined());
  h_grad = c10::make_optional(torch::empty_like(h));
  const int64_t B = h.size(0);
  const T* b_grad_data = b_grad.data_ptr<T>();
  const c10::complex<T>* h_data = h.data_ptr<c10::complex<T>>();
  c10::complex<T>* h_grad_data = h_grad->data_ptr<c10::complex<T>>();

  if (B == 1) {
    EMAParametersBatchSize1CUDABwdKernel<T>
        <<<dim3(D, N), num_threads, 0, cuda_stream>>>(
            N, L, w_grad_data, b_grad_data, p_data, log_q_data, gamma_data,
            h_data, p_grad_data, q_grad_data, gamma_grad_data, h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  EMAWeightCUDABwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, w_grad_data, p_data, log_q_data, gamma_data, p_grad_data,
      q_grad_data, gamma_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (B <= 8) {
    DISPATCH_BATCH_CUDA_KERNEL(EMABiasCUDABwdKernel, T, B, dim3(D, N),
                               num_threads, 0, cuda_stream, D, N, L,
                               b_grad_data, log_q_data, gamma_data, h_data,
                               q_grad_data, gamma_grad_data, h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  TORCH_CHECK(v.defined());
  torch::Tensor b_grad_complex =
      b_grad.to(c10::toComplexType(b_grad.scalar_type()));
  torch::Tensor v_grad = torch::empty({D, N, L}, v.options());

  const c10::complex<T>* b_grad_complex_data =
      b_grad_complex.data_ptr<c10::complex<T>>();
  const c10::complex<T>* v_data = v.data_ptr<c10::complex<T>>();
  c10::complex<T>* v_grad_data = v_grad.data_ptr<c10::complex<T>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T> kAlpha(1);
  const c10::complex<T> kBeta(0);

  if constexpr (std::is_same<T, float>::value) {
    TORCH_CUDABLAS_CHECK(cublasCgemm3mStridedBatched(
        handle, CUBLAS_OP_C, CUBLAS_OP_N, N, B, L,
        reinterpret_cast<const cuComplex*>(&kAlpha),
        reinterpret_cast<const cuComplex*>(v_data + 1), L + 1, N * (L + 1),
        reinterpret_cast<const cuComplex*>(b_grad_complex_data), D * L, L,
        reinterpret_cast<const cuComplex*>(&kBeta),
        reinterpret_cast<cuComplex*>(h_grad_data), D * N, N, D));
    TORCH_CUDABLAS_CHECK(cublasCgemm3mStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, L, N, B,
        reinterpret_cast<const cuComplex*>(&kAlpha),
        reinterpret_cast<const cuComplex*>(b_grad_complex_data), D * L, L,
        reinterpret_cast<const cuComplex*>(h_data), D * N, N,
        reinterpret_cast<const cuComplex*>(&kBeta),
        reinterpret_cast<cuComplex*>(v_grad_data), L, N * L, D));
  } else {
    TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
        handle, CUBLAS_OP_C, CUBLAS_OP_N, N, B, L,
        reinterpret_cast<const cuDoubleComplex*>(&kAlpha),
        reinterpret_cast<const cuDoubleComplex*>(v_data + 1), L + 1,
        N * (L + 1),
        reinterpret_cast<const cuDoubleComplex*>(b_grad_complex_data), D * L, L,
        reinterpret_cast<const cuDoubleComplex*>(&kBeta),
        reinterpret_cast<cuDoubleComplex*>(h_grad_data), D * N, N, D));
    TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, L, N, B,
        reinterpret_cast<const cuDoubleComplex*>(&kAlpha),
        reinterpret_cast<const cuDoubleComplex*>(b_grad_complex_data), D * L, L,
        reinterpret_cast<const cuDoubleComplex*>(h_data), D * N, N,
        reinterpret_cast<const cuDoubleComplex*>(&kBeta),
        reinterpret_cast<cuDoubleComplex*>(v_grad_data), L, N * L, D));
  }

  EMAVandermondeCUDABwdKernel<T><<<dim3(D, N), num_threads, 0, cuda_stream>>>(
      N, L, log_q_data, gamma_data, v_grad_data, q_grad_data, gamma_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef DISPATCH_BATCH_CUDA_KERNEL

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>>
EMAParametersCUDAFwd(const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h, int64_t L) {
  const int64_t D = p.size(0);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor w = torch::empty(
      {D, L}, p.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> b = c10::nullopt;
  c10::optional<torch::Tensor> v = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCUDAFwd", [&]() {
    EMAParametersCUDAFwdImpl<scalar_t>(
        *(p.expect_contiguous()), *(log_q.expect_contiguous()),
        *(gamma.expect_contiguous()), *(h_maybe_owned->expect_contiguous()), L,
        w, b, v);
  });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>>(
      std::move(w), std::move(b), std::move(v));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAParametersCUDABwd(const torch::Tensor& w_grad,
                     const c10::optional<torch::Tensor>& b_grad,
                     const torch::Tensor& p, const torch::Tensor& log_q,
                     const torch::Tensor& gamma,
                     const c10::optional<torch::Tensor>& h,
                     const c10::optional<torch::Tensor>& v) {
  c10::MaybeOwned<torch::Tensor> b_grad_maybe_owned =
      at::borrow_from_optional_tensor(b_grad);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  c10::MaybeOwned<torch::Tensor> v_maybe_owned =
      at::borrow_from_optional_tensor(v);
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> h_grad = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "EMAParametersCUDABwd", [&]() {
    EMAParametersCUDABwdImpl<scalar_t>(
        *(w_grad.expect_contiguous()),
        *(b_grad_maybe_owned->expect_contiguous()), *(p.expect_contiguous()),
        *(log_q.expect_contiguous()), *(gamma.expect_contiguous()),
        *(h_maybe_owned->expect_contiguous()),
        *(v_maybe_owned->expect_contiguous()), p_grad, q_grad, gamma_grad,
        h_grad);
  });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(p_grad), std::move(q_grad), std::move(gamma_grad),
      std::move(h_grad));
}

}  // namespace ops
}  // namespace mega2
