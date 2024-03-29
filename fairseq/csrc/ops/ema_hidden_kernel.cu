#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>

#include <type_traits>

#include "blas.h"
#include "cuda_utils.cuh"
#include "ops/ema_hidden.h"
#include "utils.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void EMAHiddenBatchSize1CUDAFwdKernel(
    int64_t D, int64_t N, int64_t L, const T* __restrict__ x,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  __shared__ T_ACC sum_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* sum_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum_shared);

  const int64_t d = blockIdx.x;
  const int64_t n = blockIdx.y;
  const int64_t index = d * N + n;

  const T_ACC p_v = p[index];
  const c10::complex<T_ACC> log_q_v = log_q[index];

  c10::complex<T_ACC> sum(T_ACC(0));
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC x_v = static_cast<T_ACC>(x[d * L + i]);
    const c10::complex<T_ACC> qw =
        c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - i - 1));
    sum += qw * x_v;
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum = cuda_utils::WarpReduceComplexSum<T_ACC>(sum);
  } else {
    sum = cuda_utils::BlockReduceComplexSum<T_ACC>(sum, sum_shared_ptr);
  }

  if (threadIdx.x == 0) {
    sum *= p_v;
    if (h != nullptr) {
      const c10::complex<T_ACC> qw =
          c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L));
      sum += qw * h[index];
    }
    y[index] = sum;
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenCUDAFwdKernel(
    int64_t D, int64_t N, int64_t L, const T* __restrict__ x,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h,
    c10::complex<T_ACC>* __restrict__ y) {
  __shared__ T_ACC sum_shared[B][cuda_utils::kWarpSize * 2];

  const int64_t d = blockIdx.x;
  const int64_t n = blockIdx.y;

  const T_ACC p_v = p[d * N + n];
  const c10::complex<T_ACC> log_q_v = log_q[d * N + n];

  c10::complex<T_ACC> sum[B];
#pragma unroll
  for (int64_t batch = 0; batch < B; ++batch) {
    sum[batch] = c10::complex<T_ACC>(0);
  }

  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const c10::complex<T_ACC> qw =
        c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - i - 1));
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      const T_ACC x_v = static_cast<T_ACC>(x[(batch * D + d) * L + i]);
      sum[batch] += qw * x_v;
    }
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum[batch] = cuda_utils::WarpReduceComplexSum<T_ACC>(sum[batch]);
    }
  } else {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum[batch] = cuda_utils::BlockReduceComplexSum<T_ACC>(
          sum[batch],
          reinterpret_cast<c10::complex<T_ACC>*>(sum_shared[batch]));
    }
  }

  if (threadIdx.x == 0) {
    const c10::complex<T_ACC> qw =
        h == nullptr ? c10::complex<T_ACC>(T_ACC(0))
                     : c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L));
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum[batch] =
          sum[batch] * p_v + (h == nullptr ? c10::complex<T_ACC>(T_ACC(0))
                                           : qw * h[(batch * D + d) * N + n]);
      y[(batch * D + d) * N + n] = sum[batch];
    }
  }
}

template <typename T>
__global__ void EMAHiddenWeightCUDAFwdKernel(
    int64_t L, const T* __restrict__ p,
    const c10::complex<T>* __restrict__ log_q,
    c10::complex<T>* __restrict__ v) {
  const int64_t i = blockIdx.x;
  const T p_v = p[i];
  const c10::complex<T> log_q_v = log_q[i];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T> qw =
        c10_complex_math::exp(log_q_v * static_cast<T>(L - j - 1));
    v[i * L + j] = p_v * qw;
  }
}

template <typename T>
__global__ void EMAHiddenBiasCUDAFwdKernel(
    int64_t B, int64_t D, int64_t N, int64_t L,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ h, c10::complex<T>* __restrict__ y) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D * N) {
    return;
  }
  const c10::complex<T> qw =
      c10_complex_math::exp(log_q[i] * static_cast<T>(L));
  for (int64_t b = 0; b < B; ++b) {
    y[b * D * N + i] += h[b * D * N + i] * qw;
  }
}

template <typename T, typename T_ACC>
__global__ void EMAHiddenInputBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    T* __restrict__ x_grad) {
  __shared__ T_ACC w_shared[cuda_utils::kWarpSize * 2];  // y_grad.conj() * p
  __shared__ T_ACC q_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* w_ptr = reinterpret_cast<c10::complex<T_ACC>*>(w_shared);
  c10::complex<T_ACC>* q_ptr = reinterpret_cast<c10::complex<T_ACC>*>(q_shared);

  const int64_t i = blockIdx.x;

  if (threadIdx.x < N) {
    w_ptr[threadIdx.x] =
        std::conj(y_grad[i * N + threadIdx.x]) * p[i * N + threadIdx.x];
    q_ptr[threadIdx.x] = log_q[i * N + threadIdx.x];
  }
  __syncthreads();

  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    T_ACC sum = T_ACC(0);
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          c10_complex_math::exp(q_ptr[k] * static_cast<T_ACC>(L - j - 1));
      sum += (w_ptr[k] * qw).real();
    }
    x_grad[i * L + j] = static_cast<T>(sum);
  }
}

template <typename T, typename T_ACC>
__global__ void EMAHiddenBatchSize1CUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T_ACC>* __restrict__ y_grad,
    const T* __restrict__ x, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h, T_ACC* __restrict__ p_grad,
    c10::complex<T_ACC>* __restrict__ q_grad, c10::complex<T_ACC>* h_grad) {
  __shared__ T_ACC sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T_ACC sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum1_shared);
  c10::complex<T_ACC>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T_ACC>*>(sum2_shared);

  const int64_t d = blockIdx.x;
  const int64_t n = blockIdx.y;
  const int64_t index = d * N + n;
  const c10::complex<T_ACC> dy = std::conj(y_grad[index]);
  const T_ACC p_v = p[index];
  const c10::complex<T_ACC> log_q_v = log_q[index];
  const c10::complex<T_ACC> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T_ACC> sum1(T_ACC(0));
  c10::complex<T_ACC> sum2(T_ACC(0));
  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const T_ACC x_v = static_cast<T_ACC>(x[d * L + i]);
    const c10::complex<T_ACC> qw1 =
        i == L - 1
            ? c10::complex<T_ACC>(T_ACC(0))
            : c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - i - 2));
    const c10::complex<T_ACC> qw2 =
        i == L - 1 ? c10::complex<T_ACC>(T_ACC(1)) : qw1 * q_v;
    sum1 += x_v * qw2;
    sum2 += x_v * qw1 * static_cast<T_ACC>(L - i - 1);
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T_ACC>(sum2);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T_ACC>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T_ACC>(sum2, sum2_shared_ptr);
  }

  if (threadIdx.x == 0) {
    p_grad[index] = (dy * sum1).real();
    q_grad[index] = std::conj(dy * p_v * sum2);
    if (h != nullptr) {
      const c10::complex<T_ACC> qw1 =
          c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - 1));
      const c10::complex<T_ACC> qw2 = qw1 * q_v;
      q_grad[index] += std::conj(dy * h[index] * qw1 * static_cast<T_ACC>(L));
      h_grad[index] = std::conj(dy * qw2);
    }
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenInputCUDABwdKernel(
    int64_t D, int64_t N, int64_t L,
    const c10::complex<T_ACC>* __restrict__ y_grad, const T_ACC* __restrict__ p,
    const c10::complex<T_ACC>* __restrict__ log_q, T* __restrict__ x_grad) {
  __shared__ T_ACC
      w_shared[B * cuda_utils::kWarpSize * 2];  // y_grad.conj() * p
  __shared__ T_ACC q_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T_ACC>* w_ptr = reinterpret_cast<c10::complex<T_ACC>*>(w_shared);
  c10::complex<T_ACC>* q_ptr = reinterpret_cast<c10::complex<T_ACC>*>(q_shared);

  const int64_t i = blockIdx.x;

  for (int64_t j = threadIdx.x; j < B * N; j += blockDim.x) {
    const int64_t batch = j / N;
    const int64_t n = j % N;
    w_ptr[j] = std::conj(y_grad[(batch * D + i) * N + n]) * p[i * N + n];
    if (batch == 0) {
      q_ptr[n] = log_q[i * N + n];
    }
  }
  __syncthreads();

  T_ACC sum[B];
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum[batch] = T_ACC(0);
    }
    for (int64_t k = 0; k < N; ++k) {
      const c10::complex<T_ACC> qw =
          c10_complex_math::exp(q_ptr[k] * static_cast<T_ACC>(L - j - 1));
#pragma unroll
      for (int64_t batch = 0; batch < B; ++batch) {
        sum[batch] += (w_ptr[batch * N + k] * qw).real();
      }
    }
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      x_grad[(batch * D + i) * L + j] = static_cast<T>(sum[batch]);
    }
  }
}

template <typename T, typename T_ACC, int64_t B>
__global__ void EMAHiddenCUDABwdKernel(
    int64_t D, int64_t N, int64_t L,
    const c10::complex<T_ACC>* __restrict__ y_grad, const T* __restrict__ x,
    const T_ACC* __restrict__ p, const c10::complex<T_ACC>* __restrict__ log_q,
    const c10::complex<T_ACC>* __restrict__ h, T_ACC* __restrict__ p_grad,
    c10::complex<T_ACC>* __restrict__ q_grad, c10::complex<T_ACC>* h_grad) {
  __shared__ T_ACC sum1_shared[B][cuda_utils::kWarpSize * 2];
  __shared__ T_ACC sum2_shared[B][cuda_utils::kWarpSize * 2];

  const int64_t d = blockIdx.x;
  const int64_t n = blockIdx.y;
  const T_ACC p_v = p[d * N + n];
  const c10::complex<T_ACC> log_q_v = log_q[d * N + n];
  const c10::complex<T_ACC> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T_ACC> sum1[B];
  c10::complex<T_ACC> sum2[B];
#pragma unroll
  for (int64_t batch = 0; batch < B; ++batch) {
    sum1[batch] = c10::complex<T_ACC>(0);
    sum2[batch] = c10::complex<T_ACC>(0);
  }

  for (int64_t i = threadIdx.x; i < L; i += blockDim.x) {
    const c10::complex<T_ACC> qw1 =
        i == L - 1
            ? c10::complex<T_ACC>(T_ACC(0))
            : c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - i - 2));
    const c10::complex<T_ACC> qw2 =
        i == L - 1 ? c10::complex<T_ACC>(T_ACC(1)) : qw1 * q_v;
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      const T_ACC x_v = static_cast<T_ACC>(x[(batch * D + d) * L + i]);
      sum1[batch] += x_v * qw2;
      sum2[batch] += x_v * qw1 * static_cast<T_ACC>(L - i - 1);
    }
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum1[batch] = cuda_utils::WarpReduceComplexSum<T_ACC>(sum1[batch]);
      sum2[batch] = cuda_utils::WarpReduceComplexSum<T_ACC>(sum2[batch]);
    }
  } else {
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      sum1[batch] = cuda_utils::BlockReduceComplexSum<T_ACC>(
          sum1[batch],
          reinterpret_cast<c10::complex<T_ACC>*>(sum1_shared[batch]));
      sum2[batch] = cuda_utils::BlockReduceComplexSum<T_ACC>(
          sum2[batch],
          reinterpret_cast<c10::complex<T_ACC>*>(sum2_shared[batch]));
    }
  }

  if (threadIdx.x == 0) {
    const c10::complex<T_ACC> qw1 =
        h == nullptr
            ? c10::complex<T_ACC>(T_ACC(0))
            : c10_complex_math::exp(log_q_v * static_cast<T_ACC>(L - 1));
    const c10::complex<T_ACC> qw2 = qw1 * q_v;
    T_ACC dp = T_ACC(0);
    c10::complex<T_ACC> dq(T_ACC(0));
#pragma unroll
    for (int64_t batch = 0; batch < B; ++batch) {
      const c10::complex<T_ACC> dy = std::conj(y_grad[(batch * D + d) * N + n]);
      dp += (dy * sum1[batch]).real();
      dq += dy * p_v * sum2[batch];
      if (h != nullptr) {
        dq += dy * h[d * N + n] * qw1 * static_cast<T_ACC>(L);
        h_grad[(batch * D + d) * N + n] = std::conj(dy * qw2);
      }
    }
    p_grad[d * N + n] = dp;
    q_grad[d * N + n] = std::conj(dq);
  }
}

template <typename T>
__global__ void EMAHiddenWeightCUDABwdKernel(
    int64_t N, int64_t L, const c10::complex<T>* __restrict__ v_grad,
    const T* __restrict__ p, const c10::complex<T>* __restrict__ log_q,
    T* __restrict__ p_grad, c10::complex<T>* __restrict__ q_grad) {
  __shared__ T sum1_shared[cuda_utils::kWarpSize * 2];
  __shared__ T sum2_shared[cuda_utils::kWarpSize * 2];
  c10::complex<T>* sum1_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum1_shared);
  c10::complex<T>* sum2_shared_ptr =
      reinterpret_cast<c10::complex<T>*>(sum2_shared);

  const int64_t i = blockIdx.x;
  const T p_v = p[i];
  const c10::complex<T> log_q_v = log_q[i];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);

  c10::complex<T> sum1(T(0));
  c10::complex<T> sum2(T(0));
  for (int64_t j = threadIdx.x; j < L; j += blockDim.x) {
    const c10::complex<T> dv = v_grad[i * L + j];
    const c10::complex<T> qw1 =
        j == L - 1 ? c10::complex<T>(T(0))
                   : c10_complex_math::exp(log_q_v * static_cast<T>(L - j - 2));
    const c10::complex<T> qw2 = j == L - 1 ? c10::complex<T>(T(1)) : qw1 * q_v;

    sum1 += dv * qw2;
    sum2 += dv * qw1 * static_cast<T>(L - j - 1);
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = cuda_utils::WarpReduceComplexSum<T>(sum1);
    sum2 = cuda_utils::WarpReduceComplexSum<T>(sum2);
  } else {
    sum1 = cuda_utils::BlockReduceComplexSum<T>(sum1, sum1_shared_ptr);
    sum2 = cuda_utils::BlockReduceComplexSum<T>(sum2, sum2_shared_ptr);
  }

  if (threadIdx.x == 0) {
    p_grad[i] = sum1.real();
    q_grad[i] = std::conj(p_v * sum2);
  }
}

template <typename T>
__global__ void EMAHiddenBiasCUDABwdKernel(
    int64_t B, int64_t D, int64_t N, int64_t L,
    const c10::complex<T>* __restrict__ y_grad,
    const c10::complex<T>* __restrict__ log_q,
    const c10::complex<T>* __restrict__ h, c10::complex<T>* __restrict__ q_grad,
    c10::complex<T>* __restrict__ h_grad) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D * N) {
    return;
  }

  const c10::complex<T> log_q_v = log_q[i];
  const c10::complex<T> q_v = c10_complex_math::exp(log_q_v);
  const c10::complex<T> qw1 =
      c10_complex_math::exp(log_q_v * static_cast<T>(L - 1));
  const c10::complex<T> qw2 = qw1 * q_v;
  c10::complex<T> sum(T(0));
  for (int64_t b = 0; b < B; ++b) {
    const int64_t index = b * D * N + i;
    const c10::complex<T> dy = std::conj(y_grad[index]);
    sum += dy * h[index];
    h_grad[index] = std::conj(dy * qw2);
  }
  q_grad[i] += std::conj(sum * qw1 * static_cast<T>(L));
}

#define DISPATCH_BATCH_CUDA_KERNEL(KernelFunc, T, T_ACC, B, dg, db,      \
                                   shared_memory_size, cuda_stream, ...) \
  do {                                                                   \
    switch (B) {                                                         \
      case 2: {                                                          \
        KernelFunc<T, T_ACC, 2>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 3: {                                                          \
        KernelFunc<T, T_ACC, 3>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 4: {                                                          \
        KernelFunc<T, T_ACC, 4>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 5: {                                                          \
        KernelFunc<T, T_ACC, 5>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 6: {                                                          \
        KernelFunc<T, T_ACC, 6>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 7: {                                                          \
        KernelFunc<T, T_ACC, 7>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
      case 8: {                                                          \
        KernelFunc<T, T_ACC, 8>                                          \
            <<<dg, db, shared_memory_size, cuda_stream>>>(__VA_ARGS__);  \
        break;                                                           \
      }                                                                  \
    }                                                                    \
  } while (0)

template <typename T>
void EMAHiddenCUDAFwdImpl(const torch::Tensor& x, const torch::Tensor& p,
                          const torch::Tensor& log_q, const torch::Tensor& h,
                          torch::Tensor& y, c10::optional<torch::Tensor>& v) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);

  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* h_data =
      h.defined() ? h.data_ptr<c10::complex<T_ACC>>() : nullptr;
  c10::complex<T_ACC>* y_data = y.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  const int64_t num_threads = L < cuda_utils::kCUDABlockReduceNumThreads
                                  ? cuda_utils::kWarpSize
                                  : cuda_utils::kCUDABlockReduceNumThreads;

  if (B == 1) {
    EMAHiddenBatchSize1CUDAFwdKernel<T, T_ACC>
        <<<dim3(D, N), num_threads, 0, cuda_stream>>>(
            D, N, L, x_data, p_data, log_q_data, h_data, y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (B <= 8) {
    DISPATCH_BATCH_CUDA_KERNEL(EMAHiddenCUDAFwdKernel, T, T_ACC, B, dim3(D, N),
                               num_threads, 0, cuda_stream, D, N, L, x_data,
                               p_data, log_q_data, h_data, y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  torch::Tensor x_c = x.to(log_q.scalar_type());
  v = c10::make_optional(torch::empty({D, N, L}, log_q.options()));
  const c10::complex<T_ACC>* x_c_data = x_c.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* v_data = v->data_ptr<c10::complex<T_ACC>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T_ACC> kAlpha(1);
  const c10::complex<T_ACC> kBeta(0);

  EMAHiddenWeightCUDAFwdKernel<T_ACC>
      <<<D * N, num_threads, 0, cuda_stream>>>(L, p_data, log_q_data, v_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kT, D, B, N, L, kAlpha,
      x_c_data, D * L, L, v_data, L, N * L, kBeta, y_data, D * N, N);

  if (h_data != nullptr) {
    const int64_t M = utils::DivUp(D * N, cuda_utils::kCUDANumThreads);
    EMAHiddenBiasCUDAFwdKernel<T_ACC>
        <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, D, N, L, log_q_data, h_data, y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T>
void EMAHiddenCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x,
                          const torch::Tensor& p, const torch::Tensor& log_q,
                          const torch::Tensor& h, const torch::Tensor& v,
                          torch::Tensor& x_grad, torch::Tensor& p_grad,
                          torch::Tensor& q_grad,
                          c10::optional<torch::Tensor>& h_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  const int64_t L = x.size(2);
  TORCH_CHECK(N <= cuda_utils::kWarpSize);

  const c10::complex<T_ACC>* y_grad_data =
      y_grad.data_ptr<c10::complex<T_ACC>>();
  const T* x_data = x.data_ptr<T>();
  const T_ACC* p_data = p.data_ptr<T_ACC>();
  const c10::complex<T_ACC>* log_q_data = log_q.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* h_data =
      h.defined() ? h.data_ptr<c10::complex<T_ACC>>() : nullptr;
  T* x_grad_data = x_grad.data_ptr<T>();
  T_ACC* p_grad_data = p_grad.data_ptr<T_ACC>();
  c10::complex<T_ACC>* q_grad_data = q_grad.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* h_grad_data = nullptr;

  if (h.defined()) {
    h_grad = c10::make_optional(torch::empty_like(h));
    h_grad_data = h_grad->data_ptr<c10::complex<T_ACC>>();
  }

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  const int64_t num_threads = L < cuda_utils::kCUDABlockReduceNumThreads
                                  ? cuda_utils::kWarpSize
                                  : cuda_utils::kCUDABlockReduceNumThreads;

  if (B == 1) {
    EMAHiddenInputBatchSize1CUDABwdKernel<T, T_ACC>
        <<<D, num_threads, 0, cuda_stream>>>(N, L, y_grad_data, p_data,
                                             log_q_data, x_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    EMAHiddenBatchSize1CUDABwdKernel<T, T_ACC>
        <<<dim3(D, N), num_threads, 0, cuda_stream>>>(
            N, L, y_grad_data, x_data, p_data, log_q_data, h_data, p_grad_data,
            q_grad_data, h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (B <= 8) {
    DISPATCH_BATCH_CUDA_KERNEL(EMAHiddenInputCUDABwdKernel, T, T_ACC, B, D,
                               num_threads, 0, cuda_stream, D, N, L,
                               y_grad_data, p_data, log_q_data, x_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    DISPATCH_BATCH_CUDA_KERNEL(EMAHiddenCUDABwdKernel, T, T_ACC, B, dim3(D, N),
                               num_threads, 0, cuda_stream, D, N, L,
                               y_grad_data, x_data, p_data, log_q_data, h_data,
                               p_grad_data, q_grad_data, h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  TORCH_CHECK(v.defined());
  torch::Tensor y_grad_conj = torch::conj_physical(y_grad);
  torch::Tensor x_c = x.to(v.scalar_type());
  torch::Tensor x_grad_c = torch::empty({B, D, L}, v.options());
  torch::Tensor v_grad = torch::empty_like(v);
  const c10::complex<T_ACC>* y_grad_conj_data =
      y_grad_conj.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* x_c_data = x_c.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* v_data = v.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* x_grad_c_data = x_grad_c.data_ptr<c10::complex<T_ACC>>();
  c10::complex<T_ACC>* v_grad_data = v_grad.data_ptr<c10::complex<T_ACC>>();

  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const c10::complex<T_ACC> kAlpha(1);
  const c10::complex<T_ACC> kBeta(0);

  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kN, blas::TransposeOp::kN, D, B, L, N, kAlpha,
      y_grad_conj_data, D * N, N, v_data, L, N * L, kBeta, x_grad_c_data, D * L,
      L);
  blas::GemmStridedBatchedCUDA<c10::complex<T_ACC>>(
      handle, blas::TransposeOp::kC, blas::TransposeOp::kN, D, N, L, B, kAlpha,
      y_grad_data, D * N, N, x_c_data, D * L, L, kBeta, v_grad_data, L, N * L);

  // TODO: Optimize this.
  x_grad = torch::real(x_grad_c).to(x.scalar_type());

  EMAHiddenWeightCUDABwdKernel<T_ACC><<<D * N, num_threads, 0, cuda_stream>>>(
      N, L, v_grad_data, p_data, log_q_data, p_grad_data, q_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (h_data != nullptr) {
    const int64_t M = utils::DivUp(D * N, cuda_utils::kCUDANumThreads);
    EMAHiddenBiasCUDABwdKernel<T_ACC>
        <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            B, D, N, L, y_grad_data, log_q_data, h_data, q_grad_data,
            h_grad_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

#undef DISPATCH_BATCH_CUDA_KERNEL

}  // namespace

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> EMAHiddenCUDAFwd(
    const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& log_q,
    const c10::optional<torch::Tensor>& h) {
  const int64_t B = x.size(0);
  const int64_t D = x.size(1);
  const int64_t N = p.size(1);
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  torch::Tensor y = torch::empty(
      {B, D, N}, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> v = c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCUDAFwd", [&]() {
        EMAHiddenCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(p.expect_contiguous()),
            *(log_q.expect_contiguous()), *(h_maybe_owned->expect_contiguous()),
            y, v);
      });

  return std::make_tuple<torch::Tensor, c10::optional<torch::Tensor>>(
      std::move(y), std::move(v));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>>
EMAHiddenCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                 const torch::Tensor& p, const torch::Tensor& log_q,
                 const c10::optional<torch::Tensor>& h,
                 const c10::optional<torch::Tensor>& v) {
  c10::MaybeOwned<torch::Tensor> h_maybe_owned =
      at::borrow_from_optional_tensor(h);
  c10::MaybeOwned<torch::Tensor> v_maybe_owned =
      at::borrow_from_optional_tensor(v);
  torch::Tensor x_grad = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor p_grad = torch::empty_like(
      p, p.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor q_grad = torch::empty_like(
      log_q, log_q.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> h_grad = c10::nullopt;
  if (h.has_value()) {
    h_grad = c10::make_optional(torch::empty_like(
        *h, h->options().memory_format(at::MemoryFormat::Contiguous)));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "EMAHiddenCUDABwd", [&]() {
        EMAHiddenCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(p.expect_contiguous()), *(log_q.expect_contiguous()),
            *(h_maybe_owned->expect_contiguous()),
            *(v_maybe_owned->expect_contiguous()), x_grad, p_grad, q_grad,
            h_grad);
      });

  return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         c10::optional<torch::Tensor>>(
      std::move(x_grad), std::move(p_grad), std::move(q_grad),
      std::move(h_grad));
}

}  // namespace ops
}  // namespace mega2
