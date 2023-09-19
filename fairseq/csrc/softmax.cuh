#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <thrust/pair.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <cmath>
#include <limits>

#include "cuda_utils.cuh"

namespace mega2 {
namespace softmax {

constexpr int64_t kMaxSoftmaxSize = 8192;
constexpr int64_t kUnroll = 4;

template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionSoftmaxFwdKernel(int64_t outer_size,
                                          int64_t inner_size, bool causal_mask,
                                          const T* x, T* y) {
  constexpr int64_t kElementsPerThread = kCapacity / kNumThreads;
  constexpr T_ACC kInf = std::numeric_limits<T_ACC>::infinity();

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC x_acc[kElementsPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int32_t r = blockIdx.x;
  const int64_t n = causal_mask ? inner_size - outer_size + r + 1 : inner_size;
  T_ACC d = T_ACC(0);
  T_ACC m = -kInf;  // attn mask is -inf.

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    const T_ACC v =
        idx < n ? static_cast<T_ACC>(x[i * inner_size + idx]) : -kInf;
    x_acc[j] = v;
    m = c10::cuda::compat::max(m, v);
  }
  if constexpr (kNumThreads <= cuda_utils::kWarpSize) {
    m = cuda_utils::WarpReduceMax<T_ACC>(m);
  } else {
    m = cuda_utils::BlockReduceMax<T_ACC>(m, shm);
  }
  if (threadIdx.x == 0) {
    shm[0] = m;
  }
  __syncthreads();

  m = shm[0];

  if (std::isinf(m)) {
#pragma unroll
    for (int64_t j = 0; j < kElementsPerThread; ++j) {
      const int64_t idx = j * blockDim.x + threadIdx.x;
      if (idx < inner_size) {
        y[i * inner_size + idx] = T(0);
      }
    }
    return;
  }

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const T_ACC v = c10::cuda::compat::exp(x_acc[j] - m);
    x_acc[j] = v;
    d += v;
  }
  if constexpr (kNumThreads <= cuda_utils::kWarpSize) {
    d = at::native::cuda_utils::WarpReduceSum<T_ACC>(d);
  } else {
    d = at::native::cuda_utils::BlockReduceSum<T_ACC>(d, shm);
  }
  if (threadIdx.x == 0) {
    shm[0] = d;
  }
  __syncthreads();

  const T_ACC c = shm[0] == T_ACC(0) ? T_ACC(0) : T_ACC(1) / shm[0];
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    if (idx < inner_size) {
      y[i * inner_size + idx] = static_cast<T>(x_acc[j] * c);
    }
  }
}

template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionDropKeySoftmaxFwdKernel(
    at::PhiloxCudaState philox_args, int64_t outer_size, int64_t inner_size,
    bool causal_mask, T_ACC dropout, bool inverted_dropout,
    const T* __restrict x, T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kCapacity / kNumThreads;
  constexpr int64_t kCapacityPerThread = std::max(kElementsPerThread, kUnroll);
  constexpr T_ACC kInf = std::numeric_limits<T_ACC>::infinity();

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC x_acc[kCapacityPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int32_t r = blockIdx.x;
  const int64_t n = causal_mask ? inner_size - outer_size + r + 1 : inner_size;

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(seed, i * blockDim.x + threadIdx.x, offset, &state);

#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread; j += kUnroll) {
    const float4 randv = curand_uniform4(&state);
    x_acc[j + 0] = randv.x < dropout ? -kInf : T_ACC(0);
    x_acc[j + 1] = randv.y < dropout ? -kInf : T_ACC(0);
    x_acc[j + 2] = randv.z < dropout ? -kInf : T_ACC(0);
    x_acc[j + 3] = randv.w < dropout ? -kInf : T_ACC(0);
  }

  const T_ACC scale =
      inverted_dropout ? T_ACC(1) / (T_ACC(1) - dropout) : T_ACC(1);
  T_ACC d = T_ACC(0);
  T_ACC m = -kInf;  // attn mask is -inf.

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    // const T_ACC v = idx < n
    //                     ? static_cast<T_ACC>(x[i * inner_size + idx]) +
    //                     x_acc[j] : -kInf;
    const T_ACC v =
        idx < n ? static_cast<T_ACC>(x[i * inner_size + idx]) * scale + x_acc[j]
                : -kInf;
    x_acc[j] = v;
    m = c10::cuda::compat::max(m, v);
  }
  if constexpr (kNumThreads <= cuda_utils::kWarpSize) {
    m = cuda_utils::WarpReduceMax<T_ACC>(m);
  } else {
    m = cuda_utils::BlockReduceMax<T_ACC>(m, shm);
  }
  if (threadIdx.x == 0) {
    shm[0] = m;
  }
  __syncthreads();

  m = shm[0];

  if (std::isinf(m)) {
#pragma unroll
    for (int64_t j = 0; j < kElementsPerThread; ++j) {
      const int64_t idx = j * blockDim.x + threadIdx.x;
      if (idx < inner_size) {
        y[i * inner_size + idx] = T(0);
      }
    }
    return;
  }

#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const T_ACC x = x_acc[j];
    const T_ACC v = c10::cuda::compat::exp(x_acc[j] - m);
    x_acc[j] = v;
    d += v;
  }
  if constexpr (kNumThreads <= cuda_utils::kWarpSize) {
    d = at::native::cuda_utils::WarpReduceSum<T_ACC>(d);
  } else {
    d = at::native::cuda_utils::BlockReduceSum<T_ACC>(d, shm);
  }
  if (threadIdx.x == 0) {
    shm[0] = d;
  }
  __syncthreads();

  const T_ACC c = shm[0] == T_ACC(0) ? T_ACC(0) : T_ACC(1) / shm[0];
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    if (idx < inner_size) {
      y[i * inner_size + idx] = static_cast<T>(x_acc[j] * c);
    }
  }
}

template <typename T, typename T_ACC, int64_t kCapacity, int64_t kNumThreads>
__global__ void AttentionSoftmaxBwdKernel(int64_t outer_size,
                                          int64_t inner_size, bool causal_mask,
                                          T_ACC scale, const T* y_grad,
                                          const T* __restrict__ y, T* x_grad) {
  constexpr int kElementsPerThread = kCapacity / kNumThreads;

  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  T_ACC p_acc[kElementsPerThread];
  T_ACC o_acc[kElementsPerThread];

  const int64_t i = blockIdx.y * outer_size + blockIdx.x;
  const int32_t r = blockIdx.x;
  const int64_t n = causal_mask ? inner_size - outer_size + r + 1 : inner_size;

  T_ACC sum = T_ACC(0);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    const T_ACC g =
        idx < n ? static_cast<T_ACC>(y_grad[i * inner_size + idx]) : T_ACC(0);
    const T_ACC p =
        idx < n ? static_cast<T_ACC>(y[i * inner_size + idx]) : T_ACC(0);
    const T_ACC o = p * g;
    p_acc[j] = p;
    o_acc[j] = o;
    sum += o;
  }
  if constexpr (kNumThreads <= cuda_utils::kWarpSize) {
    sum = at::native::cuda_utils::WarpReduceSum<T_ACC>(sum);
  } else {
    sum = at::native::cuda_utils::BlockReduceSum<T_ACC>(sum, shm);
  }
  if (threadIdx.x == 0) {
    shm[0] = sum;
  }
  __syncthreads();

  sum = shm[0];
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    const int64_t idx = j * blockDim.x + threadIdx.x;
    if (idx < inner_size) {
      // x_grad[i * inner_size + idx] = static_cast<T>(o_acc[j] - y_acc[j] *
      // sum);
      x_grad[i * inner_size + idx] =
          static_cast<T>((o_acc[j] - p_acc[j] * sum) * scale);
    }
  }
}

}  // namespace softmax

#define DISPATCH_ATTENTION_SOFTMAX_CUDA_KERNEL(                               \
    KernelFunc, T, T_ACC, block_size, inner_size, shm_size, cuda_stream, ...) \
  do {                                                                        \
    if (inner_size <= 32) {                                                   \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32>, block_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 64) {                                            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32>, block_size, 32,  \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 128) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32>, block_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 256) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32>, block_size, 32, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 512) {                                           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64>, block_size, 64, \
                               shm_size, cuda_stream, __VA_ARGS__);           \
    } else if (inner_size <= 1024) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128>, block_size,   \
                               128, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 2048) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256>, block_size,   \
                               256, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 4096) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512>, block_size,   \
                               512, shm_size, cuda_stream, __VA_ARGS__);      \
    } else if (inner_size <= 8192) {                                          \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8192, 1024>, block_size,  \
                               1024, shm_size, cuda_stream, __VA_ARGS__);     \
    } else {                                                                  \
      TORCH_CHECK(false);                                                     \
    }                                                                         \
  } while (0)

}  // namespace mega2
