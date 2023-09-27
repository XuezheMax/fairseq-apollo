#pragma once

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <cstdint>
#include <limits>

namespace mega2 {
namespace cuda_utils {

constexpr int64_t kWarpSize = 32;
constexpr int64_t kCUDANumThreads = 128;
constexpr int64_t kCUDABlockReduceNumThreads = 512;
constexpr int64_t kColwiseThreshold = 256;
constexpr int64_t kMaxStaticSharedMemorySize = 49152;

struct __align__(8) BF16x4 {
  __nv_bfloat16 x0;
  __nv_bfloat16 x1;
  __nv_bfloat16 x2;
  __nv_bfloat16 x3;
};

struct __align__(16) BF16x8 {
  __nv_bfloat16 x0;
  __nv_bfloat16 x1;
  __nv_bfloat16 x2;
  __nv_bfloat16 x3;
  __nv_bfloat16 x4;
  __nv_bfloat16 x5;
  __nv_bfloat16 x6;
  __nv_bfloat16 x7;
};

template <typename T>
C10_HOST_DEVICE thrust::pair<T, T> Fast2Sum(T a, T b) {
  const T s = a + b;
  const T z = s - a;
  const T t = b - z;
  return thrust::make_pair(s, t);
}

template <typename T>
C10_HOST_DEVICE thrust::pair<T, T> KahanAdd(T x, T sum, T c) {
  const T y = x - c;
  const T t = sum + y;
  c = t - sum - y;
  return thrust::make_pair(t, c);
}

template <typename T>
C10_HOST_DEVICE T Cube(T x) {
  return x * x * x;
}

template <typename T>
C10_HOST_DEVICE thrust::tuple<int64_t, T, T> WelfordUpdate(int64_t m0, T m1,
                                                           T m2, T x) {
  ++m0;
  const T coef = T(1) / static_cast<T>(m0);
  const T delta1 = x - m1;
  m1 += coef * delta1;
  const T delta2 = delta1 * (x - m1) - m2;
  m2 += coef * delta2;
  return thrust::make_tuple(m0, m1, m2);
}

template <typename T>
C10_HOST_DEVICE thrust::tuple<int64_t, T, T> WelfordCombine(int64_t a_m0,
                                                            T a_m1, T a_m2,
                                                            int64_t b_m0,
                                                            T b_m1, T b_m2) {
  const int64_t m0 = a_m0 + b_m0;
  const T c1 = m0 == 0 ? T(0) : static_cast<T>(a_m0) / static_cast<T>(m0);
  const T c2 = m0 == 0 ? T(0) : static_cast<T>(b_m0) / static_cast<T>(m0);
  const T delta = b_m1 - a_m1;
  const T m1 = c1 * a_m1 + c2 * b_m1;
  const T m2 = c1 * a_m2 + c2 * b_m2 + (c1 * delta) * (c2 * delta);
  return thrust::make_tuple(m0, m1, m2);
}

template <typename T>
__inline__ __device__ T WarpReduceMax(T x) {
#pragma unroll
  for (int64_t offset = (kWarpSize >> 1); offset > 0; offset >>= 1) {
    x = c10::cuda::compat::max(x, WARP_SHFL_DOWN(x, offset));
  }
  return x;
}

template <typename T>
__inline__ __device__ T BlockReduceMax(T x, T* shm) {
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % kWarpSize;
  const int64_t wid = tid / kWarpSize;
  const int64_t num_warps = blockDim.x / kWarpSize;
  x = WarpReduceMax<T>(x);
  __syncthreads();
  if (lid == 0) {
    shm[wid] = x;
  }
  __syncthreads();
  x = tid < num_warps ? shm[lid] : -std::numeric_limits<T>::infinity();
  if (wid == 0) {
    x = WarpReduceMax<T>(x);
  }
  return x;
}

template <typename T>
__inline__ __device__ thrust::tuple<int64_t, T, T> WarpReduceMoments(int64_t m0,
                                                                     T m1,
                                                                     T m2) {
#pragma unroll
  for (int64_t offset = (kWarpSize >> 1); offset > 0; offset >>= 1) {
    int64_t n = m0;
    m0 += WARP_SHFL_DOWN(m0, offset);
    const T c1 = m0 == 0 ? T(0) : static_cast<T>(n) / static_cast<T>(m0);
    const T c2 = m0 == 0 ? T(0) : T(1) - c1;
    const T u = WARP_SHFL_DOWN(m1, offset);
    const T v = WARP_SHFL_DOWN(m2, offset);
    const T delta = u - m1;
    m1 = c1 * m1 + c2 * u;
    m2 = c1 * m2 + c2 * v + (c1 * delta) * (c2 * delta);
  }
  return thrust::make_tuple(m0, m1, m2);
}

template <typename T>
__inline__ __device__ thrust::tuple<int64_t, T, T> BlockReduceMoments(
    int64_t m0, T m1, T m2, int64_t* m0_shared, T* m1_shared, T* m2_shared) {
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % kWarpSize;
  const int64_t wid = tid / kWarpSize;
  const int64_t num_warps = blockDim.x / kWarpSize;
  thrust::tie(m0, m1, m2) = WarpReduceMoments(m0, m1, m2);
  __syncthreads();
  if (lid == 0) {
    m0_shared[wid] = m0;
    m1_shared[wid] = m1;
    m2_shared[wid] = m2;
  }
  __syncthreads();
  m0 = tid < num_warps ? m0_shared[lid] : 0;
  m1 = tid < num_warps ? m1_shared[lid] : T(0);
  m2 = tid < num_warps ? m2_shared[lid] : T(0);
  if (wid == 0) {
    thrust::tie(m0, m1, m2) = WarpReduceMoments(m0, m1, m2);
  }
  return thrust::make_tuple(m0, m1, m2);
}

template <typename T>
__inline__ __device__ c10::complex<T> WarpReduceComplexSum(c10::complex<T> x) {
  T u = x.real();
  T v = x.imag();
#pragma unroll
  for (int64_t offset = (kWarpSize >> 1); offset > 0; offset >>= 1) {
    u += WARP_SHFL_DOWN(u, offset);
    v += WARP_SHFL_DOWN(v, offset);
  }
  return c10::complex<T>(u, v);
}

template <typename T>
__inline__ __device__ c10::complex<T> BlockReduceComplexSum(
    c10::complex<T> x, c10::complex<T>* shared_mem) {
  const int64_t tid = threadIdx.x;
  const int64_t lid = tid % kWarpSize;
  const int64_t wid = tid / kWarpSize;
  const int64_t num_warps = blockDim.x / kWarpSize;
  x = WarpReduceComplexSum(x);
  __syncthreads();
  if (lid == 0) {
    shared_mem[wid] = x;
  }
  __syncthreads();
  x = tid < num_warps ? shared_mem[lid] : c10::complex<T>(0);
  if (wid == 0) {
    x = WarpReduceComplexSum(x);
  }
  return x;
}

template <class KernelFunc, class... Args>
void LaunchKernel(KernelFunc kernel, dim3 dg, dim3 db, int64_t shared_mem_size,
                  cudaStream_t cuda_stream, Args... args) {
  if (shared_mem_size > cuda_utils::kMaxStaticSharedMemorySize) {
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
  }
  kernel<<<dg, db, shared_mem_size, cuda_stream>>>(args...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cuda_utils
}  // namespace mega2
