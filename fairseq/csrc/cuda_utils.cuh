#pragma once

#include <c10/macros/Macros.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <cstdint>

namespace mega2 {
namespace cuda_utils {

constexpr int64_t kWarpSize = 32;
constexpr int64_t kCUDANumThreads = 128;
constexpr int64_t kCUDABlockReduceNumThreads = 512;
constexpr int64_t kColwiseThreshold = 256;

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

}  // namespace cuda_utils
}  // namespace mega2
