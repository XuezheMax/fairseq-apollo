#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>
#include <thrust/swap.h>

#include <ATen/cuda/DeviceUtils.cuh>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "cuda_utils.cuh"
#include "twiddle_factor.cuh"

namespace mega2 {
namespace fft {

constexpr int kFFTMaxLength = 8192;
constexpr int kFFTNumThreads = 1024;

constexpr __device__ int Log2(int x) {
  switch (x) {
    case 1: {
      return 0;
    }
    case 2: {
      return 1;
    }
    case 4: {
      return 2;
    }
    case 8: {
      return 3;
    }
    case 16: {
      return 4;
    }
    case 32: {
      return 5;
    }
    case 64: {
      return 6;
    }
    case 128: {
      return 7;
    }
    case 256: {
      return 8;
    }
    case 512: {
      return 9;
    }
    case 1024: {
      return 10;
    }
    case 2048: {
      return 11;
    }
    case 4096: {
      return 12;
    }
    case 8192: {
      return 13;
    }
    default: {
      return -1;
    }
  }
}

template <typename T>
__inline__ __device__ c10::complex<T> Mul1i(c10::complex<T> x) {
  return c10::complex<T>(-x.imag(), x.real());
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void LoadAsComplexImpl(
    const T* __restrict__ src, int64_t size,
    c10::complex<T_ACC>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(kFFTSize - 1 - idx) >> (32 - kNumBits));
      const int64_t d = 2 * kFFTSize - size;
      const int64_t p = idx * 2;
      const int64_t q = idx * 2 + 1;
      const T_ACC x0 = p >= d ? static_cast<T_ACC>(src[p - d]) : T_ACC(0);
      const T_ACC x1 = q >= d ? static_cast<T_ACC>(src[q - d]) : T_ACC(0);
      dst[rev] = c10::complex<T_ACC>(x1, x0);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      const int64_t p = idx * 2;
      const int64_t q = idx * 2 + 1;
      const T_ACC x0 = p < size ? static_cast<T_ACC>(src[p]) : T_ACC(0);
      const T_ACC x1 = q < size ? static_cast<T_ACC>(src[q]) : T_ACC(0);
      dst[rev] = c10::complex<T_ACC>(x0, x1);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP32x2Impl(
    const float* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const float2* src2 = reinterpret_cast<const float2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? src2[idx] : kZero2;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP32x4Impl(
    const float* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float4 kZero4 = {0.0f, 0.0f, 0.0f, 0.0f};
  const float4* src4 = reinterpret_cast<const float4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float4 v4 = idx * 4 < size ? src4[idx] : kZero4;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 2;
      const int64_t q = size / 2 - 1 - idx * 2 - 1;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      dst[rev0] = c10::complex<float>(v4.y, v4.x);
      dst[rev1] = c10::complex<float>(v4.w, v4.z);
    } else {
      const int64_t rev0 = (__brev(idx * 2 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 2 + 1) >> (32 - kNumBits));
      dst[rev0] = c10::complex<float>(v4.x, v4.y);
      dst[rev1] = c10::complex<float>(v4.z, v4.w);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexFP16x2Impl(
    const at::Half* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const __half2* src2 = reinterpret_cast<const __half2*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const float2 v2 = idx * 2 < size ? __half22float2(src2[idx]) : kZero2;
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x2Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  constexpr float2 kZero2 = {0.0f, 0.0f};
  const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const float2 v2 = idx * 2 < size ? __bfloat1622float2(src2[idx]) : kZero2;
#else
    const __nv_bfloat162 x2 = src2[idx];
    const float2 v2 = idx * 2 < size ? make_float2(__bfloat162float(x2.x),
                                                   __bfloat162float(x2.y))
                                     : kZero2;
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (kFlip) {
      const int64_t rev = (__brev(size / 2 - 1 - idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.y, v2.x);
    } else {
      const int64_t rev = (__brev(idx) >> (32 - kNumBits));
      dst[rev] = c10::complex<float>(v2.x, v2.y);
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x4Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  const __nv_bfloat16 kZero = __float2bfloat16(0.0f);
  const cuda_utils::BF16x4 kZero4 = {kZero, kZero, kZero, kZero};
  const cuda_utils::BF16x4* src4 =
      reinterpret_cast<const cuda_utils::BF16x4*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 2; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x4 v4 = idx * 4 < size ? src4[idx] : kZero4;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 2;
      const int64_t q = size / 2 - 1 - idx * 2 - 1;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v4.x1), __bfloat162float(v4.x0));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v4.x3), __bfloat162float(v4.x2));
    } else {
      const int64_t rev0 = (__brev(idx * 2 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 2 + 1) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v4.x0), __bfloat162float(v4.x1));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v4.x2), __bfloat162float(v4.x3));
    }
  }
}

template <int64_t kFFTSize, int64_t kNumThreads, bool kFlip = false>
__inline__ __device__ void LoadAsComplexBF16x8Impl(
    const at::BFloat16* __restrict__ src, int64_t size,
    c10::complex<float>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int64_t kNumBits = Log2(kFFTSize);
  const __nv_bfloat16 kZero = __float2bfloat16(0.0f);
  const cuda_utils::BF16x8 kZero8 = {kZero, kZero, kZero, kZero,
                                     kZero, kZero, kZero, kZero};
  const cuda_utils::BF16x8* src8 =
      reinterpret_cast<const cuda_utils::BF16x8*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread / 4; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const cuda_utils::BF16x8 v8 = idx * 8 < size ? src8[idx] : kZero8;
    if constexpr (kFlip) {
      const int64_t p = size / 2 - 1 - idx * 4;
      const int64_t q = size / 2 - 1 - idx * 4 - 1;
      const int64_t r = size / 2 - 1 - idx * 4 - 2;
      const int64_t s = size / 2 - 1 - idx * 4 - 3;
      const int64_t rev0 = (__brev(p) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(q) >> (32 - kNumBits));
      const int64_t rev2 = (__brev(r) >> (32 - kNumBits));
      const int64_t rev3 = (__brev(s) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v8.x1), __bfloat162float(v8.x0));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v8.x3), __bfloat162float(v8.x2));
      dst[rev2] =
          c10::complex<float>(__bfloat162float(v8.x5), __bfloat162float(v8.x4));
      dst[rev3] =
          c10::complex<float>(__bfloat162float(v8.x7), __bfloat162float(v8.x6));
    } else {
      const int64_t rev0 = (__brev(idx * 4 + 0) >> (32 - kNumBits));
      const int64_t rev1 = (__brev(idx * 4 + 1) >> (32 - kNumBits));
      const int64_t rev2 = (__brev(idx * 4 + 2) >> (32 - kNumBits));
      const int64_t rev3 = (__brev(idx * 4 + 3) >> (32 - kNumBits));
      dst[rev0] =
          c10::complex<float>(__bfloat162float(v8.x0), __bfloat162float(v8.x1));
      dst[rev1] =
          c10::complex<float>(__bfloat162float(v8.x2), __bfloat162float(v8.x3));
      dst[rev2] =
          c10::complex<float>(__bfloat162float(v8.x4), __bfloat162float(v8.x5));
      dst[rev3] =
          c10::complex<float>(__bfloat162float(v8.x6), __bfloat162float(v8.x7));
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void LoadAsComplex(
    const T* __restrict__ src, int64_t size,
    c10::complex<T_ACC>* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;

  if constexpr (std::is_same<T, float>::value &&
                std::is_same<T_ACC, float>::value) {
    if (kElementsPerThread % 2 == 0 &&
        reinterpret_cast<uintptr_t>(src) % sizeof(float4) == 0 &&
        size % 4 == 0) {
      LoadAsComplexFP32x4Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (reinterpret_cast<uintptr_t>(src) % sizeof(float2) == 0 &&
               size % 2 == 0) {
      LoadAsComplexFP32x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<float, float, kFFTSize, kNumThreads, kFlip>(src, size,
                                                                    dst);
    }
  } else if constexpr (std::is_same<T, at::Half>::value &&
                       std::is_same<T_ACC, float>::value) {
    if (reinterpret_cast<uintptr_t>(src) % sizeof(__half2) == 0 &&
        size % 2 == 0) {
      LoadAsComplexFP16x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<at::Half, float, kFFTSize, kNumThreads, kFlip>(
          src, size, dst);
    }
  } else if constexpr (std::is_same<T, at::BFloat16>::value &&
                       std::is_same<T_ACC, float>::value) {
    if (kElementsPerThread % 4 == 0 &&
        reinterpret_cast<uintptr_t>(src) % sizeof(cuda_utils::BF16x8) == 0 &&
        size % 8 == 0) {
      LoadAsComplexBF16x8Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (kElementsPerThread % 2 == 0 &&
               reinterpret_cast<uintptr_t>(src) % sizeof(cuda_utils::BF16x4) ==
                   0 &&
               size % 4 == 0) {
      LoadAsComplexBF16x4Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else if (reinterpret_cast<uintptr_t>(src) % sizeof(__nv_bfloat162) == 0 &&
               size % 2 == 0) {
      LoadAsComplexBF16x2Impl<kFFTSize, kNumThreads, kFlip>(src, size, dst);
    } else {
      LoadAsComplexImpl<at::BFloat16, float, kFFTSize, kNumThreads, kFlip>(
          src, size, dst);
    }
  } else {
    LoadAsComplexImpl<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, dst);
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsRealImpl1(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  const T_ACC* src_real = reinterpret_cast<const T_ACC*>(src);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    const int64_t p = idx * 2;
    const int64_t q = idx * 2 + 1;
    if constexpr (kFlip) {
      if (p < size) {
        dst[p] = static_cast<T>(src_real[size - 1 - p] * scale);
      }
      if (q < size) {
        dst[q] = static_cast<T>(src_real[size - 1 - q] * scale);
      }
    } else {
      if (p < size) {
        dst[p] = static_cast<T>(src_real[p] * scale);
      }
      if (q < size) {
        dst[q] = static_cast<T>(src_real[q] * scale);
      }
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsRealImpl2(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  constexpr int64_t kElementsPerThread = kFFTSize / kNumThreads;
  c10::complex<T>* dst_complex = reinterpret_cast<c10::complex<T>*>(dst);
#pragma unroll
  for (int64_t i = 0; i < kElementsPerThread; ++i) {
    const int64_t idx = i * blockDim.x + threadIdx.x;
    if (idx * 2 < size) {
      if constexpr (kFlip) {
        const c10::complex<T_ACC> v = src[size / 2 - 1 - idx] * scale;
        dst_complex[idx] =
            c10::complex<T>(static_cast<T>(v.imag()), static_cast<T>(v.real()));
      } else if constexpr (std::is_same<T, T_ACC>::value) {
        dst_complex[idx] = src[idx] * scale;
      } else {
        const c10::complex<T_ACC> v = src[idx] * scale;
        dst_complex[idx] =
            c10::complex<T>(static_cast<T>(v.real()), static_cast<T>(v.imag()));
      }
    }
  }
}

template <typename T, typename T_ACC, int64_t kFFTSize, int64_t kNumThreads,
          bool kFlip = false>
__inline__ __device__ void SaveAsReal(
    const c10::complex<T_ACC>* __restrict__ src, int64_t size, T_ACC scale,
    T* __restrict__ dst) {
  if (reinterpret_cast<uintptr_t>(dst) % sizeof(c10::complex<T>) == 0 &&
      size % 2 == 0) {
    SaveAsRealImpl2<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, scale,
                                                            dst);
  } else {
    SaveAsRealImpl1<T, T_ACC, kFFTSize, kNumThreads, kFlip>(src, size, scale,
                                                            dst);
  }
}

template <typename T, bool kIFFT = false>
__inline__ __device__ c10::complex<T> WarpFFTImpl(c10::complex<T> x) {
#pragma unroll
  for (int offset = 1; offset < cuda_utils::kWarpSize; offset <<= 1) {
    const int r = (threadIdx.x & (offset - 1));
    const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(offset, r);
    const T u = WARP_SHFL_XOR(x.real(), offset);
    const T v = WARP_SHFL_XOR(x.imag(), offset);
    const c10::complex<T> y(u, v);
    // x = (threadIdx.x & offset) ? (y - x * (kIFFT ? w : std::conj(w)))
    //                            : (x + y * (kIFFT ? w : std::conj(w)));
    if constexpr (kIFFT) {
      x = (threadIdx.x & offset) ? (y - x * w) : (x + y * w);
    } else {
      x = (threadIdx.x & offset) ? (y - x * std::conj(w))
                                 : (x + y * std::conj(w));
    }
  }
  return x;
}

template <typename T, int N, bool kIFFT = false>
__inline__ __device__ void BlockFFTImpl(c10::complex<T>* shared_mem) {
  constexpr int D = Log2(cuda_utils::kWarpSize);
  constexpr int K = Log2(N);

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    shared_mem[i] = WarpFFTImpl<T, kIFFT>(shared_mem[i]);
  }
  // __syncthreads();

#pragma unroll
  for (int i = D; i < K; ++i) {
    __syncthreads();
    const int m = (1 << i);
    for (int j = threadIdx.x; j < N / 2; j += blockDim.x) {
      const int r = (j & (m - 1));
      const int p = ((j >> i) << (i + 1)) + r;
      const int q = p + m;
      const c10::complex<T> w = cuda_utils::TwiddleFactor<T>(m, r);
      const c10::complex<T> u = shared_mem[p];
      const c10::complex<T> v = shared_mem[q] * (kIFFT ? w : std::conj(w));
      shared_mem[p] = u + v;
      shared_mem[q] = u - v;
    }
    // __syncthreads();
  }
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void BlockRFFT(const T* X, int N, bool flip,
                                     c10::complex<T_ACC>* Y,
                                     c10::complex<T_ACC>* shared_mem) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;

  if (flip) {
    LoadAsComplex<T, T_ACC, kFFTSize, kNumThreads, true>(X, N, shared_mem);
  } else {
    LoadAsComplex<T, T_ACC, kFFTSize, kNumThreads, false>(X, N, shared_mem);
  }
  __syncthreads();

  BlockFFTImpl<T_ACC, kFFTSize, /*kIFFT=*/false>(shared_mem);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = idx == 0 ? 0 : kFFTSize - idx;
    const c10::complex<T_ACC> w =
        std::conj(cuda_utils::TwiddleFactor<T_ACC>(kFFTSize, idx));
    const c10::complex<T_ACC> z1 = shared_mem[idx];
    const c10::complex<T_ACC> z2 = std::conj(shared_mem[rev]);
    const c10::complex<T_ACC> zx = (z1 + z2) * T_ACC(0.5);
    const c10::complex<T_ACC> zy = Mul1i(z1 - z2) * w * T_ACC(-0.5);
    Y[idx] = zx + zy;
    if (idx == 0) {
      Y[kFFTSize] = zx - zy;
    }
    // Y[idx] = idx == 0 ? c10::complex<T_ACC>(zx.real() + zy.real(),
    //                                         zx.real() - zy.real())
    //                   : zx + zy;
  }
  // __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void BlockIRFFT(const c10::complex<T_ACC>* X, int N,
                                      bool flip, T* Y,
                                      c10::complex<T_ACC>* shared_mem) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(idx) >> (32 - kNumBits));
    const c10::complex<T_ACC> w =
        cuda_utils::TwiddleFactor<T_ACC>(kFFTSize, idx);
    const c10::complex<T_ACC> z1 = X[idx];
    const c10::complex<T_ACC> z2 = std::conj(X[kFFTSize - idx]);
    // const c10::complex<T_ACC> z1 =
    //     idx == 0 ? c10::complex<T_ACC>(X[0].real()) : X[idx];
    // const c10::complex<T_ACC> z2 = idx == 0 ?
    // c10::complex<T_ACC>(X[0].imag())
    //                                         : std::conj(X[kFFTSize - idx]);
    const c10::complex<T_ACC> zx = (z1 + z2);
    const c10::complex<T_ACC> zy = (z1 - z2) * w;
    shared_mem[rev] = (zx + Mul1i(zy)) * T_ACC(0.5);
  }
  __syncthreads();

  BlockFFTImpl<T_ACC, kFFTSize, /*kIFFT=*/true>(shared_mem);
  __syncthreads();

  constexpr T_ACC coef = T_ACC(1) / static_cast<T_ACC>(kFFTSize);
  if (flip) {
    SaveAsReal<T, T_ACC, kFFTSize, kNumThreads, true>(shared_mem, N, coef, Y);
  } else {
    SaveAsReal<T, T_ACC, kFFTSize, kNumThreads, false>(shared_mem, N, coef, Y);
  }
}

}  // namespace fft
}  // namespace mega2
