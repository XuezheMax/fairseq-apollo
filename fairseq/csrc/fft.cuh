#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>
#include <thrust/swap.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "cuda_utils.cuh"

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
      return 0;
    }
  }
}

template <typename T>
__inline__ __device__ c10::complex<T> PolarPi(T k) {
  constexpr T kPi = T(M_PI);
  T s;
  T c;
  c10::cuda::compat::sincos(k * kPi, &s, &c);
  return c10::complex<T>(c, s);
}

template <>
__inline__ __device__ c10::complex<float> PolarPi(float k) {
  constexpr float kPi = float(M_PI);
  float s;
  float c;
  // sincospif(k, &s, &c);
  __sincosf(k * kPi, &s, &c);
  return c10::complex<float>(c, s);
}

template <>
__inline__ __device__ c10::complex<double> PolarPi(double k) {
  double s;
  double c;
  sincospi(k, &s, &c);
  return c10::complex<double>(c, s);
}

template <typename T>
__inline__ __device__ c10::complex<T> Mul1i(c10::complex<T> x) {
  return c10::complex<T>(-x.imag(), x.real());
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void LoadComplex0(const T* __restrict__ X, int N,
                                        c10::complex<T_ACC>* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);
  constexpr c10::complex<T_ACC> kZero(T_ACC(0), T_ACC(0));

  const c10::complex<T>* X_complex =
      reinterpret_cast<const c10::complex<T>*>(X);
  const int M = N / 2;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(idx) >> (32 - kNumBits));
    if constexpr (std::is_same<T, T_ACC>::value) {
      Y[rev] = idx < M ? X_complex[idx] : kZero;
    } else {
      const T_ACC x =
          idx < M ? static_cast<T_ACC>(X_complex[idx].real()) : T_ACC(0);
      const T_ACC y =
          idx < M ? static_cast<T_ACC>(X_complex[idx].imag()) : T_ACC(0);
      Y[rev] = c10::complex<T_ACC>(x, y);
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void LoadComplex1(const T* __restrict__ X, int N,
                                        c10::complex<T_ACC>* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(idx) >> (32 - kNumBits));
    const int p = 2 * idx;
    const int q = 2 * idx + 1;
    const T_ACC x = p < N ? static_cast<T_ACC>(X[p]) : T_ACC(0);
    const T_ACC y = q < N ? static_cast<T_ACC>(X[q]) : T_ACC(0);
    Y[rev] = c10::complex<T_ACC>(x, y);
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void LoadFlippedComplex0(
    const T* __restrict__ X, int N, c10::complex<T_ACC>* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);
  constexpr c10::complex<T_ACC> kZero(T_ACC(0), T_ACC(0));

  const c10::complex<T>* X_complex =
      reinterpret_cast<const c10::complex<T>*>(X);
  const int M = N / 2;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(M - 1 - idx) >> (32 - kNumBits));
    if constexpr (std::is_same<T, T_ACC>::value) {
      const c10::complex<T_ACC> x = idx < M ? X_complex[idx] : kZero;
      Y[rev] = c10::complex<T_ACC>(x.imag(), x.real());
    } else {
      const T_ACC x =
          idx < M ? static_cast<T_ACC>(X_complex[idx].real()) : T_ACC(0);
      const T_ACC y =
          idx < M ? static_cast<T_ACC>(X_complex[idx].imag()) : T_ACC(0);
      Y[rev] = c10::complex<T_ACC>(y, x);
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void LoadFlippedComplex1(
    const T* __restrict__ X, int N, c10::complex<T_ACC>* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);

  T_ACC* Y_ptr = reinterpret_cast<T_ACC*>(Y);

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int p = 2 * idx;
    const int q = 2 * idx + 1;
    const T_ACC x = p < N ? static_cast<T_ACC>(X[p]) : T_ACC(0);
    const T_ACC y = q < N ? static_cast<T_ACC>(X[q]) : T_ACC(0);
    Y_ptr[p < N ? N - 1 - p : p] = x;
    Y_ptr[q < N ? N - 1 - q : q] = y;
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(idx) >> (32 - kNumBits));
    if (idx < rev) {
      thrust::swap(Y[idx], Y[rev]);
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void DumpComplex0(
    const c10::complex<T_ACC>* __restrict__ X, int N, T_ACC coef,
    T* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  c10::complex<T>* Y_complex = reinterpret_cast<c10::complex<T>*>(Y);
  const int M = N / 2;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    if (idx < M) {
      const c10::complex<T_ACC> x = X[idx] * coef;
      if constexpr (std::is_same<T, T_ACC>::value) {
        Y_complex[idx] = x;
      } else {
        Y_complex[idx] =
            c10::complex<T>(static_cast<T>(x.real()), static_cast<T>(x.imag()));
      }
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void DumpComplex1(
    const c10::complex<T_ACC>* __restrict__ X, int N, T_ACC coef,
    T* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  const T_ACC* X_ptr = reinterpret_cast<const T_ACC*>(X);
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int p = 2 * idx;
    const int q = 2 * idx + 1;
    if (p < N) {
      Y[p] = static_cast<T>(X_ptr[p] * coef);
    }
    if (q < N) {
      Y[q] = static_cast<T>(X_ptr[q] * coef);
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void DumpFlippedComplex0(
    const c10::complex<T_ACC>* __restrict__ X, int N, T_ACC coef,
    T* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  c10::complex<T>* Y_complex = reinterpret_cast<c10::complex<T>*>(Y);
  const int M = N / 2;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    if (idx < M) {
      const c10::complex<T_ACC> x = X[M - 1 - idx] * coef;
      Y_complex[idx] =
          c10::complex<T>(static_cast<T>(x.imag()), static_cast<T>(x.real()));
    }
  }
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void DumpFlippedComplex1(
    const c10::complex<T_ACC>* __restrict__ X, int N, T_ACC coef,
    T* __restrict__ Y) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  const T_ACC* X_ptr = reinterpret_cast<const T_ACC*>(X);
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int p = 2 * idx;
    const int q = 2 * idx + 1;
    if (p < N) {
      Y[p] = static_cast<T>(X_ptr[N - 1 - p] * coef);
    }
    if (q < N) {
      Y[q] = static_cast<T>(X_ptr[N - 1 - q] * coef);
    }
  }
  __syncthreads();
}

template <typename T, int N, bool kIFFT = false>
__inline__ __device__ void BlockFFTImpl(c10::complex<T>* shared_mem) {
  constexpr int K = Log2(N);
  // constexpr T kPi = T(M_PI);

#pragma unroll
  for (int i = 0; i < K; ++i) {
    const int m = (1 << i);
    for (int j = threadIdx.x; j < N / 2; j += blockDim.x) {
      const int r = (j & (m - 1));
      const int p = ((j >> i) << (i + 1)) + r;
      const int q = p + m;
      const T k = static_cast<T>(r) / static_cast<T>(m);
      const c10::complex<T> w = PolarPi(kIFFT ? k : -k);
      const c10::complex<T> u = shared_mem[p];
      const c10::complex<T> v = shared_mem[q] * w;
      shared_mem[p] = u + v;
      shared_mem[q] = u - v;
    }
    __syncthreads();
  }
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void BlockRFFT(const T* X, int N, bool flip,
                                     c10::complex<T_ACC>* Y,
                                     c10::complex<T_ACC>* shared_mem) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  // constexpr T_ACC kPi = T_ACC(M_PI);

  if (flip) {
    if (N & 1) {
      LoadFlippedComplex1<T, T_ACC, kFFTSize, kNumThreads>(X, N, shared_mem);
    } else {
      LoadFlippedComplex0<T, T_ACC, kFFTSize, kNumThreads>(X, N, shared_mem);
    }
  } else {
    if (N & 1) {
      LoadComplex1<T, T_ACC, kFFTSize, kNumThreads>(X, N, shared_mem);
    } else {
      LoadComplex0<T, T_ACC, kFFTSize, kNumThreads>(X, N, shared_mem);
    }
  }

  BlockFFTImpl<T_ACC, kFFTSize, /*kIFFT=*/false>(shared_mem);

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = idx == 0 ? 0 : kFFTSize - idx;
    const c10::complex<T_ACC> w =
        PolarPi(-static_cast<T_ACC>(idx) / static_cast<T_ACC>(kFFTSize));
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
  __syncthreads();
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__inline__ __device__ void BlockIRFFT(const c10::complex<T_ACC>* X, int N,
                                      bool flip, T* Y,
                                      c10::complex<T_ACC>* shared_mem) {
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr int kNumBits = Log2(kFFTSize);
  // constexpr T_ACC kPi = T_ACC(M_PI);

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    const int rev = (__brev(idx) >> (32 - kNumBits));
    const c10::complex<T_ACC> w =
        PolarPi(static_cast<T_ACC>(idx) / static_cast<T_ACC>(kFFTSize));
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

  constexpr T_ACC coef = T_ACC(1) / static_cast<T_ACC>(kFFTSize);
  if (flip) {
    if (N & 1) {
      DumpFlippedComplex1<T, T_ACC, kFFTSize, kNumThreads>(shared_mem, N, coef,
                                                           Y);
    } else {
      DumpFlippedComplex0<T, T_ACC, kFFTSize, kNumThreads>(shared_mem, N, coef,
                                                           Y);
    }
  } else {
    if (N & 1) {
      DumpComplex1<T, T_ACC, kFFTSize, kNumThreads>(shared_mem, N, coef, Y);
    } else {
      DumpComplex0<T, T_ACC, kFFTSize, kNumThreads>(shared_mem, N, coef, Y);
    }
  }
}

}  // namespace fft
}  // namespace mega2
