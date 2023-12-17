#pragma once

#include <c10/cuda/CUDAMathCompat.h>
#include <thrust/pair.h>

#include "register_utils.cuh"

namespace mega2 {
namespace activations {

template <typename T>
__inline__ __device__ T Sigmoid(T x) {
  return T(1) / (T(1) + c10::cuda::compat::exp(-x));
}

template <>
__inline__ __device__ float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

template <typename T>
__inline__ __device__ T Tanh(T x) {
  return c10::cuda::compat::tanh(x);
}

// https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/Eigen/src/Core/MathFunctionsImpl.h#L26
template <>
__inline__ __device__ float Tanh(float x) {
  // return __tanhf(x);
  constexpr float kAlpha1 = 4.89352455891786e-03f;
  constexpr float kAlpha3 = 6.37261928875436e-04f;
  constexpr float kAlpha5 = 1.48572235717979e-05f;
  constexpr float kAlpha7 = 5.12229709037114e-08f;
  constexpr float kAlpha9 = -8.60467152213735e-11f;
  constexpr float kAlpha11 = 2.00018790482477e-13f;
  constexpr float kAlpha13 = -2.76076847742355e-16f;

  constexpr float kBeta0 = 4.89352518554385e-03f;
  constexpr float kBeta2 = 2.26843463243900e-03f;
  constexpr float kBeta4 = 1.18534705686654e-04f;
  constexpr float kBeta6 = 1.19825839466702e-06f;

  const float x1 = fmaxf(fminf(x, 9.0f), -9.0f);
  const float x2 = x1 * x1;

  float p = x2 * kAlpha13 + kAlpha11;
  p = x2 * p + kAlpha9;
  p = x2 * p + kAlpha7;
  p = x2 * p + kAlpha5;
  p = x2 * p + kAlpha3;
  p = x2 * p + kAlpha1;
  p = x1 * p;

  float q = x2 * kBeta6 + kBeta4;
  q = x2 * q + kBeta2;
  q = x2 * q + kBeta0;

  return p / q;
}

template <typename T>
__inline__ __device__ T Swish(T x) {
  return x * Sigmoid<T>(x);
}

template <typename T>
__inline__ __device__ T SwiGLU(T x1, T x2) {
  return Swish<T>(x1) * Tanh<T>(x2);
}

template <typename T>
__inline__ __device__ T SwishBwd(T y_grad, T x) {
  const T s = Sigmoid<T>(x);
  return y_grad * (s + x * s * (T(1) - s));
}

template <typename T>
__inline__ __device__ thrust::pair<T, T> SwiGLUBwd(T y_grad, T x1, T x2) {
  const T s = Sigmoid<T>(x1);
  const T t = Tanh<T>(x2);
  const T x1_grad = y_grad * t * (s + x1 * s * (T(1) - s));
  const T x2_grad = y_grad * x1 * s * (T(1) - t * t);
  return thrust::make_pair<T, T>(x1_grad, x2_grad);
}

}  // namespace activations
}  // namespace mega2
