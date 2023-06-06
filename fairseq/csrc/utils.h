#pragma once

#include <c10/util/llvmMathExtras.h>
#include <torch/torch.h>

#include <cstring>
#include <tuple>
#include <utility>

namespace py = pybind11;

namespace mega2 {
namespace utils {

constexpr int64_t kChunkSize = 16;

template <typename T>
T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

// https://github.com/pytorch/pytorch/blob/eb0971cfe9b05940978bed73d6e2b43aea49fc84/aten/src/ATen/native/cpu/utils.h#L93
template <typename T>
T CeilLog2(T x) {
  if (x <= 2) {
    return 1;
  }
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<T>(c10::llvm::findLastSet(static_cast<uint64_t>(x) - 1) +
                        1);
}

template <typename T>
std::pair<T, T> Fast2Sum(T a, T b) {
  const T s = a + b;
  const T z = s - a;
  const T t = b - z;
  return std::make_pair(s, t);
}

template <typename T>
std::pair<T, T> KahanAdd(T x, T sum, T c) {
  const T y = x - c;
  const T t = sum + y;
  c = t - sum - y;
  return std::make_pair(sum, c);
}

template <typename T>
T Cube(T x) {
  return x * x * x;
}

template <typename T>
std::tuple<int64_t, T, T> WelfordUpdate(int64_t m0, T m1, T m2, T x) {
  ++m0;
  const T coef = T(1) / static_cast<T>(m0);
  const T delta1 = x - m1;
  m1 += coef * delta1;
  const T delta2 = delta1 * (x - m1) - m2;
  m2 += coef * delta2;
  return std::make_tuple(m0, m1, m2);
}

template <typename T>
std::tuple<int64_t, T, T> WelfordCombine(int64_t a_m0, T a_m1, T a_m2,
                                         int64_t b_m0, T b_m1, T b_m2) {
  const int64_t m0 = a_m0 + b_m0;
  const T c1 = m0 == 0 ? T(0) : T(a_m0) / static_cast<T>(m0);
  const T c2 = m0 == 0 ? T(0) : T(b_m0) / static_cast<T>(m0);
  const T delta = b_m1 - a_m1;
  const T m1 = c1 * a_m1 + c2 * b_m1;
  const T m2 = c1 * a_m2 + c2 * b_m2 + (c1 * delta) * (c2 * delta);
  return std::make_tuple(m0, m1, m2);
}

}  // namespace utils
}  // namespace mega2
