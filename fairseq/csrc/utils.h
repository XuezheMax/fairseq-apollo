#pragma once

#include <torch/torch.h>

#include <cstring>
#include <tuple>
#include <utility>

namespace py = pybind11;

namespace mega2 {
namespace utils {

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

}  // namespace utils
}  // namespace mega2
