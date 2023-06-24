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
  return std::make_pair(t, c);
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

// TODO: Optimize by using Vec256.
template <typename T, typename T_ACC>
std::tuple<int64_t, T_ACC, T_ACC> RowwiseMoments(int64_t N, const T* X,
                                                 const bool* padding_mask) {
  const int64_t num_chunks = utils::DivUp(N, utils::kChunkSize);
  const int64_t depth = utils::CeilLog2(num_chunks);

  std::vector<int64_t> m0_stk(depth, 0);
  std::vector<T_ACC> m1_stk(depth, T_ACC(0));
  std::vector<T_ACC> m2_stk(depth, T_ACC(0));
  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t l = i * utils::kChunkSize;
    const int64_t r = std::min(l + utils::kChunkSize, N);
    for (int64_t j = l; j < r; ++j) {
      const T_ACC x = static_cast<T_ACC>(X[j]);
      const bool mask = padding_mask != nullptr && padding_mask[j];
      const auto [_, u, v] =
          utils::WelfordUpdate(m0_stk[0], m1_stk[0], m2_stk[0], x);
      m0_stk[0] += mask ? 0 : 1;
      m1_stk[0] = mask ? m1_stk[0] : u;
      m2_stk[0] = mask ? m2_stk[0] : v;
    }

    int64_t cnt = i + 1;
    for (int64_t j = 1; j < depth && (cnt & 1) == 0; ++j) {
      std::tie(m0_stk[j], m1_stk[j], m2_stk[j]) =
          utils::WelfordCombine(m0_stk[j], m1_stk[j], m2_stk[j], m0_stk[j - 1],
                                m1_stk[j - 1], m2_stk[j - 1]);
      m0_stk[j - 1] = 0;
      m1_stk[j - 1] = T_ACC(0);
      m2_stk[j - 1] = T_ACC(0);
      cnt >>= 1;
    }
  }
  for (int64_t i = 1; i < depth; ++i) {
    std::tie(m0_stk[0], m1_stk[0], m2_stk[0]) = utils::WelfordCombine(
        m0_stk[0], m1_stk[0], m2_stk[0], m0_stk[i], m1_stk[i], m2_stk[i]);
  }

  return std::make_tuple(m0_stk[0], m1_stk[0], m2_stk[0]);
}

}  // namespace utils
}  // namespace mega2
