#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <thrust/tuple.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_utils.cuh"
#include "ops/timestep_norm.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDAFwdSmallKernel(
    int64_t L, int64_t H, const T* X, const int64_t* prev_count,
    const T* prev_mean, const T* prev_var, const T* gamma, const T* beta,
    const bool* padding_mask, T_ACC eps, T* Y, int64_t* count, T* mean, T* var,
    T* cummean, T* cumrstd) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T* Y_ptr = Y + i * L * H;
  T* m1_ptr = mean + i * H;
  T* m2_ptr = var + i * H;
  T* cu_ptr = cummean + i * L * H;
  T* cr_ptr = cumrstd + i * L * H;

  int64_t m0 = prev_count[i];
  T_ACC m1 = static_cast<T_ACC>(prev_mean[i * H + k]);
  T_ACC m2 = static_cast<T_ACC>(prev_var[i * H + k]);

  // TODO: Improve this.
  for (int64_t j = 0; j < L; ++j) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const T_ACC w = static_cast<T_ACC>(gamma[k]);
    const T_ACC b = static_cast<T_ACC>(beta[k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const auto moments = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    Y_ptr[j * H + k] = mask ? T(0) : static_cast<T>((x - m1) * rstd * w + b);
    cu_ptr[j * H + k] = static_cast<T>(m1);
    cr_ptr[j * H + k] = static_cast<T>(rstd);
  }
  if (k == 0) {
    count[i] = m0;
  }
  m1_ptr[k] = static_cast<T>(m1);
  m2_ptr[k] = static_cast<T>(m2);
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDAFwdLargeKernel(
    int64_t L, int64_t H, const int64_t chunk_size, const T* X,
    const int64_t* prev_count, const T* prev_mean, const T* prev_var,
    const T* gamma, const T* beta, const bool* padding_mask, T_ACC eps, T* Y,
    int64_t* count, T* mean, T* var, T* cummean, T* cumrstd) {
  __shared__ int64_t
      m0_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m2_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (k >= H) {
    return;
  }

  const T* X_ptr = X + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T* Y_ptr = Y + i * L * H;
  T* m1_ptr = mean + i * H;
  T* m2_ptr = var + i * H;
  T* cu_ptr = cummean + i * L * H;
  T* cr_ptr = cumrstd + i * L * H;

  int64_t m0 = 0;
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);
  for (int64_t j = l; j < r; ++j) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const auto moments = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
  }

  m0_shared[threadIdx.y][threadIdx.x] = m0;
  m1_shared[threadIdx.y][threadIdx.x] = m1;
  m2_shared[threadIdx.y][threadIdx.x] = m2;
  __syncthreads();

  int64_t offset = 1;
  for (int64_t d = cuda_utils::kWarpSize >> 1; d > 0; d >>= 1) {
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      thrust::tie(m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
                  m2_shared[bi][threadIdx.x]) =
          cuda_utils::WelfordCombine(
              m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
              m2_shared[bi][threadIdx.x], m0_shared[ai][threadIdx.x],
              m1_shared[ai][threadIdx.x], m2_shared[ai][threadIdx.x]);
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    m0_shared[cuda_utils::kWarpSize - 1][threadIdx.x] = prev_count[i];
    m1_shared[cuda_utils::kWarpSize - 1][threadIdx.x] =
        static_cast<T_ACC>(prev_mean[i * H + k]);
    m2_shared[cuda_utils::kWarpSize - 1][threadIdx.x] =
        static_cast<T_ACC>(prev_var[i * H + k]);
  }
  __syncthreads();
  for (int64_t d = 1; d < cuda_utils::kWarpSize; d <<= 1) {
    offset >>= 1;
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      const int64_t am0 = m0_shared[ai][threadIdx.x];
      const T_ACC am1 = m1_shared[ai][threadIdx.x];
      const T_ACC am2 = m2_shared[ai][threadIdx.x];
      m0_shared[ai][threadIdx.x] = m0_shared[bi][threadIdx.x];
      m1_shared[ai][threadIdx.x] = m1_shared[bi][threadIdx.x];
      m2_shared[ai][threadIdx.x] = m2_shared[bi][threadIdx.x];
      thrust::tie(m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
                  m2_shared[bi][threadIdx.x]) =
          cuda_utils::WelfordCombine(m0_shared[bi][threadIdx.x],
                                     m1_shared[bi][threadIdx.x],
                                     m2_shared[bi][threadIdx.x], am0, am1, am2);
    }
    __syncthreads();
  }

  m0 = m0_shared[threadIdx.y][threadIdx.x];
  m1 = m1_shared[threadIdx.y][threadIdx.x];
  m2 = m2_shared[threadIdx.y][threadIdx.x];
  for (int64_t j = l; j < r; ++j) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const T_ACC w = static_cast<T_ACC>(gamma[k]);
    const T_ACC b = static_cast<T_ACC>(beta[k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const auto moments = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    Y_ptr[j * H + k] = mask ? T(0) : static_cast<T>((x - m1) * rstd * w + b);
    cu_ptr[j * H + k] = static_cast<T>(m1);
    cr_ptr[j * H + k] = static_cast<T>(rstd);
  }
  if (threadIdx.y == cuda_utils::kWarpSize - 1) {
    if (k == 0) {
      count[i] = m0;
    }
    m1_ptr[k] = static_cast<T>(m1);
    m2_ptr[k] = static_cast<T>(m2);
  }
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDABwdKernel(
    int64_t L, int64_t H, const T* Y_grad, const T* mean_grad,
    const T* var_grad, const T* X, const T* prev_mean, const int64_t* count,
    const T* cummean, const T* cumrstd, const T* gamma,
    const bool* padding_mask, T* X_grad, T* prev_mean_grad, T* prev_var_grad,
    T_ACC* gamma_grad, T_ACC* beta_grad) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= H) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * H;
  const T* mean_grad_ptr = mean_grad + i * H;
  const T* var_grad_ptr = var_grad + i * H;
  const T* X_ptr = X + i * L * H;
  const T* mean_ptr = cummean + i * L * H;
  const T* rstd_ptr = cumrstd + i * L * H;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;

  T* X_grad_ptr = X_grad + i * L * H;
  T* m1_grad_ptr = prev_mean_grad + i * H;
  T* m2_grad_ptr = prev_var_grad + i * H;
  T_ACC* w_grad_ptr = gamma_grad + i * H;
  T_ACC* b_grad_ptr = beta_grad + i * H;

  int64_t m0 = count[i];
  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[k]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[k]);

  w_grad_ptr[k] = T_ACC(0);
  b_grad_ptr[k] = T_ACC(0);

  // TODO: Improve this.
  for (int64_t j = L - 1; j >= 0; --j) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
    const T_ACC prev_u = static_cast<T_ACC>(j == 0 ? prev_mean[i * H + k]
                                                   : mean_ptr[(j - 1) * H + k]);
    const T_ACC u = static_cast<T_ACC>(mean_ptr[j * H + k]);
    const T_ACC r = static_cast<T_ACC>(rstd_ptr[j * H + k]);
    const T_ACC w = static_cast<T_ACC>(gamma[k]);
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
    const T_ACC dy_rstd = dy * r;
    const T_ACC delta = x - u;
    const T_ACC dv =
        v_grad - (T_ACC(0.5) * dy * w * delta * cuda_utils::Cube(r));
    const T_ACC du = u_grad - (w * dy_rstd + coef * dv * (x - prev_u));
    const T_ACC dx = w * dy_rstd + dv * coef * (delta + x - prev_u) + coef * du;
    X_grad_ptr[j * H + k] = mask ? T(0) : static_cast<T>(dx);
    u_grad = mask ? u_grad : (T_ACC(1) - coef) * du - coef * delta * dv;
    v_grad = mask ? v_grad : (T_ACC(1) - coef) * dv;
    w_grad_ptr[k] += mask ? T_ACC(0) : dy_rstd * delta;
    b_grad_ptr[k] += mask ? T_ACC(0) : dy;
    m0 -= mask ? 0 : 1;
  }

  m1_grad_ptr[k] = static_cast<T>(u_grad);
  m2_grad_ptr[k] = static_cast<T>(v_grad);
}

template <typename T, typename T_ACC>
__global__ void GammaBetaCUDABwdKernel(int64_t B, int64_t H,
                                       const T_ACC* dw_internal,
                                       const T_ACC* db_internal, T* dw, T* db) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    w_grad += dw_internal[i * H + j];
    b_grad += db_internal[i * H + j];
  }
  dw[j] = static_cast<T>(w_grad);
  db[j] = static_cast<T>(b_grad);
}

template <typename T, typename T_ACC>
__global__ void RowwiseMomentsKernel(int64_t N, const T* X, T_ACC* mean,
                                     T_ACC* var) {
  __shared__ int64_t m0_shared[cuda_utils::kWarpSize];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize];
  __shared__ T_ACC m2_shared[cuda_utils::kWarpSize];

  const int64_t i = blockIdx.x;
  const T* X_ptr = X + i * N;

  int64_t m0 = 0;
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);
  for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
    thrust::tie(m0, m1, m2) = cuda_utils::WelfordUpdate(m0, m1, m2, x);
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    thrust::tie(m0, m1, m2) = cuda_utils::WarpReduceMoments(m0, m1, m2);
  } else {
    thrust::tie(m0, m1, m2) = cuda_utils::BlockReduceMoments(
        m0, m1, m2, m0_shared, m1_shared, m2_shared);
  }
  if (threadIdx.x == 0) {
    mean[i] = m1;
    var[i] = m2;
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsSmallKernel(
    int64_t L, int64_t num_groups, const int64_t* prev_count,
    const T* prev_mean, const T* prev_var, const T_ACC* group_mean,
    const T_ACC* group_var, const bool* padding_mask, T_ACC eps, int64_t* count,
    T* mean, T* var, T_ACC* cummean, T_ACC* cumrstd) {
  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  int64_t m0 = prev_count[b];
  T_ACC m1 = static_cast<T_ACC>(prev_mean[b * num_groups + g]);
  T_ACC m2 = static_cast<T_ACC>(prev_var[b * num_groups + g]);
  for (int64_t i = 0; i < L; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const auto moments = cuda_utils::WelfordCombine(m0, m1, m2, 1, gu, gv);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    cummean_ptr[i * num_groups + g] = m1;
    cumrstd_ptr[i * num_groups + g] = rstd;
  }

  if (g == 0) {
    count[b] = m0;
  }
  mean[b * num_groups + g] = static_cast<T>(m1);
  var[b * num_groups + g] = static_cast<T>(m2);
}

template <typename T, typename T_ACC>
__global__ void ColwiseCumMomentsLargeKernel(
    int64_t L, int64_t num_groups, int64_t chunk_size,
    const int64_t* prev_count, const T* prev_mean, const T* prev_var,
    const T_ACC* group_mean, const T_ACC* group_var, const bool* padding_mask,
    T_ACC eps, int64_t* count, T* mean, T* var, T_ACC* cummean,
    T_ACC* cumrstd) {
  __shared__ int64_t
      m0_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m2_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t b = blockIdx.y;
  const int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = threadIdx.y * chunk_size;
  const int64_t r = min(l + chunk_size, L);
  if (g >= num_groups) {
    return;
  }

  const T_ACC* gu_ptr = group_mean + b * L * num_groups;
  const T_ACC* gv_ptr = group_var + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  T_ACC* cummean_ptr = cummean + b * L * num_groups;
  T_ACC* cumrstd_ptr = cumrstd + b * L * num_groups;

  int64_t m0 = 0;
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const auto moments = cuda_utils::WelfordCombine(m0, m1, m2, 1, gu, gv);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
  }

  m0_shared[threadIdx.y][threadIdx.x] = m0;
  m1_shared[threadIdx.y][threadIdx.x] = m1;
  m2_shared[threadIdx.y][threadIdx.x] = m2;
  __syncthreads();

  int64_t offset = 1;
  for (int64_t d = cuda_utils::kWarpSize >> 1; d > 0; d >>= 1) {
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      thrust::tie(m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
                  m2_shared[bi][threadIdx.x]) =
          cuda_utils::WelfordCombine(
              m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
              m2_shared[bi][threadIdx.x], m0_shared[ai][threadIdx.x],
              m1_shared[ai][threadIdx.x], m2_shared[ai][threadIdx.x]);
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    m0_shared[cuda_utils::kWarpSize - 1][threadIdx.x] = prev_count[b];
    m1_shared[cuda_utils::kWarpSize - 1][threadIdx.x] =
        static_cast<T_ACC>(prev_mean[b * num_groups + g]);
    m2_shared[cuda_utils::kWarpSize - 1][threadIdx.x] =
        static_cast<T_ACC>(prev_var[b * num_groups + g]);
  }
  __syncthreads();
  for (int64_t d = 1; d < cuda_utils::kWarpSize; d <<= 1) {
    offset >>= 1;
    if (threadIdx.y < d) {
      const int64_t ai = offset * (2 * threadIdx.y + 1) - 1;
      const int64_t bi = offset * (2 * threadIdx.y + 2) - 1;
      const int64_t am0 = m0_shared[ai][threadIdx.x];
      const T_ACC am1 = m1_shared[ai][threadIdx.x];
      const T_ACC am2 = m2_shared[ai][threadIdx.x];
      m0_shared[ai][threadIdx.x] = m0_shared[bi][threadIdx.x];
      m1_shared[ai][threadIdx.x] = m1_shared[bi][threadIdx.x];
      m2_shared[ai][threadIdx.x] = m2_shared[bi][threadIdx.x];
      thrust::tie(m0_shared[bi][threadIdx.x], m1_shared[bi][threadIdx.x],
                  m2_shared[bi][threadIdx.x]) =
          cuda_utils::WelfordCombine(m0_shared[bi][threadIdx.x],
                                     m1_shared[bi][threadIdx.x],
                                     m2_shared[bi][threadIdx.x], am0, am1, am2);
    }
    __syncthreads();
  }

  m0 = m0_shared[threadIdx.y][threadIdx.x];
  m1 = m1_shared[threadIdx.y][threadIdx.x];
  m2 = m2_shared[threadIdx.y][threadIdx.x];
  for (int64_t i = l; i < r; ++i) {
    const T_ACC gu = gu_ptr[i * num_groups + g];
    const T_ACC gv = gv_ptr[i * num_groups + g];
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const auto moments = cuda_utils::WelfordCombine(m0, m1, m2, 1, gu, gv);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    cummean_ptr[i * num_groups + g] = m1;
    cumrstd_ptr[i * num_groups + g] = rstd;
  }
  if (threadIdx.y == cuda_utils::kWarpSize - 1) {
    if (g == 0) {
      count[b] = m0;
    }
    mean[b * num_groups + g] = static_cast<T>(m1);
    var[b * num_groups + g] = static_cast<T>(m2);
  }
}

template <typename T, typename T_ACC>
__global__ void GroupTimestepNormCUDAFwdKernel(int64_t L, int64_t H,
                                               int64_t num_groups, const T* X,
                                               const T_ACC* cummean,
                                               const T_ACC* cumrstd,
                                               const T* gamma, const T* beta,
                                               const bool* padding_mask, T* Y) {
  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t l = blockIdx.x;

  const T* X_ptr = X + (b * L + l) * H;
  const T_ACC* cummean_ptr = cummean + (b * L + l) * num_groups;
  const T_ACC* cumrstd_ptr = cumrstd + (b * L + l) * num_groups;
  const bool mask = padding_mask != nullptr && padding_mask[b * L + l];
  T* Y_ptr = Y + (b * L + l) * H;

  if (mask) {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      Y_ptr[i] = T(0);
    }
  } else {
    for (int64_t i = threadIdx.x; i < H; i += blockDim.x) {
      const int64_t g = i / D;
      const T_ACC x = static_cast<T_ACC>(X_ptr[i]);
      const T_ACC u = cummean_ptr[g];
      const T_ACC r = cumrstd_ptr[g];
      const T_ACC w = static_cast<T_ACC>(gamma[i]);
      const T_ACC b = static_cast<T_ACC>(beta[i]);
      Y_ptr[i] = static_cast<T>((x - u) * r * w + b);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void RowwiseInternalGradientsKernel(int64_t H, int64_t num_groups,
                                               const T* Y_grad, const T* X,
                                               const T_ACC* mean,
                                               const T* gamma, T_ACC* ds,
                                               T_ACC* db) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize];

  const int64_t D = H / num_groups;
  const int64_t i = blockIdx.x;
  const T* Y_grad_ptr = Y_grad + i * D;
  const T* X_ptr = X + i * D;
  const T* gamma_ptr = gamma + (i % num_groups) * D;
  const T_ACC u = mean[i];

  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t j = threadIdx.x; j < D; j += blockDim.x) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j]);
    const T_ACC w = static_cast<T_ACC>(gamma_ptr[j]);
    sum1 += dy * (x - u) * w;
    sum2 += dy * w;
  }
  if (blockDim.x <= cuda_utils::kWarpSize) {
    sum1 = at::native::cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = at::native::cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    sum1 = at::native::cuda_utils::BlockReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = at::native::cuda_utils::BlockReduceSum<T_ACC>(sum2, db_shared);
  }
  if (threadIdx.x == 0) {
    ds[i] = sum1;
    db[i] = sum2;
  }
}

template <typename T, typename T_ACC>
__global__ void GroupTimestepNormCUDABwdKernel(
    int64_t L, int64_t H, int64_t num_groups, const T* Y_grad,
    const T* mean_grad, const T* var_grad, const T* X, const T* prev_mean,
    const int64_t* count, const T_ACC* group_mean, const T_ACC* cummean,
    const T_ACC* cumrstd, const T* gamma, const bool* padding_mask,
    const T_ACC* ds, const T_ACC* db, T* X_grad, T* prev_mean_grad,
    T* prev_var_grad, T_ACC* gamma_grad, T_ACC* beta_grad) {
  const int64_t D = H / num_groups;
  const int64_t b = blockIdx.y;
  const int64_t h = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t g = h / D;
  const int64_t d = h % D;
  const T_ACC cg = T_ACC(1) / T_ACC(D);
  if (h >= H) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + b * L * H;
  const T* mean_grad_ptr = mean_grad + b * num_groups;
  const T* var_grad_ptr = var_grad + b * num_groups;
  const T* X_ptr = X + b * L * H;
  const T_ACC* group_mean_ptr = group_mean + b * L * num_groups;
  const T_ACC* mean_ptr = cummean + b * L * num_groups;
  const T_ACC* rstd_ptr = cumrstd + b * L * num_groups;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + b * L;
  const T_ACC* ds_ptr = ds + b * L * num_groups;
  const T_ACC* db_ptr = db + b * L * num_groups;

  T* X_grad_ptr = X_grad + b * L * H;
  T* m1_grad_ptr = prev_mean_grad + b * num_groups;
  T* m2_grad_ptr = prev_var_grad + b * num_groups;
  T_ACC* w_grad_ptr = gamma_grad + b * H;
  T_ACC* b_grad_ptr = beta_grad + b * H;

  int64_t m0 = count[b];
  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[g]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[g]);

  w_grad_ptr[h] = T_ACC(0);
  b_grad_ptr[h] = T_ACC(0);

  // TODO: Improve this.
  for (int64_t i = L - 1; i >= 0; --i) {
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[i * H + h]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[i * H + h]);
    const T_ACC prev_u = i == 0
                             ? static_cast<T_ACC>(prev_mean[b * num_groups + g])
                             : mean_ptr[(i - 1) * num_groups + g];
    const T_ACC ux = group_mean_ptr[i * num_groups + g];
    const T_ACC u = mean_ptr[i * num_groups + g];
    const T_ACC r = rstd_ptr[i * num_groups + g];
    const T_ACC w = static_cast<T_ACC>(gamma[h]);
    const bool mask = mask_ptr != nullptr && mask_ptr[i];
    const T_ACC c1 = static_cast<T_ACC>(m0 - 1) / static_cast<T_ACC>(m0);
    const T_ACC c2 = T_ACC(1) / static_cast<T_ACC>(m0);

    const T_ACC du = u_grad - r * db_ptr[i * num_groups + g];
    const T_ACC dv =
        v_grad - T_ACC(0.5) * cuda_utils::Cube(r) * ds_ptr[i * num_groups + g];
    const T_ACC dux = c2 * du + T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    const T_ACC dvx = c2 * dv;
    const T_ACC dx = dy * r * w + dux * cg + T_ACC(2) * dvx * cg * (x - ux);
    X_grad_ptr[i * H + h] = mask ? T(0) : static_cast<T>(dx);
    u_grad = mask ? u_grad : c1 * du - T_ACC(2) * c1 * c2 * dv * (ux - prev_u);
    v_grad = mask ? v_grad : c1 * dv;
    w_grad_ptr[h] += mask ? T_ACC(0) : dy * (x - u) * r;
    b_grad_ptr[h] += mask ? T_ACC(0) : dy;
    m0 -= mask ? 0 : 1;
  }

  if (d == 0) {
    m1_grad_ptr[g] = static_cast<T>(u_grad);
    m2_grad_ptr[g] = static_cast<T>(v_grad);
  }
}

template <typename T>
void TimestepNormCUDAFwdImpl(
    const torch::Tensor& X, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, double eps, torch::Tensor& Y,
    torch::Tensor& count, torch::Tensor& mean, torch::Tensor& var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T* cummean_data = cummean.data_ptr<T>();
  T* cumrstd_data = cumrstd.data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
    TimestepNormCUDAFwdSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, H, X_data, prev_count_data, prev_mean_data, prev_var_data,
            gamma_data, beta_data, padding_mask_data, static_cast<T_ACC>(eps),
            Y_data, count_data, mean_data, var_data, cummean_data,
            cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(H, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize);
    TimestepNormCUDAFwdLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, H, chunk_size, X_data, prev_count_data,
                          prev_mean_data, prev_var_data, gamma_data, beta_data,
                          padding_mask_data, static_cast<T_ACC>(eps), Y_data,
                          count_data, mean_data, var_data, cummean_data,
                          cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T>
void TimestepNormCUDABwdImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
    const torch::Tensor& gamma, const torch::Tensor& padding_mask,
    torch::Tensor& X_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);

  torch::Tensor w_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor b_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T* cummean_data = cummean.data_ptr<T>();
  const T* cumrstd_data = cumrstd.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* w_grad_data = w_grad.data_ptr<T_ACC>();
  T_ACC* b_grad_data = b_grad.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  TimestepNormCUDABwdKernel<T, T_ACC>
      <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, Y_grad_data, mean_grad_data, var_grad_data, X_data,
          prev_mean_data, count_data, cummean_data, cumrstd_data, gamma_data,
          padding_mask_data, X_grad_data, prev_mean_grad_data,
          prev_var_grad_data, w_grad_data, b_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, w_grad_data, b_grad_data, gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupTimestepNormCUDAFwdImpl(
    const torch::Tensor& X, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, int64_t num_groups, double eps,
    torch::Tensor& Y, torch::Tensor& count, torch::Tensor& mean,
    torch::Tensor& var, torch::Tensor& group_mean, torch::Tensor& group_var,
    torch::Tensor& cummean, torch::Tensor& cumrstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;

  const T* X_data = X.data_ptr<T>();
  const int64_t* prev_count_data = prev_count.data_ptr<int64_t>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const T* prev_var_data = prev_var.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T* mean_data = mean.data_ptr<T>();
  T* var_data = var.data_ptr<T>();
  T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  T_ACC* group_var_data = group_var.data_ptr<T_ACC>();
  T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t num_threads = (D < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    RowwiseMomentsKernel<T, T_ACC>
        <<<B * L * num_groups, num_threads, 0, cuda_stream>>>(
            D, X_data, group_mean_data, group_var_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t num_threads = (num_groups < cuda_utils::kCUDANumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDANumThreads);
    const int64_t M = utils::DivUp(num_groups, num_threads);
    ColwiseCumMomentsSmallKernel<T, T_ACC>
        <<<dim3(M, B), num_threads, 0, cuda_stream>>>(
            L, num_groups, prev_count_data, prev_mean_data, prev_var_data,
            group_mean_data, group_var_data, padding_mask_data,
            static_cast<T_ACC>(eps), count_data, mean_data, var_data,
            cummean_data, cumrstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(num_groups, cuda_utils::kWarpSize);
    const int64_t chunk_size = utils::DivUp(L, cuda_utils::kWarpSize);
    ColwiseCumMomentsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, num_groups, chunk_size, prev_count_data,
                          prev_mean_data, prev_var_data, group_mean_data,
                          group_var_data, padding_mask_data,
                          static_cast<T_ACC>(eps), count_data, mean_data,
                          var_data, cummean_data, cumrstd_data);
  }
  GroupTimestepNormCUDAFwdKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, num_groups, X_data, cummean_data, cumrstd_data, gamma_data,
          beta_data, padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void GroupTimestepNormCUDABwdImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& group_mean, const torch::Tensor& cummean,
    const torch::Tensor& cumrstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, int64_t num_groups,
    torch::Tensor& X_grad, torch::Tensor& prev_mean_grad,
    torch::Tensor& prev_var_grad, torch::Tensor& gamma_grad,
    torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t D = H / num_groups;

  torch::Tensor ds =
      torch::empty({B, L, num_groups},
                   X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db =
      torch::empty({B, L, num_groups},
                   X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor w_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor b_grad = torch::empty(
      {B, H}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* prev_mean_data = prev_mean.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* group_mean_data = group_mean.data_ptr<T_ACC>();
  const T_ACC* cummean_data = cummean.data_ptr<T_ACC>();
  const T_ACC* cumrstd_data = cumrstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* prev_mean_grad_data = prev_mean_grad.data_ptr<T>();
  T* prev_var_grad_data = prev_var_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  T_ACC* w_grad_data = w_grad.data_ptr<T_ACC>();
  T_ACC* b_grad_data = b_grad.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  {
    const int64_t num_threads = (D < cuda_utils::kCUDABlockReduceNumThreads
                                     ? cuda_utils::kWarpSize
                                     : cuda_utils::kCUDABlockReduceNumThreads);
    RowwiseInternalGradientsKernel<T, T_ACC>
        <<<B * L * num_groups, num_threads, 0, cuda_stream>>>(
            H, num_groups, Y_grad_data, X_data, cummean_data, gamma_data,
            ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t M = utils::DivUp(H, cuda_utils::kCUDANumThreads);
  GroupTimestepNormCUDABwdKernel<T, T_ACC>
      <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, H, num_groups, Y_grad_data, mean_grad_data, var_grad_data, X_data,
          prev_mean_data, count_data, group_mean_data, cummean_data,
          cumrstd_data, gamma_data, padding_mask_data, ds_data, db_data,
          X_grad_data, prev_mean_grad_data, prev_var_grad_data, w_grad_data,
          b_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, H, w_grad_data, b_grad_data, gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
TimestepNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& prev_count,
                    const torch::Tensor& prev_mean,
                    const torch::Tensor& prev_var, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps) {
  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t N = X.size(2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, N},
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, N},
      prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "TimestepNormCUDAFwd", [&]() {
        TimestepNormCUDAFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
            mean, var, cummean, cumrstd);
      });
  return std::make_tuple(Y, count, mean, var, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                    const torch::Tensor& var_grad, const torch::Tensor& X,
                    const torch::Tensor& prev_mean, const torch::Tensor& count,
                    const torch::Tensor& cummean, const torch::Tensor& cumrstd,
                    const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& padding_mask) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad,
      mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "TimestepNormCUDABwd", [&]() {
        TimestepNormCUDABwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(X.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(cummean.expect_contiguous()), *(cumrstd.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
            prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(X_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
GroupTimestepNormCUDAFwd(const torch::Tensor& X,
                         const torch::Tensor& prev_count,
                         const torch::Tensor& prev_mean,
                         const torch::Tensor& prev_var,
                         const torch::Tensor& gamma, const torch::Tensor& beta,
                         const c10::optional<torch::Tensor>& padding_mask,
                         int64_t num_groups, double eps) {
  const int64_t B = X.size(0);
  const int64_t L = X.size(1);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count = torch::empty_like(
      prev_count,
      prev_count.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor mean = torch::empty_like(
      prev_mean,
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor var = torch::empty_like(
      prev_var, prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(X.scalar_type(), true);
  torch::Tensor group_mean = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor group_var = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cummean = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumrstd = torch::empty(
      {B, L, num_groups},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupTimestepNormCUDAFwd",
      [&]() {
        GroupTimestepNormCUDAFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups, eps,
            Y, count, mean, var, group_mean, group_var, cummean, cumrstd);
      });

  return std::make_tuple(Y, count, mean, var, group_mean, cummean, cumrstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
GroupTimestepNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& prev_mean, const torch::Tensor& count,
    const torch::Tensor& group_mean, const torch::Tensor& cummean,
    const torch::Tensor& cumrstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, int64_t num_groups) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_mean_grad = torch::empty_like(
      mean_grad,
      mean_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor prev_var_grad = torch::empty_like(
      var_grad, var_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "GroupTimestepNormCUDABwd",
      [&]() {
        GroupTimestepNormCUDABwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(mean_grad.expect_contiguous()),
            *(var_grad.expect_contiguous()), *(X.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(count.expect_contiguous()),
            *(group_mean.expect_contiguous()), *(cummean.expect_contiguous()),
            *(cumrstd.expect_contiguous()), *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), num_groups,
            X_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(X_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

}  // namespace ops
}  // namespace mega2
