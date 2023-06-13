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

#include <ATen/native/cuda/Loops.cuh>
#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_utils.cuh"
#include "ops/sequence_norm.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void ColwiseMomentsSmallKernel(int64_t L, int64_t N, const T* X,
                                          const bool* padding_mask, T_ACC eps,
                                          int64_t* count, T_ACC* mean,
                                          T_ACC* rstd) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* X_ptr = X + i * L * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* mean_ptr = mean + i * N;
  T_ACC* rstd_ptr = rstd + i * N;

  int64_t m0 = 0;
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);
  for (int64_t j = 0; j < L; ++j) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    const auto moments = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
  }
  if (k == 0) {
    count[i] = m0;
  }
  mean_ptr[k] = m1;
  rstd_ptr[k] = c10::cuda::compat::rsqrt(m2 + eps);
}

template <typename T, typename T_ACC>
__global__ void ColwiseMomentsLargeKernel(int64_t L, int64_t N, const T* X,
                                          const bool* padding_mask, T_ACC eps,
                                          int64_t* count, T_ACC* mean,
                                          T_ACC* rstd) {
  __shared__ int64_t
      m0_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m1_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC m2_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* X_ptr = X + i * L * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* mean_ptr = mean + i * N;
  T_ACC* rstd_ptr = rstd + i * N;

  int64_t m0 = 0;
  T_ACC m1 = T_ACC(0);
  T_ACC m2 = T_ACC(0);
  for (int64_t j = threadIdx.y; j < L; j += blockDim.y) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    const auto moments = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    m0 = mask ? m0 : thrust::get<0>(moments);
    m1 = mask ? m1 : thrust::get<1>(moments);
    m2 = mask ? m2 : thrust::get<2>(moments);
  }
  m0_shared[threadIdx.y][threadIdx.x] = m0;
  m1_shared[threadIdx.y][threadIdx.x] = m1;
  m2_shared[threadIdx.y][threadIdx.x] = m2;
  __syncthreads();

  for (int64_t offset = cuda_utils::kWarpSize >> 1; offset >= 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      thrust::tie(m0_shared[threadIdx.y][threadIdx.x],
                  m1_shared[threadIdx.y][threadIdx.x],
                  m2_shared[threadIdx.y][threadIdx.x]) =
          cuda_utils::WelfordCombine(
              m0_shared[threadIdx.y][threadIdx.x],
              m1_shared[threadIdx.y][threadIdx.x],
              m2_shared[threadIdx.y][threadIdx.x],
              m0_shared[threadIdx.y + offset][threadIdx.x],
              m1_shared[threadIdx.y + offset][threadIdx.x],
              m2_shared[threadIdx.y + offset][threadIdx.x]);
    }
    __syncthreads();
  }

  if (threadIdx.y == 0) {
    if (k == 0) {
      count[i] = m0_shared[0][0];
    }
    mean_ptr[k] = m1_shared[0][threadIdx.x];
    rstd_ptr[k] = c10::cuda::compat::rsqrt(m2_shared[0][threadIdx.x] + eps);
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDAFwdKernel(int64_t L, int64_t N, const T* X,
                                          const T_ACC* mean, const T_ACC* rstd,
                                          const T* gamma, const T* beta,
                                          const bool* padding_mask, T* Y) {
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;
  const T* X_ptr = X + (i * L + j) * N;
  const T_ACC* mean_ptr = mean + i * N;
  const T_ACC* rstd_ptr = rstd + i * N;
  T* Y_ptr = Y + (i * L + j) * N;
  const bool mask = padding_mask && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < N; k += blockDim.x) {
      Y_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < N; k += blockDim.x) {
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC b = static_cast<T_ACC>(beta[k]);
      Y_ptr[k] = static_cast<T>((x - mean_ptr[k]) * rstd_ptr[k] * w + b);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsSmallKernel(int64_t L, int64_t N,
                                                    const T* Y_grad, const T* X,
                                                    const T_ACC* mean,
                                                    const bool* padding_mask,
                                                    T_ACC* ds, T_ACC* db) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * N;
  const T* X_ptr = X + i * L * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* ds_ptr = ds + i * N;
  T_ACC* db_ptr = db + i * N;

  const T_ACC u = mean[i * N + k];
  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t j = 0; j < L; ++j) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * N + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    sum1 += mask ? T_ACC(0) : dy * (x - u);
    sum2 += mask ? T_ACC(0) : dy;
  }
  ds_ptr[k] = sum1;
  db_ptr[k] = sum2;
}

template <typename T, typename T_ACC>
__global__ void ColwiseInternalGradientsLargeKernel(int64_t L, int64_t N,
                                                    const T* Y_grad, const T* X,
                                                    const T_ACC* mean,
                                                    const bool* padding_mask,
                                                    T_ACC* ds, T_ACC* db) {
  __shared__ T_ACC ds_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  __shared__ T_ACC db_shared[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];

  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * N;
  const T* X_ptr = X + i * L * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T_ACC* ds_ptr = ds + i * N;
  T_ACC* db_ptr = db + i * N;

  const T_ACC u = mean[i * N + k];
  T_ACC sum1 = T_ACC(0);
  T_ACC sum2 = T_ACC(0);
  for (int64_t j = threadIdx.y; j < L; j += blockDim.y) {
    const bool mask = mask_ptr != nullptr && mask_ptr[j];
    const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * N + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    sum1 += mask ? T_ACC(0) : dy * (x - u);
    sum2 += mask ? T_ACC(0) : dy;
  }
  ds_shared[threadIdx.y][threadIdx.x] = sum1;
  db_shared[threadIdx.y][threadIdx.x] = sum2;
  __syncthreads();

  for (int64_t offset = cuda_utils::kWarpSize >> 1; offset >= 1; offset >>= 1) {
    if (threadIdx.y < offset) {
      ds_shared[threadIdx.y][threadIdx.x] +=
          ds_shared[threadIdx.y + offset][threadIdx.x];
      db_shared[threadIdx.y][threadIdx.x] +=
          db_shared[threadIdx.y + offset][threadIdx.x];
    }
    __syncthreads();
  }

  if (threadIdx.y == 0) {
    ds_ptr[k] = ds_shared[0][threadIdx.x];
    db_ptr[k] = db_shared[0][threadIdx.x];
  }
}

template <typename T, typename T_ACC>
__global__ void SequenceNormCUDABwdKernel(
    int64_t L, int64_t N, const T* Y_grad, const T* X, const int64_t* count,
    const T_ACC* mean, const T_ACC* rstd, const T* gamma,
    const bool* padding_mask, const T_ACC* ds, const T_ACC* db, T* X_grad) {
  const int64_t i = blockIdx.y;
  const int64_t j = blockIdx.x;
  const T* Y_grad_ptr = Y_grad + (i * L + j) * N;
  const T* X_ptr = X + (i * L + j) * N;
  const T_ACC* mean_ptr = mean + i * N;
  const T_ACC* rstd_ptr = rstd + i * N;
  const T_ACC* ds_ptr = ds + i * N;
  const T_ACC* db_ptr = db + i * N;
  T* X_grad_ptr = X_grad + (i * L + j) * N;

  // const int64_t cnt = count[i];
  const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count[i]);
  const bool mask = padding_mask && padding_mask[i * L + j];
  if (mask) {
    for (int64_t k = threadIdx.x; k < N; k += blockDim.x) {
      X_grad_ptr[k] = T(0);
    }
  } else {
    for (int64_t k = threadIdx.x; k < N; k += blockDim.x) {
      const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[k]);
      const T_ACC x = static_cast<T_ACC>(X_ptr[k]);
      const T_ACC u = mean_ptr[k];
      const T_ACC r = rstd_ptr[k];
      const T_ACC w = static_cast<T_ACC>(gamma[k]);
      const T_ACC dv = -T_ACC(0.5) * cuda_utils::Cube(r) * w * ds_ptr[k];
      const T_ACC du = -r * w * db_ptr[k];
      const T_ACC dx = r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
      X_grad_ptr[k] = static_cast<T>(dx);
    }
  }
}

template <typename T, typename T_ACC>
__global__ void GammaBetaCUDABwdKernel(int64_t B, int64_t N, const T_ACC* rstd,
                                       const T_ACC* ds, const T_ACC* db,
                                       T* gamma_grad, T* beta_grad) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    const T_ACC r = rstd[i * N + j];
    w_grad += ds[i * N + j] * r;
    b_grad += db[i * N + j];
  }
  gamma_grad[j] = static_cast<T>(w_grad);
  beta_grad[j] = static_cast<T>(b_grad);
}

template <typename T>
void SequenceNormCUDAFwdImpl(const torch::Tensor& X, const torch::Tensor& gamma,
                             const torch::Tensor& beta,
                             const torch::Tensor& padding_mask, double eps,
                             torch::Tensor& Y, torch::Tensor& count,
                             torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t N = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(N, cuda_utils::kCUDANumThreads);
    ColwiseMomentsSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, N, X_data, padding_mask_data, static_cast<T_ACC>(eps),
            count_data, mean_data, rstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(N, cuda_utils::kWarpSize);
    ColwiseMomentsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, N, X_data, padding_mask_data,
                          static_cast<T_ACC>(eps), count_data, mean_data,
                          rstd_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  SequenceNormCUDAFwdKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, N, X_data, mean_data, rstd_data, gamma_data, beta_data,
          padding_mask_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void SequenceNormCUDABwdImpl(const torch::Tensor& Y_grad,
                             const torch::Tensor& X, const torch::Tensor& count,
                             const torch::Tensor& mean,
                             const torch::Tensor& rstd,
                             const torch::Tensor& gamma,
                             const torch::Tensor& padding_mask,
                             torch::Tensor& X_grad, torch::Tensor& gamma_grad,
                             torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t N = X.size(2);

  torch::Tensor ds = torch::empty(
      {B, N}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor db = torch::empty(
      {B, N}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T_ACC* mean_data = mean.data_ptr<T_ACC>();
  const T_ACC* rstd_data = rstd.data_ptr<T_ACC>();
  const T* gamma_data = gamma.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* X_grad_data = X_grad.data_ptr<T>();
  T* gamma_grad_data = gamma_grad.data_ptr<T>();
  T* beta_grad_data = beta_grad.data_ptr<T>();
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  if (L < cuda_utils::kColwiseThreshold) {
    const int64_t M = utils::DivUp(N, cuda_utils::kCUDANumThreads);
    ColwiseInternalGradientsSmallKernel<T, T_ACC>
        <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
            L, N, Y_grad_data, X_data, mean_data, padding_mask_data, ds_data,
            db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int64_t M = utils::DivUp(N, cuda_utils::kWarpSize);
    ColwiseInternalGradientsLargeKernel<T, T_ACC>
        <<<dim3(M, B), dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize), 0,
           cuda_stream>>>(L, N, Y_grad_data, X_data, mean_data,
                          padding_mask_data, ds_data, db_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  SequenceNormCUDABwdKernel<T, T_ACC>
      <<<dim3(L, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, N, Y_grad_data, X_data, count_data, mean_data, rstd_data,
          gamma_data, padding_mask_data, ds_data, db_data, X_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t M = utils::DivUp(N, cuda_utils::kCUDANumThreads);
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, N, rstd_data, ds_data, db_data, gamma_grad_data, beta_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCUDAFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                    const torch::Tensor& beta,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps) {
  const int64_t B = X.size(0);
  const int64_t N = X.size(2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count =
      torch::empty({B}, X.options()
                            .dtype(torch::kInt64)
                            .memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toAccumulateType(X.scalar_type(), true);
  torch::Tensor mean = torch::empty(
      {B, N},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty(
      {B, N},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCUDAFwd", [&]() {
        SequenceNormCUDAFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(gamma.expect_contiguous()),
            *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
            mean, rstd);
      });

  return std::make_tuple(Y, count, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCUDAFwd", [&]() {
        SequenceNormCUDABwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
            *(count.expect_contiguous()), *(mean.expect_contiguous()),
            *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
            gamma_grad, beta_grad);
      });

  return std::make_tuple(X_grad, gamma_grad, beta_grad);
}

}  // namespace ops
}  // namespace mega2
