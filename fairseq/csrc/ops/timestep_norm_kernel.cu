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

#include <cstring>
#include <tuple>
#include <vector>

#include "cuda_utils.cuh"
#include "ops/timestep_norm.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDAFwdKernel(int64_t L, int64_t N, const T* X,
                                          const int64_t* prev_count,
                                          const T* prev_mean, const T* prev_var,
                                          const T* gamma, const T* beta,
                                          const bool* padding_mask, T_ACC eps,
                                          T* Y, int64_t* count, T* mean, T* var,
                                          T* cummean, T* cumvar) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* X_ptr = X + i * L * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;
  T* Y_ptr = Y + i * L * N;
  T* m1_ptr = mean + i * N;
  T* m2_ptr = var + i * N;
  T* cm1_ptr = cummean + i * (L + 1) * N;
  T* cm2_ptr = cumvar + i * (L + 1) * N;

  int64_t m0 = prev_count[i];
  int64_t m0_out = m0;
  T_ACC m1 = static_cast<T_ACC>(prev_mean[i * N + k]);
  T_ACC m2 = static_cast<T_ACC>(prev_var[i * N + k]);
  cm1_ptr[k] = prev_mean[i * N + k];
  cm2_ptr[k] = prev_var[i * N + k];

  // TODO: Improve this.
  for (int64_t j = 0; j < L; ++j) {
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    const T_ACC w = static_cast<T_ACC>(gamma[k]);
    const T_ACC b = static_cast<T_ACC>(beta[k]);
    const bool mask = mask_ptr == nullptr ? false : mask_ptr[j];
    thrust::tie(m0, m1, m2) = cuda_utils::WelfordUpdate(m0, m1, m2, x);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    Y_ptr[j * N + k] = mask ? T(0) : static_cast<T>((x - m1) * rstd * w + b);
    m0_out = mask ? m0_out : m0;
    m1_ptr[k] = mask ? m1_ptr[k] : static_cast<T>(m1);
    m2_ptr[k] = mask ? m2_ptr[k] : static_cast<T>(m2);
    cm1_ptr[(j + 1) * N + k] = static_cast<T>(m1);
    cm2_ptr[(j + 1) * N + k] = static_cast<T>(m2);
  }

  if (k == 0) {
    count[i] = m0_out;
  }
}

template <typename T, typename T_ACC>
__global__ void TimestepNormCUDABwdKernel(
    int L, int N, const T* Y_grad, const T* mean_grad, const T* var_grad,
    const T* X, const int64_t* count, const T* cummean, const T* cumvar,
    const T* gamma, const bool* padding_mask, T_ACC eps, T* X_grad,
    T* prev_mean_grad, T* prev_var_grad, T_ACC* gamma_grad, T_ACC* beta_grad) {
  const int64_t i = blockIdx.y;
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) {
    return;
  }

  const T* Y_grad_ptr = Y_grad + i * L * N;
  const T* mean_grad_ptr = mean_grad + i * N;
  const T* var_grad_ptr = var_grad + i * N;
  const T* X_ptr = X + i * L * N;
  const T* m1_ptr = cummean + i * (L + 1) * N;
  const T* m2_ptr = cumvar + i * (L + 1) * N;
  const bool* mask_ptr =
      padding_mask == nullptr ? nullptr : padding_mask + i * L;

  T* X_grad_ptr = X_grad + i * L * N;
  T* m1_grad_ptr = prev_mean_grad + i * N;
  T* m2_grad_ptr = prev_var_grad + i * N;
  T_ACC* w_grad_ptr = gamma_grad + i * N;
  T_ACC* b_grad_ptr = beta_grad + i * N;

  int64_t m0 = count[i];
  T_ACC u_grad = static_cast<T_ACC>(mean_grad_ptr[k]);
  T_ACC v_grad = static_cast<T_ACC>(var_grad_ptr[k]);

  w_grad_ptr[k] = T_ACC(0);
  b_grad_ptr[k] = T_ACC(0);

  // TODO: Improve this.
  for (int64_t j = L - 1; j >= 0; --j) {
    const T_ACC y_grad = static_cast<T_ACC>(Y_grad_ptr[j * N + k]);
    const T_ACC x = static_cast<T_ACC>(X_ptr[j * N + k]);
    const T_ACC prev_m1 = static_cast<T_ACC>(m1_ptr[j * N + k]);
    const T_ACC m1 = static_cast<T_ACC>(m1_ptr[(j + 1) * N + k]);
    const T_ACC m2 = static_cast<T_ACC>(m2_ptr[(j + 1) * N + k]);
    const T_ACC w = static_cast<T_ACC>(gamma[k]);
    const bool mask = mask_ptr == nullptr ? false : mask_ptr[j];
    const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(m0);
    const T_ACC rstd = c10::cuda::compat::rsqrt(m2 + eps);
    const T_ACC dy_rstd = y_grad * rstd;
    const T_ACC delta = x - m1;
    const T_ACC dm2 =
        v_grad - (T_ACC(0.5) * y_grad * w * delta * cuda_utils::Cube(rstd));
    const T_ACC dm1 = u_grad - (w * dy_rstd + coef * dm2 * (x - prev_m1));
    const T_ACC x_grad =
        w * dy_rstd + dm2 * coef * (delta + x - prev_m1) + coef * dm1;

    X_grad_ptr[j * N + k] = mask ? T(0) : static_cast<T>(x_grad);
    u_grad = mask ? u_grad : (T_ACC(1) - coef) * dm1 - coef * delta * dm2;
    v_grad = mask ? v_grad : (T_ACC(1) - coef) * dm2;
    w_grad_ptr[k] += mask ? T_ACC(0) : dy_rstd * delta;
    b_grad_ptr[k] += mask ? T_ACC(0) : y_grad;
    m0 -= mask ? 0 : 1;
  }

  m1_grad_ptr[k] = static_cast<T>(u_grad);
  m2_grad_ptr[k] = static_cast<T>(v_grad);
}

template <typename T, typename T_ACC>
__global__ void GammaBetaCUDABwdKernel(int64_t B, int64_t N,
                                       const T_ACC* dw_internal,
                                       const T_ACC* db_internal, T* dw, T* db) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) {
    return;
  }
  T_ACC w_grad = T_ACC(0);
  T_ACC b_grad = T_ACC(0);
  for (int64_t i = 0; i < B; ++i) {
    w_grad += dw_internal[i * N + j];
    b_grad += db_internal[i * N + j];
  }
  dw[j] = static_cast<T>(w_grad);
  db[j] = static_cast<T>(b_grad);
}

template <typename T>
void TimestepNormCUDAFwdImpl(
    const torch::Tensor& X, const torch::Tensor& prev_count,
    const torch::Tensor& prev_mean, const torch::Tensor& prev_var,
    const torch::Tensor& gamma, const torch::Tensor& beta,
    const torch::Tensor& padding_mask, double eps, torch::Tensor& Y,
    torch::Tensor& count, torch::Tensor& mean, torch::Tensor& var,
    torch::Tensor& cummean, torch::Tensor& cumvar) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t N = X.size(2);
  const int64_t M =
      (N + cuda_utils::kCUDANumThreads - 1) / cuda_utils::kCUDANumThreads;

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
  T* cumvar_data = cumvar.data_ptr<T>();

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  TimestepNormCUDAFwdKernel<T, T_ACC>
      <<<dim3(M, B), cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          L, N, X_data, prev_count_data, prev_mean_data, prev_var_data,
          gamma_data, beta_data, padding_mask_data, static_cast<T_ACC>(eps),
          Y_data, count_data, mean_data, var_data, cummean_data, cumvar_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void TimestepNormCUDABwdImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
    const torch::Tensor& var_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& cummean,
    const torch::Tensor& cumvar, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, double eps, torch::Tensor& X_grad,
    torch::Tensor& prev_mean_grad, torch::Tensor& prev_var_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t N = X.size(2);
  const int64_t M =
      (N + cuda_utils::kCUDANumThreads - 1) / cuda_utils::kCUDANumThreads;

  torch::Tensor w_grad = torch::empty(
      {B, N}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  torch::Tensor b_grad = torch::empty(
      {B, N}, gamma.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const T* mean_grad_data = mean_grad.data_ptr<T>();
  const T* var_grad_data = var_grad.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const int64_t* count_data = count.data_ptr<int64_t>();
  const T* cummean_data = cummean.data_ptr<T>();
  const T* cumvar_data = cumvar.data_ptr<T>();
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
          L, N, Y_grad_data, mean_grad_data, var_grad_data, X_data, count_data,
          cummean_data, cumvar_data, gamma_data, padding_mask_data,
          static_cast<T_ACC>(eps), X_grad_data, prev_mean_grad_data,
          prev_var_grad_data, w_grad_data, b_grad_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  GammaBetaCUDABwdKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDANumThreads, 0, cuda_stream>>>(
          B, N, w_grad_data, b_grad_data, gamma_grad_data, beta_grad_data);
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
      {B, L + 1, N},
      prev_mean.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor cumvar = torch::empty(
      {B, L + 1, N},
      prev_var.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "TimestepNormCUDAFwd", [&]() {
        TimestepNormCUDAFwdImpl<scalar_t>(
            *(X.expect_contiguous()), *(prev_count.expect_contiguous()),
            *(prev_mean.expect_contiguous()), *(prev_var.expect_contiguous()),
            *(gamma.expect_contiguous()), *(beta.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
            mean, var, cummean, cumvar);
      });
  return std::make_tuple(Y, count, mean, var, cummean, cumvar);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
TimestepNormCUDABwd(const torch::Tensor& Y_grad, const torch::Tensor& mean_grad,
                    const torch::Tensor& var_grad, const torch::Tensor& X,
                    const torch::Tensor& count, const torch::Tensor& cummean,
                    const torch::Tensor& cumvar, const torch::Tensor& gamma,
                    const c10::optional<torch::Tensor>& padding_mask,
                    double eps) {
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
            *(var_grad.expect_contiguous()), *(X.expect_contiguous()), count,
            *(cummean.expect_contiguous()), *(cumvar.expect_contiguous()),
            *(gamma.expect_contiguous()),
            *(padding_mask_maybe_owned->expect_contiguous()), eps, X_grad,
            prev_mean_grad, prev_var_grad, gamma_grad, beta_grad);
      });
  return std::make_tuple(X_grad, prev_mean_grad, prev_var_grad, gamma_grad,
                         beta_grad);
}

}  // namespace ops
}  // namespace mega2
