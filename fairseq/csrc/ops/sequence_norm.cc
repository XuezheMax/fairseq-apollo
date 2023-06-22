#include "ops/sequence_norm.h"

#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace mega2 {
namespace ops {

namespace {

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

template <typename T, typename T_ACC>
void ColwiseMoments(int64_t row, int64_t col, const T* X,
                    const bool* padding_mask, int64_t* m0_stk, T_ACC* m1_stk,
                    T_ACC* m2_stk) {
  const int64_t num_chunks = utils::DivUp(row, utils::kChunkSize);
  const int64_t depth = utils::CeilLog2(num_chunks);
  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t l = i * utils::kChunkSize;
    const int64_t r = std::min(l + utils::kChunkSize, row);
    for (int64_t j = l; j < r; ++j) {
      const bool mask = padding_mask != nullptr && padding_mask[j];
      for (int64_t k = 0; k < col; ++k) {
        const T_ACC x = static_cast<T_ACC>(X[j * col + k]);
        const auto [_, u, v] =
            utils::WelfordUpdate(m0_stk[0], m1_stk[k], m2_stk[k], x);
        m1_stk[k] = mask ? m1_stk[k] : u;
        m2_stk[k] = mask ? m2_stk[k] : v;
      }
      m0_stk[0] += mask ? 0 : 1;
    }
    int64_t cnt = i + 1;
    for (int64_t j = 1; j < depth && (cnt & 1) == 0; ++j) {
      int64_t m0 = 0;
      for (int64_t k = 0; k < col; ++k) {
        const int64_t p = j * col + k;
        const int64_t q = (j - 1) * col + k;
        std::tie(m0, m1_stk[p], m2_stk[p]) =
            utils::WelfordCombine(m0_stk[j], m1_stk[p], m2_stk[p],
                                  m0_stk[j - 1], m1_stk[q], m2_stk[q]);
        m1_stk[q] = T_ACC(0);
        m2_stk[q] = T_ACC(0);
      }
      m0_stk[j] += m0_stk[j - 1];
      m0_stk[j - 1] = 0;
      cnt >>= 1;
    }
  }
  for (int64_t i = 1; i < depth; ++i) {
    int64_t m0 = 0;
    for (int64_t j = 0; j < col; ++j) {
      std::tie(m0, m1_stk[j], m2_stk[j]) =
          utils::WelfordCombine(m0_stk[0], m1_stk[j], m2_stk[j], m0_stk[i],
                                m1_stk[i * col + j], m2_stk[i * col + j]);
    }
    m0_stk[0] += m0_stk[i];
  }
}

template <typename T, typename T_ACC>
std::pair<T_ACC, T_ACC> RowwiseInternalGradients(int64_t N, const T* Y_grad,
                                                 const T* X, T_ACC mean,
                                                 const bool* padding_mask) {
  const int64_t num_chunks = utils::DivUp(N, utils::kChunkSize);
  const int64_t depth = utils::CeilLog2(num_chunks);

  std::vector<T_ACC> ds_stk(depth, T_ACC(0));
  std::vector<T_ACC> db_stk(depth, T_ACC(0));
  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t l = i * utils::kChunkSize;
    const int64_t r = std::min(l + utils::kChunkSize, N);
    for (int64_t j = l; j < r; ++j) {
      const T_ACC dy = static_cast<T_ACC>(Y_grad[j]);
      const T_ACC x = static_cast<T_ACC>(X[j]);
      const bool mask = padding_mask != nullptr && padding_mask[j];
      ds_stk[0] += mask ? T_ACC(0) : dy * (x - mean);
      db_stk[0] += mask ? T_ACC(0) : dy;
    }
    int64_t cnt = i + 1;
    for (int64_t j = 1; j < depth && (cnt & 1) == 0; ++j) {
      ds_stk[j] += ds_stk[j - 1];
      db_stk[j] += db_stk[j - 1];
      ds_stk[j - 1] = T_ACC(0);
      db_stk[j - 1] = T_ACC(0);
      cnt >>= 1;
    }
  }
  for (int64_t j = 1; j < depth; ++j) {
    ds_stk[0] += ds_stk[j];
    db_stk[0] += db_stk[j];
  }

  return std::make_pair(ds_stk[0], db_stk[0]);
}

template <typename T, typename T_ACC>
void ColwiseInternalGradients(int64_t row, int64_t col, const T* Y_grad,
                              const T* X, const T_ACC* mean,
                              const bool* padding_mask, T_ACC* ds_stk,
                              T_ACC* db_stk) {
  const int64_t num_chunks = utils::DivUp(row, utils::kChunkSize);
  const int64_t depth = utils::CeilLog2(num_chunks);
  for (int64_t i = 0; i < num_chunks; ++i) {
    const int64_t l = i * utils::kChunkSize;
    const int64_t r = std::min(l + utils::kChunkSize, row);
    for (int64_t j = l; j < r; ++j) {
      const bool mask = padding_mask != nullptr && padding_mask[j];
      for (int64_t k = 0; k < col; ++k) {
        const T_ACC dy = static_cast<T_ACC>(Y_grad[j * col + k]);
        const T_ACC x = static_cast<T_ACC>(X[j * col + k]);
        const T_ACC u = mean[k];
        ds_stk[k] += mask ? T_ACC(0) : dy * (x - u);
        db_stk[k] += mask ? T_ACC(0) : dy;
      }
    }
    int64_t cnt = i + 1;
    for (int64_t j = 1; j < depth && (cnt & 1) == 0; ++j) {
      for (int64_t k = 0; k < col; ++k) {
        const int64_t p = j * col + k;
        const int64_t q = (j - 1) * col + k;
        ds_stk[p] += ds_stk[q];
        db_stk[p] += db_stk[q];
        ds_stk[q] = T_ACC(0);
        db_stk[q] = T_ACC(0);
      }
      cnt >>= 1;
    }
  }
  for (int64_t i = 1; i < depth; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      ds_stk[j] += ds_stk[i * col + j];
      db_stk[j] += db_stk[i * col + j];
    }
  }
}

template <typename T>
void SequenceNormCPUFwdBLHImpl(const torch::Tensor& X,
                               const torch::Tensor& gamma,
                               const torch::Tensor& beta,
                               const torch::Tensor& padding_mask, double eps,
                               torch::Tensor& Y, torch::Tensor& count,
                               torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t depth = utils::CeilLog2(utils::DivUp(L, utils::kChunkSize));

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  std::vector<int64_t> m0(B * depth, int64_t(0));
  std::vector<T_ACC> m1(B * depth * H, T_ACC(0));
  std::vector<T_ACC> m2(B * depth * H, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* X_ptr = X_data + i * L * H;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;
      T* Y_ptr = Y_data + i * L * H;
      T_ACC* mean_ptr = mean_data + i * H;
      T_ACC* rstd_ptr = rstd_data + i * H;
      int64_t* m0_ptr = m0.data() + i * depth;
      T_ACC* m1_ptr = m1.data() + i * depth * H;
      T_ACC* m2_ptr = m2.data() + i * depth * H;

      ColwiseMoments(L, H, X_ptr, mask_ptr, m0_ptr, m1_ptr, m2_ptr);
      count_data[i] = m0_ptr[0];
      for (int64_t j = 0; j < H; ++j) {
        const T_ACC rstd =
            T_ACC(1) / std::sqrt(m2_ptr[j] + static_cast<T_ACC>(eps));
        mean_ptr[j] = m1_ptr[j];
        rstd_ptr[j] = rstd;
      }
      for (int64_t j = 0; j < L; ++j) {
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        for (int64_t k = 0; k < H; ++k) {
          const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
          const T_ACC w = static_cast<T_ACC>(gamma_data[k]);
          const T_ACC b = static_cast<T_ACC>(beta_data[k]);
          Y_ptr[j * H + k] =
              mask ? T(0)
                   : static_cast<T>((x - mean_ptr[k]) * rstd_ptr[k] * w + b);
        }
      }
    }
  });
}

template <typename T>
void SequenceNormCPUFwdBHLImpl(const torch::Tensor& X,
                               const torch::Tensor& gamma,
                               const torch::Tensor& beta,
                               const torch::Tensor& padding_mask, double eps,
                               torch::Tensor& Y, torch::Tensor& count,
                               torch::Tensor& mean, torch::Tensor& rstd) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);

  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  const bool* padding_mask_data =
      padding_mask.defined() ? padding_mask.data_ptr<bool>() : nullptr;

  T* Y_data = Y.data_ptr<T>();
  int64_t* count_data = count.data_ptr<int64_t>();
  T_ACC* mean_data = mean.data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd.data_ptr<T_ACC>();

  at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t b = i / H;
      const int64_t h = i % H;
      const T* X_ptr = X_data + i * L;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + b * L;
      T* Y_ptr = Y_data + i * L;
      const auto [m0, m1, m2] = RowwiseMoments<T, T_ACC>(L, X_ptr, mask_ptr);
      const T_ACC rstd = T_ACC(1) / std::sqrt(m2 + static_cast<T_ACC>(eps));
      const T_ACC weight = static_cast<T_ACC>(gamma_data[h]);
      const T_ACC bias = static_cast<T_ACC>(beta_data[h]);
      for (int64_t j = 0; j < L; ++j) {
        const T_ACC x = static_cast<T_ACC>(X_ptr[j]);
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        Y_ptr[j] =
            mask ? T(0) : static_cast<T>((x - m1) * rstd * weight + bias);
      }
      if (h == 0) {
        count_data[b] = m0;
      }
      mean_data[i] = m1;
      rstd_data[i] = rstd;
    }
  });
}

template <typename T>
void SequenceNormCPUBwdBLHImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, torch::Tensor& X_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t L = X.size(1);
  const int64_t H = X.size(2);
  const int64_t depth = utils::CeilLog2(utils::DivUp(L, utils::kChunkSize));

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

  std::vector<T_ACC> ds(B * depth * H, T_ACC(0));
  std::vector<T_ACC> db(B * depth * H, T_ACC(0));

  at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T* Y_grad_ptr = Y_grad_data + i * L * H;
      const T* X_ptr = X_data + i * L * H;
      // const int64_t cnt = count_data[i];
      const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count_data[i]);
      const T_ACC* mean_ptr = mean_data + i * H;
      const T_ACC* rstd_ptr = rstd_data + i * H;
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + i * L;
      T* X_grad_ptr = X_grad_data + i * L * H;
      T_ACC* ds_ptr = ds.data() + i * depth * H;
      T_ACC* db_ptr = db.data() + i * depth * H;

      ColwiseInternalGradients(L, H, Y_grad_ptr, X_ptr, mean_ptr, mask_ptr,
                               ds_ptr, db_ptr);
      for (int64_t j = 0; j < L; ++j) {
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        for (int64_t k = 0; k < H; ++k) {
          const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j * H + k]);
          const T_ACC x = static_cast<T_ACC>(X_ptr[j * H + k]);
          const T_ACC u = mean_ptr[k];
          const T_ACC r = rstd_ptr[k];
          const T_ACC w = static_cast<T_ACC>(gamma_data[k]);
          const T_ACC dv = -T_ACC(0.5) * utils::Cube(r) * w * ds_ptr[k];
          const T_ACC du = -r * w * db_ptr[k];
          const T_ACC dx =
              r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
          X_grad_ptr[j * H + k] = mask ? T(0) : static_cast<T>(dx);
        }
      }
    }
  });

  for (int64_t i = 0; i < H; ++i) {
    ds[i] *= rstd_data[i];
  }
  for (int64_t i = 1; i < B; ++i) {
    const T_ACC* rstd_ptr = rstd_data + i * H;
    const T_ACC* ds_ptr = ds.data() + i * depth * H;
    const T_ACC* db_ptr = db.data() + i * depth * H;
    for (int64_t j = 0; j < H; ++j) {
      ds[j] += ds_ptr[j] * rstd_ptr[j];
      db[j] += db_ptr[j];
    }
  }
  for (int64_t i = 0; i < H; ++i) {
    gamma_grad_data[i] = static_cast<T>(ds[i]);
    beta_grad_data[i] = static_cast<T>(db[i]);
  }
}

template <typename T>
void SequenceNormCPUBwdBHLImpl(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const torch::Tensor& padding_mask, torch::Tensor& X_grad,
    torch::Tensor& gamma_grad, torch::Tensor& beta_grad) {
  using T_ACC = at::opmath_type<T>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);

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

  std::vector<T_ACC> dgamma(B * H, T_ACC(0));
  std::vector<T_ACC> dbeta(B * H, T_ACC(0));

  at::parallel_for(0, B * H, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t b = i / H;
      const int64_t h = i % H;
      const T* Y_grad_ptr = Y_grad_data + i * L;
      const T* X_ptr = X_data + i * L;
      // const int64_t cnt = count_data[i];
      const T_ACC coef = T_ACC(1) / static_cast<T_ACC>(count_data[b]);
      const T_ACC u = mean_data[i];
      const T_ACC r = rstd_data[i];
      const bool* mask_ptr =
          padding_mask_data == nullptr ? nullptr : padding_mask_data + b * L;
      T* X_grad_ptr = X_grad_data + i * L;

      const auto [ds, db] =
          RowwiseInternalGradients(L, Y_grad_ptr, X_ptr, u, mask_ptr);

      for (int64_t j = 0; j < L; ++j) {
        const T_ACC dy = static_cast<T_ACC>(Y_grad_ptr[j]);
        const T_ACC x = static_cast<T_ACC>(X_ptr[j]);
        const T_ACC w = static_cast<T_ACC>(gamma_data[h]);
        const bool mask = mask_ptr != nullptr && mask_ptr[j];
        const T_ACC dv = -T_ACC(0.5) * utils::Cube(r) * w * ds;
        const T_ACC du = -r * w * db;
        const T_ACC dx =
            r * w * dy + T_ACC(2) * coef * (x - u) * dv + coef * du;
        X_grad_ptr[j] = mask ? T(0) : static_cast<T>(dx);
      }
      dgamma[i] = ds * r;
      dbeta[i] = db;
    }
  });

  for (int64_t i = 1; i < B; ++i) {
    for (int64_t j = 0; j < H; ++j) {
      dgamma[j] += dgamma[i * H + j];
      dbeta[j] += dbeta[i * H + j];
    }
  }
  for (int64_t i = 0; i < H; ++i) {
    gamma_grad_data[i] = static_cast<T>(dgamma[i]);
    beta_grad_data[i] = static_cast<T>(dbeta[i]);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormCPUFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                   const torch::Tensor& beta,
                   const c10::optional<torch::Tensor>& padding_mask, double eps,
                   bool length_last) {
  const int64_t B = X.size(0);
  const int64_t H = X.size(length_last ? 1 : 2);

  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor count =
      torch::empty({B}, X.options()
                            .dtype(torch::kInt64)
                            .memory_format(at::MemoryFormat::Contiguous));

  const auto acc_type = at::toOpMathType(X.scalar_type());
  torch::Tensor mean = torch::empty(
      {B, H},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor rstd = torch::empty(
      {B, H},
      X.options().dtype(acc_type).memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCPUFwd", [&]() {
        if (length_last) {
          SequenceNormCPUFwdBHLImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
              mean, rstd);
        } else {
          SequenceNormCPUFwdBLHImpl<scalar_t>(
              *(X.expect_contiguous()), *(gamma.expect_contiguous()),
              *(beta.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), eps, Y, count,
              mean, rstd);
        }
      });

  return std::make_tuple(Y, count, mean, rstd);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SequenceNormFwd(const torch::Tensor& X, const torch::Tensor& gamma,
                const torch::Tensor& beta,
                const c10::optional<torch::Tensor>& padding_mask, double eps,
                bool length_last) {
  if (X.device().type() == torch::kCUDA) {
    return SequenceNormCUDAFwd(X, gamma, beta, padding_mask, eps, length_last);
  } else {
    return SequenceNormCPUFwd(X, gamma, beta, padding_mask, eps, length_last);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormCPUBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last) {
  c10::MaybeOwned<torch::Tensor> padding_mask_maybe_owned =
      at::borrow_from_optional_tensor(padding_mask);

  torch::Tensor X_grad = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor gamma_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor beta_grad = torch::empty_like(
      gamma, gamma.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "SequenceNormCPUFwd", [&]() {
        if (length_last) {
          SequenceNormCPUBwdBHLImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
              gamma_grad, beta_grad);
        } else {
          SequenceNormCPUBwdBLHImpl<scalar_t>(
              *(Y_grad.expect_contiguous()), *(X.expect_contiguous()),
              *(count.expect_contiguous()), *(mean.expect_contiguous()),
              *(rstd.expect_contiguous()), *(gamma.expect_contiguous()),
              *(padding_mask_maybe_owned->expect_contiguous()), X_grad,
              gamma_grad, beta_grad);
        }
      });

  return std::make_tuple(X_grad, gamma_grad, beta_grad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SequenceNormBwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X,
    const torch::Tensor& count, const torch::Tensor& mean,
    const torch::Tensor& rstd, const torch::Tensor& gamma,
    const c10::optional<torch::Tensor>& padding_mask, bool length_last) {
  if (X.device().type() == torch::kCUDA) {
    return SequenceNormCUDABwd(Y_grad, X, count, mean, rstd, gamma,
                               padding_mask, length_last);
  } else {
    return SequenceNormCPUBwd(Y_grad, X, count, mean, rstd, gamma, padding_mask,
                              length_last);
  }
}

void DefineSequenceNormOp(py::module& m) {
  m.def("sequence_norm_fwd", &SequenceNormFwd, "SequenceNorm forward")
      .def("sequence_norm_bwd", &SequenceNormBwd, "SequenceNorm backward");
}

}  // namespace ops
}  // namespace mega2
