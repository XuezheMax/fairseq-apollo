#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
// #include <curand.h>
// #include <curand_kernel.h>
// #include <curand_philox4x32_x.h>
#include <thrust/tuple.h>

#include <mutex>
#include <tuple>
#include <type_traits>

#include "activations.cuh"
#include "blas.h"
#include "cuda_utils.cuh"
#include "ops/ffn_swiglu.h"
#include "random_utils.cuh"
#include "register_utils.cuh"
#include "utils.h"

namespace mega2 {
namespace ops {

namespace {

constexpr int64_t kMaxElementsPerThread = 8;

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwishFwdKernel(int64_t size, const T* __restrict__ x,
                               T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC y_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + i * kBlockSize, cur_size, cur_size, T_ACC(0), y_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    y_acc[j] = activations::Swish<T_ACC>(y_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(y_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwishDropoutFwdKernel(at::PhiloxCudaState philox_args,
                                      int64_t size, const T* __restrict__ x,
                                      T_ACC dropout, T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC y_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + i * kBlockSize, cur_size, cur_size, T_ACC(0), y_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    y_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : y_acc[j + 0];
    y_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : y_acc[j + 1];
    y_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : y_acc[j + 2];
    y_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : y_acc[j + 3];
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    y_acc[j] = activations::Swish<T_ACC>(y_acc[j]) * coef;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(y_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUFwdKernel(int64_t size, const T* __restrict__ xw,
                                const T* __restrict__ xv, T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC xw_acc[kElementsPerThread];
  T_ACC xv_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xw + i * kBlockSize, cur_size, cur_size, T_ACC(0), xw_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xv + i * kBlockSize, cur_size, cur_size, T_ACC(0), xv_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    xw_acc[j] = activations::SwiGLU<T_ACC>(xw_acc[j], xv_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(xw_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUDropoutFwdKernel(at::PhiloxCudaState philox_args,
                                       int64_t size, const T* __restrict__ xw,
                                       const T* __restrict__ xv, T_ACC dropout,
                                       T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC xw_acc[kCapacityPerThread];
  T_ACC xv_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xw + i * kBlockSize, cur_size, cur_size, T_ACC(0), xw_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xv + i * kBlockSize, cur_size, cur_size, T_ACC(0), xv_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    xw_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : xw_acc[j + 0];
    xw_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : xw_acc[j + 1];
    xw_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : xw_acc[j + 2];
    xw_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : xw_acc[j + 3];
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    xw_acc[j] = activations::SwiGLU<T_ACC>(xw_acc[j], xv_acc[j]) * coef;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(xw_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwishBwdKernel(int64_t size, const T* __restrict__ y_grad,
                               const T* __restrict__ x,
                               T* __restrict__ x_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC d_acc[kElementsPerThread];
  T_ACC x_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), d_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + i * kBlockSize, cur_size, cur_size, T_ACC(0), x_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    d_acc[j] = activations::SwishBwd<T_ACC>(d_acc[j], x_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(d_acc, cur_size,
                                                     x_grad + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwishDropoutBwdKernel(at::PhiloxCudaState philox_args,
                                      int64_t size,
                                      const T* __restrict__ y_grad,
                                      const T* __restrict__ x, T_ACC dropout,
                                      T* __restrict__ x_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC d_acc[kCapacityPerThread];
  T_ACC x_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), d_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x + i * kBlockSize, cur_size, cur_size, T_ACC(0), x_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    d_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : d_acc[j + 0] * coef;
    d_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : d_acc[j + 1] * coef;
    d_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : d_acc[j + 2] * coef;
    d_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : d_acc[j + 3] * coef;
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    d_acc[j] = activations::SwishBwd<T_ACC>(d_acc[j], x_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(d_acc, cur_size,
                                                     x_grad + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUBwdKernel(int64_t size, const T* __restrict__ y_grad,
                                const T* __restrict__ xw,
                                const T* __restrict__ xv,
                                T* __restrict__ w_grad,
                                T* __restrict__ v_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC d_acc[kElementsPerThread];
  T_ACC w_acc[kElementsPerThread];
  T_ACC v_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), d_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xw + i * kBlockSize, cur_size, cur_size, T_ACC(0), w_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xv + i * kBlockSize, cur_size, cur_size, T_ACC(0), v_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    thrust::tie(w_acc[j], v_acc[j]) =
        activations::SwiGLUBwd<T_ACC>(d_acc[j], w_acc[j], v_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(w_acc, cur_size,
                                                     w_grad + i * kBlockSize);
  register_utils::Save<T_ACC, T, kElementsPerThread>(v_acc, cur_size,
                                                     v_grad + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUDropoutBwdKernel(
    at::PhiloxCudaState philox_args, int64_t size, const T* __restrict__ y_grad,
    const T* __restrict__ xw, const T* __restrict__ xv, T_ACC dropout,
    T* __restrict__ w_grad, T* __restrict__ v_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC d_acc[kCapacityPerThread];
  T_ACC w_acc[kCapacityPerThread];
  T_ACC v_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), d_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xw + i * kBlockSize, cur_size, cur_size, T_ACC(0), w_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      xv + i * kBlockSize, cur_size, cur_size, T_ACC(0), v_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    d_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : d_acc[j + 0] * coef;
    d_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : d_acc[j + 1] * coef;
    d_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : d_acc[j + 2] * coef;
    d_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : d_acc[j + 3] * coef;
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    thrust::tie(w_acc[j], v_acc[j]) =
        activations::SwiGLUBwd<T_ACC>(d_acc[j], w_acc[j], v_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(w_acc, cur_size,
                                                     w_grad + i * kBlockSize);
  register_utils::Save<T_ACC, T, kElementsPerThread>(v_acc, cur_size,
                                                     v_grad + i * kBlockSize);
}

#define DISPATCH_ELEMENTWISE_CUDA_KERNEL(KernelFunc, T, T_ACC, size,          \
                                         cuda_stream, ...)                    \
  do {                                                                        \
    if (!std::is_same<T_ACC, double>::value && size >= 4096) {                \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 4096);           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4096, 512>, num_blocks,   \
                               512, 0, cuda_stream, __VA_ARGS__);             \
    } else if (size >= 2048) {                                                \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 2048);           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2048, 256>, num_blocks,   \
                               256, 0, cuda_stream, __VA_ARGS__);             \
    } else if (size >= 1024) {                                                \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 1024);           \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1024, 128>, num_blocks,   \
                               128, 0, cuda_stream, __VA_ARGS__);             \
    } else if (size >= 512) {                                                 \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 512);            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 512, 64>, num_blocks, 64, \
                               0, cuda_stream, __VA_ARGS__);                  \
    } else if (size >= 256) {                                                 \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 256);            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 256, 32>, num_blocks, 32, \
                               0, cuda_stream, __VA_ARGS__);                  \
    } else if (size >= 128) {                                                 \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 128);            \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 128, 32>, num_blocks, 32, \
                               0, cuda_stream, __VA_ARGS__);                  \
    } else if (size >= 64) {                                                  \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 64);             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 64, 32>, num_blocks, 32,  \
                               0, cuda_stream, __VA_ARGS__);                  \
    } else {                                                                  \
      const int64_t num_blocks = utils::DivUp<int64_t>(size, 32);             \
      cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 32, 32>, num_blocks, 32,  \
                               0, cuda_stream, __VA_ARGS__);                  \
    }                                                                         \
  } while (false)

template <typename T>
void FFNSwiGLUCUDAFwdImpl(const torch::Tensor& x, const torch::Tensor& w1,
                          const torch::Tensor& b1, const torch::Tensor& w2,
                          const torch::Tensor& b2, const torch::Tensor& w3,
                          const torch::Tensor& b3, double dropout,
                          torch::Tensor& y, torch::Tensor& h, torch::Tensor& hw,
                          c10::optional<torch::Tensor>& hv, int64_t& seed,
                          int64_t& offset) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t inner_size = x.size(-1);
  const int64_t outer_size = x.numel() / inner_size;
  const int64_t hidden_size = w1.size(0);
  if (w3.defined()) {
    TORCH_CHECK(hv.has_value());
  }

  const T* x_data = x.data_ptr<T>();
  const T* w1_data = w1.data_ptr<T>();
  const T* b1_data = b1.defined() ? b1.data_ptr<T>() : nullptr;
  const T* w2_data = w2.data_ptr<T>();
  const T* b2_data = b2.defined() ? b2.data_ptr<T>() : nullptr;
  const T* w3_data = w3.defined() ? w3.data_ptr<T>() : nullptr;
  const T* b3_data = b3.defined() ? b3.data_ptr<T>() : nullptr;
  T* y_data = y.data_ptr<T>();
  T* h_data = h.data_ptr<T>();
  T* hw_data = hw.data_ptr<T>();
  T* hv_data = hv.has_value() ? hv->data_ptr<T>() : nullptr;

  seed = -1;
  offset = -1;
  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  if (b1_data == nullptr) {
    blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kT,
                      outer_size, hidden_size, inner_size, /*alpha=*/T_ACC(1),
                      x_data, inner_size, w1_data, inner_size,
                      /*beta=*/T_ACC(0), hw_data, hidden_size);
  } else {
    blas::GemmAndBiasCUDA<T>(handle, cuda_stream, blas::TransposeOp::kN,
                             blas::TransposeOp::kT, outer_size, hidden_size,
                             inner_size, /*alpha=*/T_ACC(1), x_data, inner_size,
                             w1_data, inner_size, b1_data, /*beta=*/T_ACC(0),
                             hw_data, hidden_size);
  }

  if (w3_data == nullptr) {
    if (dropout == 0.0) {
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwishFwdKernel, T, T_ACC, outer_size * hidden_size, cuda_stream,
          outer_size * hidden_size, hw_data, h_data);
    } else {
      auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
      at::PhiloxCudaState rng_engine_inputs;
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(kMaxElementsPerThread);
      }
      const auto random_state = random_utils::HostUnpack(rng_engine_inputs);
      seed = std::get<0>(random_state);
      offset = std::get<1>(random_state);
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwishDropoutFwdKernel, T, T_ACC, outer_size * hidden_size,
          cuda_stream, rng_engine_inputs, outer_size * hidden_size, hw_data,
          dropout, h_data);
    }
  } else {
    if (b3_data == nullptr) {
      blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kT,
                        outer_size, hidden_size, inner_size,
                        /*alpha=*/T_ACC(1), x_data, inner_size, w3_data,
                        inner_size,
                        /*beta=*/T_ACC(0), hv_data, hidden_size);
    } else {
      blas::GemmAndBiasCUDA<T>(
          handle, cuda_stream, blas::TransposeOp::kN, blas::TransposeOp::kT,
          outer_size, hidden_size, inner_size,
          /*alpha=*/T_ACC(1), x_data, inner_size, w3_data, inner_size, b3_data,
          /*beta=*/T_ACC(0), hv_data, hidden_size);
    }
    if (dropout == 0.0) {
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwiGLUFwdKernel, T, T_ACC, outer_size * hidden_size, cuda_stream,
          outer_size * hidden_size, hw_data, hv_data, h_data);
    } else {
      auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
      at::PhiloxCudaState rng_engine_inputs;
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(kMaxElementsPerThread);
      }
      const auto random_state = random_utils::HostUnpack(rng_engine_inputs);
      seed = std::get<0>(random_state);
      offset = std::get<1>(random_state);
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwiGLUDropoutFwdKernel, T, T_ACC, outer_size * hidden_size,
          cuda_stream, rng_engine_inputs, outer_size * hidden_size, hw_data,
          hv_data, dropout, h_data);
    }
  }

  if (b2_data == nullptr) {
    blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kT,
                      outer_size, inner_size, hidden_size, /*alpha=*/T_ACC(1),
                      h_data, hidden_size, w2_data, hidden_size,
                      /*beta=*/T_ACC(0), y_data, inner_size);
  } else {
    blas::GemmAndBiasCUDA<T>(handle, cuda_stream, blas::TransposeOp::kN,
                             blas::TransposeOp::kT, outer_size, inner_size,
                             hidden_size, /*alpha=*/T_ACC(1), h_data,
                             hidden_size, w2_data, hidden_size, b2_data,
                             /*beta=*/T_ACC(0), y_data, inner_size);
  }
}

template <typename T>
void FFNSwiGLUCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x,
                          const torch::Tensor& w1, const torch::Tensor& w2,
                          const torch::Tensor& w3, const torch::Tensor& h,
                          double dropout, int64_t seed, int64_t offset,
                          torch::Tensor& x_grad, torch::Tensor& w1_grad,
                          c10::optional<torch::Tensor>& b1_grad,
                          torch::Tensor& w2_grad,
                          c10::optional<torch::Tensor>& b2_grad,
                          c10::optional<torch::Tensor>& w3_grad,
                          c10::optional<torch::Tensor>& b3_grad,
                          torch::Tensor& hw, c10::optional<torch::Tensor>& hv) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t inner_size = x.size(-1);
  const int64_t outer_size = x.numel() / inner_size;
  const int64_t hidden_size = w1.size(0);
  TORCH_CHECK(outer_size > 1);

  torch::Tensor h_grad = torch::empty_like(hw);

  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* x_data = x.data_ptr<T>();
  const T* w1_data = w1.data_ptr<T>();
  const T* w2_data = w2.data_ptr<T>();
  const T* w3_data = w3.defined() ? w3.data_ptr<T>() : nullptr;
  const T* h_data = h.data_ptr<T>();
  T* x_grad_data = x_grad.data_ptr<T>();
  T* w1_grad_data = w1_grad.data_ptr<T>();
  T* b1_grad_data = b1_grad.has_value() ? b1_grad->data_ptr<T>() : nullptr;
  T* w2_grad_data = w2_grad.data_ptr<T>();
  T* b2_grad_data = b2_grad.has_value() ? b2_grad->data_ptr<T>() : nullptr;
  T* w3_grad_data = w3_grad.has_value() ? w3_grad->data_ptr<T>() : nullptr;
  T* b3_grad_data = b3_grad.has_value() ? b3_grad->data_ptr<T>() : nullptr;
  T* hw_data = hw.data_ptr<T>();
  T* hv_data = hv.has_value() ? hv->data_ptr<T>() : nullptr;
  T* h_grad_data = h_grad.data_ptr<T>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  torch::globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // dL/dw2 and dL/db2
  if (b2_grad_data == nullptr) {
    blas::GemmCUDA<T>(handle, blas::TransposeOp::kT, blas::TransposeOp::kN,
                      inner_size, hidden_size, outer_size, /*alpha=*/T_ACC(1),
                      y_grad_data, inner_size, h_data, hidden_size,
                      /*beta=*/T_ACC(0), w2_grad_data, hidden_size);
  } else {
    blas::GemmAndBiasGradCUDA<T>(
        handle, cuda_stream, blas::TransposeOp::kT, blas::TransposeOp::kN,
        inner_size, hidden_size, outer_size, /*alpha=*/T_ACC(1), y_grad_data,
        inner_size, h_data, hidden_size,
        /*beta=*/T_ACC(0), w2_grad_data, hidden_size, b2_grad_data);
  }
  // dL/dh
  blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kN,
                    outer_size, hidden_size, inner_size,
                    /*alpha=*/T_ACC(1), y_grad_data, inner_size, w2_data,
                    hidden_size, /*beta=*/T_ACC(0), h_grad_data, hidden_size);

  if (w3_data == nullptr) {
    if (dropout == 0.0) {
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwishBwdKernel, T, T_ACC, outer_size * hidden_size, cuda_stream,
          outer_size * hidden_size, h_grad_data, hw_data, hw_data);
    } else {
      at::PhiloxCudaState rng_engine_inputs(seed, offset);
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwishDropoutBwdKernel, T, T_ACC, outer_size * hidden_size,
          cuda_stream, rng_engine_inputs, outer_size * hidden_size, h_grad_data,
          hw_data, dropout, hw_data);
    }
  } else {
    if (dropout == 0.0) {
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(SwiGLUBwdKernel, T, T_ACC,
                                       outer_size * hidden_size, cuda_stream,
                                       outer_size * hidden_size, h_grad_data,
                                       hw_data, hv_data, hw_data, hv_data);
    } else {
      at::PhiloxCudaState rng_engine_inputs(seed, offset);
      DISPATCH_ELEMENTWISE_CUDA_KERNEL(
          SwiGLUDropoutBwdKernel, T, T_ACC, outer_size * hidden_size,
          cuda_stream, rng_engine_inputs, outer_size * hidden_size, h_grad_data,
          hw_data, hv_data, dropout, hw_data, hv_data);
    }
  }
  // dL/dw1 and dL/db1
  if (b1_grad_data == nullptr) {
    blas::GemmCUDA<T>(handle, blas::TransposeOp::kT, blas::TransposeOp::kN,
                      hidden_size, inner_size, outer_size, /*alpha=*/T_ACC(1),
                      hw_data, hidden_size, x_data, inner_size,
                      /*beta=*/T_ACC(0), w1_grad_data, inner_size);
  } else {
    blas::GemmAndBiasGradCUDA<T>(
        handle, cuda_stream, blas::TransposeOp::kT, blas::TransposeOp::kN,
        hidden_size, inner_size, outer_size, /*alpha=*/T_ACC(1), hw_data,
        hidden_size, x_data, inner_size,
        /*beta=*/T_ACC(0), w1_grad_data, inner_size, b1_grad_data);
  }
  if (w3_data != nullptr) {
    // dL/dw3 and dL/db3
    if (b3_grad_data == nullptr) {
      blas::GemmCUDA<T>(handle, blas::TransposeOp::kT, blas::TransposeOp::kN,
                        hidden_size, inner_size, outer_size, /*alpha=*/T_ACC(1),
                        hv_data, hidden_size, x_data, inner_size,
                        /*beta=*/T_ACC(0), w3_grad_data, inner_size);
    } else {
      blas::GemmAndBiasGradCUDA<T>(
          handle, cuda_stream, blas::TransposeOp::kT, blas::TransposeOp::kN,
          hidden_size, inner_size, outer_size, /*alpha=*/T_ACC(1), hv_data,
          hidden_size, x_data, inner_size,
          /*beta=*/T_ACC(0), w3_grad_data, inner_size, b3_grad_data);
    }
  }
  // dL/dx
  blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kN,
                    outer_size, inner_size, hidden_size, /*alpha=*/T_ACC(1),
                    hw_data, hidden_size, w1_data, inner_size,
                    /*beta=*/T_ACC(0), x_grad_data, inner_size);
  if (w3_data != nullptr) {
    blas::GemmCUDA<T>(handle, blas::TransposeOp::kN, blas::TransposeOp::kN,
                      outer_size, inner_size, hidden_size, /*alpha=*/T_ACC(1),
                      hv_data, hidden_size, w3_data, inner_size,
                      /*beta=*/T_ACC(1), x_grad_data, inner_size);
  }
}

#undef DISPATCH_ELEMENTWISE_CUDA_KERNEL

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           c10::optional<torch::Tensor>, int64_t, int64_t>
FFNSwiGLUCUDAFwd(const torch::Tensor& x, const torch::Tensor& w1,
                 const c10::optional<torch::Tensor>& b1,
                 const torch::Tensor& w2,
                 const c10::optional<torch::Tensor>& b2,
                 const c10::optional<torch::Tensor>& w3,
                 const c10::optional<torch::Tensor>& b3, double dropout) {
  const int64_t N = x.size(-1);
  const int64_t H = w1.size(0);
  TORCH_CHECK(w1.size(1) == N);
  if (b1.has_value()) {
    TORCH_CHECK(b1->size(0) == H);
  }
  TORCH_CHECK(w2.size(0) == N);
  TORCH_CHECK(w2.size(1) == H);
  if (b2.has_value()) {
    TORCH_CHECK(b2->size(0) == N);
  }
  if (w3.has_value()) {
    TORCH_CHECK(w3->size(0) == H);
    TORCH_CHECK(w3->size(1) == N);
  }
  if (b3.has_value()) {
    TORCH_CHECK(b1.has_value());
    TORCH_CHECK(w3.has_value());
    TORCH_CHECK(b3->size(0) == H);
  }

  c10::MaybeOwned<torch::Tensor> b1_maybe_owned =
      at::borrow_from_optional_tensor(b1);
  c10::MaybeOwned<torch::Tensor> b2_maybe_owned =
      at::borrow_from_optional_tensor(b2);
  c10::MaybeOwned<torch::Tensor> w3_maybe_owned =
      at::borrow_from_optional_tensor(w3);
  c10::MaybeOwned<torch::Tensor> b3_maybe_owned =
      at::borrow_from_optional_tensor(b3);

  torch::Tensor y = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));

  auto h_size = x.sizes().vec();
  h_size.back() = H;
  torch::Tensor h = torch::empty(
      h_size, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor hw = torch::empty(
      h_size, x.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> hv =
      w3.has_value()
          ? c10::make_optional(torch::empty(
                h_size,
                x.options().memory_format(at::MemoryFormat::Contiguous)))
          : c10::nullopt;

  int64_t seed = -1;
  int64_t offset = -1;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "FFNSwiGLUCUDAFwd", [&]() {
        FFNSwiGLUCUDAFwdImpl<scalar_t>(
            *(x.expect_contiguous()), *(w1.expect_contiguous()),
            *(b1_maybe_owned->expect_contiguous()), *(w2.expect_contiguous()),
            *(b2_maybe_owned->expect_contiguous()),
            *(w3_maybe_owned->expect_contiguous()),
            *(b3_maybe_owned->expect_contiguous()), dropout, y, h, hw, hv, seed,
            offset);
      });

  return std::make_tuple(y, h, hw, hv, seed, offset);
}

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
           torch::Tensor, c10::optional<torch::Tensor>,
           c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
FFNSwiGLUCUDABwd(const torch::Tensor& y_grad, const torch::Tensor& x,
                 const torch::Tensor& w1,
                 const c10::optional<torch::Tensor>& b1,
                 const torch::Tensor& w2,
                 const c10::optional<torch::Tensor>& b2,
                 const c10::optional<torch::Tensor>& w3,
                 const c10::optional<torch::Tensor>& b3, const torch::Tensor& h,
                 double dropout, int64_t seed, int64_t offset,
                 torch::Tensor& hw, c10::optional<torch::Tensor>& hv) {
  const int64_t N = x.size(-1);
  const int64_t H = w1.size(0);
  TORCH_CHECK(w1.size(1) == N);
  if (b1.has_value()) {
    TORCH_CHECK(b1->size(0) == H);
  }
  TORCH_CHECK(w2.size(0) == N);
  TORCH_CHECK(w2.size(1) == H);
  if (b2.has_value()) {
    TORCH_CHECK(b2->size(0) == N);
  }
  if (w3.has_value()) {
    TORCH_CHECK(w3->size(0) == H);
    TORCH_CHECK(w3->size(1) == N);
  }
  if (b3.has_value()) {
    TORCH_CHECK(b1.has_value());
    TORCH_CHECK(w3.has_value());
    TORCH_CHECK(b3->size(0) == H);
  }

  c10::MaybeOwned<torch::Tensor> w3_maybe_owned =
      at::borrow_from_optional_tensor(w3);

  torch::Tensor x_grad = torch::empty_like(
      x, x.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor w1_grad = torch::empty_like(
      w1, w1.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> b1_grad =
      b1.has_value()
          ? c10::make_optional(torch::empty_like(
                *b1, b1->options().memory_format(at::MemoryFormat::Contiguous)))
          : c10::nullopt;
  torch::Tensor w2_grad = torch::empty_like(
      w2, w2.options().memory_format(at::MemoryFormat::Contiguous));
  c10::optional<torch::Tensor> b2_grad =
      b2.has_value()
          ? c10::make_optional(torch::empty_like(
                *b2, b2->options().memory_format(at::MemoryFormat::Contiguous)))
          : c10::nullopt;
  c10::optional<torch::Tensor> w3_grad =
      w3.has_value()
          ? c10::make_optional(torch::empty_like(
                *w3, w3->options().memory_format(at::MemoryFormat::Contiguous)))
          : c10::nullopt;
  c10::optional<torch::Tensor> b3_grad =
      b3.has_value()
          ? c10::make_optional(torch::empty_like(
                *b3, b3->options().memory_format(at::MemoryFormat::Contiguous)))
          : c10::nullopt;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "FFNSwiGLUCUDABwd", [&]() {
        FFNSwiGLUCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x.expect_contiguous()),
            *(w1.expect_contiguous()), *(w2.expect_contiguous()),
            *(w3_maybe_owned->expect_contiguous()), *(h.expect_contiguous()),
            dropout, seed, offset, x_grad, w1_grad, b1_grad, w2_grad, b2_grad,
            w3_grad, b3_grad, hw, hv);
      });

  return std::make_tuple(x_grad, w1_grad, b1_grad, w2_grad, b2_grad, w3_grad,
                         b3_grad);
}

}  // namespace ops
}  // namespace mega2
