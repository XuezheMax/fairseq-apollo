#include <ATen/AccumulateType.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <thrust/tuple.h>

#include <ATen/native/cuda/Loops.cuh>
#include <type_traits>

#include "activations.cuh"
#include "ops/swiglu.h"
#include "random_utils.cuh"

namespace mega2 {
namespace ops {

namespace {

constexpr int64_t kMaxElementsPerThread = 8;

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUFwdKernel(int64_t size, const T* __restrict__ x1,
                                const T* __restrict__ x2, T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC x1_acc[kElementsPerThread];
  T_ACC x2_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x1 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x1_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x2 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x2_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    x1_acc[j] = activations::SwiGLU<T_ACC>(x1_acc[j], x2_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x1_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUDropoutFwdKernel(at::PhiloxCudaState philox_args,
                                       int64_t size, const T* __restrict__ x1,
                                       const T* __restrict__ x2, T_ACC dropout,
                                       T* __restrict__ y) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC x1_acc[kCapacityPerThread];
  T_ACC x2_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);

  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x1 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x1_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x2 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x2_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    x1_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : x1_acc[j + 0];
    x1_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : x1_acc[j + 1];
    x1_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : x1_acc[j + 2];
    x1_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : x1_acc[j + 3];
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    x1_acc[j] = activations::SwiGLU<T_ACC>(x1_acc[j], x2_acc[j]) * coef;
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x1_acc, cur_size,
                                                     y + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUBwdKernel(int64_t size, const T* __restrict__ y_grad,
                                const T* __restrict__ x1,
                                const T* __restrict__ x2,
                                T* __restrict__ x1_grad,
                                T* __restrict__ x2_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  T_ACC dy_acc[kElementsPerThread];
  T_ACC x1_acc[kElementsPerThread];
  T_ACC x2_acc[kElementsPerThread];

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), dy_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x1 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x1_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x2 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x2_acc);
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    thrust::tie(x1_acc[j], x2_acc[j]) =
        activations::SwiGLUBwd<T_ACC>(dy_acc[j], x1_acc[j], x2_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x1_acc, cur_size,
                                                     x1_grad + i * kBlockSize);
  register_utils::Save<T_ACC, T, kElementsPerThread>(x2_acc, cur_size,
                                                     x2_grad + i * kBlockSize);
}

template <typename T, typename T_ACC, int64_t kBlockSize, int64_t kNumThreads>
__global__ void SwiGLUDropoutBwdKernel(
    at::PhiloxCudaState philox_args, int64_t size, const T* __restrict__ y_grad,
    const T* __restrict__ x1, const T* __restrict__ x2, T_ACC dropout,
    T* __restrict__ x1_grad, T* __restrict__ x2_grad) {
  constexpr int64_t kElementsPerThread = kBlockSize / kNumThreads;
  constexpr int64_t kCapacityPerThread =
      std::max(kElementsPerThread, random_utils::kRandomUnroll);
  T_ACC dy_acc[kCapacityPerThread];
  T_ACC x1_acc[kCapacityPerThread];
  T_ACC x2_acc[kCapacityPerThread];

  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, index, offset, &state);

  const int64_t i = blockIdx.x;
  const int64_t cur_size = std::min(kBlockSize, size - i * kBlockSize);
  const T_ACC coef = T_ACC(1) / (T_ACC(1) - dropout);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      y_grad + i * kBlockSize, cur_size, cur_size, T_ACC(0), dy_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x1 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x1_acc);
  register_utils::Load<T, T_ACC, kElementsPerThread>(
      x2 + i * kBlockSize, cur_size, cur_size, T_ACC(0), x2_acc);
#pragma unroll
  for (int64_t j = 0; j < kCapacityPerThread;
       j += random_utils::kRandomUnroll) {
    const float4 rand4 = curand_uniform4(&state);
    dy_acc[j + 0] = rand4.x < dropout ? T_ACC(0) : dy_acc[j + 0] * coef;
    dy_acc[j + 1] = rand4.y < dropout ? T_ACC(0) : dy_acc[j + 1] * coef;
    dy_acc[j + 2] = rand4.z < dropout ? T_ACC(0) : dy_acc[j + 2] * coef;
    dy_acc[j + 3] = rand4.w < dropout ? T_ACC(0) : dy_acc[j + 3] * coef;
  }
#pragma unroll
  for (int64_t j = 0; j < kElementsPerThread; ++j) {
    thrust::tie(x1_acc[j], x2_acc[j]) =
        activations::SwiGLUBwd<T_ACC>(dy_acc[j], x1_acc[j], x2_acc[j]);
  }
  register_utils::Save<T_ACC, T, kElementsPerThread>(x1_acc, cur_size,
                                                     x1_grad + i * kBlockSize);
  register_utils::Save<T_ACC, T, kElementsPerThread>(x2_acc, cur_size,
                                                     x2_grad + i * kBlockSize);
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
void SwiGLUCUDAFwdImpl(const torch::Tensor& x1, const torch::Tensor& x2,
                       double dropout, torch::Tensor& y, int64_t& seed,
                       int64_t& offset) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t size = x1.numel();
  const T* x1_data = x1.data_ptr<T>();
  const T* x2_data = x2.data_ptr<T>();
  T* y_data = y.data_ptr<T>();

  seed = -1;
  offset = -1;

  at::cuda::OptionalCUDAGuard guard(at::device_of(x1));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (dropout == 0.0) {
    DISPATCH_ELEMENTWISE_CUDA_KERNEL(SwiGLUFwdKernel, T, T_ACC, size,
                                     cuda_stream, size, x1_data, x2_data,
                                     y_data);
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
        SwiGLUDropoutFwdKernel, T, T_ACC, size, cuda_stream, rng_engine_inputs,
        size, x1_data, x2_data, static_cast<T_ACC>(dropout), y_data);
  }

  // at::TensorIterator iter = at::TensorIterator::binary_op(y, x1, x2);
  // at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x1, T x2) -> T {
  //   const T_ACC x1_acc = static_cast<T_ACC>(x1);
  //   const T_ACC x2_acc = static_cast<T_ACC>(x2);
  //   T_ACC y = activations::SwiGLU<T_ACC>(x1_acc, x2_acc);
  //   return static_cast<T>(y);
  // });
}

template <typename T>
void SwiGLUCUDABwdImpl(const torch::Tensor& y_grad, const torch::Tensor& x1,
                       const torch::Tensor& x2, double dropout, int64_t seed,
                       int64_t offset, torch::Tensor& x1_grad,
                       torch::Tensor& x2_grad) {
  using T_ACC = at::acc_type<T, /*is_cuda=*/true>;

  const int64_t size = y_grad.numel();
  const T* y_grad_data = y_grad.data_ptr<T>();
  const T* x1_data = x1.data_ptr<T>();
  const T* x2_data = x2.data_ptr<T>();
  T* x1_grad_data = x1_grad.data_ptr<T>();
  T* x2_grad_data = x2_grad.data_ptr<T>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(x1));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (dropout == 0.0) {
    DISPATCH_ELEMENTWISE_CUDA_KERNEL(SwiGLUBwdKernel, T, T_ACC, size,
                                     cuda_stream, size, y_grad_data, x1_data,
                                     x2_data, x1_grad_data, x2_grad_data);
  } else {
    at::PhiloxCudaState rng_engine_inputs(seed, offset);
    DISPATCH_ELEMENTWISE_CUDA_KERNEL(SwiGLUDropoutBwdKernel, T, T_ACC, size,
                                     cuda_stream, rng_engine_inputs, size,
                                     y_grad_data, x1_data, x2_data, dropout,
                                     x1_grad_data, x2_grad_data);
  }
}

#undef DISPATCH_ELEMENTWISE_CUDA_KERNEL

}  // namespace

std::tuple<torch::Tensor, int64_t, int64_t> SwiGLUCUDAFwd(
    const torch::Tensor& x1, const torch::Tensor& x2, double dropout) {
  TORCH_CHECK(x1.sizes() == x2.sizes());
  torch::Tensor y = torch::empty_like(
      x1, x1.options().memory_format(at::MemoryFormat::Contiguous));
  int64_t seed = -1;
  int64_t offset = -1;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x1.scalar_type(), "SwiGLUCUDAFwd", [&]() {
        SwiGLUCUDAFwdImpl<scalar_t>(*(x1.expect_contiguous()),
                                    *(x2.expect_contiguous()), dropout, y, seed,
                                    offset);
      });

  return std::make_tuple(y, seed, offset);
}

std::tuple<torch::Tensor, torch::Tensor> SwiGLUCUDABwd(
    const torch::Tensor& y_grad, const torch::Tensor& x1,
    const torch::Tensor& x2, double dropout, int64_t seed, int64_t offset) {
  TORCH_CHECK(y_grad.sizes() == x1.sizes());
  TORCH_CHECK(x1.sizes() == x2.sizes());

  torch::Tensor x1_grad = torch::empty_like(
      x1, x1.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor x2_grad = torch::empty_like(
      x2, x2.options().memory_format(at::MemoryFormat::Contiguous));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x1.scalar_type(), "SwiGLUCUDABwd", [&]() {
        SwiGLUCUDABwdImpl<scalar_t>(
            *(y_grad.expect_contiguous()), *(x1.expect_contiguous()),
            *(x2.expect_contiguous()), dropout, seed, offset, x1_grad, x2_grad);
      });

  return std::make_tuple(x1_grad, x2_grad);
}

}  // namespace ops
}  // namespace mega2
