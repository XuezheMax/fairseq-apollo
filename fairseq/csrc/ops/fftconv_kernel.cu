#include <ATen/AccumulateType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <type_traits>
#include <vector>

#include "cuda_utils.cuh"
#include "fft.cuh"
#include "ops/fftconv.h"
#include "utils.h"

namespace mega2 {
namespace ops {

namespace {

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__global__ void RFFTCUDAKernel(const T* __restrict__ X, int N, bool flip,
                               c10::complex<T_ACC>* __restrict__ Y) {
  constexpr int kRFFTSize = kFFTSize + 1;

  extern __shared__ float shared_mem[];
  c10::complex<T_ACC>* shared_data =
      reinterpret_cast<c10::complex<T_ACC>*>(shared_mem);

  const int b = blockIdx.x;
  const T* X_data = X + b * N;
  c10::complex<T_ACC>* Y_data = Y + b * kRFFTSize;

  fft::BlockRFFT<T, T_ACC, kFFTSize, kNumThreads>(X_data, N, flip, Y_data,
                                                  shared_data);
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__global__ void FFTConvCUDAFwdKernel(
    int H, int L, const T* __restrict__ X,
    const c10::complex<T_ACC>* __restrict__ K_f, T* __restrict__ Y,
    c10::complex<T_ACC>* __restrict__ X_f) {
  constexpr int kRFFTSize = kFFTSize + 1;
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;

  extern __shared__ float shared_mem[];
  c10::complex<T_ACC>* shared_data =
      reinterpret_cast<c10::complex<T_ACC>*>(shared_mem);

  const int b = blockIdx.y;
  const int h = blockIdx.x;

  const T* X_data = X + (b * H + h) * L;
  const c10::complex<T_ACC>* K_f_data = K_f + h * kRFFTSize;
  T* Y_data = Y + (b * H + h) * L;
  c10::complex<T_ACC>* X_f_data = X_f + (b * H + h) * kRFFTSize;

  fft::BlockRFFT<T, T_ACC, kFFTSize, kNumThreads>(
      X_data, L, /*flip=*/false, shared_data, shared_data + kFFTSize);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    X_f_data[idx] = shared_data[idx];
    shared_data[idx] *= K_f_data[idx];
  }
  if (threadIdx.x == 0) {
    X_f_data[kFFTSize] = shared_data[kFFTSize];
    shared_data[kFFTSize] *= K_f_data[kFFTSize];
  }
  __syncthreads();

  fft::BlockIRFFT<T, T_ACC, kFFTSize, kNumThreads>(
      shared_data, L, /*flip=*/false, Y_data, shared_data + kFFTSize);
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__global__ void FFTConvCUDABwdKernel(
    int H, int L, const T* __restrict__ Y_grad,
    const c10::complex<T_ACC>* __restrict__ X_f,
    const c10::complex<T_ACC>* __restrict__ K_f, T* __restrict__ X_grad,
    c10::complex<T_ACC>* __restrict__ K_grad_f) {
  constexpr int kRFFTSize = kFFTSize + 1;
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;

  extern __shared__ float shared_mem[];
  c10::complex<T_ACC>* shared_data =
      reinterpret_cast<c10::complex<T_ACC>*>(shared_mem);

  const int b = blockIdx.y;
  const int h = blockIdx.x;

  const T* Y_grad_data = Y_grad + (b * H + h) * L;
  const c10::complex<T_ACC>* X_f_data = X_f + (b * H + h) * kRFFTSize;
  const c10::complex<T_ACC>* K_f_data = K_f + h * kRFFTSize;
  T* X_grad_data = X_grad + (b * H + h) * L;
  c10::complex<T_ACC>* K_grad_f_data = K_grad_f + (b * H + h) * kRFFTSize;

  fft::BlockRFFT<T, T_ACC, kFFTSize, kNumThreads>(
      Y_grad_data, L, /*flip=*/true, shared_data, shared_data + kFFTSize);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    K_grad_f_data[idx] = shared_data[idx] * X_f_data[idx];
    shared_data[idx] *= K_f_data[idx];
  }
  if (threadIdx.x == 0) {
    K_grad_f_data[kFFTSize] = shared_data[kFFTSize] * X_f_data[kFFTSize];
    shared_data[kFFTSize] *= K_f_data[kFFTSize];
  }
  __syncthreads();

  fft::BlockIRFFT<T, T_ACC, kFFTSize, kNumThreads>(
      shared_data, L, /*flip=*/true, X_grad_data, shared_data + kFFTSize);
}

template <typename T, typename T_ACC, int kFFTSize, int kNumThreads>
__global__ void FFTConvKernelCUDABwdKernel(
    int B, int H, int L, const c10::complex<T_ACC>* __restrict__ K_grad_f,
    T* __restrict__ K_grad) {
  constexpr int kRFFTSize = kFFTSize + 1;
  constexpr int kElementsPerThread = kFFTSize / kNumThreads;
  constexpr c10::complex<T_ACC> kZero(T_ACC(0), T_ACC(0));

  extern __shared__ float shared_mem[];
  c10::complex<T_ACC>* shared_data =
      reinterpret_cast<c10::complex<T_ACC>*>(shared_mem);

  const int h = blockIdx.x;

  T* K_grad_data = K_grad + h * L;

#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    const int idx = i * blockDim.x + threadIdx.x;
    shared_data[idx] = kZero;
  }
  if (threadIdx.x == 0) {
    shared_data[kFFTSize] = kZero;
  }
  for (int b = 0; b < B; ++b) {
    const c10::complex<T_ACC>* K_grad_f_data =
        K_grad_f + (b * H + h) * kRFFTSize;
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      const int idx = i * blockDim.x + threadIdx.x;
      shared_data[idx] += K_grad_f_data[idx];
    }
    if (threadIdx.x == 0) {
      shared_data[kFFTSize] += K_grad_f_data[kFFTSize];
    }
  }
  __syncthreads();

  fft::BlockIRFFT<T, T_ACC, kFFTSize, kNumThreads>(
      shared_data, L, /*flip=*/true, K_grad_data, shared_data + kFFTSize);
}

int64_t ComputeFFTSize(int64_t N) {
  return std::max((int64_t(1) << utils::CeilLog2(N)), cuda_utils::kWarpSize);
}

template <typename T>
void RFFTCUDAImpl(const torch::Tensor& X, bool flip, torch::Tensor& Y) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t N = X.size(-1);
  const int64_t B = X.numel() / N;
  const int64_t fft_size = ComputeFFTSize(N);

  const T* X_data = X.data_ptr<T>();
  c10::complex<T_ACC>* Y_data = Y.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(X));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shared_size = fft_size * sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 32, 32>, B, 32,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 64, 32>, B, 32,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 128, 64>, B, 64,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 256, 128>, B, 128,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 512, 256>, B, 256,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 1024, 512>, B, 512,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 2048, 1024>, B, 1024,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 4096, 1024>, B, 1024,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(RFFTCUDAKernel<T, T_ACC, 8192, 1024>, B, 1024,
                             shared_size, cuda_stream, X_data, N, flip, Y_data);
  }
}

template <typename T>
void FFTConvCUDAFwdImpl(const torch::Tensor& X, const torch::Tensor& K_f,
                        torch::Tensor& Y, torch::Tensor& X_f) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = X.size(0);
  const int64_t H = X.size(1);
  const int64_t L = X.size(2);
  const int64_t fft_size = ComputeFFTSize(L);

  const T* X_data = X.data_ptr<T>();
  const c10::complex<T_ACC>* K_f_data = K_f.data_ptr<c10::complex<T_ACC>>();
  T* Y_data = Y.data_ptr<T>();
  c10::complex<T_ACC>* X_f_data = X_f.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(X));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shared_size = (fft_size * 2) * sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 32, 32>, dim3(H, B),
                             32, shared_size, cuda_stream, H, L, X_data,
                             K_f_data, Y_data, X_f_data);
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 64, 32>, dim3(H, B),
                             32, shared_size, cuda_stream, H, L, X_data,
                             K_f_data, Y_data, X_f_data);
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 128, 64>,
                             dim3(H, B), 64, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 256, 128>,
                             dim3(H, B), 128, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 512, 256>,
                             dim3(H, B), 256, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 1024, 512>,
                             dim3(H, B), 512, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 2048, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 4096, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(FFTConvCUDAFwdKernel<T, T_ACC, 8192, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             X_data, K_f_data, Y_data, X_f_data);
  }
}

template <typename T>
void FFTConvCUDABwdImpl(const torch::Tensor& Y_grad, const torch::Tensor& X_f,
                        const torch::Tensor& K_f, const torch::Dtype& K_dtype,
                        torch::Tensor& X_grad, torch::Tensor& K_grad) {
  using T_ACC = at::acc_type<T, true>;

  const int64_t B = Y_grad.size(0);
  const int64_t H = Y_grad.size(1);
  const int64_t L = Y_grad.size(2);
  const int64_t fft_size = X_f.size(2) - 1;

  torch::Tensor K_grad_f = torch::empty_like(X_f);

  const T* Y_grad_data = Y_grad.data_ptr<T>();
  const c10::complex<T_ACC>* X_f_data = X_f.data_ptr<c10::complex<T_ACC>>();
  const c10::complex<T_ACC>* K_f_data = K_f.data_ptr<c10::complex<T_ACC>>();
  T* X_grad_data = X_grad.data_ptr<T>();
  c10::complex<T_ACC>* K_grad_f_data = K_grad_f.data_ptr<c10::complex<T_ACC>>();

  at::cuda::OptionalCUDAGuard guard(at::device_of(X_f));
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  const int64_t shared_size = (fft_size * 2) * sizeof(c10::complex<T_ACC>);

  if (fft_size == 32) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 32, 32>, dim3(H, B),
                             32, shared_size, cuda_stream, H, L, Y_grad_data,
                             X_f_data, K_f_data, X_grad_data, K_grad_f_data);

    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 32, 32>, H,
                               32, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 32, 32>,
                               H, 32, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 64) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 64, 32>, dim3(H, B),
                             32, shared_size, cuda_stream, H, L, Y_grad_data,
                             X_f_data, K_f_data, X_grad_data, K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 64, 32>, H,
                               32, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 64, 32>,
                               H, 32, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 128) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 128, 64>,
                             dim3(H, B), 64, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 128, 64>, H,
                               64, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());

    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 128, 64>, H, 64, shared_size,
          cuda_stream, B, H, L, K_grad_f_data, K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 256) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 256, 128>,
                             dim3(H, B), 128, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 256, 128>,
                               H, 128, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 256, 128>, H, 128,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 512) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 512, 256>,
                             dim3(H, B), 256, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 512, 256>,
                               H, 256, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 512, 256>, H, 256,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 1024) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 1024, 512>,
                             dim3(H, B), 512, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 1024, 512>,
                               H, 512, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 1024, 512>, H, 512,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 2048) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 2048, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 2048, 1024>,
                               H, 1024, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 2048, 1024>, H, 1024,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 4096) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 4096, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 4096, 1024>,
                               H, 1024, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 4096, 1024>, H, 1024,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  } else if (fft_size == 8192) {
    cuda_utils::LaunchKernel(FFTConvCUDABwdKernel<T, T_ACC, 8192, 1024>,
                             dim3(H, B), 1024, shared_size, cuda_stream, H, L,
                             Y_grad_data, X_f_data, K_f_data, X_grad_data,
                             K_grad_f_data);
    if (K_dtype == Y_grad.scalar_type()) {
      cuda_utils::LaunchKernel(FFTConvKernelCUDABwdKernel<T, T_ACC, 8192, 1024>,
                               H, 1024, shared_size, cuda_stream, B, H, L,
                               K_grad_f_data, K_grad.data_ptr<T>());
    } else {
      cuda_utils::LaunchKernel(
          FFTConvKernelCUDABwdKernel<T_ACC, T_ACC, 8192, 1024>, H, 1024,
          shared_size, cuda_stream, B, H, L, K_grad_f_data,
          K_grad.data_ptr<T_ACC>());
    }
  }
}

}  // namespace

torch::Tensor RFFTCUDA(const torch::Tensor& X, bool flip) {
  std::vector<int64_t> sizes = X.sizes().vec();
  const int64_t L = sizes.back();
  const int64_t rfft_size = ComputeFFTSize(L) + 1;
  sizes.back() = rfft_size;
  TORCH_CHECK(L <= fft::kFFTMaxLength);
  const auto complex_type =
      X.scalar_type() == at::kDouble ? at::kComplexDouble : at::kComplexFloat;
  torch::Tensor Y =
      torch::empty(sizes, X.options()
                              .dtype(complex_type)
                              .memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "RFFT",
      [&]() { RFFTCUDAImpl<scalar_t>(*(X.expect_contiguous()), flip, Y); });
  return Y;
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDAFwd(
    const torch::Tensor& X, const torch::Tensor& K_f) {
  std::vector<int64_t> sizes = X.sizes().vec();
  const int64_t L = sizes.back();
  const int64_t rfft_size = ComputeFFTSize(L) + 1;
  sizes.back() = rfft_size;
  TORCH_CHECK(L <= fft::kFFTMaxLength);
  const auto complex_type =
      X.scalar_type() == at::kDouble ? at::kComplexDouble : at::kComplexFloat;
  torch::Tensor Y = torch::empty_like(
      X, X.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor X_f =
      torch::empty(sizes, X.options()
                              .dtype(complex_type)
                              .memory_format(at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, X.scalar_type(), "FFTConvFwd", [&]() {
        FFTConvCUDAFwdImpl<scalar_t>(*(X.expect_contiguous()),
                                     *(K_f.expect_contiguous()), Y, X_f);
      });
  return std::make_tuple(Y, X_f);
}

std::tuple<torch::Tensor, torch::Tensor> FFTConvCUDABwd(
    const torch::Tensor& Y_grad, const torch::Tensor& X_f,
    const torch::Tensor& K_f, const torch::Dtype& K_dtype) {
  const int64_t H = Y_grad.size(1);
  const int64_t L = Y_grad.size(2);
  torch::Tensor X_grad = torch::empty_like(
      Y_grad, Y_grad.options().memory_format(at::MemoryFormat::Contiguous));
  torch::Tensor K_grad =
      torch::empty({H, L}, Y_grad.options().dtype(K_dtype).memory_format(
                               at::MemoryFormat::Contiguous));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, Y_grad.scalar_type(), "FFTConvBwd", [&]() {
        FFTConvCUDABwdImpl<scalar_t>(
            *(Y_grad.expect_contiguous()), *(X_f.expect_contiguous()),
            *(K_f.expect_contiguous()), K_dtype, X_grad, K_grad);
      });
  return std::make_tuple(X_grad, K_grad);
}

}  // namespace ops
}  // namespace mega2
