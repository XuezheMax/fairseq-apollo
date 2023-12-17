#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <ATen/cuda/detail/UnpackRaw.cuh>
#include <tuple>

namespace mega2 {
namespace random_utils {

constexpr int64_t kRandomUnroll = 4;

// Copied from
// https://github.com/pytorch/pytorch/blob/d1c092ae1b03d10dc5264383ba6002fa7d8ffdf4/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L16
// at::cuda::philox::unpack is a device only function before PyTorch 2.1.
// TODO: Remove this later.
__inline__ std::tuple<uint64_t, uint64_t> HostUnpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to
    // "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire
    // kernel. For most threads' reads it will hit in cache, so it shouldn't
    // hurt performance.
    return std::make_tuple(
        static_cast<uint64_t>(*arg.seed_.ptr),
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

}  // namespace random_utils
}  // namespace mega2
