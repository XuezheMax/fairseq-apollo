#include "blas.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/complex.h>
#include <cublasLt.h>

#include <type_traits>

namespace mega2 {
namespace blas {

namespace {

constexpr cublasOperation_t ToCuBLASOp(TransposeOp op) {
  switch (op) {
    case TransposeOp::kN: {
      return CUBLAS_OP_N;
    }
    case TransposeOp::kT: {
      return CUBLAS_OP_T;
    }
    case TransposeOp::kC: {
      return CUBLAS_OP_C;
    }
    default: {
      TORCH_CHECK(false);
    }
  }
}

}  // namespace

template <>
void GemmCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                     TransposeOp transb, int64_t m, int64_t n, int64_t k,
                     float alpha, const float* a, int64_t lda, const float* b,
                     int64_t ldb, float beta, float* c, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasSgemm(handle, ToCuBLASOp(transb),
                                   ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
                                   a, lda, &beta, c, ldc));
}

template <>
void GemmCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                      TransposeOp transb, int64_t m, int64_t n, int64_t k,
                      double alpha, const double* a, int64_t lda,
                      const double* b, int64_t ldb, double beta, double* c,
                      int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasDgemm(handle, ToCuBLASOp(transb),
                                   ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
                                   a, lda, &beta, c, ldc));
}

template <>
void GemmCUDA<at::Half>(cublasHandle_t handle, TransposeOp transa,
                        TransposeOp transb, int64_t m, int64_t n, int64_t k,
                        float alpha, const at::Half* a, int64_t lda,
                        const at::Half* b, int64_t ldb, float beta, at::Half* c,
                        int64_t ldc) {
  TORCH_CUDABLAS_CHECK(
      cublasGemmEx(handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
                   &alpha, b, CUDA_R_16F, ldb, a, CUDA_R_16F, lda, &beta, c,
                   CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmCUDA<at::BFloat16>(cublasHandle_t handle, TransposeOp transa,
                            TransposeOp transb, int64_t m, int64_t n, int64_t k,
                            float alpha, const at::BFloat16* a, int64_t lda,
                            const at::BFloat16* b, int64_t ldb, float beta,
                            at::BFloat16* c, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(
      cublasGemmEx(handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
                   &alpha, b, CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda, &beta, c,
                   CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Adapted from
// https://github.com/pytorch/pytorch/blob/3cb6cf1e8aa594268fbb8ca365c77c695d828e76/aten/src/ATen/cuda/CUDABlas.cpp#L677.
template <typename T>
void GemmAndBiasCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                     TransposeOp transa, TransposeOp transb, int64_t m,
                     int64_t n, int64_t k, at::opmath_type<T> alpha, const T* a,
                     int64_t lda, const T* b, int64_t ldb, const T* bias,
                     at::opmath_type<T> beta, T* c, int64_t ldc) {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};

  cudaDataType_t data_type = CUDA_R_32F;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  if constexpr (std::is_same<T, float>::value) {
    if (at::globalContext().allowTF32CuBLAS()) {
      compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
  } else if constexpr (std::is_same<T, double>::value) {
    data_type = CUDA_R_64F;
    scale_type = CUDA_R_64F;
    compute_type = CUBLAS_COMPUTE_64F;
  } else if constexpr (std::is_same<T, at::Half>::value) {
    data_type = CUDA_R_16F;
  } else if constexpr (std::is_same<T, at::BFloat16>::value) {
    data_type = CUDA_R_16BF;
  }

  const cublasOperation_t cublas_transa = ToCuBLASOp(transa);
  const cublasOperation_t cublas_transb = ToCuBLASOp(transb);
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &cublas_transb, sizeof(cublas_transb)));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &cublas_transa, sizeof(cublas_transa)));

  if (bias != nullptr) {
    constexpr cublasLtEpilogue_t kEpilogue = CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &kEpilogue, sizeof(kEpilogue)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }

  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &a_desc, data_type, cublas_transa == CUBLAS_OP_N ? k : m,
      cublas_transa == CUBLAS_OP_N ? m : k, lda));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_desc, data_type, cublas_transb == CUBLAS_OP_N ? n : k,
      cublas_transb == CUBLAS_OP_N ? k : n, ldb));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&c_desc, data_type, n, m, ldc));

  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize,
      sizeof(kWorkspaceSize)));

  auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  auto workspace = cuda_allocator->allocate(kWorkspaceSize);

  cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(handle);
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt_handle, op_desc, b_desc, a_desc, c_desc, c_desc, preference, 1,
      &heuristic_result, &returned_results));
  if (returned_results == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  TORCH_CUDABLAS_CHECK(cublasLtMatmul(lt_handle, op_desc, &alpha, b, b_desc, a,
                                      a_desc, &beta, c, c_desc, c, c_desc,
                                      &heuristic_result.algo, workspace.get(),
                                      kWorkspaceSize, cuda_stream));

  if (preference != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  }
  if (c_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  }
  if (b_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  }
  if (a_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  }
  if (op_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  }
}

template void GemmAndBiasCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                              TransposeOp transa, TransposeOp transb, int64_t m,
                              int64_t n, int64_t k, float alpha, const float* a,
                              int64_t lda, const float* b, int64_t ldb,
                              const float* bias, float beta, float* c,
                              int64_t ldc);

template void GemmAndBiasCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                              TransposeOp transa, TransposeOp transb, int64_t m,
                              int64_t n, int64_t k, double alpha,
                              const double* a, int64_t lda, const double* b,
                              int64_t ldb, const double* bias, double beta,
                              double* c, int64_t ldc);

template void GemmAndBiasCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                              TransposeOp transa, TransposeOp transb, int64_t m,
                              int64_t n, int64_t k, float alpha,
                              const at::Half* a, int64_t lda, const at::Half* b,
                              int64_t ldb, const at::Half* bias, float beta,
                              at::Half* c, int64_t ldc);

template void GemmAndBiasCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                              TransposeOp transa, TransposeOp transb, int64_t m,
                              int64_t n, int64_t k, float alpha,
                              const at::BFloat16* a, int64_t lda,
                              const at::BFloat16* b, int64_t ldb,
                              const at::BFloat16* bias, float beta,
                              at::BFloat16* c, int64_t ldc);

template <typename T>
void GemmAndBiasGradCUDA(cublasHandle_t handle, cudaStream_t cuda_stream,
                         TransposeOp transa, TransposeOp transb, int64_t m,
                         int64_t n, int64_t k, at::opmath_type<T> alpha,
                         const T* a, int64_t lda, const T* b, int64_t ldb,
                         at::opmath_type<T> beta, T* c, int64_t ldc,
                         T* bias_grad) {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};

  cudaDataType_t data_type = CUDA_R_32F;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  if constexpr (std::is_same<T, float>::value) {
    if (at::globalContext().allowTF32CuBLAS()) {
      compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
  } else if constexpr (std::is_same<T, double>::value) {
    data_type = CUDA_R_64F;
    scale_type = CUDA_R_64F;
    compute_type = CUBLAS_COMPUTE_64F;
  } else if constexpr (std::is_same<T, at::Half>::value) {
    data_type = CUDA_R_16F;
  } else if constexpr (std::is_same<T, at::BFloat16>::value) {
    data_type = CUDA_R_16BF;
  }

  const cublasOperation_t cublas_transa = ToCuBLASOp(transa);
  const cublasOperation_t cublas_transb = ToCuBLASOp(transb);
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &cublas_transb, sizeof(cublas_transb)));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &cublas_transa, sizeof(cublas_transa)));

  if (bias_grad != nullptr) {
    constexpr cublasLtEpilogue_t kEpilogue = CUBLASLT_EPILOGUE_BGRADB;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &kEpilogue, sizeof(kEpilogue)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_grad,
        sizeof(bias_grad)));
  }

  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &a_desc, data_type, cublas_transa == CUBLAS_OP_N ? k : m,
      cublas_transa == CUBLAS_OP_N ? m : k, lda));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_desc, data_type, cublas_transb == CUBLAS_OP_N ? n : k,
      cublas_transb == CUBLAS_OP_N ? k : n, ldb));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&c_desc, data_type, n, m, ldc));

  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize,
      sizeof(kWorkspaceSize)));

  auto* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  auto workspace = cuda_allocator->allocate(kWorkspaceSize);

  cublasLtHandle_t lt_handle = reinterpret_cast<cublasLtHandle_t>(handle);
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt_handle, op_desc, b_desc, a_desc, c_desc, c_desc, preference, 1,
      &heuristic_result, &returned_results));
  if (returned_results == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  TORCH_CUDABLAS_CHECK(cublasLtMatmul(lt_handle, op_desc, &alpha, b, b_desc, a,
                                      a_desc, &beta, c, c_desc, c, c_desc,
                                      &heuristic_result.algo, workspace.get(),
                                      kWorkspaceSize, cuda_stream));

  if (preference != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  }
  if (c_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  }
  if (b_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  }
  if (a_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  }
  if (op_desc != nullptr) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  }
}

template void GemmAndBiasGradCUDA(cublasHandle_t handle,
                                  cudaStream_t cuda_stream, TransposeOp transa,
                                  TransposeOp transb, int64_t m, int64_t n,
                                  int64_t k, float alpha, const float* a,
                                  int64_t lda, const float* b, int64_t ldb,
                                  float beta, float* c, int64_t ldc,
                                  float* bias_grad);

template void GemmAndBiasGradCUDA(cublasHandle_t handle,
                                  cudaStream_t cuda_stream, TransposeOp transa,
                                  TransposeOp transb, int64_t m, int64_t n,
                                  int64_t k, double alpha, const double* a,
                                  int64_t lda, const double* b, int64_t ldb,
                                  double beta, double* c, int64_t ldc,
                                  double* bias_grad);

template void GemmAndBiasGradCUDA(cublasHandle_t handle,
                                  cudaStream_t cuda_stream, TransposeOp transa,
                                  TransposeOp transb, int64_t m, int64_t n,
                                  int64_t k, float alpha, const at::Half* a,
                                  int64_t lda, const at::Half* b, int64_t ldb,
                                  float beta, at::Half* c, int64_t ldc,
                                  at::Half* bias_grad);

template void GemmAndBiasGradCUDA(cublasHandle_t handle,
                                  cudaStream_t cuda_stream, TransposeOp transa,
                                  TransposeOp transb, int64_t m, int64_t n,
                                  int64_t k, float alpha, const at::BFloat16* a,
                                  int64_t lda, const at::BFloat16* b,
                                  int64_t ldb, float beta, at::BFloat16* c,
                                  int64_t ldc, at::BFloat16* bias_grad);

template <>
void GemmBatchedCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                            TransposeOp transb, int64_t batch_size, int64_t m,
                            int64_t n, int64_t k, float alpha,
                            const float** a_array, int64_t lda,
                            const float** b_array, int64_t ldb, float beta,
                            float** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasSgemmBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b_array,
      ldb, a_array, lda, &beta, c_array, ldc, batch_size));
}

template <>
void GemmStridedBatchedCUDA<float>(cublasHandle_t handle, TransposeOp transa,
                                   TransposeOp transb, int64_t batch_size,
                                   int64_t m, int64_t n, int64_t k, float alpha,
                                   const float* a, int64_t lda,
                                   int64_t batch_stride_a, const float* b,
                                   int64_t ldb, int64_t batch_stride_b,
                                   float beta, float* c, int64_t ldc,
                                   int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
      batch_stride_b, a, lda, batch_stride_a, &beta, c, ldc, batch_stride_c,
      batch_size));
}

template <>
void GemmBatchedCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                             TransposeOp transb, int64_t batch_size, int64_t m,
                             int64_t n, int64_t k, double alpha,
                             const double** a_array, int64_t lda,
                             const double** b_array, int64_t ldb, double beta,
                             double** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasDgemmBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b_array,
      ldb, a_array, lda, &beta, c_array, ldc, batch_size));
}

template <>
void GemmStridedBatchedCUDA<double>(cublasHandle_t handle, TransposeOp transa,
                                    TransposeOp transb, int64_t batch_size,
                                    int64_t m, int64_t n, int64_t k,
                                    double alpha, const double* a, int64_t lda,
                                    int64_t batch_stride_a, const double* b,
                                    int64_t ldb, int64_t batch_stride_b,
                                    double beta, double* c, int64_t ldc,
                                    int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b, ldb,
      batch_stride_b, a, lda, batch_stride_a, &beta, c, ldc, batch_stride_c,
      batch_size));
}

template <>
void GemmBatchedCUDA<at::Half>(cublasHandle_t handle, TransposeOp transa,
                               TransposeOp transb, int64_t batch_size,
                               int64_t m, int64_t n, int64_t k, float alpha,
                               const at::Half** a_array, int64_t lda,
                               const at::Half** b_array, int64_t ldb,
                               float beta, at::Half** c_array, int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha,
      reinterpret_cast<const void**>(b_array), CUDA_R_16F, ldb,
      reinterpret_cast<const void**>(a_array), CUDA_R_16F, lda, &beta,
      reinterpret_cast<void**>(c_array), CUDA_R_16F, ldc, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<at::Half>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha,
    const at::Half* a, int64_t lda, int64_t batch_stride_a, const at::Half* b,
    int64_t ldb, int64_t batch_stride_b, float beta, at::Half* c, int64_t ldc,
    int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b,
      CUDA_R_16F, ldb, batch_stride_b, a, CUDA_R_16F, lda, batch_stride_a,
      &beta, c, CUDA_R_16F, ldc, batch_stride_c, batch_size, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmBatchedCUDA<at::BFloat16>(cublasHandle_t handle, TransposeOp transa,
                                   TransposeOp transb, int64_t batch_size,
                                   int64_t m, int64_t n, int64_t k, float alpha,
                                   const at::BFloat16** a_array, int64_t lda,
                                   const at::BFloat16** b_array, int64_t ldb,
                                   float beta, at::BFloat16** c_array,
                                   int64_t ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha,
      reinterpret_cast<const void**>(b_array), CUDA_R_16BF, ldb,
      reinterpret_cast<const void**>(a_array), CUDA_R_16BF, lda, &beta,
      reinterpret_cast<void**>(c_array), CUDA_R_16BF, ldc, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<at::BFloat16>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k, float alpha,
    const at::BFloat16* a, int64_t lda, int64_t batch_stride_a,
    const at::BFloat16* b, int64_t ldb, int64_t batch_stride_b, float beta,
    at::BFloat16* c, int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k, &alpha, b,
      CUDA_R_16BF, ldb, batch_stride_b, a, CUDA_R_16BF, lda, batch_stride_a,
      &beta, c, CUDA_R_16BF, ldc, batch_stride_c, batch_size,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <>
void GemmStridedBatchedCUDA<c10::complex<float>>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    c10::complex<float> alpha, const c10::complex<float>* a, int64_t lda,
    int64_t batch_stride_a, const c10::complex<float>* b, int64_t ldb,
    int64_t batch_stride_b, c10::complex<float> beta, c10::complex<float>* c,
    int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasCgemm3mStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
      reinterpret_cast<const cuComplex*>(&alpha),
      reinterpret_cast<const cuComplex*>(b), ldb, batch_stride_b,
      reinterpret_cast<const cuComplex*>(a), lda, batch_stride_a,
      reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, batch_stride_c, batch_size));
}

template <>
void GemmStridedBatchedCUDA<c10::complex<double>>(
    cublasHandle_t handle, TransposeOp transa, TransposeOp transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    c10::complex<double> alpha, const c10::complex<double>* a, int64_t lda,
    int64_t batch_stride_a, const c10::complex<double>* b, int64_t ldb,
    int64_t batch_stride_b, c10::complex<double> beta, c10::complex<double>* c,
    int64_t ldc, int64_t batch_stride_c) {
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, ToCuBLASOp(transb), ToCuBLASOp(transa), n, m, k,
      reinterpret_cast<const cuDoubleComplex*>(&alpha),
      reinterpret_cast<const cuDoubleComplex*>(b), ldb, batch_stride_b,
      reinterpret_cast<const cuDoubleComplex*>(a), lda, batch_stride_a,
      reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, batch_stride_c, batch_size));
}

}  // namespace blas
}  // namespace mega2
