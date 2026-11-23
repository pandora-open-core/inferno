// dk_macros.h — the c10 replacement layer for transcribed ATen code.
//
// During transcription (see PORTED_FROM.md) these substitute textually:
//   TORCH_CHECK(cond, msg...)        -> DK_CHECK(cond, msg)
//   AT_CUDA_CHECK / C10_CUDA_CHECK   -> DK_CUDA_CHECK(expr)
//   TORCH_CUDABLAS_CHECK             -> DK_CUBLAS_CHECK(expr)
//   AT_CUDNN_CHECK                   -> DK_CUDNN_CHECK(expr)
//
// All failures throw std::runtime_error; cxx maps exceptions crossing the
// bridge into Rust `Result::Err`.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#define DK_CHECK(cond, msg)                                                    \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::ostringstream dk_oss__;                                             \
      dk_oss__ << "dk-cuda-ffi check failed: " << (msg) << " (" << #cond       \
               << ") at " << __FILE__ << ":" << __LINE__;                      \
      throw std::runtime_error(dk_oss__.str());                                \
    }                                                                          \
  } while (0)

#define DK_CUDA_CHECK(expr)                                                    \
  do {                                                                         \
    cudaError_t dk_err__ = (expr);                                             \
    if (dk_err__ != cudaSuccess) {                                             \
      std::ostringstream dk_oss__;                                             \
      dk_oss__ << "CUDA error: " << cudaGetErrorString(dk_err__) << " from "   \
               << #expr << " at " << __FILE__ << ":" << __LINE__;              \
      throw std::runtime_error(dk_oss__.str());                                \
    }                                                                          \
  } while (0)

#define DK_CUBLAS_CHECK(expr)                                                  \
  do {                                                                         \
    cublasStatus_t dk_st__ = (expr);                                           \
    if (dk_st__ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream dk_oss__;                                             \
      dk_oss__ << "cuBLAS error " << static_cast<int>(dk_st__) << " from "     \
               << #expr << " at " << __FILE__ << ":" << __LINE__;              \
      throw std::runtime_error(dk_oss__.str());                                \
    }                                                                          \
  } while (0)

#define DK_CUDNN_CHECK(expr)                                                   \
  do {                                                                         \
    cudnnStatus_t dk_st__ = (expr);                                            \
    if (dk_st__ != CUDNN_STATUS_SUCCESS) {                                     \
      std::ostringstream dk_oss__;                                             \
      dk_oss__ << "cuDNN error: " << cudnnGetErrorString(dk_st__) << " from "  \
               << #expr << " at " << __FILE__ << ":" << __LINE__;              \
      throw std::runtime_error(dk_oss__.str());                                \
    }                                                                          \
  } while (0)
