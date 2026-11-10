# Provenance map: dk-cuda-ffi ← pytorch @ 540f3a53054 (v2.14.0a0)

Transcribed, not `#include`d: each C++ function body is copied from the
upstream file with the c10 macro/type layer replaced by `cpp/shim/` (see the
substitution tables in `dk_macros.h` / `dk_types.h`). Complex-dtype
instantiations are deleted, not shimmed. Global-context flags (TF32,
deterministic, benchmark) become explicit parameters.

License: BSD-3-Clause (`cpp/LICENSE.PYTORCH` → `cpp/LICENSE.pytorch`).

| dk-cuda-ffi | upstream (pytorch) | status |
|---|---|---|
| `cpp/ping.{h,cpp}` | — (original) | Phase 0 |
| `cpp/shim/dk_macros.h` | replaces `c10/util/Exception.h`, `ATen/cuda/Exceptions.h` macros | Phase 0 |
| `cpp/shim/dk_types.h` | replaces `c10/util/Half.h`, `BFloat16.h`, `ATen/OpMathType.h` | Phase 0 |
| `cpp/shim/dk_handles.{h,cpp}` | `aten/src/ATen/cudnn/Handle.{h,cpp}`, `ATen/cuda/CublasHandlePool.cpp` (pattern) | Phase 5 |
| `cpp/blas.{h,cpp}` | `aten/src/ATen/cuda/CUDABlas.{h,cpp}` (gemm/bgemm/gemm_and_bias/gemv/dot/trsm/trsmBatched/getrfBatched/getrsBatched; macro-expanded per dtype, complex dropped) | Phase 5 |
| `cpp/descriptors.{h,cpp}` | `aten/src/ATen/cudnn/Descriptors.{h,cpp}` | Phase 5 |
| `cpp/conv.{h,cpp}` | `aten/src/ATen/native/cudnn/Conv_v8.cpp:1463,1508,1550` (`raw_cudnn_convolution_{forward,backward_input,backward_weight}_out`), Tensor& params rewritten to ptr+sizes+strides+dtype | Phase 5 |
| `cpp/batchnorm.{h,cpp}` | authored raw layer; reference `aten/src/ATen/native/cudnn/BatchNorm.cpp:227,255,408` | Phase 5 |
| `cpp/vendor/cudnn_frontend/` | `third_party/cudnn_frontend` at pytorch's pinned commit (MIT) | Phase 5 |

Rust-side ports (logic, not C++):

| dk-cuda-ffi | upstream | status |
|---|---|---|
| `src/blas.rs` strides→(transa,transb,lda) normalizer | `aten/src/ATen/native/cuda/cuBlasCommonArgs.h` | Phase 5 |
