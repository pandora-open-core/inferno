// dk_types.h — type substitutions for transcribed ATen code.
//
//   at::Half      -> __half          (cuda_fp16.h)
//   at::BFloat16  -> __nv_bfloat16   (cuda_bf16.h)
//   c10::complex  -> DROPPED — no curated op needs complex; the ATen macro
//                    instantiations for complex dtypes are deleted during
//                    transcription rather than shimmed.
//   at::opmath_type<T> -> dk::opmath_t<T> below.
//
// Global-context flags (TF32, deterministic, benchmark) do NOT get a global
// here: they become explicit function parameters at every entry point, per
// the port plan.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dk {

// ATen's opmath: reduced-precision types accumulate in float.
template <typename T> struct opmath {
  using type = T;
};
template <> struct opmath<__half> {
  using type = float;
};
template <> struct opmath<__nv_bfloat16> {
  using type = float;
};
template <typename T> using opmath_t = typename opmath<T>::type;

// dtype tags crossing the cxx bridge as u8 (must match src/lib.rs DkDtype).
enum class Dtype : uint8_t {
  F32 = 0,
  F64 = 1,
  F16 = 2,
  BF16 = 3,
};

} // namespace dk
