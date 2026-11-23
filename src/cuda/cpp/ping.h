// Phase-0 link smoke test: proves the cxx bridge compiles, the CUDA headers
// resolve, and libcudart links. Returns the CUDA runtime version (e.g. 12040)
// or throws.
#pragma once

#include <cstdint>

namespace dk {
namespace cuda {

std::int32_t ping();

} // namespace cuda
} // namespace dk
