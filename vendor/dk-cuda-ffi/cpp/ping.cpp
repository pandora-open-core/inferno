#include "cpp/ping.h"
#include "cpp/shim/dk_macros.h"

namespace dk {
namespace cuda {

std::int32_t ping() {
  int version = 0;
  DK_CUDA_CHECK(cudaRuntimeGetVersion(&version));
  return static_cast<std::int32_t>(version);
}

} // namespace cuda
} // namespace dk
