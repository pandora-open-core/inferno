// Original DeepKernels driver adapter; no ATen/KeOps source transcription.
#include "cpp/pqo3.h"
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <cmath>
namespace dk { namespace cuda {
static void check(CUresult r) {
  if (r != CUDA_SUCCESS) {
    const char* name = nullptr; cuGetErrorName(r, &name);
    throw std::runtime_error(name ? name : "CUDA driver failure");
  }
}
static std::uint64_t page_stride(std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t dim, std::uint32_t page_tokens, std::uint32_t page_count, std::uint32_t tokens) {
  if (!heads || heads > 65536 || !kv_heads || heads % kv_heads ||
      (dim != 32 && dim != 64 && dim != 128 && dim != 256) ||
      !page_tokens || !page_count || page_count > 256 || !tokens ||
      std::uint64_t(tokens) > std::uint64_t(page_tokens) * page_count)
    throw std::runtime_error("invalid PQO3 geometry");
  const std::uint64_t bytes = std::uint64_t(2) * page_tokens * kv_heads * (dim / 32) * 14;
  if (bytes > (256 * 1024 * 1024) / page_count)
    throw std::runtime_error("PQO3 packed buffer exceeds bound");
  return bytes;
}
rust::String device_fingerprint(std::int32_t ordinal) {
  check(cuInit(0)); CUdevice device; check(cuDeviceGet(&device, ordinal));
  int major, minor, driver;
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  check(cuDriverGetVersion(&driver));
  return rust::String("sm_" + std::to_string(major) + std::to_string(minor) + ":driver=" + std::to_string(driver));
}
void launch_pqo3(rust::Slice<const std::uint8_t> image, std::uint64_t query,
                 std::uint64_t pages, std::uint64_t output, std::uint64_t stream,
                 std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
                 std::uint32_t page_tokens, std::uint32_t page_count, std::uint32_t tokens) {
  page_stride(heads, kv_heads, dim, page_tokens, page_count, tokens);
  if (!query || !pages || !output || image.size() < 4 || image.size() > 64 * 1024 * 1024 ||
      image[0] != 0x7f || image[1] != 'E' || image[2] != 'L' || image[3] != 'F')
    throw std::runtime_error("invalid PQO3 addresses or CUBIN image");
  CUcontext context; check(cuCtxGetCurrent(&context));
  if (!context) throw std::runtime_error("caller must bind CUDA context");
  CUmodule module; check(cuModuleLoadData(&module, image.data()));
  try {
    CUfunction kernel; check(cuModuleGetFunction(&kernel, module, "dk_pqo3_decode_attention"));
    void* args[] = {&query, &pages, &output, &heads, &kv_heads, &dim, &page_tokens, &tokens};
    check(cuLaunchKernel(kernel, heads, 1, 1, dim, 1, 1, 0,
                        reinterpret_cast<CUstream>(stream), args, nullptr));
    // Explicit synchronous API: device allocations and module remain live until
    // completion, including error paths. Async integration needs owned events.
    check(cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
  } catch (...) { cuModuleUnload(module); throw; }
  check(cuModuleUnload(module));
}
}}

#include <vector>
namespace dk { namespace cuda {
rust::Vec<float> pqo3_host(rust::Slice<const std::uint8_t> image,
  rust::Slice<const float> query, rust::Slice<const std::uint8_t> packed,
  std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
  std::uint32_t page_tokens, std::uint32_t page_count, std::uint32_t tokens,
  std::int32_t ordinal) {
  const std::uint64_t page_bytes = page_stride(heads, kv_heads, dim, page_tokens, page_count, tokens);
  if (query.size() != std::uint64_t(heads) * dim || packed.size() != page_bytes * page_count)
    throw std::runtime_error("PQO3 host buffer bounds");
  for (float value : query)
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite PQO3 query");
  // Reserve before retaining a context: allocation failure must not leak it.
  std::vector<CUdeviceptr> allocations;
  allocations.reserve(4);
  check(cuInit(0)); CUdevice device; check(cuDeviceGet(&device,ordinal));
  CUcontext previous, context; check(cuCtxGetCurrent(&previous));
  check(cuDevicePrimaryCtxRetain(&context,device));
  auto cleanup = [&]() { for (auto p: allocations) cuMemFree(p);
    cuCtxSetCurrent(previous); cuDevicePrimaryCtxRelease(device); };
  try {
    check(cuCtxSetCurrent(context));
    auto alloc = [&](std::size_t bytes) { CUdeviceptr p;check(cuMemAlloc(&p,bytes));allocations.push_back(p);return p; };
    CUdeviceptr q=alloc(query.size()*sizeof(float)), data=alloc(packed.size());
    CUdeviceptr out=alloc(query.size()*sizeof(float)), table=alloc(page_count*sizeof(CUdeviceptr));
    check(cuMemcpyHtoD(q,query.data(),query.size()*sizeof(float)));
    check(cuMemcpyHtoD(data,packed.data(),packed.size()));
    std::vector<CUdeviceptr> pointers;
    for (unsigned i=0;i<page_count;++i) pointers.push_back(data+i*page_bytes);
    check(cuMemcpyHtoD(table,pointers.data(),pointers.size()*sizeof(CUdeviceptr)));
    launch_pqo3(image,q,table,out,0,heads,kv_heads,dim,page_tokens,page_count,tokens);
    rust::Vec<float> result; result.reserve(query.size());
    for (std::size_t i=0;i<query.size();++i) result.push_back(0.f);
    check(cuMemcpyDtoH(result.data(),out,query.size()*sizeof(float)));
    cleanup();return result;
  } catch (...) {cleanup();throw;}
}
}}
