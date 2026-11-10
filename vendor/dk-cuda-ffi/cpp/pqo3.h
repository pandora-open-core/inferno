#pragma once
#include "rust/cxx.h"
#include <cstdint>
namespace dk { namespace cuda {
rust::String device_fingerprint(std::int32_t device);
void launch_pqo3(rust::Slice<const std::uint8_t> image, std::uint64_t query,
                 std::uint64_t pages, std::uint64_t output, std::uint64_t stream,
                 std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
                 std::uint32_t page_tokens, std::uint32_t page_count, std::uint32_t tokens);
}}
namespace dk { namespace cuda {
rust::Vec<float> pqo3_host(rust::Slice<const std::uint8_t> image,
  rust::Slice<const float> query, rust::Slice<const std::uint8_t> packed,
  std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
  std::uint32_t page_tokens, std::uint32_t page_count, std::uint32_t tokens,
  std::int32_t device);
}}
