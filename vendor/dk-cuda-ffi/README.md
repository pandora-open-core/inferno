# dk-cuda-ffi

cxx bridge over ATen-derived cuBLAS/cuDNN kernels (transcribed from pytorch/aten, BSD-3 — see cpp/LICENSE.pytorch and PORTED_FROM.md).

## Organisation

Roles: shared-infrastructure. Explicit root workspace member: yes.

Roles describe intended engineering responsibility, not a deployment audit or a claim of production readiness. See [the crate catalogue](../README.md).

## Source map

- [src/bridge.rs](src/bridge.rs)
- [src/lib.rs](src/lib.rs)

Declared feature names: `cuda`, `default`. See Cargo.toml for propagation and defaults.

## Validation

From the complete repository, run `cargo check -p dk-cuda-ffi` and `cargo test -p dk-cuda-ffi` with the backend features required by your deployment. These commands are guidance, not a report of checks run here. Model tests may require fitted artifacts.

## PQO3 attention entry points

The existing `cpp/pqo3.cpp` driver adapter is now included in the CUDA build and
linked against `libcuda`. The CXX bridge and public Rust exports match the model
call sites in `dk-mistral-tq::attention::Kernel`:

- `device_fingerprint(device)` returns the architecture/driver cache namespace.
- `Pqo3Launch` describes device pointers, a caller-owned stream and grouped-query
  packed-page geometry.
- `unsafe launch_pqo3(image, launch)` runs `dk_pqo3_decode_attention` in the
  caller's bound context and synchronizes before unloading its module.
- `unsafe pqo3_host(image, query, packed, launch, device)` owns temporary device
  buffers, binds/restores the selected primary context and returns host output.
  It ignores the launch's address/stream fields and uses its geometry only.

Both paths validate head grouping, supported dimensions, token extent, at most
256 pages, and checked packed sizes capped at 256 MiB. Host input/output lengths
and finite floats are checked. Images must have a CUBIN ELF header and fit 64 MiB;
this is a bound, **not** verification of arbitrary executable code. The unsafe
contract requires a complete trusted compiler-produced image with the exact
kernel ABI. Device-pointer calls additionally require valid nonaliasing buffer
extents and the matching context/stream. This API is synchronous; it does not
return an event or promise asynchronous lifetime management.

Without `cuda`, all four entry points (including ping) return `CudaUnavailable`
and no C++ is compiled. Portable tests also cover invalid/oversized geometry.
The CUDA build still needs the toolkit and existing cuBLAS/cuDNN headers/libs;
link-time driver stubs are not a working runtime driver. Run model qualification
on a real device with the existing independent oracle:

```sh
cargo test --manifest-path crates/dk-mistral-tq/Cargo.toml --features cuda \
  fused_gpu_matches_independent_cpu_oracle -- --nocapture
```

Portable FFI tests passed; an isolated CUDA-feature Rust/CXX compile check also
passed against NVIDIA CUDA 12.4 headers, including the actual `pqo3.cpp` source.
That check does not qualify full toolkit/model linking or GPU execution. The
standalone workspace CPU/CUDA attention JIT process is separately documented
in `pandora-codegen/README.md`; it does not change PQO3's packed KV ABI.
