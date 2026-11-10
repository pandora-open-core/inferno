# dk-tensor

Lightweight autograd-free tensor: CPU storage always, cudarc CUDA storage behind the `cuda` feature. The value type dk-autograd's Variable wraps.

## Organisation

Roles: shared-infrastructure. Explicit root workspace member: yes.

Roles describe intended engineering responsibility, not a deployment audit or a claim of production readiness. See [the crate catalogue](../README.md).

## Source map

- [src/cache_model.rs](src/cache_model.rs)
- [src/cpu_ops.rs](src/cpu_ops.rs)
- [src/dtype.rs](src/dtype.rs)
- [src/error.rs](src/error.rs)
- [src/kernels.rs](src/kernels.rs)
- [src/lib.rs](src/lib.rs)
- [src/pairwise.rs](src/pairwise.rs)
- [src/panel.rs](src/panel.rs)
- [src/storage.rs](src/storage.rs)
- [src/tensor.rs](src/tensor.rs)

Declared feature names: `cuda`, `default`. See Cargo.toml for propagation and defaults.

## Validation

From the complete repository, run `cargo check -p dk-tensor` and `cargo test -p dk-tensor` with the backend features required by your deployment. These commands are guidance, not a report of checks run here. Model tests may require fitted artifacts.
