# Inferno tensor

Lightweight autograd-free tensor: CPU storage always, cudarc CUDA storage behind the `cuda` feature. The value type dk-autograd's Variable wraps.

These sources are included in `inferno-rs` as `inferno::tensor`.
They are no longer a separate workspace package.

## Source map

- [src/cache_model.rs](cache_model.rs)
- [src/cpu_ops.rs](cpu_ops.rs)
- [src/dtype.rs](dtype.rs)
- [src/error.rs](error.rs)
- [src/kernels.rs](kernels.rs)
- [src/lib.rs](mod.rs)
- [src/pairwise.rs](pairwise.rs)
- [src/panel.rs](panel.rs)
- [src/storage.rs](storage.rs)
- [src/tensor.rs](tensor.rs)

Declared feature names: `cuda`, `default`. See the root Cargo.toml for propagation and defaults.

## Validation

From the complete repository, run `cargo check -p inferno-rs` and `cargo test -p inferno-rs` with the backend features required by your deployment. These commands are guidance, not a report of checks run here. Model tests may require fitted artifacts.
