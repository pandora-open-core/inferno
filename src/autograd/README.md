# Inferno autograd

From-first-principles autograd engine: a Rust port of torch/csrc/autograd's scheduling core (Node/Edge/GraphTask/ReadyQueue/Engine) over dk-tensor, with a curated hand-port of derivatives.yaml formulas.

These sources are included in `inferno-rs` as `inferno::autograd`.
They are no longer a separate workspace package.

## Source map

- [src/accumulate_grad.rs](accumulate_grad.rs)
- [src/anomaly.rs](anomaly.rs)
- [src/edge.rs](edge.rs)
- [src/engine.rs](engine.rs)
- [src/gradcheck.rs](gradcheck.rs)
- [src/graph_root.rs](graph_root.rs)
- [src/graph_task.rs](graph_task.rs)
- [src/input_buffer.rs](input_buffer.rs)
- [src/input_metadata.rs](input_metadata.rs)
- [src/lib.rs](mod.rs)
- [src/node.rs](node.rs)
- [src/ready_queue.rs](ready_queue.rs)
- [src/saved_variable.rs](saved_variable.rs)
- [src/variable.rs](variable.rs)

Direct internal dependencies: `dk-cuda-ffi`, `dk-tensor`.

Declared feature names: `cuda`, `default`. See the root Cargo.toml for propagation and defaults.

## Validation

From the complete repository, run `cargo check -p inferno-rs` and `cargo test -p inferno-rs` with the backend features required by your deployment. These commands are guidance, not a report of checks run here. Model tests may require fitted artifacts.
