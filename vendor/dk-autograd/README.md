# dk-autograd

From-first-principles autograd engine: a Rust port of torch/csrc/autograd's scheduling core (Node/Edge/GraphTask/ReadyQueue/Engine) over dk-tensor, with a curated hand-port of derivatives.yaml formulas.

## Organisation

Roles: production-training. Explicit root workspace member: yes.

Roles describe intended engineering responsibility, not a deployment audit or a claim of production readiness. See [the crate catalogue](../README.md).

## Source map

- [src/accumulate_grad.rs](src/accumulate_grad.rs)
- [src/anomaly.rs](src/anomaly.rs)
- [src/edge.rs](src/edge.rs)
- [src/engine.rs](src/engine.rs)
- [src/gradcheck.rs](src/gradcheck.rs)
- [src/graph_root.rs](src/graph_root.rs)
- [src/graph_task.rs](src/graph_task.rs)
- [src/input_buffer.rs](src/input_buffer.rs)
- [src/input_metadata.rs](src/input_metadata.rs)
- [src/lib.rs](src/lib.rs)
- [src/node.rs](src/node.rs)
- [src/ready_queue.rs](src/ready_queue.rs)
- [src/saved_variable.rs](src/saved_variable.rs)
- [src/variable.rs](src/variable.rs)

Direct internal dependencies: `dk-cuda-ffi`, `dk-tensor`.

Declared feature names: `cuda`, `default`. See Cargo.toml for propagation and defaults.

## Validation

From the complete repository, run `cargo check -p dk-autograd` and `cargo test -p dk-autograd` with the backend features required by your deployment. These commands are guidance, not a report of checks run here. Model tests may require fitted artifacts.
