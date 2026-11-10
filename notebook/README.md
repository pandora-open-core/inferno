# Model development

## The model development loop

```rust
use inferno::prelude::*;
let device = Device::Cpu;
let mut model = Model::new(&device, |vb| nn::linear(1, 1, vb.pp("output")))?;
let mut optimizer = model.adam(0.01)?;
let x = Tensor::new(&[[0f32], [1.]], &device)?;
let y = x.affine(2., 1.)?;
model.train();
let loss = nn::loss::mse(&model.forward(&x)?, &y)?;
optimizer.backward_step(&loss)?;
model.eval();
let prediction = model.forward(&x)?;
```

A Rust struct owns its child layers; `impl Module` is its `forward` method.
`Model` owns the parameter store and gives you `parameters`, `adam`, `train`,
`eval`, `save`, and `load`. Build children using the supplied builder with named
`pp` scopes. The full trainable example is `examples/notebook_mlp.rs`.
For dropout or mode-dependent behavior implement `ModuleT` instead of `Module`;
the wrapper passes the train/eval flag to `forward_t`. `eval` does not disable
autograd. Candle constructs gradients per backward call, so there is no
accumulating `.grad` buffer or `zero_grad` step. Save/load acts on parameter
values, not architecture, mode or optimizer state.

The prelude consistently uses **Candle tensors**. Existing root `inferno::Tensor`
and `Variable` remain the distinct dk backend. The notebook `GaussianProcess`
accepts Candle CPU tensors and explicitly copies to the existing exact GP
reference implementation, returning F64 Candle predictions. Use `fit` then
`predict`; `examples/notebook_gp.rs` prints latent means and variances. It is a
small-data fixed-hyperparameter GP, without an autograd path across that copy.
Advanced custom kernels/means still use the root `gp::ExactGp` API.

## Verify

Run `cargo test --locked --test notebook` and the `notebook_mlp` and
`notebook_gp` examples from the repository root. Each example is an ordinary
Rust program and can run in a terminal or a notebook with shell-command support.
