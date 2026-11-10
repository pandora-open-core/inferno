# Inferno

Inferno is an **alpha, open-source Rust SDK** for tensors, automatic
differentiation, neural models, spectral VAEs and Gaussian processes. It is
released under [MPL-2.0](LICENSE) as part of Pandora-AI's open-core ecosystem.
APIs may change. No trained weights are bundled. Linux CPU is the primary
validation target; optional GPU paths require separate qualification.

Created and contributed by **Liam Dziga Douglas Giles**, founder and CEO of
**Pandora-AI Proprietary Ltd.** Third-party components retain their own
copyrights and licenses.

## Install

Keep the two SDK checkouts beside each other:

```text
workspace/
  pandora-open-core/inferno
  pandora-open-core/pudding
```

Add Inferno to your application:

```toml
[dependencies]
inferno = { package = "inferno-rs", path = "../inferno-open-source" }
```

Use Rust 1.95 or newer and a C/C++ toolchain. Pudding resolves from the sibling
checkout. Candle and the tensor backends are vendored; consumers need no Cargo
patch table. Registry publication is disabled. This source distribution requires
both checkouts; it is not yet a standalone Git dependency.

## Build a neural model

```rust
use inferno::prelude::*;

let device = Device::Cpu;
let mut model = Model::new(&device, |vb| nn::linear(1, 1, vb.pp("output")))?;
let mut optimizer = model.adam(0.01)?;
let x = Tensor::new(&[[0f32], [1.]], &device)?;
let target = x.affine(2., 1.)?;
model.train();
let loss = nn::loss::mse(&model.forward(&x)?, &target)?;
optimizer.backward_step(&loss)?;
model.eval();
let prediction = model.forward(&x)?;
```

`Model` owns registered parameters and supports training mode, optimizers and
weight save/load. Use the supplied named VarBuilder for each child layer.
See the [model guide](notebook/README.md).

## Model components

Curated VAE, GP and stochastic components are available through
`inferno::pandorapurple` (formerly `inferno::rustysecrets`).

| Component | Public entry point | Scope |
| --- | --- | --- |
| Neural model | `inferno::prelude::Model` | Candle layers, registered parameters and optimizers |
| Kernel-space multitask GP | `inferno::models::PandoraGP` | Alias of the existing differentiable GP using kernel-network features and a prepared posterior |
| Spectral VAE | `inferno::models::PandoraVae` | Direct rename of generic `SpectralVae`; `inferno::model::Vae` selects the default components |
| Small exact GP | `inferno::prelude::GaussianProcess` | CPU RBF conditioning with fixed hyperparameters |
| Custom GP | `inferno::gp::ExactGp` | Composable kernels, means and Gaussian observation noise |
| Latent dynamics | `inferno::stochastics`, `inferno::flows` | Langevin and bridge components with caller-supplied objectives |

`PandoraGP` and `PandoraVae` name the existing model components. They add no new
model architecture, fit API, configuration class or checkpoint format.
The GP retains the packed kernel-network feature layout and posterior setup;
it does not accept arbitrary `[n, features]` and `[n, tasks]` training matrices
through a general-purpose `fit` method. `PandoraVae` retains `SpectralVae`'s
generic encoder, Dirichlet and decoder types and its existing construction API.

```sh
cargo run --locked --example notebook_mlp
cargo run --locked --example notebook_gp
cargo run --locked --example vae
cargo run --locked --example custom_gp
```

## Check computations with Pudding

```rust
use inferno::{diagnostics, pudding::{Check, Engine}};
use inferno::prelude::*;

let x = Tensor::new(&[0.25f64, 0.75], &Device::Cpu)?;
let observation = diagnostics::snapshot(&x)?;
let mut checks = Engine::new();
checks.check("finite input", Check::Finite(&observation));
assert!(checks.finish().passed());
```

Pudding is imported directly and re-exported as `inferno::pudding`.
`diagnostics::check_operator` compares a small operator's multiplication against
its dense representation; `check_solution` observes a linear-solve residual.
Snapshots copy values to CPU/f64 and detach from autograd. These diagnostics
produce runtime observations; they do not certify solver correctness or gradients.

## Covariance kernels

`inferno::covariance` provides CPU RBF, Matérn 1/2, 3/2 and 5/2, linear,
polynomial, periodic, rational-quadratic, constant and white-noise kernels,
plus sum and product composition. This F64 reference API copies tensor values
and does not preserve gradients. Matérn supports per-feature lengthscales.

`inferno::kernels::standard::Kernel` provides the same families on Candle
F32/F64 tensors with scalar tensor hyperparameters. Inputs and hyperparameters
retain first-order gradients. Matérn uses a zero subgradient at coincident
points; Matérn 1/2 is not differentiable there. White noise compares feature
vectors for equality, including duplicate rows, and has no input derivative.
For independent observation noise, use the likelihood instead.

The legacy neural kernel and differentiable multitask GP remain available.
The newer protected coupling kernel is excluded. Symbolic reductions use tiled
Candle evaluation; generated CUDA/KeOps backends and the Cloud client have been
removed. Default execution follows the same path, while workloads that enabled
generated GPU reductions may run more slowly.

## Tensor types and alpha limits

`inferno::prelude` and `inferno::candle` use Candle tensors. Root
`inferno::Tensor` uses dk-tensor. Keep a single tensor family within a computation.
The explicit cross-engine adapter supports CPU first-order F32/F64 gradients.
Sinkhorn gradients replay only the final 16 updates. See the
[advanced API](docs/PUBLIC_ML.md), [GP development guide](docs/DEVELOPER_GUIDE.md)
and [architecture](docs/ARCHITECTURE.md).

## Development

```sh
python3 tools/check_public_scope.py
python3 tools/check_consumer.py
cargo test --workspace --locked
cargo doc -p inferno-rs --no-deps --locked
```

The external consumer check builds a separate application and exercises the
Candle API and Pudding integration. Vendored Candle licensing and source changes
are documented in [its provenance](vendor/candle/PROVENANCE.md).

See [contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md).
