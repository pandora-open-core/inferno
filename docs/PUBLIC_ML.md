# Using the ML library

For the owning Pandora model wrappers, fitting and checkpoints, start with the
[SDK overview](../README.md). This guide covers the retained advanced components.

Use `inferno::candle::{Tensor, Device, DType}` with `inferno::nn::{VarMap,
VarBuilder, Optimizer}` for Candle-native models. VAE components retain their
`load` constructors, which accept VarBuilder: a VarMap-backed builder initialises
trainable parameters and a tensor-backed builder loads caller-owned weights.
Use model components directly and choose your own loss, optimiser and loop.
There is no built-in application trainer or command-line model runner.

`inferno::model::SpectralVae` is generic over Encoder, Dirichlet and Decoder traits;
`Vae` selects the retained concrete implementations. EWMA, latent SSM, Fourier
time embeddings, sparse decoder routing, normalisation, initialisation and noise
value types remain because the core VAE directly depends on them. These are
not evidence that the full private pipeline is included.

Legacy GP components live under `inferno::pandorapurple::gp`. `SvgpPredictor` uses
packed CPU features and historical layout conventions; `DifferentiableGp` uses
Candle tensors. The separate root `inferno::gp::ExactGp` is a small extensible CPU
reference GP. Layout constants describe the retained implementation; arbitrary
architectures should implement component traits instead of assuming those widths.

Standard covariance families are available in two tensor APIs. The CPU
`covariance` module provides RBF, Matérn 1/2, 3/2 and 5/2, linear, polynomial,
periodic, rational-quadratic, constant and white-noise kernels. It returns F64
values and does not preserve gradients. `kernels::standard::Kernel` provides
these families on Candle F32/F64 tensors, with scalar tensor hyperparameters
that may be supplied from `Var` leaves for first-order training. Matérn has a
zero input subgradient at coincident points; white noise uses feature equality
and has no input derivative. Both APIs use equal feature vectors, including
duplicate rows, as the same white-noise location. Observation-index noise
belongs in the Gaussian likelihood.

For latent dynamics, supply closures to `LangevinSDE::new(forward, score, sigma)`.
The closures can call the VAE or GP. `step` accepts physical time, a positive dt,
a Direction and a standard-normal noise tensor. Backward drift is `b - sigma² s`
and backward stepping multiplies drift by negative dt. Sigma is scalar and
constant in space; this API does not implement the divergence correction for
state-dependent diffusion. Caller-owned noise makes sampling reproducible.
See `examples/open_core.rs`.

Compute a component plan with `kernels::sinkhorn::sinkhorn_coupling_t`, then use
`flows::latent_geoc_drift` for its barycentric latent displacement.
`GmmSchrodingerBridge::fit` computes its own diagonal Bures cost and component
coupling using this same routine. The bridge supplies forward/backward analytic
drifts and marginal quantities. No bridge objective or bridge trainer is exposed.
Sinkhorn runs the requested fixed iteration count for its plan, with detached
warm-up and gradients through the final 16 scaling updates. This bounds Candle
graph depth and stack use. The replay gradient is an approximation to the full
fixed-point derivative; it is not an implicit or converged-envelope gradient.
Very sharp or slowly converging plans may need a different gradient estimator.

Actual coupling tensors may weight `model::decoder::bridge_weighted_reconstruction`
and `model::dirichlet::bridge_weighted_divergence`. The latter is categorical KL
between Dirichlet mean probabilities, **not** KL between full Dirichlet densities.
Both keep coupling gradients. Neither automatically inserts a training objective.
`model::FlowMatchingModel` is a separate velocity-network wrapper with a
straight-path conditional flow-matching loss, independent of the VAE/GP path.

The generalised engine is `inferno::autograd`; its `Variable` wraps the exact
same dk-tensor package as `inferno::Tensor`. `candle_apply(&[&variable], |xs| ... )`
connects a single-output Candle function to that graph using a first-order VJP.
All differentiable parameters must be explicit inputs. Captured parameters are
constants from the generalised engine's perspective. The adapter supports CPU
F32/F64 only, uses copies, honours no_grad, and releases its saved Candle graph
after non-retaining backward. See `examples/autograd.rs`.

Candle-native models still use Candle's gradient tape internally. This integration
does not replace every Candle operation with a dk-autograd operation. The explicit
VJP node lets both engines participate in one first-order gradient computation.

`LatentGeocDrift` accepts a caller-supplied transition matrix, offset and inverse
controllability Gramian. `ground_cost` builds the quadratic reference cost;
solve that cost at temperature 2, then pass its plan to `drift`. For the Brownian
reference, Phi=I, offset=0 and inverse Gramian=I/tau yield displacement/tau.
The plain `latent_geoc_drift` helper is only the unit-horizon barycentric
special case. Automatic linearisation, reference fitting, unbalanced marginal
relaxation and private GEOC orchestration are excluded.

A trainable VAE can be constructed without checkpoint files; run
`cargo run --locked --example vae` for a VarMap-backed example.

For a parameterised Candle calculation inside the generalised engine, pass both
inputs and parameters explicitly, for example:

```rust
let loss = inferno::candle_apply(&[&input, &weight], |xs| {
    xs[0].matmul(&xs[1])?.sqr()?.mean_all()
})?;
inferno::autograd::backward(&[loss], None, false)?;
let weight_gradient = weight.grad();
```

Captured model parameters do not become dk-autograd inputs automatically.

For transposed 1D convolution, use `model::conv_t::forward` or
`model::conv_t::conv_transpose1d_diff`. The VAE already uses this differentiable
composition. Native transposed 1D convolution still lacks backward support
in the vendor snapshot. The former PR #4044 dependency also showed an unpadded
CPU fast-path discrepancy against an independent scatter reference. See
[development checks](../README.md#development) for the dependency and test status.
