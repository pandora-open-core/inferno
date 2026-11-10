# Building Gaussian process components

Start with `cargo run --example custom_gp`. It demonstrates a custom kernel,
composed Matérn covariance, linear mean, custom noise, bounded cache and numerical
hyperparameter gradients. `cargo doc -p inferno-rs --no-deps --open` opens the API.
No account, paid request, checkpoint or GPU is required for CPU mathematics.
`cargo run --example custom_components` demonstrates named implementations of
`Mean`, a polynomial `Covariance`, a Laplace `Likelihood`, and a custom
capacity-limited `Cache` from scratch. The mean, covariance and cache are
wired into an exact GP with Gaussian noise.

## Choose a tensor family

The reference APIs below use `inferno::Tensor` (dk-tensor), CPU and f64 computation.
They are for small, single-output datasets: conditioning costs O(n³) time and
O(n²) memory. Marginal prediction currently constructs an m×m prior matrix;
full posterior covariance also costs O(m²) storage. Set resource limits in your
application before accepting untrusted dimensions. These APIs do not provide
autodiff, GPU support, sparse inference or automatic hyperparameter optimization.

For differentiable, variational and larger models use `inferno::pandorapurple` as
one consistent namespace. Its `gp`, `model`, `likelihoods`, `operators`,
`distributions` and `stochastics` modules retain the source
implementations. Candle tensors and root dk-tensor tensors are different types.
The large inherited stack has a broader experimental surface than the tested
reference path; inclusion does not constitute a numerical audit of every method.

## Custom means and kernels

Implement `mean::Mean::evaluate` returning one finite mean per input row. Use
`Zero`, `Constant`, `Linear` or `FunctionMean` for quick experiments. A custom
mean is subtracted from training targets and added to predictions.

Implement `covariance::Covariance::covariance(x, y)` returning a finite `[n,m]`
matrix for inputs `[n,d]`, `[m,d]`. `FunctionCovariance` wraps a fallible closure.
Your function must define a symmetric positive semidefinite kernel for every
input set, with consistent cross-covariances. Checks on a training matrix cannot
prove this global property. Kernels and means must remain unchanged after
conditioning, including any captured interior state. Noise must also be stable.

Built-ins: isotropic `Rbf`; ARD `Matern` with smoothness 1/2, 3/2, 5/2; `Sum`
and `Product`. Positive variances and lengthscales are constructor-checked.
Compose kernels by nesting `Sum` and `Product`. Boxed traits permit runtime
selection. For new kernels, test symmetry, diagonal values, PSD behavior on
representative inputs, extreme parameters and cross-input consistency.

## Likelihoods and inference

`likelihood::Likelihood` is a scalar log-density seam, with `Gaussian`, stable
logistic `Bernoulli`, and `FunctionLikelihood`. A custom likelihood can return
negative infinity for an impossible event. It must not return NaN or positive
infinity. `quadrature::normal_expectation` supports Gaussian expectations for
finite real functions; `developer::gaussian_kl` provides scalar Gaussian KL.

Exact GP conditioning accepts only independent additive Gaussian noise through
`GaussianNoise`. Its `variances(inputs)` must return one finite nonnegative
variance per row, using `Gaussian` or `FunctionNoise`, or your own struct. This
supports input-dependent (heteroscedastic) noise known at prediction inputs.
It does not support correlated noise, noisy input locations or arbitrary
non-Gaussian exact conditioning. A Bernoulli GP needs approximate inference in
the source-native stack or a new inference implementation.

`ExactGp::new` preserves the zero-mean, homoscedastic API.
`ExactGp::with_components` accepts any compatible kernel, mean and Gaussian
noise. `condition` retains the Cholesky factor and solves; it never explicitly
inverts a matrix. Jitter is a fixed numerical diagonal addition with no automatic
escalation; singular or indefinite systems return an error. The reported log
marginal likelihood includes jitter in the training covariance.

`predict` returns latent marginal variances, `predict_observed` adds future
observation noise, and `posterior_covariance` returns joint latent covariance.
Neither prediction mode adds jitter to future observations. Small negative
marginal variances within floating point tolerance are clamped to zero.

## Custom caches

Implement `cache::Cache<K,V>` to use your own storage policy, or use the owned
`LruCache` (entry-bounded, linear lookup, capacity zero disables storage).
`CachedCovariance::new(kernel, custom_cache, revision)` plugs a custom cache
directly into `ExactGp`; `CachedCovariance::lru(kernel, capacity, revision)`
provides a shortcut. Conditioning and prediction then share the same cached
matrices. Parameter access through `kernel_mut` clears entries first, and
`set_revision` invalidates external dependencies. The adapter serializes cache
misses with a mutex; a kernel must not recursively reenter the same adapter.

`cache::covariance` also supports direct use, caching detached covariance matrices, checking
shape and finite values. Errors are not cached. `CovarianceKey` stores exact
input values, dtypes and shapes, plus a caller-supplied revision. Use a distinct revision
for every parameter set, kernel identity and external dependency. Reusing a
revision for different kernels gives stale results. Call `clear` when lifecycle
boundaries are simpler than revision management. Cache memory is bounded by
entry count, not bytes; enforce matrix size limits in the application.

Autograd caches need a different policy: use the source-native
`pandorapurple::stochastics::gp_cache` and vendored `dk-transport-cache` generation
namespaces. Never put tensors with active gradient graphs in the reference cache.

## Checking custom objectives

`developer::finite_difference` evaluates a fallible scalar objective at central
perturbations. Use it to validate gradients or explore log-hyperparameters;
transform positive parameters with `exp` inside the objective. This is a
numerical diagnostic, not an optimizer. Verify convergence over multiple step
sizes. Function errors and nonfinite objective values propagate as errors.

## Validation

Run `cargo test -p inferno-rs --locked` for reference, operator,
likelihood and local HTTP contract tests. Tests never call a paid service. Run
`cargo test --workspace --locked` for inherited workspace tests; external fixture
and GPU gates remain documented in their source. Run the checks against the revision you intend to use.

## Posterior draws and cross validation

`Posterior::sample_latent(inputs, innovations, jitter)` transforms caller-provided
independent N(0,1) innovations of shape `[draws, points]` into joint latent draws.
This keeps RNG choice and seeding under your control. The sampling-only jitter
is explicit and can regularize a singular joint covariance; it is not observation
noise. Passing arbitrary numbers instead of standard normals is just a linear
transformation, not Gaussian sampling.

`Posterior::leave_one_out()` returns exact leave-one-out predictive means and
variances without separate model refits. Those variances include training
Gaussian observation noise and the training jitter, since both are present in
the fitted covariance. With zero jitter they match predictions of future
observations from leave-one-out refits. Kernel parameters are held fixed; this
is not nested cross validation with hyperparameter reoptimization.
