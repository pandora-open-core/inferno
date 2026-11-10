//! Inferno — Observation likelihood interfaces and multitask Gaussian noise models.

use candle_core::{Result, Tensor};
use crate::distributions::MultivariateNormal;
pub mod multitask_gaussian;
pub use multitask_gaussian::MultitaskGaussianLikelihood;
/// The observation model `p(y | f)`.
pub trait Likelihood {
    /// `E_{q(f)}[log p(y | f)]` — the ELL term of the ELBO.
    ///
    /// `y`, `mean`, `variance` are `[..., N, T]` (or `[..., N]` for a
    /// single task). Returns the per-datapoint total `[..., N]`.
    fn expected_log_prob(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor>;

    /// `log q(y)` under the noise-marginalised predictive → `[..., N]`.
    fn log_marginal(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor>;

    /// `q(f) → q(y)`: add the observation noise to a latent distribution.
    fn marginal(&self, latent: &MultivariateNormal) -> Result<MultivariateNormal>;
}
