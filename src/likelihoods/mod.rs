//! Inferno — Observation likelihood interfaces and multitask Gaussian noise models.

use crate::distributions::MultivariateNormal;
use candle_core::{Result, Tensor};
pub mod multitask_gaussian;
pub use multitask_gaussian::{MultitaskGaussianLikelihood, TaskNoiseRank};
/// Observation model p(y | f), extracted without the application adapters.
pub trait Likelihood {
    fn expected_log_prob(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor>;
    fn log_marginal(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor>;
    fn marginal(&self, latent: &MultivariateNormal) -> Result<MultivariateNormal>;
}
