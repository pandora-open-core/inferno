//! Inferno — Gaussian observation-noise models for the CPU reference Gaussian process.

use crate::{ensure, Result};
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    variance: f64,
}
impl Gaussian {
    pub fn new(variance: f64) -> Result<Self> {
        ensure(
            variance.is_finite() && variance > 0.0,
            "noise variance must be finite and positive",
        )?;
        Ok(Self { variance })
    }
    pub fn variance(&self) -> f64 {
        self.variance
    }
    pub fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        ensure(
            observation.is_finite() && latent.is_finite(),
            "likelihood inputs must be finite",
        )?;
        let value = -0.5
            * ((observation - latent).powi(2) / self.variance
                + self.variance.ln()
                + (2.0 * std::f64::consts::PI).ln());
        ensure(value.is_finite(), "log probability overflowed")?;
        Ok(value)
    }
}

/// Scalar likelihood extension seam. Non-Gaussian likelihoods require approximate
/// inference; implementing this trait alone does not make exact GP inference valid.
pub trait Likelihood {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64>;
}
impl Likelihood for Gaussian {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        Gaussian::log_prob(self, observation, latent)
    }
}
/// Independent additive Gaussian observation noise, expressed as variances.
/// Exact inference relies on this contract; arbitrary likelihoods are not accepted.
pub trait GaussianNoise {
    fn variances(&self, inputs: &crate::Tensor) -> Result<Vec<f64>>;
}
impl GaussianNoise for Gaussian {
    fn variances(&self, inputs: &crate::Tensor) -> Result<Vec<f64>> {
        Ok(vec![self.variance; crate::matrix(inputs)?.0])
    }
}
/// Logistic Bernoulli likelihood using stable softplus arithmetic.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bernoulli;
impl Likelihood for Bernoulli {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        ensure(
            (observation == 0.0 || observation == 1.0) && latent.is_finite(),
            "Bernoulli requires binary observations and finite logits",
        )?;
        let z = if observation == 1.0 { -latent } else { latent };
        Ok(-z.max(0.0) - (-z.abs()).exp().ln_1p())
    }
}

/// Closure adapter for input-dependent independent Gaussian noise.
pub struct FunctionNoise<F>(pub F);
impl<F: Fn(&crate::Tensor) -> Result<Vec<f64>>> GaussianNoise for FunctionNoise<F> {
    fn variances(&self, inputs: &crate::Tensor) -> Result<Vec<f64>> {
        (self.0)(inputs)
    }
}
/// Closure adapter for scalar custom likelihoods. Return log densities, not probabilities.
pub struct FunctionLikelihood<F>(pub F);
impl<F: Fn(f64, f64) -> Result<f64>> Likelihood for FunctionLikelihood<F> {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        ensure(
            observation.is_finite() && latent.is_finite(),
            "likelihood inputs must be finite",
        )?;
        let result = (self.0)(observation, latent)?;
        ensure(
            !result.is_nan() && result != f64::INFINITY,
            "invalid log likelihood",
        )?;
        Ok(result)
    }
}
impl<T: GaussianNoise + ?Sized> GaussianNoise for Box<T> {
    fn variances(&self, inputs: &crate::Tensor) -> Result<Vec<f64>> {
        (**self).variances(inputs)
    }
}
impl<T: Likelihood + ?Sized> Likelihood for Box<T> {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        (**self).log_prob(observation, latent)
    }
}
