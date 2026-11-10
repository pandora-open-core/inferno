//! Inferno — Implement custom mean, likelihood and cache components using library interfaces.

use inferno::{
    cache::{Cache, CachedCovariance, CovarianceKey},
    covariance::Covariance,
    gp::ExactGp,
    likelihood::{Gaussian, Likelihood},
    mean::Mean,
    DType, Error, Result, Tensor,
};
use std::{collections::HashMap, hash::Hash};
struct QuadraticMean;
impl Mean for QuadraticMean {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        if inputs.shape().len() != 2 || inputs.shape()[1] != 1 {
            return Err(Error("quadratic mean expects [n,1]".into()));
        }
        inputs
            .to_vec_f64()?
            .into_iter()
            .map(|x| {
                let value = x * x;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(Error("mean overflow".into()))
                }
            })
            .collect()
    }
}
// Degree-two polynomial covariance: (1 + x*y)^2 is PSD on real inputs.
struct PolynomialCovariance;
impl Covariance for PolynomialCovariance {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        if x.shape().len() != 2 || y.shape().len() != 2 || x.shape()[1] != 1 || y.shape()[1] != 1 {
            return Err(Error(
                "polynomial covariance expects [n,1] and [m,1]".into(),
            ));
        }
        let xv = x.to_vec_f64()?;
        let yv = y.to_vec_f64()?;
        if !xv.iter().chain(&yv).all(|v| v.is_finite()) {
            return Err(Error("nonfinite covariance input".into()));
        }
        let values: Vec<_> = xv
            .iter()
            .flat_map(|a| yv.iter().map(move |b| (1.0 + a * b).powi(2)))
            .collect();
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error("covariance overflow".into()));
        }
        Ok(Tensor::from_f64_slice(
            &values,
            &[x.shape()[0], y.shape()[0]],
            DType::F64,
        )?)
    }
}
struct Laplace {
    scale: f64,
}
impl Likelihood for Laplace {
    fn log_prob(&self, observation: f64, latent: f64) -> Result<f64> {
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !observation.is_finite()
            || !latent.is_finite()
        {
            return Err(Error("invalid Laplace parameters".into()));
        }
        // Log density may legitimately be negative infinity in extreme tails.
        Ok(-(observation - latent).abs() / self.scale - std::f64::consts::LN_2 - self.scale.ln())
    }
}
// A deliberately simple custom policy: clear all entries when capacity is hit.
// Substitute a byte-bounded or domain-specific policy behind this same trait.
struct ScratchCache<K, V> {
    entries: HashMap<K, V>,
    capacity: usize,
}
impl<K: Eq + Hash, V> Cache<K, V> for ScratchCache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }
    fn insert(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.entries.clear();
        }
        self.entries.insert(key, value);
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
}
fn main() -> Result<()> {
    let x = Tensor::from_f64_slice(&[0.0, 1.0, 2.0], &[3, 1], DType::F64)?;
    println!("custom prior mean: {:?}", QuadraticMean.evaluate(&x)?);
    println!(
        "custom likelihood log density: {}",
        Laplace { scale: 0.5 }.log_prob(1.0, 0.8)?
    );
    // A Laplace likelihood is not additive Gaussian noise and cannot be passed
    // to ExactGp::condition: choose an approximate inference method instead.
    let cache: ScratchCache<CovarianceKey, Tensor> = ScratchCache {
        entries: HashMap::new(),
        capacity: 2,
    };
    let kernel = CachedCovariance::new(PolynomialCovariance, cache, 0);
    let targets = Tensor::from_f64_slice(&[0.0, 1.0, 4.0], &[3], DType::F64)?;
    let posterior = ExactGp::with_components(kernel, QuadraticMean, Gaussian::new(0.1)?, 1e-9)?
        .condition(&x, &targets)?;
    println!(
        "custom mean/kernel/cache GP: {:?}",
        posterior.predict(&x)?.mean.to_vec_f64()?
    );
    Ok(())
}
