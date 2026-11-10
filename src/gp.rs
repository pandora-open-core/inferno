//! Inferno — Exact Gaussian-process conditioning with caller-supplied kernels, means and noise.

use crate::{
    covariance::Covariance,
    ensure,
    likelihood::{Gaussian, GaussianNoise},
    matrix,
    mean::{Mean, Zero},
    DType, Result, Tensor,
};
pub struct ExactGp<K, M = Zero, N = Gaussian> {
    kernel: K,
    likelihood: N,
    mean: M,
    jitter: f64,
}
pub struct Posterior<K, M = Zero, N = Gaussian> {
    model: ExactGp<K, M, N>,
    inputs: Tensor,
    lower: Vec<f64>,
    alpha: Vec<f64>,
    observations: Vec<f64>,
    log_marginal_likelihood: f64,
}
pub struct Prediction {
    pub mean: Tensor,
    /// Marginal latent variances; excludes observation noise and numerical jitter.
    pub variance: Tensor,
}
impl<K: Covariance> ExactGp<K> {
    pub fn new(kernel: K, likelihood: Gaussian, jitter: f64) -> Result<Self> {
        ensure(
            jitter.is_finite() && jitter >= 0.0,
            "jitter must be finite and nonnegative",
        )?;
        Ok(Self {
            kernel,
            likelihood,
            mean: Zero,
            jitter,
        })
    }
}
impl<K: Covariance, M: Mean, N: GaussianNoise> ExactGp<K, M, N> {
    /// Construct an exact GP with a custom mean and independent Gaussian noise.
    pub fn with_components(kernel: K, mean: M, likelihood: N, jitter: f64) -> Result<Self> {
        ensure(jitter.is_finite() && jitter >= 0.0, "invalid jitter")?;
        Ok(Self {
            kernel,
            mean,
            likelihood,
            jitter,
        })
    }
    pub fn condition(self, inputs: &Tensor, targets: &Tensor) -> Result<Posterior<K, M, N>> {
        let (n, d, _) = matrix(inputs)?;
        ensure(
            n > 0 && d > 0 && targets.shape() == [n],
            "expected nonempty inputs [n,d] and targets [n]",
        )?;
        ensure(
            matches!(targets.device(), crate::Device::Cpu),
            "targets must be on CPU",
        )?;
        let mut y = targets.to_vec_f64()?;
        let observations = y.clone();
        ensure(y.iter().all(|v| v.is_finite()), "targets must be finite")?;
        let means = self.mean.evaluate(inputs)?;
        let noise = self.likelihood.variances(inputs)?;
        ensure(
            means.len() == n && means.iter().all(|v| v.is_finite()),
            "invalid training mean",
        )?;
        ensure(
            noise.len() == n && noise.iter().all(|v| v.is_finite() && *v >= 0.0),
            "invalid noise variances",
        )?;
        for (y, mean) in y.iter_mut().zip(means) {
            *y -= mean;
        }
        let covariance = self.kernel.covariance(inputs, inputs)?;
        let (rows, cols, mut a) = matrix(&covariance)?;
        ensure(
            rows == n && cols == n,
            "kernel returned incorrect covariance dimensions",
        )?;
        for i in 0..n {
            for j in 0..i {
                ensure(
                    (a[i * n + j] - a[j * n + i]).abs() <= 1e-12 * (1.0 + a[i * n + j].abs()),
                    "covariance must be symmetric",
                )?;
            }
            a[i * n + i] += noise[i] + self.jitter;
        }
        let l = cholesky(&a, n)?;
        let v = forward(&l, &y, n);
        let mut alpha = vec![0.0; n];
        for i in (0..n).rev() {
            alpha[i] =
                (v[i] - (i + 1..n).map(|j| l[j * n + i] * alpha[j]).sum::<f64>()) / l[i * n + i];
        }
        let lml = -0.5 * y.iter().zip(&alpha).map(|(a, b)| a * b).sum::<f64>()
            - (0..n).map(|i| l[i * n + i].ln()).sum::<f64>()
            - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
        ensure(
            lml.is_finite() && alpha.iter().all(|v| v.is_finite()),
            "GP conditioning overflow",
        )?;
        Ok(Posterior {
            model: self,
            inputs: inputs.clone(),
            lower: l,
            alpha,
            observations,
            log_marginal_likelihood: lml,
        })
    }
}
pub(crate) fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            if i == j {
                ensure(
                    value.is_finite() && value > 0.0,
                    "covariance is not positive definite; check kernel and jitter",
                )?;
                l[i * n + j] = value.sqrt();
            } else {
                l[i * n + j] = value / l[j * n + j];
            }
        }
    }
    Ok(l)
}
fn symmetric(a: &[f64], n: usize) -> Result<()> {
    for i in 0..n {
        for j in 0..i {
            ensure(
                (a[i * n + j] - a[j * n + i]).abs()
                    <= 1e-12 * (1.0 + a[i * n + j].abs().max(a[j * n + i].abs())),
                "covariance must be symmetric",
            )?;
        }
    }
    Ok(())
}
pub(crate) fn forward(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = (b[i] - (0..i).map(|j| l[i * n + j] * x[j]).sum::<f64>()) / l[i * n + i];
    }
    x
}
impl<K: Covariance, M: Mean, N: GaussianNoise> Posterior<K, M, N> {
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal_likelihood
    }
    pub fn predict(&self, inputs: &Tensor) -> Result<Prediction> {
        let (m, d, _) = matrix(inputs)?;
        ensure(
            d == self.inputs.shape()[1],
            "prediction feature dimension mismatch",
        )?;
        let n = self.alpha.len();
        let (rows, cols, cross) = matrix(&self.model.kernel.covariance(&self.inputs, inputs)?)?;
        ensure(
            rows == n && cols == m,
            "kernel returned incorrect cross covariance dimensions",
        )?;
        let (rows, cols, prior) = matrix(&self.model.kernel.covariance(inputs, inputs)?)?;
        ensure(
            rows == m && cols == m,
            "kernel returned incorrect prior dimensions",
        )?;
        symmetric(&prior, m)?;
        let mut means = self.model.mean.evaluate(inputs)?;
        ensure(
            means.len() == m && means.iter().all(|v| v.is_finite()),
            "invalid prediction mean",
        )?;
        let mut variances = vec![0.0; m];
        for j in 0..m {
            let k: Vec<_> = (0..n).map(|i| cross[i * m + j]).collect();
            means[j] += k.iter().zip(&self.alpha).map(|(k, a)| k * a).sum::<f64>();
            let v = forward(&self.lower, &k, n);
            let variance = prior[j * m + j] - v.iter().map(|v| v * v).sum::<f64>();
            ensure(
                variance.is_finite() && variance >= -1e-10 * (1.0 + prior[j * m + j].abs()),
                "invalid posterior variance",
            )?;
            variances[j] = variance.max(0.0);
        }
        ensure(means.iter().all(|v| v.is_finite()), "prediction overflow")?;
        Ok(Prediction {
            mean: Tensor::from_f64_slice(&means, &[m], DType::F64)?,
            variance: Tensor::from_f64_slice(&variances, &[m], DType::F64)?,
        })
    }
    /// Exact leave-one-out predictions at training points. The variances include
    /// training observation noise AND the fixed training jitter, treating it as
    /// part of the fitted covariance. No n separate GP refits are needed.
    pub fn leave_one_out(&self) -> Result<Prediction> {
        let n = self.alpha.len();
        let mut means = Vec::with_capacity(n);
        let mut variances = Vec::with_capacity(n);
        for i in 0..n {
            let mut unit = vec![0.0; n];
            unit[i] = 1.0;
            let column = forward(&self.lower, &unit, n);
            let precision = column.iter().map(|v| v * v).sum::<f64>();
            ensure(
                precision.is_finite() && precision > 0.0,
                "invalid leave-one-out precision",
            )?;
            variances.push(1.0 / precision);
            means.push(self.observations[i] - self.alpha[i] / precision);
        }
        ensure(
            means.iter().chain(&variances).all(|v| v.is_finite()),
            "leave-one-out overflow",
        )?;
        Ok(Prediction {
            mean: Tensor::from_f64_slice(&means, &[n], DType::F64)?,
            variance: Tensor::from_f64_slice(&variances, &[n], DType::F64)?,
        })
    }
    /// Joint latent draws from caller-provided independent standard normal
    /// innovations `[draws, points]`. Returns the same shape. `jitter` is an
    /// explicit sampling-only covariance stabilization, not observation noise.
    /// The caller owns RNG choice/seeding; supplying nonnormal innovations does
    /// not generate Gaussian draws. Singular covariance may need positive jitter.
    pub fn sample_latent(
        &self,
        inputs: &Tensor,
        innovations: &Tensor,
        jitter: f64,
    ) -> Result<Tensor> {
        ensure(
            jitter.is_finite() && jitter >= 0.0,
            "invalid sampling jitter",
        )?;
        let prediction = self.predict(inputs)?;
        let means = prediction.mean.to_vec_f64()?;
        let m = means.len();
        let (draws, width, z) = matrix(innovations)?;
        ensure(width == m, "sampling innovation dimension mismatch")?;
        let (_, _, mut covariance) = matrix(&self.posterior_covariance(inputs)?)?;
        for i in 0..m {
            covariance[i * m + i] += jitter;
        }
        let lower = cholesky(&covariance, m)?;
        let mut out = vec![0.0; draws * m];
        for s in 0..draws {
            for i in 0..m {
                out[s * m + i] = means[i]
                    + (0..=i)
                        .map(|j| lower[i * m + j] * z[s * m + j])
                        .sum::<f64>();
            }
        }
        ensure(
            out.iter().all(|v| v.is_finite()),
            "posterior sample overflow",
        )?;
        Ok(Tensor::from_f64_slice(&out, &[draws, m], DType::F64)?)
    }
    /// Marginals for future observations, including observation noise (never jitter).
    pub fn predict_observed(&self, inputs: &Tensor) -> Result<Prediction> {
        let mut prediction = self.predict(inputs)?;
        let mut values = prediction.variance.to_vec_f64()?;
        let noise = self.model.likelihood.variances(inputs)?;
        ensure(
            noise.len() == values.len() && noise.iter().all(|v| v.is_finite() && *v >= 0.0),
            "invalid prediction noise",
        )?;
        for (v, noise) in values.iter_mut().zip(noise) {
            *v += noise;
        }
        ensure(
            values.iter().all(|v| v.is_finite()),
            "observed variance overflow",
        )?;
        prediction.variance = Tensor::from_f64_slice(&values, &[values.len()], DType::F64)?;
        Ok(prediction)
    }
    /// Full latent covariance, O(m²) storage, for joint predictions.
    pub fn posterior_covariance(&self, inputs: &Tensor) -> Result<Tensor> {
        let (m, d, _) = matrix(inputs)?;
        ensure(
            d == self.inputs.shape()[1],
            "prediction feature dimension mismatch",
        )?;
        let n = self.alpha.len();
        let (r, c, cross) = matrix(&self.model.kernel.covariance(&self.inputs, inputs)?)?;
        ensure((r, c) == (n, m), "invalid cross covariance shape")?;
        let (r, c, mut prior) = matrix(&self.model.kernel.covariance(inputs, inputs)?)?;
        ensure((r, c) == (m, m), "invalid prior covariance shape")?;
        symmetric(&prior, m)?;
        let solved: Vec<_> = (0..m)
            .map(|j| {
                forward(
                    &self.lower,
                    &(0..n).map(|i| cross[i * m + j]).collect::<Vec<_>>(),
                    n,
                )
            })
            .collect();
        for i in 0..m {
            for j in 0..m {
                prior[i * m + j] -= solved[i]
                    .iter()
                    .zip(&solved[j])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
        }
        for i in 0..m {
            let diagonal = prior[i * m + i];
            ensure(
                diagonal.is_finite()
                    && diagonal
                        >= -1e-10
                            * (1.0 + diagonal.abs() + solved[i].iter().map(|v| v * v).sum::<f64>()),
                "invalid posterior variance",
            )?;
            prior[i * m + i] = diagonal.max(0.0);
        }
        ensure(
            prior.iter().all(|v| v.is_finite()),
            "posterior covariance overflow",
        )?;
        Ok(Tensor::from_f64_slice(&prior, &[m, m], DType::F64)?)
    }
}

/// Generic kernel seam extracted from the source SVGP module.
pub trait KernelEval: Send + Sync {
    fn kmm(&self, z: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>;
    fn kxm(
        &self,
        x: &candle_core::Tensor,
        z: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor>;
    fn kdiag(&self, x: &candle_core::Tensor, l: usize) -> candle_core::Result<candle_core::Tensor>;
}
pub mod added_loss;
pub mod prediction;
