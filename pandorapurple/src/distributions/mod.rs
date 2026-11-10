//! Inferno — Multivariate and multitask probability distributions on Candle tensors.

use std::f64::consts::PI;

use candle_core::{Result, Tensor, D};

use crate::operators::{dense_diagonal, DenseOperator, DiagOperator, LinearOperator};

pub mod multitask;
pub use multitask::MultitaskMultivariateNormal;

/// `N(μ, Σ)` with the covariance held as a [`LinearOperator`].
///
/// `mean: [..., N]`; `covar` is `[..., N, N]`.
pub struct MultivariateNormal {
    pub mean: Tensor,
    pub covar: Box<dyn LinearOperator>,
    /// Cholesky jitter used by every factorisation on this distribution.
    pub jitter: f64,
}

impl MultivariateNormal {
    pub fn new(mean: Tensor, covar: Box<dyn LinearOperator>) -> Self {
        Self { mean, covar, jitter: crate::operators::DEFAULT_JITTER }
    }

    /// Dense-covariance convenience constructor.
    pub fn from_dense(mean: Tensor, covar: Tensor) -> Self {
        Self::new(mean, Box::new(DenseOperator::new(covar)))
    }

    /// Diagonal-covariance constructor — the independent-Gaussian case,
    /// which keeps every op O(N).
    pub fn from_diag(mean: Tensor, var: Tensor) -> Self {
        Self::new(mean, Box::new(DiagOperator::new(var)))
    }

    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Event size `N`.
    pub fn event_size(&self) -> usize {
        self.covar.ncols()
    }

    /// Batch shape (everything before the event dim).
    pub fn batch_shape(&self) -> Vec<usize> {
        self.covar.batch_shape()
    }

    /// Marginal variances `diag(Σ)` → `[..., N]`.
    pub fn variance(&self) -> Result<Tensor> {
        self.covar.diagonal()
    }

    pub fn stddev(&self) -> Result<Tensor> {
        self.variance()?.sqrt()
    }

    /// `log p(x)` for `x: [..., N]` → `[...]`.
    ///
    /// One factorisation for both the quadratic form and the log-determinant.
    pub fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.event_size() as f64;
        let diff = x.broadcast_sub(&self.mean)?.unsqueeze(D::Minus1)?; // [..., N, 1]
        let (iq, logdet) = self.covar.inv_quad_logdet(&diff, self.jitter)?;
        // iq is [..., 1] (one column); drop the column axis.
        let iq = iq.squeeze(D::Minus1)?;
        let c = n * (2.0 * PI).ln();
        iq.broadcast_add(&logdet)?.affine(-0.5, -0.5 * c)
    }

    /// Differential entropy `½[n log(2πe) + log|Σ|]` → `[...]`.
    pub fn entropy(&self) -> Result<Tensor> {
        let n = self.event_size() as f64;
        let logdet = self.covar.log_abs_det(self.jitter)?;
        logdet.affine(0.5, 0.5 * n * (2.0 * PI * std::f64::consts::E).ln())
    }

    /// Reparameterised sample `μ + L ε` — differentiable in `μ` and `Σ`
    /// (the "rsample" path). Returns `[s, ..., N]`.
    pub fn rsample(&self, s: usize) -> Result<Tensor> {
        let z = self.covar.zero_mean_samples(s, self.jitter)?; // [s, ..., N]
        z.broadcast_add(&self.mean)
    }

    /// Alias for [`rsample`](Self::rsample) — every sampler here is
    /// reparameterised (there is no detached path).
    pub fn sample(&self, s: usize) -> Result<Tensor> {
        self.rsample(s)
    }

    /// `KL(self ‖ other)` for two MVNs of the same event size:
    ///
    /// ```text
    ///   ½[ tr(Σ₂⁻¹Σ₁) + (μ₂−μ₁)ᵀΣ₂⁻¹(μ₂−μ₁) − n + log|Σ₂| − log|Σ₁| ]
    /// ```
    ///
    /// The trace term is computed as `Σᵢⱼ (L₂⁻¹L₁)ᵢⱼ²` — the squared
    /// Frobenius norm of the whitened factor — so no inverse is formed and
    /// the gradient path stays through the two Choleskys.
    pub fn kl_divergence(&self, other: &MultivariateNormal) -> Result<Tensor> {
        let n = self.event_size() as f64;
        let l1 = self.covar.cholesky(self.jitter)?;
        let l2 = other.covar.cholesky(other.jitter)?;

        // tr(Σ₂⁻¹Σ₁) = ‖L₂⁻¹L₁‖²_F
        let w = crate::linalg_diff::tri_solve_diff(&l2, &l1)?;
        let trace = w.sqr()?.sum(D::Minus1)?.sum(D::Minus1)?;

        // (μ₂−μ₁)ᵀΣ₂⁻¹(μ₂−μ₁)
        let dmu = other.mean.broadcast_sub(&self.mean)?.unsqueeze(D::Minus1)?;
        let maha = crate::linalg_diff::tri_solve_diff(&l2, &dmu)?
            .sqr()?
            .sum(D::Minus2)?
            .squeeze(D::Minus1)?;

        let ld1 = dense_diagonal(&l1)?.log()?.sum(D::Minus1)?.affine(2.0, 0.0)?;
        let ld2 = dense_diagonal(&l2)?.log()?.sum(D::Minus1)?.affine(2.0, 0.0)?;

        trace
            .broadcast_add(&maha)?
            .broadcast_add(&ld2)?
            .broadcast_sub(&ld1)?
            .affine(0.5, -0.5 * n)
    }

    /// Add independent observation noise `σ²` to the diagonal — the
    /// likelihood's `marginal()` in its simplest form.
    pub fn add_diagonal_noise(&self, noise: &Tensor) -> Result<MultivariateNormal> {
        let dense = self.covar.to_dense()?;
        let d = crate::operators::diag_embed(
            &noise.broadcast_as(self.covar.diagonal()?.shape())?.contiguous()?,
        )?;
        Ok(MultivariateNormal {
            mean: self.mean.clone(),
            covar: Box::new(DenseOperator::new(dense.broadcast_add(&d)?)),
            jitter: self.jitter,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Verification of multivariate and multitask probability distributions on candle tensors.

    use super::*;
    use candle_core::{DType, Device};

    fn spd(n: usize, dev: &Device) -> Tensor {
        let a = Tensor::randn(0f32, 1f32, (n, n), dev).unwrap();
        a.matmul(&a.t().unwrap())
            .unwrap()
            .add(&Tensor::eye(n, DType::F32, dev).unwrap().affine(n as f64, 0.0).unwrap())
            .unwrap()
    }

    /// Against the closed form for a 1-D Gaussian:
    /// `log p = −½[(x−μ)²/σ² + log σ² + log 2π]`.
    #[test]
    fn log_prob_matches_the_scalar_closed_form() {
        let dev = Device::Cpu;
        let mu = Tensor::new(&[1.5f32], &dev).unwrap();
        let var = 2.25f32;
        let mvn = MultivariateNormal::from_diag(
            mu.clone(),
            Tensor::new(&[var], &dev).unwrap(),
        )
        .with_jitter(0.0);

        let x = Tensor::new(&[3.0f32], &dev).unwrap();
        let got = mvn.log_prob(&x).unwrap().to_scalar::<f32>().unwrap();
        let d = 3.0 - 1.5;
        let want = -0.5 * (d * d / var + var.ln() + (2.0 * PI as f32).ln());
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }

    /// A diagonal MVN's log_prob must equal the sum of independent
    /// univariate log-probs.
    #[test]
    fn diagonal_log_prob_equals_sum_of_univariates() {
        let dev = Device::Cpu;
        let mu = Tensor::new(&[0.0f32, -1.0, 2.0], &dev).unwrap();
        let var = Tensor::new(&[1.0f32, 4.0, 0.25], &dev).unwrap();
        let x = Tensor::new(&[0.5f32, 0.0, 1.0], &dev).unwrap();

        let got = MultivariateNormal::from_diag(mu.clone(), var.clone())
            .with_jitter(0.0)
            .log_prob(&x)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let mv: Vec<f32> = mu.to_vec1().unwrap();
        let vv: Vec<f32> = var.to_vec1().unwrap();
        let xv: Vec<f32> = x.to_vec1().unwrap();
        let want: f32 = (0..3)
            .map(|i| {
                let d = xv[i] - mv[i];
                -0.5 * (d * d / vv[i] + vv[i].ln() + (2.0 * PI as f32).ln())
            })
            .sum();
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }

    /// The diagonal fast path and the dense path must agree — same
    /// distribution, different operator.
    #[test]
    fn diag_and_dense_covariance_agree() {
        let dev = Device::Cpu;
        let mu = Tensor::randn(0f32, 1f32, 4, &dev).unwrap();
        let var = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &dev).unwrap();
        let x = Tensor::randn(0f32, 1f32, 4, &dev).unwrap();

        let a = MultivariateNormal::from_diag(mu.clone(), var.clone()).with_jitter(1e-9);
        let dense = crate::operators::diag_embed(&var).unwrap();
        let b = MultivariateNormal::from_dense(mu, dense).with_jitter(1e-9);

        let la = a.log_prob(&x).unwrap().to_scalar::<f32>().unwrap();
        let lb = b.log_prob(&x).unwrap().to_scalar::<f32>().unwrap();
        assert!((la - lb).abs() < 1e-3, "{la} vs {lb}");

        let ea = a.entropy().unwrap().to_scalar::<f32>().unwrap();
        let eb = b.entropy().unwrap().to_scalar::<f32>().unwrap();
        assert!((ea - eb).abs() < 1e-3, "entropy {ea} vs {eb}");
    }

    /// KL to itself is 0; KL between different distributions is > 0
    /// (and matches the closed form in the scalar case).
    #[test]
    fn kl_divergence_is_zero_to_self_and_matches_scalar_form() {
        let dev = Device::Cpu;
        let mu = Tensor::randn(0f32, 1f32, 3, &dev).unwrap();
        let s = spd(3, &dev);
        let p = MultivariateNormal::from_dense(mu.clone(), s.clone()).with_jitter(1e-9);
        let q = MultivariateNormal::from_dense(mu, s).with_jitter(1e-9);
        let kl = p.kl_divergence(&q).unwrap().to_scalar::<f32>().unwrap();
        assert!(kl.abs() < 1e-3, "KL(p‖p) must vanish, got {kl}");

        // scalar closed form: log(σ₂/σ₁) + (σ₁²+(μ₁−μ₂)²)/(2σ₂²) − ½
        let (m1, v1, m2, v2) = (0.5f32, 1.0f32, -0.25f32, 2.0f32);
        let a = MultivariateNormal::from_diag(
            Tensor::new(&[m1], &dev).unwrap(),
            Tensor::new(&[v1], &dev).unwrap(),
        )
        .with_jitter(0.0);
        let b = MultivariateNormal::from_diag(
            Tensor::new(&[m2], &dev).unwrap(),
            Tensor::new(&[v2], &dev).unwrap(),
        )
        .with_jitter(0.0);
        let got = a.kl_divergence(&b).unwrap().to_scalar::<f32>().unwrap();
        let want = 0.5 * (v2.ln() - v1.ln()) + (v1 + (m1 - m2).powi(2)) / (2.0 * v2) - 0.5;
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }

    /// Samples must reproduce the covariance (statistical check with a
    /// large draw — the reparameterised path is `μ + Lε`).
    #[test]
    fn samples_have_the_requested_moments() {
        let dev = Device::Cpu;
        let mu = Tensor::new(&[1.0f32, -2.0], &dev).unwrap();
        let var = Tensor::new(&[4.0f32, 0.25], &dev).unwrap();
        let mvn = MultivariateNormal::from_diag(mu.clone(), var.clone()).with_jitter(0.0);

        let s = mvn.rsample(20_000).unwrap(); // [S, 2]
        assert_eq!(s.dims(), &[20_000, 2]);
        let m = s.mean(0).unwrap();
        let c = s.broadcast_sub(&m).unwrap().sqr().unwrap().mean(0).unwrap();

        let mv: Vec<f32> = m.to_vec1().unwrap();
        let cv: Vec<f32> = c.to_vec1().unwrap();
        assert!((mv[0] - 1.0).abs() < 0.1, "mean0 {}", mv[0]);
        assert!((mv[1] + 2.0).abs() < 0.05, "mean1 {}", mv[1]);
        assert!((cv[0] - 4.0).abs() < 0.2, "var0 {}", cv[0]);
        assert!((cv[1] - 0.25).abs() < 0.05, "var1 {}", cv[1]);
    }

    #[test]
    fn batched_log_prob_returns_one_value_per_batch_element() {
        let dev = Device::Cpu;
        let mu = Tensor::randn(0f32, 1f32, (5, 3), &dev).unwrap();
        let var = Tensor::rand(0.5f32, 2.0f32, (5, 3), &dev).unwrap();
        let mvn = MultivariateNormal::from_diag(mu.clone(), var).with_jitter(0.0);
        let x = Tensor::randn(0f32, 1f32, (5, 3), &dev).unwrap();
        let lp = mvn.log_prob(&x).unwrap();
        assert_eq!(lp.dims(), &[5]);
        assert!(lp.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }

    /// Gradients must reach both the mean and the covariance through
    /// log_prob — the property that makes this trainable.
    #[test]
    fn gradients_flow_through_log_prob() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let vmu = Var::from_tensor(&Tensor::zeros(3, DType::F32, &dev).unwrap()).unwrap();
        let vcov = Var::from_tensor(&spd(3, &dev)).unwrap();
        let mvn = MultivariateNormal::from_dense(
            vmu.as_tensor().clone(),
            vcov.as_tensor().clone(),
        )
        .with_jitter(1e-6);

        let x = Tensor::randn(0f32, 1f32, 3, &dev).unwrap();
        let loss = mvn.log_prob(&x).unwrap().neg().unwrap();
        let grads = loss.backward().unwrap();

        for (name, v) in [("mean", &vmu), ("covar", &vcov)] {
            let g = grads.get(v).unwrap_or_else(|| panic!("no grad for {name}"));
            let m = g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(m.is_finite() && m > 0.0, "{name} grad {m}");
        }
    }
}
