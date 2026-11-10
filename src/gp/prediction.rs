//! Inferno — Gaussian-process prediction interfaces and posterior strategy components.

use candle_core::{Result, Tensor, D};

use crate::linalg_diff::{cholesky_diff, tri_solve_diff, tri_solve_t_diff};
use crate::operators::{dense_diagonal, LinearOperator};

/// Cached exact-GP conditioning state for one training set.
pub struct DefaultPredictionStrategy {
    /// Lower Cholesky factor of `K_XX + Σ` — `[..., N, N]`.
    pub train_chol: Tensor,
    /// `(K_XX + Σ)⁻¹ (y − μ_X)` — `[..., N, 1]`.
    pub mean_cache: Tensor,
    /// Training prior mean `[..., N]`, retained for diagnostics.
    pub train_mean: Tensor,
    pub jitter: f64,
}

impl DefaultPredictionStrategy {
    /// Build the caches from the training prior and observations.
    ///
    /// * `train_covar` — `K_XX + Σ` (noise ALREADY added; call the
    ///   likelihood's `marginal` first, as GPyTorch does).
    /// * `train_mean` — `μ_X`, `[..., N]`.
    /// * `train_labels` — `y`, `[..., N]`.
    pub fn new(
        train_covar: &dyn LinearOperator,
        train_mean: &Tensor,
        train_labels: &Tensor,
        jitter: f64,
    ) -> Result<Self> {
        let n = train_covar.ncols();
        if train_mean.dim(D::Minus1)? != n || train_labels.dim(D::Minus1)? != n {
            return Err(candle_core::Error::Msg(format!(
                "DefaultPredictionStrategy: covariance is {n}×{n} but mean/labels have \
                 length {}/{} — the usual cause is a prior mean whose shape does not \
                 match the training targets",
                train_mean.dim(D::Minus1)?,
                train_labels.dim(D::Minus1)?
            )));
        }
        let dense = train_covar.to_dense()?;
        let train_chol = cholesky_diff(&dense, jitter)?;
        let offset = train_labels
            .to_dtype(dense.dtype())?
            .sub(train_mean)?
            .unsqueeze(D::Minus1)?; // [.., N, 1]
                                    // (K+Σ)⁻¹(y−μ) = L⁻ᵀ(L⁻¹(y−μ))
        let mean_cache = tri_solve_t_diff(&train_chol, &tri_solve_diff(&train_chol, &offset)?)?;
        Ok(Self {
            train_chol,
            mean_cache,
            train_mean: train_mean.clone(),
            jitter,
        })
    }

    pub fn num_train(&self) -> usize {
        self.train_chol.dims()[self.train_chol.rank() - 1]
    }

    /// `μ* = μ_test + K_*X (K_XX + Σ)⁻¹ (y − μ_X)` — one matvec against
    /// the cache.
    ///
    /// `test_mean: [..., M]`, `test_train_covar: [..., M, N]`.
    pub fn exact_predictive_mean(
        &self,
        test_mean: &Tensor,
        test_train_covar: &Tensor,
    ) -> Result<Tensor> {
        test_train_covar
            .broadcast_matmul(&self.mean_cache)?
            .squeeze(D::Minus1)?
            .broadcast_add(test_mean)
    }

    /// `Σ* = K_** − Rᵀ R`, `R = L⁻¹ K_X*`.
    ///
    /// Formed as a root product so the correction is PSD by construction.
    ///
    /// `test_test_covar: [..., M, M]`, `test_train_covar: [..., M, N]`.
    pub fn exact_predictive_covar(
        &self,
        test_test_covar: &Tensor,
        test_train_covar: &Tensor,
    ) -> Result<Tensor> {
        // R = L⁻¹ K_X*   ([.., N, M])
        let k_xstar = test_train_covar
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?;
        let r = tri_solve_diff(&self.train_chol, &k_xstar)?;
        let correction = r
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(&r)?;
        test_test_covar.sub(&correction)
    }

    /// Both moments together — the caches are shared, so this is strictly
    /// cheaper than the two calls separately.
    pub fn exact_prediction(
        &self,
        test_mean: &Tensor,
        test_test_covar: &Tensor,
        test_train_covar: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Ok((
            self.exact_predictive_mean(test_mean, test_train_covar)?,
            self.exact_predictive_covar(test_test_covar, test_train_covar)?,
        ))
    }

    /// Predictive VARIANCES only — `[..., M]`.
    ///
    /// `diag(Σ*) = diag(K_**) − Σᵢ Rᵢ²` avoids forming the `[M, M]`
    /// correction at all, which is the right call when only error bars are
    /// wanted.
    pub fn exact_predictive_variance(
        &self,
        test_test_diag: &Tensor,
        test_train_covar: &Tensor,
    ) -> Result<Tensor> {
        let k_xstar = test_train_covar
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?;
        let r = tri_solve_diff(&self.train_chol, &k_xstar)?; // [.., N, M]
        test_test_diag.sub(&r.sqr()?.sum(D::Minus2)?)
    }

    /// `log p(y)` of the training data under the prior — the exact-GP
    /// marginal likelihood, sharing the cached factor:
    ///
    /// ```text
    ///   −½[ (y−μ)ᵀ(K+Σ)⁻¹(y−μ) + log|K+Σ| + n log 2π ]
    /// ```
    pub fn log_marginal_likelihood(&self, train_labels: &Tensor) -> Result<Tensor> {
        let n = self.num_train() as f64;
        let offset = train_labels
            .to_dtype(self.train_chol.dtype())?
            .sub(&self.train_mean)?
            .unsqueeze(D::Minus1)?;
        let quad = tri_solve_diff(&self.train_chol, &offset)?
            .sqr()?
            .sum(D::Minus2)?
            .squeeze(D::Minus1)?;
        let logdet = dense_diagonal(&self.train_chol)?
            .log()?
            .sum(D::Minus1)?
            .affine(2.0, 0.0)?;
        quad.broadcast_add(&logdet)?
            .affine(-0.5, -0.5 * n * (2.0 * std::f64::consts::PI).ln())
    }
}

#[cfg(test)]
mod tests {
    //! Verification of gaussian-process prediction interfaces and posterior strategy components.

    use super::*;
    use crate::operators::{diag_embed, DenseOperator};
    use candle_core::{DType, Device};

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// An RBF kernel block, so the test exercises a realistic PD matrix.
    fn rbf(x1: &Tensor, x2: &Tensor) -> Tensor {
        let s1 = x1
            .sqr()
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .unsqueeze(D::Minus1)
            .unwrap();
        let s2 = x2
            .sqr()
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .unsqueeze(D::Minus2)
            .unwrap();
        let cross = x1
            .matmul(
                &x2.transpose(D::Minus2, D::Minus1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        let d2 = s1
            .broadcast_add(&s2)
            .unwrap()
            .sub(&cross.affine(2.0, 0.0).unwrap())
            .unwrap();
        d2.affine(-0.5, 0.0).unwrap().exp().unwrap()
    }

    /// The cached prediction must equal the naive
    /// `K_*X (K+Σ)⁻¹ (y−μ)` computed from scratch.
    #[test]
    fn predictive_mean_matches_the_naive_formula() {
        let dev = Device::Cpu;
        let (n, m) = (6usize, 4usize);
        let xtr = Tensor::randn(0f32, 1f32, (n, 2), &dev).unwrap();
        let xte = Tensor::randn(0f32, 1f32, (m, 2), &dev).unwrap();

        let k_xx = rbf(&xtr, &xtr)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(0.1, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let k_sx = rbf(&xte, &xtr);
        let mu_tr = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let mu_te = Tensor::zeros(m, DType::F32, &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, n, &dev).unwrap();

        let strat =
            DefaultPredictionStrategy::new(&DenseOperator::new(k_xx.clone()), &mu_tr, &y, 1e-9)
                .unwrap();
        let got = strat.exact_predictive_mean(&mu_te, &k_sx).unwrap();
        assert_eq!(got.dims(), &[m]);

        // naive reference via an explicit solve
        let alpha = DenseOperator::new(k_xx)
            .cholesky_solve(&y.unsqueeze(D::Minus1).unwrap(), 1e-9)
            .unwrap();
        let want = k_sx.matmul(&alpha).unwrap().squeeze(D::Minus1).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-3,
            "mean mismatch: {}",
            max_abs(&got, &want)
        );
    }

    /// The covariance correction must match the naive form AND stay PSD.
    #[test]
    fn predictive_covar_matches_naive_and_stays_psd() {
        let dev = Device::Cpu;
        let (n, m) = (6usize, 4usize);
        let xtr = Tensor::randn(0f32, 1f32, (n, 2), &dev).unwrap();
        let xte = Tensor::randn(0f32, 1f32, (m, 2), &dev).unwrap();

        let k_xx = rbf(&xtr, &xtr)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(0.1, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let k_ss = rbf(&xte, &xte);
        let k_sx = rbf(&xte, &xtr);
        let mu = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, n, &dev).unwrap();

        let strat =
            DefaultPredictionStrategy::new(&DenseOperator::new(k_xx.clone()), &mu, &y, 1e-9)
                .unwrap();
        let got = strat.exact_predictive_covar(&k_ss, &k_sx).unwrap();

        // naive: K_** − K_*X (K+Σ)⁻¹ K_X*
        let sol = DenseOperator::new(k_xx)
            .cholesky_solve(&k_sx.t().unwrap().contiguous().unwrap(), 1e-9)
            .unwrap();
        let want = k_ss.sub(&k_sx.matmul(&sol).unwrap()).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-3,
            "covar mismatch: {}",
            max_abs(&got, &want)
        );

        // PSD: every predictive variance is non-negative — the property
        // the root formulation exists to guarantee.
        let d: Vec<f32> = dense_diagonal(&got).unwrap().to_vec1().unwrap();
        assert!(
            d.iter().all(|v| *v > -1e-5),
            "negative predictive variance: {d:?}"
        );
    }

    /// The variance-only path must equal the diagonal of the full path
    /// without ever forming the [M, M] correction.
    #[test]
    fn variance_only_path_matches_the_full_diagonal() {
        let dev = Device::Cpu;
        let (n, m) = (5usize, 3usize);
        let xtr = Tensor::randn(0f32, 1f32, (n, 2), &dev).unwrap();
        let xte = Tensor::randn(0f32, 1f32, (m, 2), &dev).unwrap();
        let k_xx = rbf(&xtr, &xtr)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(0.2, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let k_ss = rbf(&xte, &xte);
        let k_sx = rbf(&xte, &xtr);
        let mu = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, n, &dev).unwrap();

        let strat =
            DefaultPredictionStrategy::new(&DenseOperator::new(k_xx), &mu, &y, 1e-9).unwrap();
        let full = strat.exact_predictive_covar(&k_ss, &k_sx).unwrap();
        let var = strat
            .exact_predictive_variance(&dense_diagonal(&k_ss).unwrap(), &k_sx)
            .unwrap();
        assert_eq!(var.dims(), &[m]);
        assert!(max_abs(&var, &dense_diagonal(&full).unwrap()) < 1e-4);
    }

    /// Conditioning ON a training point must reproduce that point (up to
    /// the noise) and collapse its predictive variance — the sanity check
    /// that the caches are wired the right way round.
    #[test]
    fn conditioning_at_a_training_point_reproduces_it() {
        let dev = Device::Cpu;
        let n = 5usize;
        let xtr = Tensor::new(&[[0.0f32], [1.0], [2.0], [3.0], [4.0]], &dev).unwrap();
        // tiny noise so the interpolation is near-exact
        let k_xx = rbf(&xtr, &xtr)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(1e-4, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let mu = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let y = Tensor::new(&[1.0f32, -1.0, 0.5, 2.0, -0.5], &dev).unwrap();

        let strat =
            DefaultPredictionStrategy::new(&DenseOperator::new(k_xx), &mu, &y, 1e-10).unwrap();

        // predict AT the training inputs
        let k_sx = rbf(&xtr, &xtr);
        let pm = strat.exact_predictive_mean(&mu, &k_sx).unwrap();
        assert!(
            max_abs(&pm, &y) < 5e-3,
            "must interpolate: {}",
            max_abs(&pm, &y)
        );

        let var = strat
            .exact_predictive_variance(&dense_diagonal(&rbf(&xtr, &xtr)).unwrap(), &k_sx)
            .unwrap();
        let vmax = var
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            vmax < 1e-2,
            "variance must collapse at training points, got {vmax}"
        );
    }

    /// The exact log marginal likelihood must match a direct computation.
    #[test]
    fn log_marginal_likelihood_matches_direct_computation() {
        let dev = Device::Cpu;
        let n = 5usize;
        let x = Tensor::randn(0f32, 1f32, (n, 2), &dev).unwrap();
        let k = rbf(&x, &x)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(0.3, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let mu = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, n, &dev).unwrap();

        let strat =
            DefaultPredictionStrategy::new(&DenseOperator::new(k.clone()), &mu, &y, 1e-9).unwrap();
        let got = strat
            .log_marginal_likelihood(&y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // same thing through the MVN
        let want = crate::distributions::MultivariateNormal::from_dense(mu, k)
            .with_jitter(1e-9)
            .log_prob(&y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((got - want).abs() < 2e-3, "{got} vs {want}");
    }

    #[test]
    fn mismatched_label_length_is_rejected() {
        let dev = Device::Cpu;
        let k = Tensor::eye(4, DType::F32, &dev).unwrap();
        let mu = Tensor::zeros(4, DType::F32, &dev).unwrap();
        let y = Tensor::zeros(3, DType::F32, &dev).unwrap(); // wrong length
        assert!(
            DefaultPredictionStrategy::new(&DenseOperator::new(k), &mu, &y, 1e-9).is_err(),
            "a 3-vector of labels must not pair with a 4×4 covariance"
        );
    }

    /// Gradients must flow through the caches to the kernel matrix — the
    /// exact-GP marginal likelihood has to be trainable.
    #[test]
    fn gradients_flow_through_the_prediction_caches() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let n = 4usize;
        let x = Tensor::randn(0f32, 1f32, (n, 2), &dev).unwrap();
        let base = rbf(&x, &x)
            .add(
                &Tensor::eye(n, DType::F32, &dev)
                    .unwrap()
                    .affine(0.5, 0.0)
                    .unwrap(),
            )
            .unwrap();
        let v = Var::from_tensor(&base).unwrap();
        let mu = Tensor::zeros(n, DType::F32, &dev).unwrap();
        let y = Tensor::randn(0f32, 1f32, n, &dev).unwrap();

        let strat = DefaultPredictionStrategy::new(
            &DenseOperator::new(v.as_tensor().clone()),
            &mu,
            &y,
            1e-6,
        )
        .unwrap();
        let loss = strat.log_marginal_likelihood(&y).unwrap().neg().unwrap();
        let grads = loss.backward().unwrap();
        let g = grads
            .get(&v)
            .expect("gradient must reach the training covariance");
        let m = g
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite() && m > 0.0, "grad {m}");
    }
}
