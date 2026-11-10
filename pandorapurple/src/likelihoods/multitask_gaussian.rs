//! Inferno — Gaussian observation likelihoods for correlated multitask outputs.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::Likelihood;
use crate::distributions::{MultitaskMultivariateNormal, MultivariateNormal};
use crate::model::softplus;
use crate::operators::{
    DenseOperator, DiagOperator, KroneckerDiagOperator, KroneckerOperator, LinearOperator,
    RootOperator,
};

/// Inter-task noise parameterisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskNoiseRank {
    /// `rank = 0`: a diagonal `D_T` — one variance per task.
    Diagonal,
    /// `rank = k > 0`: `D_T = F Fᵀ` with `F: [T, k]` — full cross-task
    /// covariance of rank `k`.
    LowRank(usize),
}

pub struct MultitaskGaussianLikelihood {
    pub num_tasks: usize,
    pub rank: TaskNoiseRank,
    /// `raw_task_noises [T]` (Diagonal) — constrained by softplus + floor.
    raw_task_noises: Option<Tensor>,
    /// `task_noise_covar_factor [T, k]` (LowRank) — unconstrained.
    task_noise_covar_factor: Option<Tensor>,
    /// `raw_noise [1]` — the shared global `σ²`, softplus + floor.
    raw_noise: Option<Tensor>,
    /// GPyTorch's `GreaterThan(1e-4)` noise constraint floor.
    pub noise_floor: f64,
    pub has_global_noise: bool,
    pub has_task_noise: bool,
}

impl MultitaskGaussianLikelihood {
    /// Load from a `VarBuilder` rooted at the Python likelihood submodule
    /// (`raw_task_noises` / `task_noise_covar_factor` / `raw_noise`).
    pub fn load(
        vb: VarBuilder,
        num_tasks: usize,
        rank: TaskNoiseRank,
        has_global_noise: bool,
        has_task_noise: bool,
    ) -> Result<Self> {
        if !has_task_noise && !has_global_noise {
            return Err(candle_core::Error::Msg(
                "MultitaskGaussianLikelihood: at least one of has_task_noise / \
                 has_global_noise must be set — a likelihood with no noise terms \
                 is degenerate"
                    .into(),
            ));
        }
        let (mut raw_task_noises, mut task_noise_covar_factor) = (None, None);
        if has_task_noise {
            match rank {
                TaskNoiseRank::Diagonal => {
                    raw_task_noises = Some(vb.get_with_hints(
                        num_tasks,
                        "raw_task_noises",
                        candle_nn::init::Init::Const(0.0),
                    )?);
                }
                TaskNoiseRank::LowRank(k) => {
                    if k > num_tasks {
                        return Err(candle_core::Error::Msg(format!(
                            "task-noise rank ({k}) cannot exceed num_tasks ({num_tasks})"
                        )));
                    }
                    task_noise_covar_factor = Some(crate::model::parametrize::sobol_randn(&vb, "task_noise_covar_factor", (num_tasks, k), 0.0, 1.0)?);
                }
            }
        }
        let raw_noise = if has_global_noise {
            Some(vb.get_with_hints(1, "raw_noise", candle_nn::init::Init::Const(0.0))?)
        } else {
            None
        };
        Ok(Self {
            num_tasks,
            rank,
            raw_task_noises,
            task_noise_covar_factor,
            raw_noise,
            noise_floor: 1e-4,
            has_global_noise,
            has_task_noise,
        })
    }

    /// Fresh (untrained) instance — the standalone/testing constructor.
    pub fn new(
        num_tasks: usize,
        rank: TaskNoiseRank,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, dtype, device);
        Self::load(vb, num_tasks, rank, true, true)
    }

    /// Global noise `σ² = softplus(raw) + floor` → scalar.
    pub fn noise(&self) -> Result<Option<Tensor>> {
        match &self.raw_noise {
            None => Ok(None),
            Some(r) => Ok(Some(softplus(r)?.affine(1.0, self.noise_floor)?.squeeze(0)?)),
        }
    }

    /// Per-task noises `[T]` (Diagonal rank only).
    pub fn task_noises(&self) -> Result<Option<Tensor>> {
        match &self.raw_task_noises {
            None => Ok(None),
            Some(r) => Ok(Some(softplus(r)?.affine(1.0, self.noise_floor)?)),
        }
    }

    /// The inter-task noise covariance `D_T [T, T]` — diagonal or `F Fᵀ`.
    pub fn task_noise_covar(&self) -> Result<Option<Tensor>> {
        match (&self.raw_task_noises, &self.task_noise_covar_factor) {
            (Some(_), _) => Ok(Some(crate::operators::diag_embed(
                &self.task_noises()?.unwrap(),
            )?)),
            (_, Some(f)) => Ok(Some(f.matmul(&f.t()?)?)),
            _ => Ok(None),
        }
    }

    /// The task factor with the global noise FOLDED IN:
    /// `D_T + σ² I_T`.
    ///
    /// This is the algebraic step that keeps the whole noise operator
    /// Kronecker-structured (`I_n ⊗ (D_T + σ²I)`) rather than forcing an
    /// `[N·T, N·T]` sum.
    fn task_factor(&self) -> Result<Box<dyn LinearOperator>> {
        let noise = self.noise()?;
        match (&self.raw_task_noises, &self.task_noise_covar_factor) {
            // rank 0: diagonal + σ² stays diagonal
            (Some(_), _) => {
                let mut d = self.task_noises()?.unwrap(); // [T]
                if let Some(s) = &noise {
                    d = d.broadcast_add(s)?;
                }
                Ok(Box::new(DiagOperator::new(d)))
            }
            // rank k: F Fᵀ + σ²I — low-rank plus diagonal
            (_, Some(f)) => {
                let root = RootOperator::new(f.clone());
                match &noise {
                    None => Ok(Box::new(root)),
                    Some(s) => {
                        let d = Tensor::ones(self.num_tasks, f.dtype(), f.device())?
                            .broadcast_mul(s)?;
                        Ok(Box::new(crate::operators::AddedDiagOperator::new(
                            Box::new(root),
                            DiagOperator::new(d),
                        )))
                    }
                }
            }
            // task noise disabled: pure global σ²I_T
            _ => {
                let s = noise.ok_or_else(|| {
                    candle_core::Error::Msg("no noise terms configured".into())
                })?;
                let d = Tensor::ones(self.num_tasks, s.dtype(), s.device())?.broadcast_mul(&s)?;
                Ok(Box::new(DiagOperator::new(d)))
            }
        }
    }

    /// The full noise covariance over `N` data points as a lazy operator —
    /// GPyTorch's `_shaped_noise_covar`.
    ///
    /// `interleaved` selects the Kronecker order and MUST match the
    /// distribution's flattening (see the module docs).
    pub fn shaped_noise_covar(
        &self,
        n: usize,
        interleaved: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Box<dyn LinearOperator>> {
        let eye_n = DiagOperator::new(Tensor::ones(n, dtype, device)?);
        let task = self.task_factor()?;

        // When both factors are diagonal the product is diagonal — keep it
        // that way (O(N·T)) instead of going through the dense Kronecker.
        let task_is_diag = matches!(self.rank, TaskNoiseRank::Diagonal) || !self.has_task_noise;
        if task_is_diag {
            let td = DiagOperator::new(task.diagonal()?);
            let factors =
                if interleaved { vec![eye_n, td] } else { vec![td, eye_n] };
            return Ok(Box::new(KroneckerDiagOperator::new(factors)));
        }

        let factors: Vec<Box<dyn LinearOperator>> = if interleaved {
            vec![Box::new(eye_n), task]
        } else {
            vec![task, Box::new(eye_n)]
        };
        Ok(Box::new(KroneckerOperator::new(factors)))
    }

    /// `q(f) → q(y)` for a multitask latent: add `Σ_noise` to the
    /// covariance, keeping the mean and the interleaving convention.
    pub fn marginal_multitask(
        &self,
        latent: &MultitaskMultivariateNormal,
    ) -> Result<MultitaskMultivariateNormal> {
        let dense = latent.covar.to_dense()?;
        let noise = self.shaped_noise_covar(
            latent.num_data,
            latent.interleaved,
            dense.dtype(),
            dense.device(),
        )?;
        let sum = dense.broadcast_add(&noise.to_dense()?)?;
        Ok(MultitaskMultivariateNormal {
            mean: latent.mean.clone(),
            covar: Box::new(DenseOperator::new(sum)),
            num_tasks: latent.num_tasks,
            num_data: latent.num_data,
            interleaved: latent.interleaved,
            jitter: latent.jitter,
        })
    }

    /// Per-output noise variances `[T]` — the diagonal of `D_T + σ²I`,
    /// which is what the moment-based ELL needs.
    fn task_noise_diagonal(&self) -> Result<Tensor> {
        self.task_factor()?.diagonal()
    }
}

impl Likelihood for MultitaskGaussianLikelihood {
    /// `E_q[log N(y | f, Σ_noise)]` for a diagonal-in-`n` noise model:
    ///
    /// ```text
    ///   −½ Σ_t [ ((y−μ)² + σ_f²)/σ_n²  +  log σ_n²  +  log 2π ]
    /// ```
    ///
    /// The `σ_f²` term is what makes this the EXPECTED log-prob rather
    /// than a plug-in: the latent variance inflates the residual.
    fn expected_log_prob(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor> {
        let nt = self.task_noise_diagonal()?.to_dtype(mean.dtype())?; // [T]
        let resid = y.to_dtype(mean.dtype())?.sub(mean)?.sqr()?; // [.., N, T]
        let num = resid.broadcast_add(variance)?;
        let quad = num.broadcast_div(&nt)?;
        let logdet = nt.log()?;
        let c = (2.0 * std::f64::consts::PI).ln();
        quad.broadcast_add(&logdet)?
            .affine(-0.5, -0.5 * c)?
            .sum(D::Minus1) // over tasks → [.., N]
    }

    /// Plug-in predictive: the latent variance joins the noise instead of
    /// inflating the residual — `log N(y | μ, σ_f² + σ_n²)`.
    fn log_marginal(&self, y: &Tensor, mean: &Tensor, variance: &Tensor) -> Result<Tensor> {
        let nt = self.task_noise_diagonal()?.to_dtype(mean.dtype())?;
        let total = variance.broadcast_add(&nt)?;
        let resid = y.to_dtype(mean.dtype())?.sub(mean)?.sqr()?;
        let c = (2.0 * std::f64::consts::PI).ln();
        resid
            .div(&total)?
            .add(&total.log()?)?
            .affine(-0.5, -0.5 * c)?
            .sum(D::Minus1)
    }

    /// Single-task / already-flattened path: add the noise diagonal.
    fn marginal(&self, latent: &MultivariateNormal) -> Result<MultivariateNormal> {
        let n_flat = latent.event_size();
        if n_flat % self.num_tasks != 0 {
            return Err(candle_core::Error::Msg(format!(
                "MultitaskGaussianLikelihood::marginal: event size {n_flat} is not a \
                 multiple of num_tasks {}",
                self.num_tasks
            )));
        }
        let n = n_flat / self.num_tasks;
        let dense = latent.covar.to_dense()?;
        // Interleaved is GPyTorch's default for a flattened multitask vector.
        let noise = self.shaped_noise_covar(n, true, dense.dtype(), dense.device())?;
        Ok(MultivariateNormal {
            mean: latent.mean.clone(),
            covar: Box::new(DenseOperator::new(dense.broadcast_add(&noise.to_dense()?)?)),
            jitter: latent.jitter,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Verification of gaussian observation likelihoods for correlated multitask outputs.

    use super::*;
    use crate::operators::dense_diagonal;
    use candle_core::Device;

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    }

    fn lik(t: usize, rank: TaskNoiseRank, dev: &Device) -> MultitaskGaussianLikelihood {
        MultitaskGaussianLikelihood::new(t, rank, DType::F32, dev).unwrap()
    }

    /// The Kronecker fold `I_n ⊗ D_T + σ²I = I_n ⊗ (D_T + σ²I)` must hold
    /// exactly against the explicitly-built dense noise matrix.
    #[test]
    fn kronecker_fold_matches_explicit_dense_noise() {
        let dev = Device::Cpu;
        let (n, t) = (4usize, 3usize);
        let l = lik(t, TaskNoiseRank::Diagonal, &dev);

        let op = l.shaped_noise_covar(n, true, DType::F32, &dev).unwrap();
        assert_eq!(op.shape(), vec![n * t, n * t]);
        let got = op.to_dense().unwrap();

        // Explicit reference: kron(I_n, D_T) + σ²I
        let d_t = l.task_noises().unwrap().unwrap();
        let sigma2 = l.noise().unwrap().unwrap().to_scalar::<f32>().unwrap();
        let eye_n = crate::operators::diag_embed(&Tensor::ones(n, DType::F32, &dev).unwrap()).unwrap();
        let kron =
            crate::operators::kronecker::kron_dense(&eye_n, &crate::operators::diag_embed(&d_t).unwrap())
                .unwrap();
        let want = kron
            .add(
                &Tensor::eye(n * t, DType::F32, &dev)
                    .unwrap()
                    .affine(sigma2 as f64, 0.0)
                    .unwrap(),
            )
            .unwrap();
        assert!(max_abs(&got, &want) < 1e-5, "fold mismatch: {}", max_abs(&got, &want));
    }

    /// rank-0 noise must stay DIAGONAL — that is the O(N·T) claim.
    #[test]
    fn rank_zero_noise_is_diagonal() {
        let dev = Device::Cpu;
        let (n, t) = (5usize, 4usize);
        let l = lik(t, TaskNoiseRank::Diagonal, &dev);
        let op = l.shaped_noise_covar(n, true, DType::F32, &dev).unwrap();
        let dense = op.to_dense().unwrap();
        let diag = crate::operators::diag_embed(&dense_diagonal(&dense).unwrap()).unwrap();
        assert!(
            max_abs(&dense, &diag) < 1e-7,
            "rank-0 noise must have no off-diagonal mass"
        );
    }

    /// Interleaved vs task-major must produce DIFFERENT matrices (the
    /// Kronecker order matters) but the same multiset of diagonal entries.
    #[test]
    fn interleaving_changes_the_noise_layout() {
        let dev = Device::Cpu;
        let (n, t) = (3usize, 2usize);
        let l = lik(t, TaskNoiseRank::Diagonal, &dev);
        // Give the tasks distinct noises so the orders are distinguishable.
        let l = MultitaskGaussianLikelihood {
            raw_task_noises: Some(Tensor::new(&[-2.0f32, 2.0], &dev).unwrap()),
            ..l
        };

        let a = l.shaped_noise_covar(n, true, DType::F32, &dev).unwrap().diagonal().unwrap();
        let b = l.shaped_noise_covar(n, false, DType::F32, &dev).unwrap().diagonal().unwrap();
        let av: Vec<f32> = a.to_vec1().unwrap();
        let bv: Vec<f32> = b.to_vec1().unwrap();
        assert_ne!(av, bv, "the two Kronecker orders must differ");

        // interleaved (I ⊗ D): task noises cycle fastest
        assert!((av[0] - av[2]).abs() < 1e-6 && (av[1] - av[3]).abs() < 1e-6);
        // task-major (D ⊗ I): task 0's block first
        assert!((bv[0] - bv[1]).abs() < 1e-6 && (bv[0] - bv[2]).abs() < 1e-6);

        let (mut sa, mut sb) = (av.clone(), bv.clone());
        sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
        sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (x, y) in sa.iter().zip(sb.iter()) {
            assert!((x - y).abs() < 1e-6, "same entries, different order");
        }
    }

    /// Low-rank task noise must produce genuine cross-task correlation.
    #[test]
    fn low_rank_task_noise_has_off_diagonal_structure() {
        let dev = Device::Cpu;
        let t = 4usize;
        let l = lik(t, TaskNoiseRank::LowRank(2), &dev);
        let d_t = l.task_noise_covar().unwrap().unwrap();
        assert_eq!(d_t.dims(), &[t, t]);

        // D_T = F Fᵀ is symmetric PSD with (generically) non-zero off-diagonals
        let sym = d_t.sub(&d_t.t().unwrap()).unwrap().abs().unwrap().max_all().unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(sym < 1e-5, "F Fᵀ must be symmetric");
        let off = d_t
            .sub(&crate::operators::diag_embed(&dense_diagonal(&d_t).unwrap()).unwrap())
            .unwrap()
            .abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(off > 1e-6, "low-rank noise should correlate tasks");

        // and the full noise operator is NOT diagonal
        let op = l.shaped_noise_covar(3, true, DType::F32, &dev).unwrap();
        let dense = op.to_dense().unwrap();
        let dg = crate::operators::diag_embed(&dense_diagonal(&dense).unwrap()).unwrap();
        assert!(max_abs(&dense, &dg) > 1e-6, "rank-k noise must be non-diagonal");
    }

    /// The ELL must (a) reduce to the plain Gaussian log-prob when the
    /// latent variance is zero, and (b) be strictly lower when it is not —
    /// that gap is exactly the Jensen penalty.
    #[test]
    fn expected_log_prob_reduces_to_log_prob_at_zero_variance() {
        let dev = Device::Cpu;
        let t = 2usize;
        let l = lik(t, TaskNoiseRank::Diagonal, &dev);
        let y = Tensor::new(&[[0.5f32, -0.5]], &dev).unwrap();
        let m = Tensor::new(&[[0.2f32, 0.1]], &dev).unwrap();
        let zero = Tensor::zeros((1, t), DType::F32, &dev).unwrap();

        let ell0 = Likelihood::expected_log_prob(&l, &y, &m, &zero).unwrap();
        let lm0 = Likelihood::log_marginal(&l, &y, &m, &zero).unwrap();
        assert!(max_abs(&ell0, &lm0) < 1e-5, "at σ_f²=0 the two must coincide");

        // closed form check
        let nt: Vec<f32> = l.task_noise_diagonal().unwrap().to_vec1().unwrap();
        let want: f32 = [(0.5f32 - 0.2, nt[0]), (-0.5f32 - 0.1, nt[1])]
            .iter()
            .map(|(d, s)| -0.5 * (d * d / s + s.ln() + (2.0 * std::f32::consts::PI).ln()))
            .sum();
        assert!((ell0.to_vec1::<f32>().unwrap()[0] - want).abs() < 1e-4);

        // positive latent variance strictly lowers the ELL
        let v = Tensor::new(&[[0.3f32, 0.4]], &dev).unwrap();
        let ell = Likelihood::expected_log_prob(&l, &y, &m, &v).unwrap()
            .to_vec1::<f32>().unwrap()[0];
        assert!(ell < ell0.to_vec1::<f32>().unwrap()[0], "variance must cost ELL");
    }

    /// `marginal` on a multitask latent must add exactly the noise operator.
    #[test]
    fn marginal_adds_the_noise_covariance() {
        let dev = Device::Cpu;
        let (n, t) = (3usize, 2usize);
        let l = lik(t, TaskNoiseRank::Diagonal, &dev);

        let mean = Tensor::zeros((n, t), DType::F32, &dev).unwrap();
        let base = crate::operators::diag_embed(
            &Tensor::ones(n * t, DType::F32, &dev).unwrap(),
        )
        .unwrap();
        let latent = MultitaskMultivariateNormal::new(
            mean,
            Box::new(DenseOperator::new(base.clone())),
            true,
        )
        .unwrap();

        let marg = l.marginal_multitask(&latent).unwrap();
        let delta = marg.covar.to_dense().unwrap().sub(&base).unwrap();
        let noise = l.shaped_noise_covar(n, true, DType::F32, &dev).unwrap().to_dense().unwrap();
        assert!(max_abs(&delta, &noise) < 1e-6, "marginal must add exactly Σ_noise");
        // the mean is untouched
        assert!(max_abs(&marg.mean, &latent.mean) < 1e-9);
    }

    #[test]
    fn rejects_a_likelihood_with_no_noise_terms() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let e = MultitaskGaussianLikelihood::load(vb, 3, TaskNoiseRank::Diagonal, false, false);
        assert!(e.is_err(), "no-noise likelihood must be rejected");
    }

    #[test]
    fn rejects_rank_greater_than_num_tasks() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let e = MultitaskGaussianLikelihood::load(vb, 3, TaskNoiseRank::LowRank(5), true, true);
        assert!(e.is_err(), "rank > num_tasks must be rejected");
    }

    /// Gradients must reach both noise parameterisations through the ELL.
    #[test]
    fn gradients_reach_the_noise_parameters() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let t = 3usize;
        let vraw = Var::from_tensor(&Tensor::zeros(t, DType::F32, &dev).unwrap()).unwrap();
        let vglob = Var::from_tensor(&Tensor::zeros(1, DType::F32, &dev).unwrap()).unwrap();
        let l = MultitaskGaussianLikelihood {
            num_tasks: t,
            rank: TaskNoiseRank::Diagonal,
            raw_task_noises: Some(vraw.as_tensor().clone()),
            task_noise_covar_factor: None,
            raw_noise: Some(vglob.as_tensor().clone()),
            noise_floor: 1e-4,
            has_global_noise: true,
            has_task_noise: true,
        };
        let y = Tensor::randn(0f32, 1f32, (4, t), &dev).unwrap();
        let m = Tensor::randn(0f32, 1f32, (4, t), &dev).unwrap();
        let v = Tensor::rand(0.1f32, 0.5f32, (4, t), &dev).unwrap();

        let loss = Likelihood::expected_log_prob(&l, &y, &m, &v).unwrap().sum_all().unwrap().neg().unwrap();
        let grads = loss.backward().unwrap();
        for (name, var) in [("task", &vraw), ("global", &vglob)] {
            let g = grads.get(var).unwrap_or_else(|| panic!("no grad for {name} noise"));
            let mx = g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(mx.is_finite() && mx > 0.0, "{name} grad {mx}");
        }
    }
}
