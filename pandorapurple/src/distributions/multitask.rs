//! Inferno — Multitask Gaussian distributions with structured covariance operators.

use candle_core::{Result, Tensor, D};

use super::MultivariateNormal;
use crate::operators::{DenseOperator, LinearOperator};

/// `N(M, Σ)` over an `[..., N, T]` matrix of outputs.
pub struct MultitaskMultivariateNormal {
    /// `[..., N, T]`
    pub mean: Tensor,
    /// Covariance over the flattened `N·T` vector.
    pub covar: Box<dyn LinearOperator>,
    pub num_tasks: usize,
    pub num_data: usize,
    /// Flattening convention — see the module docs.
    pub interleaved: bool,
    pub jitter: f64,
}

impl MultitaskMultivariateNormal {
    /// `mean: [..., N, T]`, `covar: [..., N·T, N·T]`.
    pub fn new(mean: Tensor, covar: Box<dyn LinearOperator>, interleaved: bool) -> Result<Self> {
        let d = mean.dims();
        let (num_data, num_tasks) = (d[d.len() - 2], d[d.len() - 1]);
        let nt = covar.ncols();
        if nt != num_data * num_tasks {
            return Err(candle_core::Error::Msg(format!(
                "MultitaskMultivariateNormal: covariance is {nt}×{nt} but mean is \
                 [{num_data}, {num_tasks}] (expected {}×{})",
                num_data * num_tasks,
                num_data * num_tasks
            )));
        }
        Ok(Self {
            mean,
            covar,
            num_tasks,
            num_data,
            interleaved,
            jitter: crate::operators::DEFAULT_JITTER,
        })
    }

    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// GPyTorch `from_batch_mvn(mvn, task_dim)`: reinterpret a BATCH of
    /// per-task MVNs as one multitask MVN.
    ///
    /// Input mean is `[T, N]` (task-major batch) and covariance `[T, N, N]`.
    /// The tasks are treated as INDEPENDENT — the resulting `N·T` covariance
    /// is block-diagonal by task — which is precisely the assumption
    /// `_assemble_mvn` makes on its cheap (non-`full_cross`) path: cross-task
    /// correlation lives implicitly in the shared latent prior, not in this
    /// matrix.
    pub fn from_batch_mvn(mean_tn: &Tensor, covar_tnn: &Tensor) -> Result<Self> {
        let (t, n) = mean_tn.dims2()?;
        // mean [T, N] → [N, T]
        let mean = mean_tn.t()?.contiguous()?;

        // Block-diagonal over tasks in NON-interleaved order (task-major),
        // which is the order `[T, N, N]` already implies.
        let dtype = covar_tnn.dtype();
        let dev = covar_tnn.device();
        let mut rows = Vec::with_capacity(t);
        for i in 0..t {
            let mut blocks = Vec::with_capacity(t);
            for j in 0..t {
                blocks.push(if i == j {
                    covar_tnn.narrow(0, i, 1)?.squeeze(0)?
                } else {
                    Tensor::zeros((n, n), dtype, dev)?
                });
            }
            let refs: Vec<&Tensor> = blocks.iter().collect();
            rows.push(Tensor::cat(&refs, D::Minus1)?); // [N, N·T]
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let full = Tensor::cat(&refs, D::Minus2)?; // [N·T, N·T]

        Self::new(mean, Box::new(DenseOperator::new(full)), false)
    }

    /// The flattened mean in this distribution's own convention → `[..., N·T]`.
    pub fn flat_mean(&self) -> Result<Tensor> {
        let m = if self.interleaved {
            self.mean.clone() // [.., N, T] → row-major flatten is n·T + t ✓
        } else {
            self.mean.transpose(D::Minus2, D::Minus1)?.contiguous()? // [.., T, N]
        };
        let d = m.dims();
        let mut out: Vec<usize> = d[..d.len() - 2].to_vec();
        out.push(self.num_data * self.num_tasks);
        m.contiguous()?.reshape(out)
    }

    /// Collapse to a plain [`MultivariateNormal`] over the flattened vector.
    pub fn to_mvn(self) -> Result<MultivariateNormal> {
        let mean = self.flat_mean()?;
        Ok(MultivariateNormal { mean, covar: self.covar, jitter: self.jitter })
    }

    /// `log p(Y)` for `y: [..., N, T]` → `[...]`.
    pub fn log_prob(&self, y: &Tensor) -> Result<Tensor> {
        let flat_y = {
            let m = if self.interleaved {
                y.clone()
            } else {
                y.transpose(D::Minus2, D::Minus1)?.contiguous()?
            };
            let d = m.dims();
            let mut out: Vec<usize> = d[..d.len() - 2].to_vec();
            out.push(self.num_data * self.num_tasks);
            m.contiguous()?.reshape(out)?
        };
        let mvn = MultivariateNormal {
            mean: self.flat_mean()?,
            covar: Box::new(DenseOperator::new(self.covar.to_dense()?)),
            jitter: self.jitter,
        };
        mvn.log_prob(&flat_y)
    }

    /// Per-output marginal variances, reshaped back to `[..., N, T]`.
    pub fn variance(&self) -> Result<Tensor> {
        let d = self.covar.diagonal()?; // [.., N·T]
        let dims = d.dims();
        let mut out: Vec<usize> = dims[..dims.len() - 1].to_vec();
        if self.interleaved {
            out.push(self.num_data);
            out.push(self.num_tasks);
            d.reshape(out)
        } else {
            out.push(self.num_tasks);
            out.push(self.num_data);
            d.reshape(out)?.transpose(D::Minus2, D::Minus1)?.contiguous()
        }
    }

    /// Reparameterised samples `[S, ..., N, T]`.
    pub fn rsample(&self, s: usize) -> Result<Tensor> {
        let z = self.covar.zero_mean_samples(s, self.jitter)?; // [S, .., N·T]
        let flat = z.broadcast_add(&self.flat_mean()?)?;
        let d = flat.dims();
        let mut out: Vec<usize> = d[..d.len() - 1].to_vec();
        if self.interleaved {
            out.push(self.num_data);
            out.push(self.num_tasks);
            flat.reshape(out)
        } else {
            out.push(self.num_tasks);
            out.push(self.num_data);
            flat.reshape(out)?.transpose(D::Minus2, D::Minus1)?.contiguous()
        }
    }
}

#[cfg(test)]
mod tests {
    //! Verification of multitask gaussian distributions with structured covariance operators.

    use super::*;
    use crate::operators::diag_embed;
    use candle_core::Device;

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    }

    /// `from_batch_mvn` must place each task's covariance on the block
    /// diagonal and transpose the mean to `[N, T]`.
    #[test]
    fn from_batch_mvn_builds_block_diagonal_and_transposes_mean() {
        let dev = Device::Cpu;
        let (t, n) = (3usize, 2usize);
        let mean_tn = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &dev).unwrap();
        let covs = Tensor::stack(
            &(0..t)
                .map(|i| {
                    diag_embed(&Tensor::new(&[(i + 1) as f32, (i + 1) as f32 * 2.0], &dev).unwrap())
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap(); // [T, N, N]

        let mt = MultitaskMultivariateNormal::from_batch_mvn(&mean_tn, &covs).unwrap();
        assert_eq!(mt.num_tasks, t);
        assert_eq!(mt.num_data, n);
        assert!(!mt.interleaved, "batch-mvn assembly is task-major");
        assert_eq!(mt.mean.dims(), &[n, t]);
        assert!(max_abs(&mt.mean, &mean_tn.t().unwrap()).abs() < 1e-6);

        // block diagonal: off-diagonal task blocks are exactly zero
        let full = mt.covar.to_dense().unwrap();
        assert_eq!(full.dims(), &[n * t, n * t]);
        let v: Vec<Vec<f32>> = full.to_vec2().unwrap();
        for ti in 0..t {
            for tj in 0..t {
                for a in 0..n {
                    for b in 0..n {
                        let got = v[ti * n + a][tj * n + b];
                        if ti != tj {
                            assert!(got.abs() < 1e-9, "off-block ({ti},{tj}) = {got}");
                        }
                    }
                }
            }
        }
        // and the diagonal carries each task's variances
        let d: Vec<f32> = crate::operators::dense_diagonal(&full).unwrap().to_vec1().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6 && (d[1] - 2.0).abs() < 1e-6);
        assert!((d[2] - 2.0).abs() < 1e-6 && (d[3] - 4.0).abs() < 1e-6);
    }

    /// The two flattening conventions must index the SAME matrix
    /// differently — this is the bug-prone part, so pin it explicitly.
    #[test]
    fn interleaved_and_task_major_flatten_differently() {
        let dev = Device::Cpu;
        // mean [N=2, T=3] with distinguishable entries
        let mean = Tensor::new(&[[10.0f32, 11.0, 12.0], [20.0, 21.0, 22.0]], &dev).unwrap();
        let cov = diag_embed(&Tensor::ones(6, candle_core::DType::F32, &dev).unwrap()).unwrap();

        let inter =
            MultitaskMultivariateNormal::new(mean.clone(), Box::new(DenseOperator::new(cov.clone())), true)
                .unwrap();
        let task_major =
            MultitaskMultivariateNormal::new(mean, Box::new(DenseOperator::new(cov)), false).unwrap();

        // interleaved: n·T + t → point 0's tasks first
        let fi: Vec<f32> = inter.flat_mean().unwrap().to_vec1().unwrap();
        assert_eq!(fi, vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0]);

        // task-major: t·N + n → task 0's points first
        let ft: Vec<f32> = task_major.flat_mean().unwrap().to_vec1().unwrap();
        assert_eq!(ft, vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0]);
    }

    /// `variance()` must round-trip back to `[N, T]` in either convention.
    #[test]
    fn variance_reshapes_back_to_n_by_t() {
        let dev = Device::Cpu;
        let mean = Tensor::zeros((2, 3), candle_core::DType::F32, &dev).unwrap();
        // interleaved diag: n·T + t
        let d = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &dev).unwrap();
        let mt = MultitaskMultivariateNormal::new(
            mean.clone(),
            Box::new(DenseOperator::new(diag_embed(&d).unwrap())),
            true,
        )
        .unwrap();
        let v = mt.variance().unwrap();
        assert_eq!(v.dims(), &[2, 3]);
        let vv: Vec<Vec<f32>> = v.to_vec2().unwrap();
        assert_eq!(vv[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vv[1], vec![4.0, 5.0, 6.0]);

        // task-major reads the same buffer as [T, N] then transposes
        let mt2 = MultitaskMultivariateNormal::new(
            mean,
            Box::new(DenseOperator::new(diag_embed(&d).unwrap())),
            false,
        )
        .unwrap();
        let v2: Vec<Vec<f32>> = mt2.variance().unwrap().to_vec2().unwrap();
        assert_eq!(v2[0], vec![1.0, 3.0, 5.0]);
        assert_eq!(v2[1], vec![2.0, 4.0, 6.0]);
    }

    /// A multitask log_prob over independent tasks must equal the sum of
    /// the per-task MVN log-probs.
    #[test]
    fn multitask_log_prob_equals_sum_over_independent_tasks() {
        let dev = Device::Cpu;
        let (t, n) = (3usize, 2usize);
        let mean_tn = Tensor::randn(0f32, 1f32, (t, n), &dev).unwrap();
        let vars = Tensor::rand(0.5f32, 2.0f32, (t, n), &dev).unwrap();
        let covs = Tensor::stack(
            &(0..t)
                .map(|i| diag_embed(&vars.narrow(0, i, 1).unwrap().squeeze(0).unwrap()).unwrap())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();

        let mt = MultitaskMultivariateNormal::from_batch_mvn(&mean_tn, &covs)
            .unwrap()
            .with_jitter(1e-9);
        let y_tn = Tensor::randn(0f32, 1f32, (t, n), &dev).unwrap();
        let got = mt.log_prob(&y_tn.t().unwrap().contiguous().unwrap()).unwrap()
            .to_scalar::<f32>().unwrap();

        let mut want = 0.0f32;
        for i in 0..t {
            let m = mean_tn.narrow(0, i, 1).unwrap().squeeze(0).unwrap();
            let v = vars.narrow(0, i, 1).unwrap().squeeze(0).unwrap();
            let y = y_tn.narrow(0, i, 1).unwrap().squeeze(0).unwrap();
            want += MultivariateNormal::from_diag(m, v)
                .with_jitter(0.0)
                .log_prob(&y)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
        }
        assert!((got - want).abs() < 2e-3, "{got} vs {want}");
    }

    #[test]
    fn rsample_returns_matrix_shaped_draws() {
        let dev = Device::Cpu;
        let mean = Tensor::new(&[[1.0f32, -1.0], [2.0, -2.0], [3.0, -3.0]], &dev).unwrap();
        let d = Tensor::ones(6, candle_core::DType::F32, &dev).unwrap();
        let mt = MultitaskMultivariateNormal::new(
            mean.clone(),
            Box::new(DenseOperator::new(diag_embed(&d).unwrap())),
            true,
        )
        .unwrap()
        .with_jitter(0.0);

        let s = mt.rsample(4000).unwrap();
        assert_eq!(s.dims(), &[4000, 3, 2]);
        let m = s.mean(0).unwrap();
        assert!(max_abs(&m, &mean) < 0.12, "sample mean off: {}", max_abs(&m, &mean));
    }

    #[test]
    fn mismatched_covariance_size_is_rejected() {
        let dev = Device::Cpu;
        let mean = Tensor::zeros((2, 3), candle_core::DType::F32, &dev).unwrap(); // needs 6×6
        let bad = Tensor::eye(5, candle_core::DType::F32, &dev).unwrap();
        let e = MultitaskMultivariateNormal::new(mean, Box::new(DenseOperator::new(bad)), true);
        assert!(e.is_err(), "a 5×5 covariance must not pair with a [2,3] mean");
    }
}
