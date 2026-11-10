//! Inferno — Structured matrix operators, tiled symbolic reductions and iterative linear solvers.

use candle_core::{DType, Result, Tensor, D};

use crate::linalg_diff::{cholesky_diff, tri_solve_diff, tri_solve_t_diff};
use crate::settings;

pub mod bakeoff;
pub mod cg;
pub mod cgcache;
pub mod ciq;
pub mod cost_model;
pub mod fixedpoint;
pub mod genred;
pub mod kronecker;
pub mod ladder;
pub mod lazy_kernel;
pub mod linsolve;
pub mod minres;
pub mod precond;
pub mod recycle;
pub mod rewrite;
pub mod schedule;
pub mod symbolic;

pub use genred::{FusedItem, FusedLse, FusedSweep, Genred, ReduceOp};
pub use kronecker::{KroneckerDiagOperator, KroneckerOperator};
pub use lazy_kernel::{
    HyperKernelFn, HyperKernelOperator, LazyKernelOperator, SymbolicKernelOperator,
};
pub use symbolic::{Axis, BlockCfg, Expr, LazyTensor, ReduceAxis};

/// Default Cholesky jitter. Matches `BridgeGpConfig::jitter`.
pub const DEFAULT_JITTER: f64 = 1e-5;

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// A (possibly structured, possibly lazy) linear operator over Candle
/// tensors.
///
/// Implementors MUST provide [`shape`](Self::shape), [`to_dense`](Self::to_dense)
/// and [`matmul`](Self::matmul). Everything else has a correct — if
/// unspecialised — default, so a new operator is cheap to add and only
/// pays for the optimisations it actually implements.
pub trait LinearOperator {
    /// Full batched shape `[..., M, N]`.
    fn shape(&self) -> Vec<usize>;

    /// Materialise as a dense `[..., M, N]` tensor. The escape hatch every
    /// default method falls back through — implement the specialised paths
    /// on any operator where this is expensive.
    fn to_dense(&self) -> Result<Tensor>;

    /// `self @ rhs`, `rhs: [..., N, C]` → `[..., M, C]`.
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor>;

    /// Rows `M`.
    fn nrows(&self) -> usize {
        let s = self.shape();
        s[s.len() - 2]
    }

    /// Columns `N`.
    fn ncols(&self) -> usize {
        let s = self.shape();
        s[s.len() - 1]
    }

    /// Leading batch dims (everything before the trailing `M, N`).
    fn batch_shape(&self) -> Vec<usize> {
        let s = self.shape();
        s[..s.len() - 2].to_vec()
    }

    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    /// `selfᵀ @ rhs`, `rhs: [..., M, C]` → `[..., N, C]`.
    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.to_dense()?
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(rhs)
    }

    /// The diagonal `[..., min(M, N)]`.
    fn diagonal(&self) -> Result<Tensor> {
        dense_diagonal(&self.to_dense()?)
    }

    /// Lower-triangular Cholesky factor `L` with `self ≈ L Lᵀ`.
    /// Differentiable (Murray's closed-form backward).
    fn cholesky(&self, jitter: f64) -> Result<Tensor> {
        cholesky_diff(&self.to_dense()?, jitter)
    }

    /// `self⁻¹ @ rhs` for SPD `self`, via `L⁻ᵀ(L⁻¹ rhs)`. Differentiable.
    ///
    /// Routing: dense Cholesky unless [`settings::iterative_solves_enabled`]
    /// is on AND the operator exceeds [`settings::max_cholesky_size`] —
    /// then the differentiable CG node ([`linsolve::lin_solve_diff`]),
    /// which trades the O(N³) factorisation for matvecs plus one adjoint
    /// solve in backward. On the routed arm, `rhs` batch dims must match
    /// the operator's exactly (the dense path broadcasts).
    fn cholesky_solve(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        if self.iterative_solve_route() {
            return linsolve::lin_solve_diff(
                &self.jittered_dense(jitter)?,
                rhs,
                &cg::CgOptions::default(),
            );
        }
        let l = self.cholesky(jitter)?;
        tri_solve_t_diff(&l, &tri_solve_diff(&l, rhs)?)
    }

    /// `log|det self| = 2 Σ log diag(L)` for SPD `self` → `[...]`.
    ///
    /// Routed arm (same gate as [`cholesky_solve`](Self::cholesky_solve)):
    /// stochastic Lanczos quadrature ([`linsolve::slq_logdet_diff`]) over
    /// [`settings::num_trace_samples`] seeded Rademacher probes — an
    /// UNBIASED ESTIMATE, not the exact value, with the matching
    /// trace-estimator gradient. Deterministic per process (fixed probe
    /// seed), so a loss is stable across backward passes.
    fn log_abs_det(&self, jitter: f64) -> Result<Tensor> {
        if self.iterative_solve_route() {
            let k = self.jittered_dense(jitter)?;
            let z = self.slq_probes(&k)?;
            return linsolve::slq_logdet_diff(&k, &z, &cg::CgOptions::default());
        }
        let l = self.cholesky(jitter)?;
        Ok(dense_diagonal(&l)?
            .log()?
            .sum(D::Minus1)?
            .affine(2.0, 0.0)?)
    }

    /// The quadratic form `rhsᵀ self⁻¹ rhs`, computed STABLY as
    /// `‖L⁻¹ rhs‖²` — no explicit inverse, and the same
    /// `cholesky_diff`/`tri_solve_diff` pair the SVGP already trains
    /// through. `rhs: [..., N, C]` → `[..., C]`. Routed arm: the CG
    /// composition [`linsolve::inv_quad_diff`], same gate as
    /// [`cholesky_solve`](Self::cholesky_solve).
    fn inv_quad(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        if self.iterative_solve_route() {
            return linsolve::inv_quad_diff(
                &self.jittered_dense(jitter)?,
                rhs,
                &cg::CgOptions::default(),
            );
        }
        let l = self.cholesky(jitter)?;
        tri_solve_diff(&l, rhs)?.sqr()?.sum(D::Minus2)
    }

    /// `(inv_quad, log_abs_det)` sharing ONE factorisation — the pairing
    /// `log_prob` needs (GPyTorch's `inv_quad_logdet`). Routed arm: the two
    /// Krylov nodes run separately (sharing their Krylov spaces is rung-4
    /// recycling); the dense arm keeps sharing its Cholesky.
    fn inv_quad_logdet(&self, rhs: &Tensor, jitter: f64) -> Result<(Tensor, Tensor)> {
        if self.iterative_solve_route() {
            let k = self.jittered_dense(jitter)?;
            let opts = cg::CgOptions::default();
            let iq = linsolve::inv_quad_diff(&k, rhs, &opts)?;
            let z = self.slq_probes(&k)?;
            let ld = linsolve::slq_logdet_diff(&k, &z, &opts)?;
            return Ok((iq, ld));
        }
        let l = self.cholesky(jitter)?;
        let iq = tri_solve_diff(&l, rhs)?.sqr()?.sum(D::Minus2)?;
        let ld = dense_diagonal(&l)?
            .log()?
            .sum(D::Minus1)?
            .affine(2.0, 0.0)?;
        Ok((iq, ld))
    }

    /// The one gate for the solve family — mirrors the CIQ routing gate so
    /// the two thresholds cannot drift apart.
    #[doc(hidden)]
    fn iterative_solve_route(&self) -> bool {
        settings::iterative_solves_enabled::value()
            && self.ncols() > settings::max_cholesky_size::value()
    }

    /// `self + jitter·I` as a dense tensor — what the routed solve nodes
    /// consume. Dense today (the trait has no leaf seam); truly lazy
    /// operators call [`linsolve::lin_solve_lazy`] directly.
    #[doc(hidden)]
    fn jittered_dense(&self, jitter: f64) -> Result<Tensor> {
        let k = self.to_dense()?;
        if jitter == 0.0 {
            return Ok(k);
        }
        let n = self.ncols();
        let eye = Tensor::eye(n, k.dtype(), k.device())?.affine(jitter, 0.0)?;
        k.broadcast_add(&eye)
    }

    /// Seeded Rademacher probes broadcast to the operator's batch shape.
    #[doc(hidden)]
    fn slq_probes(&self, k: &Tensor) -> Result<Tensor> {
        let n = self.ncols();
        let p = settings::num_trace_samples::value();
        let z = linsolve::rademacher_probes(n, p, 0x51_1D_5EED, k.dtype(), k.device())?;
        let batch = self.batch_shape();
        if batch.is_empty() {
            return Ok(z);
        }
        let mut dims = vec![1; batch.len()];
        dims.extend_from_slice(&[n, p]);
        let mut full = batch;
        full.extend_from_slice(&[n, p]);
        z.reshape(dims)?.broadcast_as(full)?.contiguous()
    }

    /// `self + diag(d)`, `d: [..., N]` (or `[]`/`[1]` for a constant).
    /// Structure-preserving where the impl overrides it.
    fn add_diagonal(&self, d: &Tensor) -> Result<AddedDiagOperator>
    where
        Self: Sized + Clone + 'static,
    {
        Ok(AddedDiagOperator::new(
            Box::new(self.clone()),
            DiagOperator::new(d.clone()),
        ))
    }

    /// Draw `s` zero-mean samples with this operator as the covariance:
    /// `L ε`, `ε ~ N(0, I)` → `[s, ..., N]`.
    fn zero_mean_samples(&self, s: usize, jitter: f64) -> Result<Tensor> {
        let l = self.cholesky(jitter)?; // [..., N, N]
        let mut dims = vec![s];
        dims.extend_from_slice(&self.batch_shape());
        dims.push(self.ncols());
        dims.push(1);
        let eps = Tensor::randn(0f32, 1f32, dims, l.device())?.to_dtype(l.dtype())?;
        l.unsqueeze(0)?.broadcast_matmul(&eps)?.squeeze(D::Minus1)
    }

    /// `R rhs` for SOME root `R` with `R Rᵀ = self` (SPD `self`).
    ///
    /// CONTRACT: which root is unspecified — the dense path applies the
    /// Cholesky factor, the CIQ path the symmetric root `self^{1/2}`. Any
    /// two roots differ by a right rotation, so everything the intended
    /// consumers compute (sample covariances, whitened residuals, quadratic
    /// forms) is invariant; callers must not assume triangularity.
    ///
    /// Routing: dense Cholesky (differentiable, the existing trained path)
    /// unless [`settings::ciq_enabled`] is on AND the operator exceeds
    /// [`settings::max_cholesky_size`] — then contour-integral quadrature
    /// ([`ciq::ciq_sqrt_matmul`]; detached, matvec-only, never densifies).
    /// With the flag off (the default) behaviour is bit-for-bit the
    /// Cholesky path at every size.
    fn sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        if !settings::ciq_enabled::value() || self.ncols() <= settings::max_cholesky_size::value() {
            self.cholesky(jitter)?.broadcast_matmul(rhs)
        } else {
            ciq::ciq_sqrt_matmul(
                |x| self.matmul(x),
                self.ncols(),
                rhs,
                &self.ciq_route_opts(),
            )
        }
    }

    /// `R⁻¹ rhs` for the same (unspecified) root `R` as
    /// [`sqrt_matmul`](Self::sqrt_matmul); same routing and the same
    /// contract caveat. Dense arm: `L⁻¹ rhs` (differentiable); CIQ arm:
    /// `R⁻¹ rhs` (detached).
    fn inv_sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        if !settings::ciq_enabled::value() || self.ncols() <= settings::max_cholesky_size::value() {
            tri_solve_diff(&self.cholesky(jitter)?, rhs)
        } else {
            ciq::ciq_inv_sqrt_matmul(
                |x| self.matmul(x),
                self.ncols(),
                rhs,
                &self.ciq_route_opts(),
            )
        }
    }

    /// Options for the trait's CIQ arms: above
    /// [`settings::min_preconditioning_size`] a rank-
    /// [`settings::max_preconditioner_size`] pivoted-Cholesky preconditioner
    /// is built from `p` matvecs + [`diagonal`](Self::diagonal) and the
    /// pipeline runs on the conjugated operator (see [`precond`]). A failed
    /// preconditioner build degrades to unpreconditioned CIQ rather than
    /// failing the solve — preconditioning is an accelerator, never a
    /// correctness dependency.
    #[doc(hidden)]
    fn ciq_route_opts(&self) -> ciq::CiqOptions {
        let mut opts = ciq::CiqOptions::default();
        let rank = settings::max_preconditioner_size::value();
        if rank > 0 && self.ncols() >= settings::min_preconditioning_size::value() {
            if let Ok(diag) = self.diagonal() {
                if let Ok(p) = precond::CiqPreconditioner::pivoted(|x| self.matmul(x), &diag, rank)
                {
                    opts.precond = Some(std::sync::Arc::new(p));
                }
            }
        }
        opts
    }

    /// A materialised root `R` with `R Rᵀ = self`, as a [`RootOperator`].
    /// Cholesky-backed (a lazy CIQ-backed root would cost `N` solves to
    /// materialise — deliberately deferred until a consumer needs it).
    fn root_decomposition(&self, jitter: f64) -> Result<RootOperator> {
        Ok(RootOperator::new(self.cholesky(jitter)?))
    }

    /// [`zero_mean_samples`](Self::zero_mean_samples) routed through
    /// [`sqrt_matmul`](Self::sqrt_matmul) — the opt-in CIQ sampling variant.
    /// The default `zero_mean_samples` stays on its dense Cholesky path;
    /// flipping the MVN `rsample` call sites onto this is its own job.
    fn zero_mean_samples_ciq(
        &self,
        s: usize,
        jitter: f64,
        opts: &ciq::CiqOptions,
    ) -> Result<Tensor> {
        let n = self.ncols();
        let mut dims = vec![s];
        dims.extend_from_slice(&self.batch_shape());
        dims.push(n);
        dims.push(1);
        let dtype = self.to_dense_dtype()?;
        let eps = Tensor::randn(0f32, 1f32, dims, &self.device()?)?.to_dtype(dtype)?;
        // ε is [s, ..., N, 1]; matmul broadcasts the operator's batch dims
        // against the leading sample axis.
        let drawn = if !settings::ciq_enabled::value() || n <= settings::max_cholesky_size::value()
        {
            self.cholesky(jitter)?
                .unsqueeze(0)?
                .broadcast_matmul(&eps)?
        } else {
            ciq::ciq_sqrt_matmul(|x| self.matmul(x), n, &eps, opts)?
        };
        drawn.squeeze(D::Minus1)
    }

    /// dtype/device probes for the sampling default. Cheap for every
    /// in-tree operator (they hold tensors); a truly lazy implementor can
    /// override.
    #[doc(hidden)]
    fn to_dense_dtype(&self) -> Result<candle_core::DType> {
        Ok(self.to_dense()?.dtype())
    }
    #[doc(hidden)]
    fn device(&self) -> Result<candle_core::Device> {
        Ok(self.to_dense()?.device().clone())
    }
}

/// Batched diagonal extraction `[..., M, N] → [..., min(M, N)]`. Candle has
/// no `diagonal`, and `gather` keeps this differentiable (an index-select
/// on the flattened last two dims would not be).
pub fn dense_diagonal(t: &Tensor) -> Result<Tensor> {
    let dims = t.dims();
    let (m, n) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let k = m.min(n);
    // Flatten [.., M, N] → [.., M*N] and gather the stride-(N+1) entries.
    let mut flat_dims: Vec<usize> = dims[..dims.len() - 2].to_vec();
    flat_dims.push(m * n);
    let flat = t.reshape(flat_dims.clone())?;
    let idx: Vec<u32> = (0..k).map(|i| (i * n + i) as u32).collect();
    let mut idx_dims: Vec<usize> = vec![1; flat_dims.len() - 1];
    idx_dims.push(k);
    let mut bshape: Vec<usize> = dims[..dims.len() - 2].to_vec();
    bshape.push(k);
    let idx = Tensor::from_vec(idx, k, t.device())?
        .reshape(idx_dims)?
        .broadcast_as(bshape)?
        .contiguous()?;
    flat.gather(&idx, D::Minus1)
}

/// Build a batched diagonal matrix from `[..., N]` → `[..., N, N]`
/// (`torch.diag_embed`). Differentiable: a broadcast multiply by an
/// identity, not a scatter.
pub fn diag_embed(d: &Tensor) -> Result<Tensor> {
    let n = d.dim(D::Minus1)?;
    let eye = Tensor::eye(n, d.dtype(), d.device())?;
    d.unsqueeze(D::Minus1)?.broadcast_mul(&eye)
}

// ---------------------------------------------------------------------------
// DenseOperator
// ---------------------------------------------------------------------------

/// A plain dense matrix — the `to_linear_operator` wrapper.
#[derive(Clone)]
pub struct DenseOperator {
    pub tensor: Tensor,
}

impl DenseOperator {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
}

impl LinearOperator for DenseOperator {
    fn shape(&self) -> Vec<usize> {
        self.tensor.dims().to_vec()
    }
    fn to_dense(&self) -> Result<Tensor> {
        Ok(self.tensor.clone())
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.tensor.broadcast_matmul(rhs)
    }
    fn diagonal(&self) -> Result<Tensor> {
        dense_diagonal(&self.tensor)
    }
}

// ---------------------------------------------------------------------------
// DiagOperator
// ---------------------------------------------------------------------------

/// `diag(d)` — stores only the diagonal `[..., N]`.
///
/// Every op is O(N): the solve is a reciprocal, the logdet a log-sum, the
/// Cholesky a square root. Nothing is ever materialised unless
/// [`to_dense`](LinearOperator::to_dense) is called explicitly.
#[derive(Clone)]
pub struct DiagOperator {
    pub diag: Tensor, // [..., N]
}

impl DiagOperator {
    pub fn new(diag: Tensor) -> Self {
        Self { diag }
    }

    /// Elementwise reciprocal — the exact inverse, still O(N).
    pub fn inverse(&self) -> Result<Self> {
        Ok(Self::new(self.diag.recip()?))
    }

    pub fn sqrt(&self) -> Result<Self> {
        Ok(Self::new(self.diag.sqrt()?))
    }
}

impl LinearOperator for DiagOperator {
    fn shape(&self) -> Vec<usize> {
        let mut s = self.diag.dims().to_vec();
        s.push(*s.last().unwrap());
        s
    }
    fn to_dense(&self) -> Result<Tensor> {
        diag_embed(&self.diag)
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        // diag(d) @ X = d ⊙ X (broadcast over the column axis)
        self.diag.unsqueeze(D::Minus1)?.broadcast_mul(rhs)
    }
    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul(rhs) // symmetric
    }
    fn diagonal(&self) -> Result<Tensor> {
        Ok(self.diag.clone())
    }
    fn cholesky(&self, jitter: f64) -> Result<Tensor> {
        diag_embed(&self.diag.affine(1.0, jitter)?.sqrt()?)
    }
    fn cholesky_solve(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        self.diag
            .affine(1.0, jitter)?
            .recip()?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(rhs)
    }
    fn log_abs_det(&self, jitter: f64) -> Result<Tensor> {
        self.diag.affine(1.0, jitter)?.log()?.sum(D::Minus1)
    }
    fn inv_quad(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let inv = self
            .diag
            .affine(1.0, jitter)?
            .recip()?
            .unsqueeze(D::Minus1)?;
        rhs.sqr()?.broadcast_mul(&inv)?.sum(D::Minus2)
    }
    /// Exact elementwise root — O(N), differentiable, no routing gate (the
    /// closed form beats both trait arms at every size).
    fn sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        self.diag
            .affine(1.0, jitter)?
            .sqrt()?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(rhs)
    }
    fn inv_sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        self.diag
            .affine(1.0, jitter)?
            .sqrt()?
            .recip()?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(rhs)
    }
    fn root_decomposition(&self, jitter: f64) -> Result<RootOperator> {
        Ok(RootOperator::new(diag_embed(
            &self.diag.affine(1.0, jitter)?.sqrt()?,
        )?))
    }
    fn to_dense_dtype(&self) -> Result<candle_core::DType> {
        Ok(self.diag.dtype())
    }
    fn device(&self) -> Result<candle_core::Device> {
        Ok(self.diag.device().clone())
    }
}

// ---------------------------------------------------------------------------
// ConstantDiagOperator
// ---------------------------------------------------------------------------

/// `c · I_n` — stores one scalar (per batch) and the size.
///
/// This is the shape the multitask likelihood's global noise `σ²I_{nt}`
/// takes; keeping it constant-diagonal is what lets it be folded INSIDE a
/// Kronecker factor (`I ⊗ D_T + σ²I = I ⊗ (D_T + σ²I)`) rather than
/// densifying the `[N·T, N·T]` product.
#[derive(Clone)]
pub struct ConstantDiagOperator {
    /// `[...]` or `[..., 1]` — the constant, batched.
    pub value: Tensor,
    pub n: usize,
}

impl ConstantDiagOperator {
    pub fn new(value: Tensor, n: usize) -> Self {
        let value = if value.rank() > 0 && value.dims()[value.rank() - 1] == 1 {
            value.squeeze(D::Minus1).unwrap_or(value)
        } else {
            value
        };
        Self { value, n }
    }

    /// Expand to an explicit [`DiagOperator`].
    pub fn to_diag(&self) -> Result<DiagOperator> {
        let mut dims = self.value.dims().to_vec();
        dims.push(self.n);
        Ok(DiagOperator::new(
            self.value
                .unsqueeze(D::Minus1)?
                .broadcast_as(dims)?
                .contiguous()?,
        ))
    }
}

impl LinearOperator for ConstantDiagOperator {
    fn shape(&self) -> Vec<usize> {
        let mut s = self.value.dims().to_vec();
        s.push(self.n);
        s.push(self.n);
        s
    }
    fn to_dense(&self) -> Result<Tensor> {
        self.to_diag()?.to_dense()
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let v = self.value.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        rhs.broadcast_mul(&v)
    }
    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul(rhs)
    }
    fn diagonal(&self) -> Result<Tensor> {
        self.to_diag()?.diagonal()
    }
    fn cholesky_solve(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let v = self.value.affine(1.0, jitter)?.recip()?;
        rhs.broadcast_mul(&v.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)
    }
    fn log_abs_det(&self, jitter: f64) -> Result<Tensor> {
        // n · log(c)
        self.value
            .affine(1.0, jitter)?
            .log()?
            .affine(self.n as f64, 0.0)
    }
    /// `√(c + jitter) · rhs` — scalar broadcast, no routing gate.
    fn sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let v = self.value.affine(1.0, jitter)?.sqrt()?;
        rhs.broadcast_mul(&v.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)
    }
    fn inv_sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let v = self.value.affine(1.0, jitter)?.sqrt()?.recip()?;
        rhs.broadcast_mul(&v.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)
    }
    fn to_dense_dtype(&self) -> Result<candle_core::DType> {
        Ok(self.value.dtype())
    }
    fn device(&self) -> Result<candle_core::Device> {
        Ok(self.value.device().clone())
    }
}

// ---------------------------------------------------------------------------
// RootOperator
// ---------------------------------------------------------------------------

/// `R Rᵀ` for a root `R: [..., N, K]` — the low-rank PSD form.
///
/// `matmul` is `R (Rᵀ x)`: two `O(NKC)` products instead of one `O(N²C)`,
/// and the `[N, N]` product is never formed. This is the shape the
/// rank-`k` task-noise factor and the whitened variational covariance take.
#[derive(Clone)]
pub struct RootOperator {
    pub root: Tensor, // [..., N, K]
}

impl RootOperator {
    pub fn new(root: Tensor) -> Self {
        Self { root }
    }
    pub fn rank(&self) -> usize {
        self.root.dims()[self.root.rank() - 1]
    }
}

impl LinearOperator for RootOperator {
    fn shape(&self) -> Vec<usize> {
        let d = self.root.dims();
        let n = d[d.len() - 2];
        let mut s = d[..d.len() - 2].to_vec();
        s.push(n);
        s.push(n);
        s
    }
    fn to_dense(&self) -> Result<Tensor> {
        self.root
            .broadcast_matmul(&self.root.transpose(D::Minus2, D::Minus1)?.contiguous()?)
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        // R (Rᵀ x) — never forms R Rᵀ
        let rt = self.root.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        self.root.broadcast_matmul(&rt.broadcast_matmul(rhs)?)
    }
    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul(rhs) // symmetric
    }
    fn diagonal(&self) -> Result<Tensor> {
        // diag(R Rᵀ) = row-wise squared norm — O(NK), not O(N²).
        self.root.sqr()?.sum(D::Minus1)
    }
}

// ---------------------------------------------------------------------------
// AddedDiagOperator
// ---------------------------------------------------------------------------

/// `A + diag(d)` — keeps `A`'s structure while carrying the diagonal.
///
/// The GP's noise-augmented covariance `K + σ²I` in its natural form.
pub struct AddedDiagOperator {
    pub op: Box<dyn LinearOperator>,
    pub diag: DiagOperator,
}

impl AddedDiagOperator {
    pub fn new(op: Box<dyn LinearOperator>, diag: DiagOperator) -> Self {
        Self { op, diag }
    }
}

impl LinearOperator for AddedDiagOperator {
    fn shape(&self) -> Vec<usize> {
        self.op.shape()
    }
    fn to_dense(&self) -> Result<Tensor> {
        let d = self.diag.to_dense()?;
        self.op.to_dense()?.broadcast_add(&d)
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        // (A + D)x = Ax + d ⊙ x — keeps A lazy.
        self.op.matmul(rhs)?.add(&self.diag.matmul(rhs)?)
    }
    fn diagonal(&self) -> Result<Tensor> {
        self.op.diagonal()?.broadcast_add(&self.diag.diag)
    }
}

// ---------------------------------------------------------------------------
// SumOperator
// ---------------------------------------------------------------------------

/// `Σᵢ Aᵢ` — lazy sum, `matmul` distributes.
pub struct SumOperator {
    pub ops: Vec<Box<dyn LinearOperator>>,
}

impl SumOperator {
    pub fn new(ops: Vec<Box<dyn LinearOperator>>) -> Self {
        assert!(!ops.is_empty(), "SumOperator needs at least one operand");
        Self { ops }
    }
}

impl LinearOperator for SumOperator {
    fn shape(&self) -> Vec<usize> {
        self.ops[0].shape()
    }
    fn to_dense(&self) -> Result<Tensor> {
        let mut acc = self.ops[0].to_dense()?;
        for o in &self.ops[1..] {
            acc = acc.broadcast_add(&o.to_dense()?)?;
        }
        Ok(acc)
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let mut acc = self.ops[0].matmul(rhs)?;
        for o in &self.ops[1..] {
            acc = acc.add(&o.matmul(rhs)?)?;
        }
        Ok(acc)
    }
    fn diagonal(&self) -> Result<Tensor> {
        let mut acc = self.ops[0].diagonal()?;
        for o in &self.ops[1..] {
            acc = acc.broadcast_add(&o.diagonal()?)?;
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------------------
// Identity helper
// ---------------------------------------------------------------------------

/// `I_n` as a [`ConstantDiagOperator`].
pub fn identity(
    n: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<ConstantDiagOperator> {
    Ok(ConstantDiagOperator::new(
        Tensor::ones((), dtype, device)?,
        n,
    ))
}

#[cfg(test)]
mod tests {
    //! Verification of structured matrix operators, tiled symbolic reductions and iterative linear solvers.

    use super::*;
    use candle_core::Device;

    fn spd(n: usize, dev: &Device) -> Tensor {
        let a = Tensor::randn(0f32, 1f32, (n, n), dev).unwrap();
        let at = a.t().unwrap();
        a.matmul(&at)
            .unwrap()
            .add(
                &Tensor::eye(n, DType::F32, dev)
                    .unwrap()
                    .affine(n as f64, 0.0)
                    .unwrap(),
            )
            .unwrap()
    }

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

    #[test]
    fn dense_diagonal_and_diag_embed_roundtrip() {
        let dev = Device::Cpu;
        let d = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let m = diag_embed(&d).unwrap();
        assert_eq!(m.dims(), &[3, 3]);
        assert!(max_abs(&dense_diagonal(&m).unwrap(), &d) < 1e-6);
        // batched
        let db = Tensor::randn(0f32, 1f32, (4, 5), &dev).unwrap();
        let mb = diag_embed(&db).unwrap();
        assert_eq!(mb.dims(), &[4, 5, 5]);
        assert!(max_abs(&dense_diagonal(&mb).unwrap(), &db) < 1e-6);
    }

    // ---- sqrt_matmul / inv_sqrt_matmul / root_decomposition ----------------

    #[test]
    fn diag_sqrt_matmul_matches_dense_reference() {
        let dev = Device::Cpu;
        let d = Tensor::new(&[2.0f32, 0.5, 4.0, 1.5], &dev).unwrap();
        let op = DiagOperator::new(d);
        let dense = DenseOperator::new(op.to_dense().unwrap());
        let x = Tensor::randn(0f32, 1f32, (4, 3), &dev).unwrap();
        let j = 1e-6;
        // Both roots are diagonal here, so they agree exactly — not just up
        // to rotation.
        assert!(
            max_abs(
                &op.sqrt_matmul(&x, j).unwrap(),
                &dense.sqrt_matmul(&x, j).unwrap()
            ) < 1e-5
        );
        assert!(
            max_abs(
                &op.inv_sqrt_matmul(&x, j).unwrap(),
                &dense.inv_sqrt_matmul(&x, j).unwrap()
            ) < 1e-5
        );
    }

    #[test]
    fn constant_diag_sqrt_matmul_matches_dense_reference() {
        let dev = Device::Cpu;
        let c = Tensor::new(2.25f32, &dev).unwrap();
        let op = ConstantDiagOperator::new(c, 5);
        let dense = DenseOperator::new(op.to_dense().unwrap());
        let x = Tensor::randn(0f32, 1f32, (5, 2), &dev).unwrap();
        assert!(
            max_abs(
                &op.sqrt_matmul(&x, 0.0).unwrap(),
                &dense.sqrt_matmul(&x, 0.0).unwrap()
            ) < 1e-5
        );
        assert!(
            max_abs(
                &op.inv_sqrt_matmul(&x, 0.0).unwrap(),
                &dense.inv_sqrt_matmul(&x, 0.0).unwrap()
            ) < 1e-5
        );
    }

    #[test]
    fn kronecker_root_is_the_kronecker_of_roots() {
        let dev = Device::Cpu;
        let a = spd(3, &dev);
        let b = spd(4, &dev);
        let op = KroneckerOperator::new(vec![
            Box::new(DenseOperator::new(a)),
            Box::new(DenseOperator::new(b)),
        ]);
        let dense = DenseOperator::new(op.to_dense().unwrap());
        let x = Tensor::randn(0f32, 1f32, (12, 2), &dev).unwrap();
        let j = 1e-6;
        // chol(A⊗B) = chol(A)⊗chol(B) exactly, so the factor-wise cycle and
        // the dense Cholesky agree to fp error.
        assert!(
            max_abs(
                &op.sqrt_matmul(&x, j).unwrap(),
                &dense.sqrt_matmul(&x, j).unwrap()
            ) < 1e-3
        );
        assert!(
            max_abs(
                &op.inv_sqrt_matmul(&x, j).unwrap(),
                &dense.inv_sqrt_matmul(&x, j).unwrap()
            ) < 1e-3
        );
    }

    #[test]
    fn root_decomposition_reconstructs_the_operator() {
        let dev = Device::Cpu;
        let a = spd(6, &dev);
        let op = DenseOperator::new(a.clone());
        let root = op.root_decomposition(0.0).unwrap();
        assert!(max_abs(&root.to_dense().unwrap(), &a) < 1e-3, "R Rᵀ ≠ A");
    }

    #[test]
    fn inv_sqrt_then_sqrt_is_identity_on_the_dense_path() {
        let dev = Device::Cpu;
        let op = DenseOperator::new(spd(6, &dev));
        let x = Tensor::randn(0f32, 1f32, (6, 2), &dev).unwrap();
        // Same root both ways on the dense path (the Cholesky factor), so
        // R(R⁻¹x) = x exactly.
        let back = op
            .sqrt_matmul(&op.inv_sqrt_matmul(&x, 1e-6).unwrap(), 1e-6)
            .unwrap();
        assert!(max_abs(&back, &x) < 1e-3, "R R⁻¹ x ≠ x");
    }

    /// The no-regression pin: below `max_cholesky_size` the flag must not
    /// change the routing — results identical with it on and off.
    #[test]
    fn ciq_disabled_or_small_n_takes_the_cholesky_path_bit_for_bit() {
        let dev = Device::Cpu;
        let op = DenseOperator::new(spd(8, &dev));
        let x = Tensor::randn(0f32, 1f32, (8, 2), &dev).unwrap();
        let off = op.sqrt_matmul(&x, 1e-6).unwrap();
        let on = {
            let _g = settings::ciq_enabled::with(true);
            op.sqrt_matmul(&x, 1e-6).unwrap()
        };
        assert_eq!(
            off.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            on.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "flag flipped the small-n path"
        );
    }

    /// Above the threshold with the flag on, the trait routes to CIQ and the
    /// symmetric root differs from the Cholesky factor per-element — but
    /// R Rᵀ is the same operator, which is the actual contract. Verified on
    /// the quadratic form.
    #[test]
    fn above_threshold_the_ciq_route_applies_a_valid_root() {
        let dev = Device::Cpu;
        let op = DenseOperator::new(spd(12, &dev));
        let x = Tensor::randn(0f32, 1f32, (12, 3), &dev).unwrap();
        let _flag = settings::ciq_enabled::with(true);
        let _size = settings::max_cholesky_size::with(4); // force the CIQ arm
        let y = op.inv_sqrt_matmul(&x, 1e-6).unwrap();
        // ‖R⁻¹x‖² per column = xᵀ K⁻¹ x, root-choice invariant.
        let got = y.sqr().unwrap().sum(D::Minus2).unwrap();
        let want = op.inv_quad(&x, 1e-6).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-2 * 12.0,
            "CIQ root broke the quadratic form"
        );
    }

    /// Above min_preconditioning_size the CIQ arms auto-build a pivoted
    /// preconditioner and switch to the CONJUGATED root — still a valid
    /// root of the operator, verified through the root-choice-invariant
    /// quadratic form.
    #[test]
    fn the_auto_preconditioned_ciq_route_still_applies_a_valid_root() {
        let dev = Device::Cpu;
        let op = DenseOperator::new(spd(16, &dev));
        let x = Tensor::randn(0f32, 1f32, (16, 2), &dev).unwrap();
        let _flag = settings::ciq_enabled::with(true);
        let _size = settings::max_cholesky_size::with(4); // force the CIQ arm
        let _pre = settings::min_preconditioning_size::with(1); // force the precond build
        let y = op.inv_sqrt_matmul(&x, 1e-6).unwrap();
        let got = y.sqr().unwrap().sum(D::Minus2).unwrap();
        let want = op.inv_quad(&x, 1e-6).unwrap();
        assert!(
            max_abs(&got, &want) < 1e-2 * 16.0,
            "auto-preconditioned CIQ root broke the quadratic form"
        );
    }

    /// The no-regression pin for the solve family: below `max_cholesky_size`
    /// the `iterative_solves_enabled` flag must not change the routing —
    /// results identical with it on and off.
    #[test]
    fn iterative_solves_below_threshold_take_the_cholesky_path_bit_for_bit() {
        let dev = Device::Cpu;
        let op = DenseOperator::new(spd(8, &dev));
        let x = Tensor::randn(0f32, 1f32, (8, 2), &dev).unwrap();
        let off = op.cholesky_solve(&x, 1e-6).unwrap();
        let (off_iq, off_ld) = op.inv_quad_logdet(&x, 1e-6).unwrap();
        let on = {
            let _g = settings::iterative_solves_enabled::with(true);
            op.cholesky_solve(&x, 1e-6).unwrap()
        };
        let (on_iq, on_ld) = {
            let _g = settings::iterative_solves_enabled::with(true);
            op.inv_quad_logdet(&x, 1e-6).unwrap()
        };
        for (a, b) in [(&off, &on), (&off_iq, &on_iq), (&off_ld, &on_ld)] {
            assert_eq!(
                a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "flag flipped the small-n solve path"
            );
        }
    }

    /// Above the threshold with the flag on, the trait routes to the Krylov
    /// nodes: the solve and quadratic form must match the dense answers to
    /// solver tolerance, the SLQ logdet must land near the exact one, and
    /// gradients must still flow.
    #[test]
    fn above_threshold_the_iterative_solve_route_matches_the_dense_answers() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let a = spd(16, &dev);
        let op = DenseOperator::new(a.clone());
        let x = Tensor::randn(0f32, 1f32, (16, 2), &dev).unwrap();
        // Dense references BEFORE flipping the flag.
        let want_solve = op.cholesky_solve(&x, 1e-6).unwrap();
        let want_iq = op.inv_quad(&x, 1e-6).unwrap();
        let want_ld: f32 = op.log_abs_det(1e-6).unwrap().to_scalar().unwrap();

        let _flag = settings::iterative_solves_enabled::with(true);
        let _size = settings::max_cholesky_size::with(4); // force the routed arm
        let got_solve = op.cholesky_solve(&x, 1e-6).unwrap();
        assert!(
            max_abs(&got_solve, &want_solve) < 1e-2,
            "routed solve drifted"
        );
        let got_iq = op.inv_quad(&x, 1e-6).unwrap();
        assert!(
            max_abs(&got_iq, &want_iq) < 1e-2 * 16.0,
            "routed inv_quad drifted"
        );
        let got_ld: f32 = op.log_abs_det(1e-6).unwrap().to_scalar().unwrap();
        assert!(
            (got_ld - want_ld).abs() < 0.3 * want_ld.abs().max(1.0),
            "routed SLQ logdet {got_ld} too far from dense {want_ld}"
        );

        // Gradients through the routed arms.
        let v = Var::from_tensor(&a).unwrap();
        let vop = DenseOperator::new(v.as_tensor().clone());
        let (iq, ld) = vop.inv_quad_logdet(&x, 1e-6).unwrap();
        let loss = iq.sum_all().unwrap().add(&ld.sum_all().unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        let g = grads
            .get(&v)
            .expect("gradient must reach K through the routed solves");
        let gmax = g
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gmax.is_finite() && gmax > 0.0,
            "routed-arm gradient: {gmax}"
        );
    }

    /// Every operator's specialised methods must agree with the dense
    /// reference computed through the default trait path.
    #[test]
    fn diag_operator_matches_dense_reference() {
        let dev = Device::Cpu;
        let d = Tensor::new(&[2.0f32, 0.5, 4.0, 1.5], &dev).unwrap();
        let op = DiagOperator::new(d.clone());
        let dense = DenseOperator::new(op.to_dense().unwrap());

        assert_eq!(op.shape(), vec![4, 4]);
        let x = Tensor::randn(0f32, 1f32, (4, 3), &dev).unwrap();
        assert!(max_abs(&op.matmul(&x).unwrap(), &dense.matmul(&x).unwrap()) < 1e-5);
        assert!(max_abs(&op.diagonal().unwrap(), &d) < 1e-6);

        // solve / logdet / inv_quad against the dense (Cholesky) path
        let s_fast = op.cholesky_solve(&x, 0.0).unwrap();
        let s_ref = dense.cholesky_solve(&x, 0.0).unwrap();
        assert!(max_abs(&s_fast, &s_ref) < 1e-4, "diag solve");

        let ld_fast = op.log_abs_det(0.0).unwrap().to_scalar::<f32>().unwrap();
        let ld_ref = dense.log_abs_det(0.0).unwrap().to_scalar::<f32>().unwrap();
        assert!((ld_fast - ld_ref).abs() < 1e-4, "{ld_fast} vs {ld_ref}");

        let q_fast = op.inv_quad(&x, 0.0).unwrap();
        let q_ref = dense.inv_quad(&x, 0.0).unwrap();
        assert!(max_abs(&q_fast, &q_ref) < 1e-3, "diag inv_quad");
    }

    #[test]
    fn constant_diag_matches_dense_reference() {
        let dev = Device::Cpu;
        let c = Tensor::new(3.0f32, &dev).unwrap();
        let op = ConstantDiagOperator::new(c, 5);
        let dense = DenseOperator::new(op.to_dense().unwrap());
        assert_eq!(op.shape(), vec![5, 5]);

        let x = Tensor::randn(0f32, 1f32, (5, 2), &dev).unwrap();
        assert!(max_abs(&op.matmul(&x).unwrap(), &dense.matmul(&x).unwrap()) < 1e-5);
        assert!(
            max_abs(
                &op.cholesky_solve(&x, 0.0).unwrap(),
                &dense.cholesky_solve(&x, 0.0).unwrap()
            ) < 1e-4
        );
        let a = op.log_abs_det(0.0).unwrap().to_scalar::<f32>().unwrap();
        let b = dense.log_abs_det(0.0).unwrap().to_scalar::<f32>().unwrap();
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }

    #[test]
    fn root_operator_never_forms_the_product() {
        let dev = Device::Cpu;
        let r = Tensor::randn(0f32, 1f32, (6, 2), &dev).unwrap();
        let op = RootOperator::new(r.clone());
        assert_eq!(op.shape(), vec![6, 6]);
        assert_eq!(op.rank(), 2);

        let dense = DenseOperator::new(op.to_dense().unwrap());
        let x = Tensor::randn(0f32, 1f32, (6, 3), &dev).unwrap();
        assert!(max_abs(&op.matmul(&x).unwrap(), &dense.matmul(&x).unwrap()) < 1e-4);
        // diag(R Rᵀ) via row norms
        assert!(max_abs(&op.diagonal().unwrap(), &dense.diagonal().unwrap()) < 1e-5);
    }

    #[test]
    fn added_diag_matches_dense_sum() {
        let dev = Device::Cpu;
        let a = spd(4, &dev);
        let d = Tensor::new(&[0.5f32, 1.0, 1.5, 2.0], &dev).unwrap();
        let op = AddedDiagOperator::new(
            Box::new(DenseOperator::new(a.clone())),
            DiagOperator::new(d.clone()),
        );
        let want = a.add(&diag_embed(&d).unwrap()).unwrap();
        assert!(max_abs(&op.to_dense().unwrap(), &want) < 1e-5);

        let x = Tensor::randn(0f32, 1f32, (4, 2), &dev).unwrap();
        let ref_op = DenseOperator::new(want.clone());
        assert!(max_abs(&op.matmul(&x).unwrap(), &ref_op.matmul(&x).unwrap()) < 1e-4);
        assert!(max_abs(&op.diagonal().unwrap(), &ref_op.diagonal().unwrap()) < 1e-5);
    }

    /// The differentiable Cholesky solve must actually invert, and the
    /// quadratic form must match the explicit `xᵀA⁻¹x`.
    #[test]
    fn cholesky_solve_and_inv_quad_are_exact() {
        let dev = Device::Cpu;
        let a = spd(5, &dev);
        let op = DenseOperator::new(a.clone());
        let x = Tensor::randn(0f32, 1f32, (5, 2), &dev).unwrap();

        // A @ (A⁻¹x) == x
        let sol = op.cholesky_solve(&x, 1e-9).unwrap();
        let back = a.matmul(&sol).unwrap();
        assert!(max_abs(&back, &x) < 1e-3, "solve must invert");

        // inv_quad == diag(xᵀ A⁻¹ x)
        let q = op.inv_quad(&x, 1e-9).unwrap();
        let want = x.t().unwrap().matmul(&sol).unwrap();
        let want_diag = dense_diagonal(&want).unwrap();
        assert!(max_abs(&q, &want_diag) < 1e-3, "inv_quad");

        // inv_quad_logdet shares one factorisation → same answers
        let (q2, ld2) = op.inv_quad_logdet(&x, 1e-9).unwrap();
        assert!(max_abs(&q, &q2) < 1e-6);
        let ld = op.log_abs_det(1e-9).unwrap();
        assert!(max_abs(&ld, &ld2) < 1e-6);
    }

    /// Batched: [L, N, N] stacks are the SVGP's per-latent layout.
    #[test]
    fn batched_operators_carry_leading_dims() {
        let dev = Device::Cpu;
        let l = 3usize;
        let mut mats = Vec::new();
        for _ in 0..l {
            mats.push(spd(4, &dev));
        }
        let a = Tensor::stack(&mats, 0).unwrap(); // [3, 4, 4]
        let op = DenseOperator::new(a.clone());
        assert_eq!(op.shape(), vec![3, 4, 4]);
        assert_eq!(op.batch_shape(), vec![3]);

        let x = Tensor::randn(0f32, 1f32, (l, 4, 2), &dev).unwrap();
        let sol = op.cholesky_solve(&x, 1e-9).unwrap();
        assert_eq!(sol.dims(), &[3, 4, 2]);
        let back = a.matmul(&sol).unwrap();
        assert!(max_abs(&back, &x) < 1e-3);

        let ld = op.log_abs_det(1e-9).unwrap();
        assert_eq!(ld.dims(), &[3]);
    }

    /// The whole point of the Candle-native choice: gradients must flow
    /// through the operator algebra (solve + logdet), so the GP island
    /// keeps training.
    #[test]
    fn gradients_flow_through_solve_and_logdet() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let base = spd(4, &dev);
        let v = Var::from_tensor(&base).unwrap();
        let op = DenseOperator::new(v.as_tensor().clone());
        let x = Tensor::randn(0f32, 1f32, (4, 1), &dev).unwrap();

        let (q, ld) = op.inv_quad_logdet(&x, 1e-6).unwrap();
        let loss = q.sum_all().unwrap().add(&ld.sum_all().unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        let g = grads
            .get(&v)
            .expect("gradient must reach the operator's tensor");
        assert_eq!(g.dims(), &[4, 4]);
        let gmax = g
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            gmax.is_finite() && gmax > 0.0,
            "gradient must be finite/non-zero: {gmax}"
        );
    }
}

/// Original dk-tensor CPU reference operators.
pub use crate::reference_operators::{conjugate_gradient, Dense, DiagonalPreconditioner};
