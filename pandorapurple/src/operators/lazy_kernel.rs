//! Inferno — Matrix operators that evaluate kernel products in tiles without materialising the full matrix.

use std::sync::Arc;

use candle_core::{Result, Tensor, D};

use super::{BlockCfg, LazyTensor, LinearOperator};
use crate::gp::KernelEval;

/// A kernel callable: `(x1 [.., N, D], x2 [.., M, D], diag) → [.., N, M]`
/// (or `[.., N]` when `diag` is set and `N == M`).
///
/// Deliberately the same shape as the existing
/// [`KernelEval`](crate::gp::KernelEval) seam so a ported kernel can be
/// dropped in without an adapter.
pub type KernelFn = Arc<dyn Fn(&Tensor, &Tensor, bool) -> Result<Tensor> + Send + Sync>;

/// Lazily-evaluated kernel matrix.
#[derive(Clone)]
pub struct LazyKernelOperator {
    pub x1: Tensor, // [.., N, D]
    pub x2: Tensor, // [.., M, D]
    kernel: KernelFn,
    /// Row-block size for `matmul`. Bounds the working set: peak memory is
    /// `O(block_rows · M)`. `None` = evaluate in one shot.
    pub block_rows: Option<usize>,
}

impl LazyKernelOperator {
    pub fn new(x1: Tensor, x2: Tensor, kernel: KernelFn) -> Self {
        Self { x1, x2, kernel, block_rows: None }
    }

    /// Set the row-block size used by `matmul` (the memory/step trade-off).
    pub fn with_block_rows(mut self, rows: usize) -> Self {
        self.block_rows = Some(rows.max(1));
        self
    }

    /// Evaluate the FULL kernel matrix — the escape hatch. Everything else
    /// on this type exists to avoid calling it.
    pub fn evaluate(&self) -> Result<Tensor> {
        (self.kernel)(&self.x1, &self.x2, false)
    }

    /// Transposed operator `K(x₂, x₁)`, still lazy.
    pub fn transpose_lazy(&self) -> Self {
        Self {
            x1: self.x2.clone(),
            x2: self.x1.clone(),
            kernel: self.kernel.clone(),
            block_rows: self.block_rows,
        }
    }
}

impl LinearOperator for LazyKernelOperator {
    /// Cheap probes off the stored inputs — the trait defaults densify,
    /// which is exactly what a lazy kernel operator above
    /// `max_cholesky_size` must never do on the sampling path.
    fn to_dense_dtype(&self) -> Result<candle_core::DType> {
        Ok(self.x1.dtype())
    }
    fn device(&self) -> Result<candle_core::Device> {
        Ok(self.x1.device().clone())
    }

    /// Batch dims are the BROADCAST of `x1` and `x2`, not `x1`'s alone.
    ///
    /// The `KernelEval` seam routinely pairs a shared `x [N, D]` with
    /// per-latent inducing inputs `z [L, M, D]`; taking the batch from `x1`
    /// reported `[N, M]` and silently dropped `L`, feeding every default
    /// trait method a wrong shape. Dormant until now only because this type
    /// had no consumers.
    fn shape(&self) -> Vec<usize> {
        let d1 = self.x1.dims();
        let d2 = self.x2.dims();
        let mut s = super::symbolic::broadcast_batch(&d1[..d1.len() - 2], &d2[..d2.len() - 2])
            .unwrap_or_else(|_| d1[..d1.len() - 2].to_vec());
        s.push(d1[d1.len() - 2]);
        s.push(d2[d2.len() - 2]);
        s
    }

    fn to_dense(&self) -> Result<Tensor> {
        self.evaluate()
    }

    /// `K @ rhs` evaluated in row blocks — the whole point of the type.
    ///
    /// Each block is `K(x₁[block], x₂) @ rhs`, so only `block_rows × M` of
    /// the kernel exists at any moment. With `block_rows = None` this is
    /// one full evaluation (still correct, just not memory-bounded).
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let n = self.nrows();
        let Some(bs) = self.block_rows else {
            return self.evaluate()?.broadcast_matmul(rhs);
        };
        if bs >= n {
            return self.evaluate()?.broadcast_matmul(rhs);
        }
        let row_axis = self.x1.rank() - 2;
        let mut blocks = Vec::with_capacity(n.div_ceil(bs));
        let mut start = 0;
        while start < n {
            let len = bs.min(n - start);
            let x1b = self.x1.narrow(row_axis, start, len)?;
            let kb = (self.kernel)(&x1b, &self.x2, false)?; // [.., len, M]
            blocks.push(kb.broadcast_matmul(rhs)?); // [.., len, C]
            start += len;
        }
        let refs: Vec<&Tensor> = blocks.iter().collect();
        Tensor::cat(&refs, D::Minus2)
    }

    /// `diag(K)` via the kernel's own diagonal mode — `O(N)` kernel
    /// evaluations, never the `[N, N]` matrix.
    fn diagonal(&self) -> Result<Tensor> {
        (self.kernel)(&self.x1, &self.x2, true)
    }

    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.transpose_lazy().matmul(rhs)
    }
}

// ---------------------------------------------------------------------------
// SymbolicKernelOperator — a formula as a LinearOperator
// ---------------------------------------------------------------------------

/// A [`LinearOperator`] backed by a symbolic
/// [`LazyTensor`](crate::operators::LazyTensor) formula.
///
/// Where [`LazyKernelOperator`] defers a closure that still materialises a
/// whole block, this defers the *expression*, so intermediates on the pair
/// axis never exist at full size either. Use it when the kernel can be
/// written as a formula; use `LazyKernelOperator` when it can only be
/// written as opaque code.
pub struct SymbolicKernelOperator {
    pub expr: LazyTensor,
    pub cfg: BlockCfg,
}

impl SymbolicKernelOperator {
    pub fn new(expr: LazyTensor) -> Self {
        Self { expr, cfg: BlockCfg::default() }
    }
    pub fn with_cfg(mut self, cfg: BlockCfg) -> Self {
        self.cfg = cfg;
        self
    }
}

impl LinearOperator for SymbolicKernelOperator {
    fn shape(&self) -> Vec<usize> {
        // `infer` is O(nodes) and cannot fail for a formula that was already
        // built; a malformed one surfaces properly from the evaluating
        // methods, so fall back to an empty shape rather than panicking here.
        match self.expr.infer() {
            Ok(m) => {
                let mut s = m.batch.clone();
                s.push(m.m.unwrap_or(0));
                s.push(m.n.unwrap_or(0));
                s
            }
            Err(_) => Vec::new(),
        }
    }

    fn to_dense(&self) -> Result<Tensor> {
        self.expr.to_dense(&self.cfg)
    }

    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.expr.matmul(rhs, &self.cfg)
    }

    /// Only the tiles the diagonal passes through — `O(N·bi)`, not `O(N²)`.
    fn diagonal(&self) -> Result<Tensor> {
        self.expr.diagonal(&self.cfg)
    }
}

// ---------------------------------------------------------------------------
// The packed-hyper seam — closure kernels join the differentiable path
// ---------------------------------------------------------------------------

/// A kernel callable with its hyperparameters as an EXPLICIT tensor
/// argument: `(x1 [.., N, D], x2 [.., M, D], hyper [H], diag) → [.., N, M]`.
///
/// The whole point of the fourth argument: a plain [`KernelFn`] hides its
/// parameters inside closure captures, where no gradient machinery can see
/// or rebind them. With hypers packed into one tensor that the closure
/// RECEIVES, "replace the parameters" is just "call the same closure with a
/// different tensor" — which is exactly the
/// [`DifferentiableKernelOp::with_grad_leaves`](super::ciq::DifferentiableKernelOp)
/// contract, so closure-based kernels get differentiable CIQ through the
/// same nested-tape route as symbolic ones. (The in-repo precedent is the
/// coupling kernel's `cost_hyper` packing for its FFI backend.)
///
/// The closure MUST build the tile from candle ops on its arguments —
/// gradients to `x1`/`x2`/`hyper` flow through candle's tape inside the
/// nested graph. A closure that detaches internally gets silently dead
/// hypers, the exact bug class the symbolic layer exists to prevent; there
/// is no way to check this from outside, so it is a documented contract.
///
/// One caveat a closure author must know: under blocked evaluation the
/// closure sees `(x1_block, x2)` — any diagonal/identity term must be
/// derived from the COORDINATES (e.g. a zero-distance indicator), never
/// from block shapes, or it lands on each block's local diagonal.
pub type HyperKernelFn =
    Arc<dyn Fn(&Tensor, &Tensor, &Tensor, bool) -> Result<Tensor> + Send + Sync>;

/// [`LazyKernelOperator`] with the packed-hyper seam: stores `(x1, x2,
/// hyper, closure)` and delegates every evaluation to a curried
/// `LazyKernelOperator`, so the blocked matmul, broadcast shape fix and
/// diag shortcut are all inherited rather than duplicated.
#[derive(Clone)]
pub struct HyperKernelOperator {
    pub x1: Tensor,
    pub x2: Tensor,
    /// Packed hyperparameters, LIVE — the gradient pack routes back to
    /// whatever tracked ops built this tensor (typically a `cat` of
    /// reparameterised raw params).
    pub hyper: Tensor,
    kernel: HyperKernelFn,
    pub block_rows: Option<usize>,
}

impl HyperKernelOperator {
    pub fn new(x1: Tensor, x2: Tensor, hyper: Tensor, kernel: HyperKernelFn) -> Self {
        Self { x1, x2, hyper, kernel, block_rows: None }
    }
    pub fn with_block_rows(mut self, rows: usize) -> Self {
        self.block_rows = Some(rows.max(1));
        self
    }

    /// Curry the hyper into a plain [`LazyKernelOperator`] over the given
    /// tensors — the delegation seam every method routes through.
    fn curried(&self, x1: Tensor, x2: Tensor, hyper: Tensor) -> LazyKernelOperator {
        let k = self.kernel.clone();
        let f: KernelFn = Arc::new(move |a: &Tensor, b: &Tensor, diag: bool| k(a, b, &hyper, diag));
        let mut op = LazyKernelOperator::new(x1, x2, f);
        if let Some(r) = self.block_rows {
            op = op.with_block_rows(r);
        }
        op
    }

    fn live(&self) -> LazyKernelOperator {
        self.curried(self.x1.clone(), self.x2.clone(), self.hyper.clone())
    }
}

impl LinearOperator for HyperKernelOperator {
    fn shape(&self) -> Vec<usize> {
        self.live().shape()
    }
    fn to_dense(&self) -> Result<Tensor> {
        self.live().to_dense()
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.live().matmul(rhs)
    }
    fn diagonal(&self) -> Result<Tensor> {
        self.live().diagonal()
    }
    fn to_dense_dtype(&self) -> Result<candle_core::DType> {
        Ok(self.x1.dtype())
    }
    fn device(&self) -> Result<candle_core::Device> {
        Ok(self.x1.device().clone())
    }
}

/// Differentiable CIQ for closure kernels. Unlike the symbolic impl below,
/// there is no formula DAG to derive gradients from — the tracked matvec
/// runs through candle's tape inside the nested graph (`O(row-blocks)` tape
/// nodes, bounded by `block_rows`). Symbolic operators remain the
/// symbolic-VJP path; this is the pragmatic on-ramp for kernels that only
/// exist as code.
impl super::ciq::DifferentiableKernelOp for HyperKernelOperator {
    fn n(&self) -> usize {
        self.ncols()
    }
    fn grad_leaves(&self) -> Vec<Tensor> {
        vec![self.x1.clone(), self.x2.clone(), self.hyper.clone()]
    }
    fn with_grad_leaves(
        &self,
        leaves: &[Tensor],
    ) -> Result<Box<dyn super::ciq::DifferentiableKernelOp>> {
        if leaves.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "HyperKernelOperator::with_grad_leaves: expected [x1, x2, hyper], got {}",
                leaves.len()
            )));
        }
        Ok(Box::new(Self {
            x1: leaves[0].clone(),
            x2: leaves[1].clone(),
            hyper: leaves[2].clone(),
            kernel: self.kernel.clone(),
            block_rows: self.block_rows,
        }))
    }
    fn matmul_detached(&self, x: &Tensor) -> Result<Tensor> {
        self.curried(self.x1.detach(), self.x2.detach(), self.hyper.detach())
            .matmul(&x.detach())
    }
    fn matmul_tracked(&self, x: &Tensor) -> Result<Tensor> {
        self.live().matmul(x)
    }
}

/// Differentiable-CIQ seam for the symbolic operator: the leaves of the
/// formula DAG are the gradient pack, `with_leaves` is the rebind, and the
/// existing tiled `matmul` (stack+sum accumulation, `O(tiles)` backward
/// memory) is the tracked matvec. `matmul_detached` re-uses
/// `detach_leaves`, so the Krylov solves allocate no tape at all.
impl super::ciq::DifferentiableKernelOp for SymbolicKernelOperator {
    fn n(&self) -> usize {
        use super::LinearOperator;
        self.ncols()
    }
    fn grad_leaves(&self) -> Vec<Tensor> {
        self.expr.leaves()
    }
    fn with_grad_leaves(&self, leaves: &[Tensor]) -> Result<Box<dyn super::ciq::DifferentiableKernelOp>> {
        Ok(Box::new(Self { expr: self.expr.with_leaves(leaves)?, cfg: self.cfg.clone() }))
    }
    fn matmul_detached(&self, x: &Tensor) -> Result<Tensor> {
        self.expr.detach_leaves().matmul(&x.detach(), &self.cfg)
    }
    /// One [`super::genred::Genred`] node: `Σ_j k_ij · x_j` as
    /// `Sum(formula · Vj(x))` over J. The tape gets ONE node and the
    /// backward runs the SYMBOLIC gradient reductions — `O(tile)` memory —
    /// instead of the raw tracked tiling's per-tile activations. `x` rides
    /// as a leaf, so its gradient (`Kᵀḡ`) comes from the same machinery.
    fn matmul_tracked(&self, x: &Tensor) -> Result<Tensor> {
        use super::genred::{Genred, ReduceOp};
        use super::symbolic::ReduceAxis;
        Genred::new(&self.expr * &LazyTensor::vj(x), ReduceOp::Sum, ReduceAxis::J)
            .with_cfg(self.cfg.clone())
            .eval()
    }
}

// ---------------------------------------------------------------------------
// KernelEval adapter
// ---------------------------------------------------------------------------

/// `K(z, z)` as a lazy operator over an existing [`KernelEval`].
///
/// Two constructors rather than one, because the seams are not
/// interchangeable the way [`KernelFn`]'s doc comment claims: `KernelEval`
/// has three methods with *different roles*, and `kdiag` takes a latent
/// COUNT rather than a second tensor. A single adapter cannot express that.
pub fn kmm_operator(k: Arc<dyn KernelEval>, z: Tensor) -> LazyKernelOperator {
    let kk = k.clone();
    LazyKernelOperator::new(
        z.clone(),
        z,
        Arc::new(move |a: &Tensor, _b: &Tensor, diag: bool| {
            if diag {
                kk.kdiag(a, a.dims()[0])
            } else {
                kk.kmm(a)
            }
        }),
    )
}

/// `K(x, z)` as a lazy operator. Rectangular, so there is no diagonal mode —
/// asking for one is a caller error, not something to silently reinterpret.
pub fn kxm_operator(k: Arc<dyn KernelEval>, x: Tensor, z: Tensor) -> LazyKernelOperator {
    let kk = k.clone();
    LazyKernelOperator::new(
        x,
        z,
        Arc::new(move |a: &Tensor, b: &Tensor, diag: bool| {
            if diag {
                return Err(candle_core::Error::Msg(
                    "kxm_operator: K(x, z) is rectangular — no diagonal mode".into(),
                ));
            }
            kk.kxm(a, b)
        }),
    )
}

#[cfg(test)]
mod tests {
    //! Verification of matrix operators that evaluate kernel products in tiles without materialising the full matrix.

    use super::*;
    use candle_core::Device;

    /// A real RBF kernel closure, in both full and diagonal modes.
    fn rbf() -> KernelFn {
        Arc::new(|x1: &Tensor, x2: &Tensor, diag: bool| {
            if diag {
                // k(xᵢ, xᵢ) = 1 for RBF — O(N), no matrix.
                let n = x1.dims()[x1.rank() - 2];
                return Tensor::ones(n, x1.dtype(), x1.device());
            }
            // ‖x₁ − x₂‖² via the expansion, then exp(−½·)
            let s1 = x1.sqr()?.sum(D::Minus1)?.unsqueeze(D::Minus1)?; // [N, 1]
            let s2 = x2.sqr()?.sum(D::Minus1)?.unsqueeze(D::Minus2)?; // [1, M]
            let cross = x1.broadcast_matmul(&x2.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
            let d2 = s1.broadcast_add(&s2)?.sub(&cross.affine(2.0, 0.0)?)?;
            d2.affine(-0.5, 0.0)?.exp()
        })
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
    }

    /// Blocked evaluation must be numerically identical to one-shot — the
    /// blocking is a memory strategy, not an approximation.
    #[test]
    fn blocked_matmul_equals_full_evaluation() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (17, 3), &dev).unwrap(); // deliberately not a multiple
        let x2 = Tensor::randn(0f32, 1f32, (11, 3), &dev).unwrap();
        let rhs = Tensor::randn(0f32, 1f32, (11, 4), &dev).unwrap();

        let full = LazyKernelOperator::new(x1.clone(), x2.clone(), rbf());
        let blocked = LazyKernelOperator::new(x1, x2, rbf()).with_block_rows(5);

        assert_eq!(full.shape(), vec![17, 11]);
        let a = full.matmul(&rhs).unwrap();
        let b = blocked.matmul(&rhs).unwrap();
        assert_eq!(a.dims(), &[17, 4]);
        assert!(max_abs(&a, &b) < 1e-5, "blocking changed the answer: {}", max_abs(&a, &b));

        // and both equal the explicit dense product
        let dense = full.evaluate().unwrap().matmul(&rhs).unwrap();
        assert!(max_abs(&a, &dense) < 1e-5);
    }

    /// The diagonal path must not touch the full matrix.
    #[test]
    fn diagonal_uses_the_kernels_diag_mode() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (6, 3), &dev).unwrap();
        let op = LazyKernelOperator::new(x.clone(), x.clone(), rbf());
        let d = op.diagonal().unwrap();
        assert_eq!(d.dims(), &[6]);
        // RBF self-covariance is 1
        let ones = Tensor::ones(6, d.dtype(), &dev).unwrap();
        assert!(max_abs(&d, &ones) < 1e-5);
        // consistent with the dense diagonal
        let dense_diag = crate::operators::dense_diagonal(&op.evaluate().unwrap()).unwrap();
        assert!(max_abs(&d, &dense_diag) < 1e-4);
    }

    #[test]
    fn transpose_is_the_swapped_kernel() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (5, 2), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (7, 2), &dev).unwrap();
        let op = LazyKernelOperator::new(x1, x2, rbf());
        let t = op.transpose_lazy();
        assert_eq!(t.shape(), vec![7, 5]);
        let kt = t.evaluate().unwrap();
        let k = op.evaluate().unwrap();
        assert!(max_abs(&kt, &k.t().unwrap()) < 1e-5);
    }


    /// Regression for the batch-broadcast bug: the `KernelEval` seam pairs a
    /// shared `x [N, D]` with per-latent inducing `z [L, M, D]`. Taking the
    /// batch from `x1` alone reported `[N, M]` and dropped `L`.
    #[test]
    fn shape_broadcasts_batch_dims_from_both_sides() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((7, 3), candle_core::DType::F32, &dev).unwrap();
        let z = Tensor::zeros((4, 5, 3), candle_core::DType::F32, &dev).unwrap();
        let op = LazyKernelOperator::new(x, z, rbf());
        assert_eq!(op.shape(), vec![4, 7, 5], "latent axis must survive");
    }

    #[test]
    fn symbolic_operator_matches_the_closure_operator() {
        use crate::operators::{BlockCfg, LazyTensor, ReduceAxis};
        let _ = ReduceAxis::J;
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (13, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (9, 3), &dev).unwrap();

        // The same RBF, written as a formula.
        let d2 = (&LazyTensor::vi(&x1) - &LazyTensor::vj(&x2)).powf(2.0).sum_last();
        let sym = SymbolicKernelOperator::new(d2.affine(-0.5, 0.0).exp())
            .with_cfg(BlockCfg::with_tiles(5, 4));
        let closure = LazyKernelOperator::new(x1, x2, rbf());

        assert_eq!(sym.shape(), closure.shape());
        assert!(max_abs(&sym.to_dense().unwrap(), &closure.to_dense().unwrap()) < 1e-5);

        let rhs = Tensor::randn(0f32, 1f32, (9, 2), &dev).unwrap();
        assert!(max_abs(&sym.matmul(&rhs).unwrap(), &closure.matmul(&rhs).unwrap()) < 1e-5);
    }

    /// A fake `KernelEval` so the adapters are exercised without pulling in
    /// a real kernel's parameter tree.
    struct FakeEval;
    impl crate::gp::KernelEval for FakeEval {
        fn kmm(&self, z: &Tensor) -> Result<Tensor> {
            let zt = z.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            z.broadcast_matmul(&zt)
        }
        fn kxm(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
            let zt = z.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            x.broadcast_matmul(&zt)
        }
        fn kdiag(&self, x: &Tensor, _l: usize) -> Result<Tensor> {
            x.sqr()?.sum(D::Minus1)
        }
    }

    #[test]
    fn kmm_adapter_routes_full_and_diagonal_modes() {
        let dev = Device::Cpu;
        let z = Tensor::randn(0f32, 1f32, (6, 3), &dev).unwrap();
        let op = kmm_operator(Arc::new(FakeEval), z.clone());
        assert_eq!(op.shape(), vec![6, 6]);

        let dense = op.to_dense().unwrap();
        assert_eq!(dense.dims(), &[6, 6]);
        // kdiag is its own computation, not an extraction from the full matrix
        let d = op.diagonal().unwrap();
        assert!(max_abs(&d, &crate::operators::dense_diagonal(&dense).unwrap()) < 1e-4);
    }

    /// `K(x, z)` is rectangular; a diagonal request is a caller error and
    /// must say so rather than returning something shaped plausibly.
    #[test]
    fn kxm_adapter_refuses_a_diagonal() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (5, 3), &dev).unwrap();
        let z = Tensor::randn(0f32, 1f32, (7, 3), &dev).unwrap();
        let op = kxm_operator(Arc::new(FakeEval), x, z);
        assert_eq!(op.shape(), vec![5, 7]);
        assert!(op.to_dense().is_ok());
        let e = op.diagonal().unwrap_err().to_string();
        assert!(e.contains("rectangular"), "{e}");
    }

    /// Gradients must survive the blocked path (each block is a real
    /// kernel call; `cat` keeps one graph).
    #[test]
    fn gradients_flow_through_blocked_matmul() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let vx = Var::from_tensor(&Tensor::randn(0f32, 1f32, (9, 3), &dev).unwrap()).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (4, 3), &dev).unwrap();
        let rhs = Tensor::randn(0f32, 1f32, (4, 1), &dev).unwrap();

        let op = LazyKernelOperator::new(vx.as_tensor().clone(), x2, rbf()).with_block_rows(4);
        let loss = op.matmul(&rhs).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g = grads.get(&vx).expect("gradient must reach x1 through the blocks");
        assert_eq!(g.dims(), &[9, 3]);
        let m = g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(m.is_finite() && m > 0.0, "grad {m}");
    }
}
