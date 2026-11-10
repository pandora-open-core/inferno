//! Inferno — Routed iterative linear solves with reusable operator interfaces.

use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor, D};
use std::sync::{Arc, Mutex};

use super::cg::{linear_cg, linear_cg_tridiag, CgOptions};
use super::ciq::{pack_leaves, unpack_leaves, DifferentiableKernelOp};
use super::precond::CiqPreconditioner;
use crate::linalg::jacobi_eigh;
use crate::linalg_diff::{square_batch_dims, storage_f64, storage_from_f64, tensor_f64};
use crate::settings;

const NO_PRECOND: Option<fn(&Tensor) -> Result<Tensor>> = None;

/// The solve options every node in this module actually runs with:
/// `tolerance` defaults to [`settings::minres_tolerance`] (NOT the raw
/// `cg_tolerance = 1.0`, which is gpytorch's stochastic-trace convention —
/// a solve node's output is read directly), threaded through
/// [`CgOptions::for_tolerance`] so the stagnation floor moves with it.
/// Options for a PURE solve. Carries the caller's preconditioner through:
/// `K⁻¹b` and `bᵀK⁻¹b` are preconditioning-INVARIANT, so `M⁻¹` changes only
/// how fast the iteration converges, never what it converges to. Every
/// forward solve, adjoint solve and `inv_quad` in this module routes here.
fn solve_opts(opts: &CgOptions) -> CgOptions {
    let tol = opts.tolerance.unwrap_or_else(settings::minres_tolerance::value);
    CgOptions {
        max_iter: opts.max_iter,
        precond: opts.precond.clone(),
        ..CgOptions::for_tolerance(tol)
    }
}

fn warn_unconverged(what: &'static str, stats: &super::cg::CgStats) {
    if !stats.converged && settings::verbose_linalg::value() {
        tracing::warn!(
            iterations = stats.iterations,
            residual = stats.residual_norm,
            "{what}: CG hit the iteration budget before tolerance"
        );
    }
}

// ---------------------------------------------------------------------------
// LinSolveOp — the dense differentiable node
// ---------------------------------------------------------------------------

/// `x = K⁻¹ b` with the solve-adjoint VJP, in the [`crate::linalg_diff`]
/// house style: forward and backward run the detached CG driver on host-f64
/// numerics; candle here is first-order-only, so nothing in `bwd` builds
/// graph.
struct LinSolveOp {
    opts: CgOptions,
}

impl CustomOp2 for LinSolveOp {
    fn name(&self) -> &'static str {
        "lin_solve_diff"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (k_data, _) = storage_f64(s1, l1)?;
        let (b_data, as_f32) = storage_f64(s2, l2)?;
        square_batch_dims(l1.shape())?; // validates squareness

        let dev = Device::Cpu;
        let k = Tensor::from_vec(k_data, l1.shape().clone(), &dev)?;
        let b = Tensor::from_vec(b_data, l2.shape().clone(), &dev)?;
        let (x, stats) =
            linear_cg(|v| k.broadcast_matmul(v), &b, None, NO_PRECOND, &solve_opts(&self.opts))?;
        warn_unconverged("lin_solve_diff", &stats);

        let x_host: Vec<f64> = x.flatten_all()?.to_vec1()?;
        Ok((storage_from_f64(x_host, as_f32), l2.shape().clone()))
    }

    fn bwd(
        &self,
        arg1: &Tensor, // K
        arg2: &Tensor, // b
        res: &Tensor,  // x = K⁻¹b — recoverable, unlike the CIQ per-shift solves
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let dev = Device::Cpu;
        let k = Tensor::from_vec(tensor_f64(arg1)?, arg1.shape().clone(), &dev)?;
        let g = Tensor::from_vec(tensor_f64(grad_res)?, grad_res.shape().clone(), &dev)?;
        let x = Tensor::from_vec(tensor_f64(res)?, res.shape().clone(), &dev)?;

        // b̄ = K⁻¹ ḡ — the one adjoint solve.
        let (u, stats) =
            linear_cg(|v| k.broadcast_matmul(v), &g, None, NO_PRECOND, &solve_opts(&self.opts))?;
        warn_unconverged("lin_solve_diff (adjoint)", &stats);

        // K̄ = −½ (u xᵀ + x uᵀ) — the matmul sums the C columns.
        let xt = x.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let ut = u.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let kbar = (u.broadcast_matmul(&xt)? + x.broadcast_matmul(&ut)?)?.affine(-0.5, 0.0)?;

        Ok((
            Some(kbar.to_dtype(arg1.dtype())?),
            Some(u.to_dtype(arg2.dtype())?),
        ))
    }
}

/// Differentiable `K⁻¹ rhs` for a DENSE symmetric-positive-definite
/// `K [..., N, N]`, `rhs [..., N]` or `[..., N, C]`. Gradients flow to both
/// arguments via the solve-adjoint VJP; the general `LinearOperator` route
/// for lazy operators is [`lin_solve_lazy`].
///
/// `K` and `rhs` must share their batch dims exactly (or both have none) —
/// gradient reduction across broadcast batch dims is deliberately not
/// implemented, matching [`super::ciq::ciq_inv_sqrt_diff`].
///
/// The symmetrised `K̄` is the CholeskyOp convention: `K` is SPD upstream,
/// so FD checks route through a symmetric construction (`K = BBᵀ + cI`)
/// and the chain rule through that construction makes the symmetrised
/// cotangent the right object. The raw Fréchet derivative for an
/// unconstrained `K` would be `−b̄ xᵀ` unsymmetrised.
pub fn lin_solve_diff(k: &Tensor, rhs: &Tensor, opts: &CgOptions) -> Result<Tensor> {
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
    let (kb, rb) = (&k.dims()[..k.rank() - 2], &rhs2.dims()[..rhs2.rank() - 2]);
    if kb != rb {
        return Err(candle_core::Error::Msg(format!(
            "lin_solve_diff: K batch dims {kb:?} must equal rhs batch dims {rb:?}"
        )));
    }
    let out = k
        .contiguous()?
        .apply_op2(&rhs2.contiguous()?, LinSolveOp { opts: opts.clone() })?;
    if is_vector { out.squeeze(D::Minus1) } else { Ok(out) }
}

/// Differentiable `rhsᵀ K⁻¹ rhs` (per column) — `rhs [..., N, C]` →
/// `[..., C]`. Pure composition over [`lin_solve_diff`]: candle's VJPs give
/// `b̄ = 2K⁻¹b`, `K̄ = −xxᵀ` with no second CustomOp to keep consistent.
pub fn inv_quad_diff(k: &Tensor, rhs: &Tensor, opts: &CgOptions) -> Result<Tensor> {
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
    let x = lin_solve_diff(k, &rhs2, opts)?;
    let q = (rhs2 * x)?.sum(D::Minus2)?;
    if is_vector { q.squeeze(D::Minus1) } else { Ok(q) }
}

// ---------------------------------------------------------------------------
// LinSolveLazyOp — solve gradients WITHOUT densifying K
// ---------------------------------------------------------------------------

/// `x = K⁻¹ rhs` for a LAZY symmetric SPD operator, with gradients to every
/// leaf parameter and to `rhs` — `K` is never densified, forward or
/// backward.
///
/// Same nested-tape pattern as [`super::ciq::ciq_inv_sqrt_lazy`], one
/// bilinear form instead of Q of them: leaves packed into the tracked first
/// argument; backward runs one detached adjoint solve `ū = K⁻¹ḡ`, then
/// takes `θ̄ = ∂/∂θ[−ūᵀ K(θ) x̂]` at frozen `ū, x̂` through ONE tracked
/// tiled matvec (`matmul_tracked` — a `Genred` node for symbolic
/// operators, whose backward evaluates the per-leaf cotangent formulas as
/// fresh tiled reductions). No symmetrisation appears: the gradient is
/// taken through the parametrisation `K(θ)`, symmetric-valued by
/// construction.
pub fn lin_solve_lazy(
    op: Arc<dyn DifferentiableKernelOp>,
    rhs: &Tensor,
    opts: &CgOptions,
) -> Result<Tensor> {
    let leaves = op.grad_leaves();
    if leaves.is_empty() {
        return Err(candle_core::Error::Msg(
            "lin_solve_lazy: operator exposes no gradient leaves — run the detached \
             linear_cg on matmul_detached instead"
                .into(),
        ));
    }
    let (packed, shapes) = pack_leaves(&leaves)?;
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
    let out = packed.contiguous()?.apply_op2(
        &rhs2.contiguous()?,
        LinSolveLazyOp { op, shapes, opts: opts.clone() },
    )?;
    if is_vector { out.squeeze(D::Minus1) } else { Ok(out) }
}

/// [`inv_quad_diff`] over a lazy operator — same composition, lazy node.
pub fn inv_quad_lazy(
    op: Arc<dyn DifferentiableKernelOp>,
    rhs: &Tensor,
    opts: &CgOptions,
) -> Result<Tensor> {
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
    let x = lin_solve_lazy(op, &rhs2, opts)?;
    let q = (rhs2 * x)?.sum(D::Minus2)?;
    if is_vector { q.squeeze(D::Minus1) } else { Ok(q) }
}

struct LinSolveLazyOp {
    op: Arc<dyn DifferentiableKernelOp>,
    shapes: Vec<Vec<usize>>,
    opts: CgOptions,
}

impl CustomOp2 for LinSolveLazyOp {
    fn name(&self) -> &'static str {
        "lin_solve_lazy"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // The packed argument's VALUES duplicate the leaves already inside
        // `self.op` — it exists to give candle a graph edge for the
        // parameter cotangent, not to carry data.
        let (b_data, as_f32) = storage_f64(s2, l2)?;
        let dev = Device::Cpu;
        let out_dtype = if as_f32 { DType::F32 } else { DType::F64 };
        let b = Tensor::from_vec(b_data, l2.shape().clone(), &dev)?.to_dtype(out_dtype)?;

        let (x, stats) = linear_cg(
            |v| self.op.matmul_detached(v),
            &b,
            None,
            NO_PRECOND,
            &solve_opts(&self.opts),
        )?;
        warn_unconverged("lin_solve_lazy", &stats);
        let x_host: Vec<f64> = x.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        Ok((storage_from_f64(x_host, as_f32), l2.shape().clone()))
    }

    fn bwd(
        &self,
        packed: &Tensor,
        rhs: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // ū = K⁻¹ ḡ, detached — doubles as rhs̄.
        let g = grad_res.detach().contiguous()?;
        let (u, stats) = linear_cg(
            |v| self.op.matmul_detached(v),
            &g,
            None,
            NO_PRECOND,
            &solve_opts(&self.opts),
        )?;
        warn_unconverged("lin_solve_lazy (adjoint)", &stats);
        let u = u.detach();
        let x = res.detach();

        // θ̄ via the nested tape: S(θ) = −ūᵀ K(θ) x̂ at frozen ū, x̂.
        let leaf_values = unpack_leaves(&packed.detach(), &self.shapes)?;
        let vars: Vec<candle_core::Var> = leaf_values
            .iter()
            .map(candle_core::Var::from_tensor)
            .collect::<Result<_>>()?;
        let leaf_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
        let tracked_op = self.op.with_grad_leaves(&leaf_tensors)?;

        let s = (u.clone() * tracked_op.matmul_tracked(&x)?)?
            .sum_all()?
            .affine(-1.0, 0.0)?;
        let grads = s.backward()?;

        let mut flat_grads = Vec::with_capacity(vars.len());
        for (var, val) in vars.iter().zip(&leaf_values) {
            match grads.get(var.as_tensor()) {
                Some(gr) => flat_grads.push(gr.flatten_all()?),
                None => flat_grads.push(val.zeros_like()?.flatten_all()?),
            }
        }
        let packed_bar = Tensor::cat(&flat_grads, 0)?.to_dtype(packed.dtype())?;
        Ok((Some(packed_bar), Some(u.to_dtype(rhs.dtype())?)))
    }
}

// ---------------------------------------------------------------------------
// SLQ log-determinant
// ---------------------------------------------------------------------------

/// Rademacher probe matrix `[n, p]` (entries ±1), deterministic in `seed` —
/// the FD-gradcheck and fwd/bwd-consistency requirement is that probes be
/// data, not hidden state, so they are drawn OUTSIDE the op and passed in.
pub fn rademacher_probes(
    n: usize,
    p: usize,
    seed: u64,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut data = Vec::with_capacity(n * p);
    for _ in 0..n * p {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push(if (s >> 62) & 1 == 1 { 1.0f64 } else { -1.0 });
    }
    Tensor::from_vec(data, (n, p), device)?.to_dtype(dtype)
}

/// SLQ estimate of `Σ_probes ‖z‖²·(ẑᵀ log(K) ẑ) / P` from the CG-tridiag
/// Ritz pairs — host-side, per batch element. `t: [P, k, k]`.
fn slq_from_tridiag(t: &Tensor, col_norms_sq: &[f64]) -> Result<f64> {
    let p = t.dims()[0];
    let k = t.dims()[1];
    let jit = settings::tridiagonal_jitter::value();
    let mut est = 0.0f64;
    for j in 0..p {
        if col_norms_sq[j] == 0.0 {
            continue; // an exactly-zero probe column carries no information
        }
        let tj: Vec<f64> = t.get(j)?.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
        // jacobi_eigh prescale rule: its convergence test is ABSOLUTE
        // (off-norm < 1e-14), so scale to unit Frobenius, decompose, scale
        // the eigenvalues back. Documented at the cg.rs spectrum test.
        let fro = tj.iter().map(|v| v * v).sum::<f64>().sqrt();
        if fro == 0.0 {
            // A nonzero probe whose tridiagonal came out identically zero
            // means CG recorded no curvature for it. The known cause is a
            // negative-curvature direction: `safe_div`'s SIGNED `den < eps`
            // guard freezes `α` to 0 when `pᵀAp < 0`, so CG-tridiag is
            // structurally blind to indefiniteness (the reason ciq.rs
            // probes spectrum bounds with direct Lanczos instead).
            // Silently skipping would return a confidently wrong logdet.
            return Err(candle_core::Error::Msg(format!(
                "slq_logdet: probe {j} produced an empty Lanczos tridiagonal — \
                 CG froze it (negative curvature?); operator not SPD?"
            )));
        }
        let scaled: Vec<f64> = tj.iter().map(|v| v / fro).collect();
        let (eig, vecs) = jacobi_eigh(&scaled, k);
        let lmax = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * fro;
        if !(lmax > 0.0) {
            return Err(candle_core::Error::Msg(format!(
                "slq_logdet: probe {j} produced a non-positive Ritz spectrum \
                 (max {lmax:.3e}) — operator not SPD?"
            )));
        }
        // Relative floor before the log, per the tridiagonal_jitter
        // contract — a near-singular direction must not send the estimate
        // to −∞ on fp noise.
        let floor = lmax * jit;
        let mut q = 0.0f64;
        for i in 0..k {
            let tau = vecs[i]; // first row of the eigenvector matrix: e₁ᵀwᵢ
            let lam = (eig[i] * fro).max(floor);
            q += tau * tau * lam.ln();
        }
        est += col_norms_sq[j] * q;
    }
    Ok(est / p as f64)
}

/// Options for the Lanczos-tridiagonal forward, which **drops** the
/// preconditioner that [`solve_opts`] carries.
///
/// Not an oversight and not a limitation of this module: a preconditioned CG
/// produces the tridiagonal of `M^{-1/2}AM^{-1/2}`, so `slq_from_tridiag`
/// would return `logdet(A) − logdet(M)`, and `host_col_norms_sq` would
/// compound that by supplying Euclidean `‖z‖²` where the preconditioned
/// quadrature weight is `zᵀMz`. Both errors are silent — a wrong logdet is
/// a plausible number.
///
/// So SLQ's forward runs unpreconditioned while its ADJOINT solve (a pure
/// solve, via [`solve_opts`]) is preconditioned. `cg::run` refuses the
/// combination outright, which makes a future edit here a hard error rather
/// than a quiet regression.
fn tridiag_opts(opts: &CgOptions, p: usize) -> CgOptions {
    CgOptions {
        n_tridiag: p,
        max_tridiag_iter: opts.max_tridiag_iter,
        precond: None,
        ..solve_opts(opts)
    }
}

/// Preconditioned SLQ `log det K`, DETACHED.
///
/// The combination `tridiag_opts` refuses, done correctly. Two independent
/// corrections are needed and each is silent on its own:
///
/// **1. The determinant shift.** A preconditioned CG runs the Lanczos process
/// for `Â = M^{-1/2}KM^{-1/2}`, so the quadrature estimates `tr ln Â` and
/// `logdet K = tr ln Â + logdet M`. That correction is closed-form —
/// [`CiqPreconditioner::log_det`].
///
/// **2. The probes, not just their norms.** This is the half that is easy to
/// get wrong. PCG on `Kx = z` is CG on `Â x̂ = M^{-1/2}z`, so the Lanczos start
/// vector is `w = M^{-1/2}z`. Hutchinson needs `E[w wᵀ] = I`; feeding Rademacher
/// `z` gives `E[w wᵀ] = M⁻¹` instead, and the estimator silently targets
/// `tr(M⁻¹ ln Â)` — a different quantity, not merely a rescaled one. So the
/// fix is not to reweight: the probes handed to the solve must be
/// `z = M^{1/2} w` for Rademacher `w`, which makes the start vector `w` again
/// and the quadrature weight the ordinary `‖w‖²`.
///
/// # Why this is not wired into `slq_logdet_diff` / `slq_logdet_lazy`
///
/// Those nodes' backward is the probe-shared trace estimator
/// `S(θ) = (ḡ/P)·Σ_z u_zᵀK(θ)z`, which is `tr(K⁻¹dK)` only for `E[zzᵀ] = I`.
/// The forward here deliberately breaks that for its own probes, so forward and
/// backward would need DIFFERENT probe sets — and "probes are explicit
/// arguments so forward, backward and the FD check share them" is the property
/// that makes those gradients checkable at all. Splitting them is a real design
/// change, not a wiring change, so this lands as a detached estimator only.
pub fn slq_logdet_preconditioned<F>(
    matmul: F,
    precond: &std::sync::Arc<CiqPreconditioner>,
    probes: &Tensor,
    opts: &CgOptions,
) -> Result<f64>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let p = probes.dim(D::Minus1)?;
    // z = M^{1/2} w, so the Lanczos start vector is w again.
    let z = precond.sqrt_apply(probes)?;
    let cg_opts = CgOptions {
        n_tridiag: p,
        max_tridiag_iter: opts.max_tridiag_iter,
        precond: Some(precond.clone()),
        allow_preconditioned_tridiag: true,
        ..solve_opts(opts)
    };
    let (_, stats, t) = linear_cg_tridiag(matmul, &z, None, NO_PRECOND, &cg_opts)?;
    warn_unconverged("slq_logdet_preconditioned", &stats);

    // The ORIGINAL probes' norms — `‖w‖²`, not `‖z‖²`.
    let norms = host_col_norms_sq(probes)?;
    let est = slq_from_tridiag(&t, &norms)?;
    let log_det_m = precond
        .log_det()?
        .to_dtype(DType::F64)?
        .sum_all()?
        .to_scalar::<f64>()?;
    Ok(est + log_det_m)
}

fn host_col_norms_sq(z: &Tensor) -> Result<Vec<f64>> {
    z.to_dtype(DType::F64)?.sqr()?.sum(D::Minus2)?.to_vec1()
}

struct SlqLogdetOp {
    opts: CgOptions,
}

impl CustomOp2 for SlqLogdetOp {
    fn name(&self) -> &'static str {
        "slq_logdet_diff"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (k_data, as_f32) = storage_f64(s1, l1)?;
        let (z_data, _) = storage_f64(s2, l2)?;
        let (bsz, n) = square_batch_dims(l1.shape())?;
        let p = l2.dims()[l2.shape().rank() - 1];

        let dev = Device::Cpu;
        let k = Tensor::from_vec(k_data, (bsz, n, n), &dev)?;
        let z = Tensor::from_vec(z_data, (bsz, n, p), &dev)?;

        let mut out = Vec::with_capacity(bsz);
        for b in 0..bsz {
            let kb = k.get(b)?;
            let zb = z.get(b)?;
            let (_, stats, t) = linear_cg_tridiag(
                |v| kb.matmul(v),
                &zb,
                None,
                NO_PRECOND,
                &tridiag_opts(&self.opts, p),
            )?;
            warn_unconverged("slq_logdet_diff", &stats);
            out.push(slq_from_tridiag(&t, &host_col_norms_sq(&zb)?)?);
        }

        let batch_dims: Vec<usize> = l1.dims()[..l1.shape().rank() - 2].to_vec();
        Ok((storage_from_f64(out, as_f32), Shape::from(batch_dims)))
    }

    fn bwd(
        &self,
        arg1: &Tensor, // K
        arg2: &Tensor, // probes — constants; no gradient
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let dev = Device::Cpu;
        let k = Tensor::from_vec(tensor_f64(arg1)?, arg1.shape().clone(), &dev)?;
        let z = Tensor::from_vec(tensor_f64(arg2)?, arg2.shape().clone(), &dev)?;
        let p = z.dim(D::Minus1)? as f64;

        // u_z = K⁻¹ z, one batched solve for every probe.
        let (u, stats) =
            linear_cg(|v| k.broadcast_matmul(v), &z, None, NO_PRECOND, &solve_opts(&self.opts))?;
        warn_unconverged("slq_logdet_diff (adjoint)", &stats);

        // K̄ = ḡ · (1/2P) Σ_z (u_z zᵀ + z u_zᵀ) — the stochastic trace
        // estimator of K⁻¹, symmetrised, sharing the forward's probes.
        let zt = z.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let ut = u.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let est = (u.broadcast_matmul(&zt)? + z.broadcast_matmul(&ut)?)?.affine(0.5 / p, 0.0)?;

        let g = Tensor::from_vec(tensor_f64(grad_res)?, grad_res.shape().clone(), &dev)?
            .unsqueeze(D::Minus1)?
            .unsqueeze(D::Minus1)?;
        let kbar = est.broadcast_mul(&g)?;
        Ok((Some(kbar.to_dtype(arg1.dtype())?), None))
    }
}

/// Differentiable stochastic-Lanczos-quadrature `log det K` for a DENSE SPD
/// `K [..., N, N]`, with probes `[..., N, P]` sharing `K`'s batch dims →
/// `[...]`. Probe columns are treated as constants (draw them with
/// [`rademacher_probes`], or pass `√N · I` columns for the exact/complete
/// estimator). See the module doc for the estimator-vs-value gradient
/// caveat.
pub fn slq_logdet_diff(k: &Tensor, probes: &Tensor, opts: &CgOptions) -> Result<Tensor> {
    let (kb, zb) = (&k.dims()[..k.rank() - 2], &probes.dims()[..probes.rank() - 2]);
    if kb != zb {
        return Err(candle_core::Error::Msg(format!(
            "slq_logdet_diff: K batch dims {kb:?} must equal probe batch dims {zb:?}"
        )));
    }
    if probes.dim(D::Minus2)? != k.dim(D::Minus1)? {
        return Err(candle_core::Error::Msg(format!(
            "slq_logdet_diff: probes have {} rows for an n = {} operator",
            probes.dim(D::Minus2)?,
            k.dim(D::Minus1)?
        )));
    }
    k.contiguous()?
        .apply_op2(&probes.contiguous()?, SlqLogdetOp { opts: opts.clone() })
}

/// [`slq_logdet_diff`] for a LAZY operator — leaf gradients via the same
/// nested-tape surrogate, `S(θ) = (ḡ/P)·Σ_z u_zᵀ K(θ) z` at frozen probe
/// solves. Unbatched (`probes [N, P]`) — the batched tridiag bookkeeping
/// is per-column, and no lazy consumer is batched today.
pub fn slq_logdet_lazy(
    op: Arc<dyn DifferentiableKernelOp>,
    probes: &Tensor,
    opts: &CgOptions,
) -> Result<Tensor> {
    if probes.rank() != 2 {
        return Err(candle_core::Error::Msg(
            "slq_logdet_lazy: probes must be [N, P] (unbatched)".into(),
        ));
    }
    if probes.dim(0)? != op.n() {
        return Err(candle_core::Error::Msg(format!(
            "slq_logdet_lazy: probes have {} rows for an n = {} operator",
            probes.dim(0)?,
            op.n()
        )));
    }
    let leaves = op.grad_leaves();
    if leaves.is_empty() {
        return Err(candle_core::Error::Msg(
            "slq_logdet_lazy: operator exposes no gradient leaves".into(),
        ));
    }
    let (packed, shapes) = pack_leaves(&leaves)?;
    packed.contiguous()?.apply_op2(
        &probes.contiguous()?,
        SlqLogdetLazyOp { op, shapes, opts: opts.clone() },
    )
}

struct SlqLogdetLazyOp {
    op: Arc<dyn DifferentiableKernelOp>,
    shapes: Vec<Vec<usize>>,
    opts: CgOptions,
}

impl CustomOp2 for SlqLogdetLazyOp {
    fn name(&self) -> &'static str {
        "slq_logdet_lazy"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (z_data, as_f32) = storage_f64(s2, l2)?;
        let dev = Device::Cpu;
        let out_dtype = if as_f32 { DType::F32 } else { DType::F64 };
        let z = Tensor::from_vec(z_data, l2.shape().clone(), &dev)?.to_dtype(out_dtype)?;
        let p = l2.dims()[1];

        let (_, stats, t) = linear_cg_tridiag(
            |v| self.op.matmul_detached(v),
            &z,
            None,
            NO_PRECOND,
            &tridiag_opts(&self.opts, p),
        )?;
        warn_unconverged("slq_logdet_lazy", &stats);
        let est = slq_from_tridiag(&t, &host_col_norms_sq(&z)?)?;
        Ok((storage_from_f64(vec![est], as_f32), Shape::from(())))
    }

    fn bwd(
        &self,
        packed: &Tensor,
        probes: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let z = probes.detach().contiguous()?;
        let p = z.dim(D::Minus1)? as f64;
        let (u, stats) = linear_cg(
            |v| self.op.matmul_detached(v),
            &z,
            None,
            NO_PRECOND,
            &solve_opts(&self.opts),
        )?;
        warn_unconverged("slq_logdet_lazy (adjoint)", &stats);
        let u = u.detach();
        let g: f64 = grad_res.to_dtype(DType::F64)?.to_scalar()?;

        // θ̄ via the nested tape: S(θ) = (ḡ/P) Σ_z u_zᵀ K(θ) z.
        let leaf_values = unpack_leaves(&packed.detach(), &self.shapes)?;
        let vars: Vec<candle_core::Var> = leaf_values
            .iter()
            .map(candle_core::Var::from_tensor)
            .collect::<Result<_>>()?;
        let leaf_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
        let tracked_op = self.op.with_grad_leaves(&leaf_tensors)?;

        let s = (u * tracked_op.matmul_tracked(&z)?)?
            .sum_all()?
            .affine(g / p, 0.0)?;
        let grads = s.backward()?;

        let mut flat_grads = Vec::with_capacity(vars.len());
        for (var, val) in vars.iter().zip(&leaf_values) {
            match grads.get(var.as_tensor()) {
                Some(gr) => flat_grads.push(gr.flatten_all()?),
                None => flat_grads.push(val.zeros_like()?.flatten_all()?),
            }
        }
        let packed_bar = Tensor::cat(&flat_grads, 0)?.to_dtype(packed.dtype())?;
        Ok((Some(packed_bar), None))
    }
}

#[cfg(test)]
mod tests {
    //! Verification of routed iterative linear solves with reusable operator interfaces.

    use super::*;
    use crate::operators::{DenseOperator, LinearOperator};
    use candle_core::{Device, Var};

    /// Dense SPD `A = BᵀB + shift·I` in f64, deterministic.
    fn spd(n: usize, shift: f64, seed: u64) -> Tensor {
        let dev = Device::Cpu;
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let b: Vec<f64> = (0..n * n).map(|_| next()).collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += b[k * n + i] * b[k * n + j];
                }
                a[i * n + j] = acc + if i == j { shift } else { 0.0 };
            }
        }
        Tensor::from_vec(a, (n, n), &dev).unwrap()
    }

    fn rhs(n: usize, c: usize) -> Tensor {
        Tensor::from_vec(
            (0..n * c).map(|i| ((i * 29 % 13) as f64) / 5.0 - 1.2).collect::<Vec<_>>(),
            (n, c),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn max_rel(x: &Tensor, y: &Tensor) -> f64 {
        let a: Vec<f64> = x.flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        let b: Vec<f64> = y.flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        let scale = b.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1e-12);
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f64, f64::max) / scale
    }

    fn tight() -> CgOptions {
        CgOptions { max_iter: Some(4000), ..CgOptions::for_tolerance(1e-12) }
    }

    /// Central finite difference — the canonical harness from linalg_diff.rs.
    fn finite_diff(
        param: &Var,
        loss: &dyn Fn(&Tensor) -> Result<f64>,
        h: f64,
    ) -> Result<Vec<f64>> {
        let base = param.as_tensor().flatten_all()?.to_vec1::<f64>()?;
        let dims = param.shape().dims().to_vec();
        let dev = param.device().clone();
        let mut grads = Vec::with_capacity(base.len());
        for i in 0..base.len() {
            let mut up = base.clone();
            up[i] += h;
            let mut dn = base.clone();
            dn[i] -= h;
            let lu = loss(&Tensor::from_vec(up, dims.clone(), &dev)?)?;
            let ld = loss(&Tensor::from_vec(dn, dims.clone(), &dev)?)?;
            grads.push((lu - ld) / (2.0 * h));
        }
        Ok(grads)
    }

    // ---- dense node ---------------------------------------------------------

    // ---- preconditioning ----------------------------------------------------

    /// `CgOptions.precond` must reach the solve, and must not move the
    /// answer: `K⁻¹b` and `bᵀK⁻¹b` are preconditioning-invariant, so `M⁻¹`
    /// buys convergence speed only. This is the test that says the transport
    /// field is wired without changing semantics.
    #[test]
    fn a_preconditioner_reaches_the_solve_without_moving_it() {
        let n = 30;
        let a = spd(n, 1.0, 21);
        let b = rhs(n, 2);
        let diag = crate::operators::dense_diagonal(&a).unwrap();
        let m = std::sync::Arc::new(
            crate::operators::precond::CiqPreconditioner::pivoted(|v| a.matmul(v), &diag, 6)
                .unwrap(),
        );

        let plain = tight();
        let pre = CgOptions { precond: Some(m), ..tight() };

        let x_plain = lin_solve_diff(&a, &b, &plain).unwrap();
        let x_pre = lin_solve_diff(&a, &b, &pre).unwrap();
        assert!(max_rel(&x_pre, &x_plain) < 1e-7, "preconditioned solve moved the answer");

        let q_plain = inv_quad_diff(&a, &b, &plain).unwrap();
        let q_pre = inv_quad_diff(&a, &b, &pre).unwrap();
        assert!(max_rel(&q_pre, &q_plain) < 1e-7, "preconditioned inv_quad moved the answer");
    }

    /// SLQ must still work with a preconditioner set, because `tridiag_opts`
    /// DROPS it — the Lanczos forward runs plain while the adjoint solve is
    /// preconditioned.
    ///
    /// If someone deletes that `precond: None`, `cg::run`'s refusal turns
    /// this into a hard error instead of an SLQ estimate silently short by
    /// `logdet(M)`. That is the whole point of the two-helper split, so the
    /// test asserts the estimate is UNCHANGED, not merely that it succeeds.
    #[test]
    fn slq_logdet_ignores_the_preconditioner_rather_than_failing_or_shifting() {
        let n = 24;
        let a = spd(n, 1.0, 33);
        // A COMPLETE probe basis: the SLQ estimator is then exact, so the
        // comparison below cannot pass by both sides sharing the same noise.
        let probes = rademacher_probes(n, n, 7, DType::F64, &Device::Cpu).unwrap();
        let diag = crate::operators::dense_diagonal(&a).unwrap();
        let m = std::sync::Arc::new(
            crate::operators::precond::CiqPreconditioner::pivoted(|v| a.matmul(v), &diag, 6)
                .unwrap(),
        );

        let plain = slq_logdet_diff(&a, &probes, &tight()).unwrap();
        let pre = slq_logdet_diff(&a, &probes, &CgOptions { precond: Some(m), ..tight() })
            .expect("SLQ must not fail merely because a preconditioner was supplied");

        let (p0, p1) = (
            plain.to_dtype(DType::F64).unwrap().to_scalar::<f64>().unwrap(),
            pre.to_dtype(DType::F64).unwrap().to_scalar::<f64>().unwrap(),
        );
        assert!(
            (p0 - p1).abs() < 1e-9 * p0.abs().max(1.0),
            "the preconditioner leaked into the tridiagonal: {p0} vs {p1} \
             (a shift of ~logdet(M) is exactly the silent bug this guards)"
        );
    }

    #[test]
    fn lin_solve_matches_the_dense_cholesky_solve() {
        let n = 14;
        let a = spd(n, 1.0, 5);
        let b = rhs(n, 3);
        let got = lin_solve_diff(&a, &b, &tight()).unwrap();
        let want = DenseOperator::new(a.clone()).cholesky_solve(&b, 0.0).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-8, "solve vs Cholesky: rel {rel:.3e}");
    }

    #[test]
    fn lin_solve_handles_batches_and_vectors() {
        let dev = Device::Cpu;
        let l = 3usize;
        let n = 8;
        let a = Tensor::stack(&(0..l).map(|i| spd(n, 1.5, 40 + i as u64)).collect::<Vec<_>>(), 0)
            .unwrap();
        let b = Tensor::from_vec(
            (0..l * n * 2).map(|i| ((i * 17 % 11) as f64) / 4.0 - 1.0).collect::<Vec<_>>(),
            (l, n, 2),
            &dev,
        )
        .unwrap();
        let got = lin_solve_diff(&a, &b, &tight()).unwrap();
        let want = DenseOperator::new(a.clone()).cholesky_solve(&b, 0.0).unwrap();
        assert!(max_rel(&got, &want) < 1e-8, "batched solve");

        // vector rhs round-trips the unsqueeze
        let v = rhs(n, 1).squeeze(D::Minus1).unwrap();
        let a1 = spd(n, 1.5, 9);
        let got_v = lin_solve_diff(&a1, &v, &tight()).unwrap();
        assert_eq!(got_v.dims(), &[n]);
        let back = a1.matmul(&got_v.unsqueeze(D::Minus1).unwrap()).unwrap();
        assert!(max_rel(&back.squeeze(D::Minus1).unwrap(), &v) < 1e-8, "A(A⁻¹v) ≠ v");
    }

    /// Pins the b̄ formula in isolation: the gradient w.r.t. `b` of
    /// `Σ (K⁻¹b) ⊙ W` is `K⁻¹W` — the adjoint solve of the cotangent.
    #[test]
    fn lin_solve_grad_b_is_the_adjoint_solve() {
        let n = 10;
        let a = spd(n, 1.2, 21);
        let rvar = Var::from_tensor(&rhs(n, 2)).unwrap();
        let w = rhs(n, 2).affine(0.7, 0.3).unwrap();

        let x = lin_solve_diff(&a, rvar.as_tensor(), &tight()).unwrap();
        let grads = x.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g_r = grads.get(rvar.as_tensor()).unwrap();

        let want = DenseOperator::new(a).cholesky_solve(&w, 0.0).unwrap();
        let rel = max_rel(g_r, &want);
        assert!(rel < 1e-8, "b̄ vs Cholesky solve of the cotangent: rel {rel:.3e}");
    }

    /// Gradcheck through the symmetric construction `K = BBᵀ + 2I` — the
    /// CholeskyOp route, which keeps FD perturbations inside the SPD cone
    /// and makes the symmetrised K̄ the right object automatically. Unlike
    /// the CIQ gradcheck there is no quadrature-parameter caveat here: at
    /// tolerance 1e-12 the solve is exact to FD precision.
    #[test]
    fn lin_solve_gradcheck_against_central_differences() {
        let dev = Device::Cpu;
        let (n, c) = (6, 2);
        let b0 = Tensor::from_vec(
            (0..n * n).map(|i| ((i * 31 % 17) as f64) / 6.0 - 1.3).collect::<Vec<_>>(),
            (n, n),
            &dev,
        )
        .unwrap();
        let bvar = Var::from_tensor(&b0).unwrap();
        let rvar = Var::from_tensor(&rhs(n, c)).unwrap();
        let w = rhs(n, c).affine(-0.4, 0.9).unwrap();

        let make_k = |bt: &Tensor| -> Result<Tensor> {
            bt.matmul(&bt.t()?)?
                .add(&Tensor::eye(n, DType::F64, bt.device())?.affine(2.0, 0.0)?)
        };
        let loss_of = |bt: &Tensor, rt: &Tensor| -> Result<f64> {
            let x = lin_solve_diff(&make_k(bt)?, rt, &tight())?;
            x.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let x = lin_solve_diff(&make_k(bvar.as_tensor()).unwrap(), rvar.as_tensor(), &tight())
            .unwrap();
        let grads = x.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g_b =
            grads.get(bvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_r =
            grads.get(rvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_b = finite_diff(&bvar, &|bt| loss_of(bt, rvar.as_tensor()), 1e-5).unwrap();
        let fd_r = finite_diff(&rvar, &|rt| loss_of(bvar.as_tensor(), rt), 1e-5).unwrap();

        for (i, (ga, gf)) in g_b.iter().zip(&fd_b).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "K-side grad[{i}]: {ga} vs FD {gf}");
        }
        for (i, (ga, gf)) in g_r.iter().zip(&fd_r).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "rhs grad[{i}]: {ga} vs FD {gf}");
        }
    }

    #[test]
    fn inv_quad_diff_matches_the_trait_inv_quad_and_carries_gradient() {
        let n = 9;
        let a = spd(n, 1.0, 3);
        let kvar = Var::from_tensor(&a).unwrap();
        let b = rhs(n, 2);

        let q = inv_quad_diff(kvar.as_tensor(), &b, &tight()).unwrap();
        let want = DenseOperator::new(a).inv_quad(&b, 0.0).unwrap();
        assert!(max_rel(&q, &want) < 1e-8, "inv_quad value");

        // K̄ = −x xᵀ per column, summed — pinned against the closed form.
        let grads = q.sum_all().unwrap().backward().unwrap();
        let g_k = grads.get(kvar.as_tensor()).unwrap();
        let x = lin_solve_diff(kvar.as_tensor(), &b, &tight()).unwrap();
        let want_k = x
            .matmul(&x.t().unwrap())
            .unwrap()
            .affine(-1.0, 0.0)
            .unwrap();
        assert!(max_rel(g_k, &want_k) < 1e-7, "inv_quad K̄ vs −xxᵀ");
    }

    // ---- lazy node ----------------------------------------------------------

    /// Symbolic RBF + δI — same fixture as the lazy-CIQ tests: TWO learnable
    /// leaves behind three leaf positions (x sits behind both vi and vj).
    fn rbf_symbolic(
        x: &Tensor,
        gamma: &Tensor,
        delta: f64,
    ) -> crate::operators::SymbolicKernelOperator {
        use crate::operators::symbolic::LazyTensor;
        let xi = LazyTensor::vi(x);
        let xj = LazyTensor::vj(x);
        let diff = &xi - &xj;
        let d2 = (&diff * &diff).sum_last();
        let k = (&d2 * &LazyTensor::pm(gamma)).affine(-1.0, 0.0).exp();
        let k = &k + &LazyTensor::eye_identity(delta);
        crate::operators::SymbolicKernelOperator::new(k)
    }

    fn points(n: usize, d: usize) -> Tensor {
        Tensor::from_vec(
            (0..n * d).map(|i| ((i * 37 % 23) as f64) / 6.0 - 1.5).collect::<Vec<_>>(),
            (n, d),
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn lazy_lin_solve_matches_the_dense_driver() {
        let (n, dim) = (12, 2);
        let x = points(n, dim);
        let gamma = Tensor::full(0.8f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let b = rhs(n, 2);

        let dense = op.to_dense().unwrap();
        let want = lin_solve_diff(&dense, &b, &tight()).unwrap();
        let got = lin_solve_lazy(Arc::new(op), &b, &tight()).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-8, "lazy vs dense solve: rel {rel:.3e}");
    }

    /// Parameter gradients through K⁻¹ with K never densified — FD-checked
    /// for the hyperparameter AND the inputs (x behind two leaf positions
    /// pins the cat-VJP summing).
    #[test]
    fn lazy_lin_solve_gradcheck_against_central_differences() {
        let (n, dim) = (8, 2);
        let xvar = Var::from_tensor(&points(n, dim)).unwrap();
        let gvar = Var::from_tensor(&Tensor::full(0.6f64, (1, 1), &Device::Cpu).unwrap()).unwrap();
        let b = rhs(n, 2);
        let w = rhs(n, 2).affine(0.5, -0.2).unwrap();

        let loss_of = |xt: &Tensor, gt: &Tensor| -> Result<f64> {
            let op = rbf_symbolic(xt, gt, 0.5);
            let y = lin_solve_lazy(Arc::new(op), &b, &tight())?;
            y.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let op = rbf_symbolic(xvar.as_tensor(), gvar.as_tensor(), 0.5);
        let y = lin_solve_lazy(Arc::new(op), &b, &tight()).unwrap();
        let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g_x =
            grads.get(xvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_g =
            grads.get(gvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_x = finite_diff(&xvar, &|t| loss_of(t, gvar.as_tensor()), 1e-5).unwrap();
        let fd_g = finite_diff(&gvar, &|t| loss_of(xvar.as_tensor(), t), 1e-5).unwrap();

        for (i, (ga, gf)) in g_x.iter().zip(&fd_x).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "x grad[{i}]: {ga} vs FD {gf}");
        }
        for (i, (ga, gf)) in g_g.iter().zip(&fd_g).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "γ grad[{i}]: {ga} vs FD {gf}");
        }
    }

    #[test]
    fn lazy_rhs_gradient_is_the_adjoint_solve() {
        let (n, dim) = (10, 2);
        let x = points(n, dim);
        let gamma = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let dense = op.to_dense().unwrap();

        let rvar = Var::from_tensor(&rhs(n, 2)).unwrap();
        let w = rhs(n, 2).affine(-0.6, 0.1).unwrap();
        let y = lin_solve_lazy(Arc::new(op), rvar.as_tensor(), &tight()).unwrap();
        let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g_r = grads.get(rvar.as_tensor()).unwrap();
        let want = DenseOperator::new(dense).cholesky_solve(&w, 0.0).unwrap();
        let rel = max_rel(g_r, &want);
        assert!(rel < 1e-7, "lazy rhs̄ vs adjoint solve: rel {rel:.3e}");
    }

    // ---- SLQ log-determinant ------------------------------------------------

    /// `√N · I` probe columns: `E[zzᵀ] = I` holds EXACTLY (not just in
    /// expectation over draws), the Hutchinson sum telescopes to the exact
    /// trace, and full-grade Lanczos makes each per-probe quadrature exact
    /// — so this pins SLQ against the dense log-determinant with a
    /// DETERMINISTIC oracle, no statistics involved.
    fn complete_probes(n: usize) -> Tensor {
        Tensor::eye(n, DType::F64, &Device::Cpu)
            .unwrap()
            .affine((n as f64).sqrt(), 0.0)
            .unwrap()
    }

    fn slq_opts(n: usize) -> CgOptions {
        CgOptions {
            max_iter: Some(4000),
            max_tridiag_iter: Some(n),
            ..CgOptions::for_tolerance(1e-12)
        }
    }

    #[test]
    fn slq_logdet_with_complete_probes_matches_the_dense_logdet() {
        let n = 8;
        let a = spd(n, 2.0, 11);
        let got = slq_logdet_diff(&a, &complete_probes(n), &slq_opts(n)).unwrap();
        let want = DenseOperator::new(a).log_abs_det(0.0).unwrap();
        let g: f64 = got.to_scalar().unwrap();
        let w: f64 = want.to_scalar().unwrap();
        assert!((g - w).abs() < 1e-6 * (1.0 + w.abs()), "SLQ {g} vs dense {w}");
    }

    /// With complete probes the trace-estimator gradient collapses to the
    /// EXACT `ḡ·K⁻¹` — pinned against the explicit inverse.
    #[test]
    fn slq_logdet_grad_with_complete_probes_is_k_inverse() {
        let n = 7;
        let a = spd(n, 2.0, 13);
        let kvar = Var::from_tensor(&a).unwrap();
        let ld = slq_logdet_diff(kvar.as_tensor(), &complete_probes(n), &slq_opts(n)).unwrap();
        let grads = ld.backward().unwrap();
        let g_k = grads.get(kvar.as_tensor()).unwrap();

        let eye = Tensor::eye(n, DType::F64, &Device::Cpu).unwrap();
        let kinv = DenseOperator::new(a).cholesky_solve(&eye, 0.0).unwrap();
        let rel = max_rel(g_k, &kinv);
        assert!(rel < 1e-6, "SLQ K̄ vs K⁻¹: rel {rel:.3e}");
    }

    /// End-to-end FD gradcheck through the symmetric construction, complete
    /// probes: value exact, gradient exact, so FD must agree tightly.
    #[test]
    fn slq_logdet_gradcheck_against_central_differences() {
        let dev = Device::Cpu;
        let n = 5;
        let b0 = Tensor::from_vec(
            (0..n * n).map(|i| ((i * 41 % 19) as f64) / 7.0 - 1.2).collect::<Vec<_>>(),
            (n, n),
            &dev,
        )
        .unwrap();
        let bvar = Var::from_tensor(&b0).unwrap();
        let make_k = |bt: &Tensor| -> Result<Tensor> {
            bt.matmul(&bt.t()?)?
                .add(&Tensor::eye(n, DType::F64, bt.device())?.affine(2.0, 0.0)?)
        };
        let loss_of = |bt: &Tensor| -> Result<f64> {
            slq_logdet_diff(&make_k(bt)?, &complete_probes(n), &slq_opts(n))?.to_scalar::<f64>()
        };

        let ld = slq_logdet_diff(&make_k(bvar.as_tensor()).unwrap(), &complete_probes(n), &slq_opts(n))
            .unwrap();
        let grads = ld.backward().unwrap();
        let g_b =
            grads.get(bvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let fd_b = finite_diff(&bvar, &loss_of, 1e-5).unwrap();
        for (i, (ga, gf)) in g_b.iter().zip(&fd_b).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "logdet grad[{i}]: {ga} vs FD {gf}");
        }
    }

    /// Rademacher probes are the production route: seeded, deterministic,
    /// statistically unbiased. Bound the spread rather than pretend
    /// exactness.
    #[test]
    fn slq_logdet_rademacher_probes_land_near_the_dense_logdet() {
        let n = 24;
        let a = spd(n, 3.0, 17);
        let want: f64 = DenseOperator::new(a.clone()).log_abs_det(0.0).unwrap().to_scalar().unwrap();
        let z = rademacher_probes(n, 64, 0xC0FFEE, DType::F64, &Device::Cpu).unwrap();
        let got: f64 = slq_logdet_diff(&a, &z, &slq_opts(20)).unwrap().to_scalar().unwrap();
        assert!(
            (got - want).abs() < 0.15 * want.abs().max(1.0),
            "SLQ Rademacher {got} too far from dense {want}"
        );
    }

    #[test]
    fn slq_logdet_handles_batches() {
        let l = 2usize;
        let n = 6;
        let mats: Vec<Tensor> = (0..l).map(|i| spd(n, 2.0, 60 + i as u64)).collect();
        let a = Tensor::stack(&mats, 0).unwrap();
        let z = complete_probes(n).unsqueeze(0).unwrap().expand((l, n, n)).unwrap().contiguous().unwrap();
        let got = slq_logdet_diff(&a, &z, &slq_opts(n)).unwrap();
        assert_eq!(got.dims(), &[l]);
        for (i, m) in mats.iter().enumerate() {
            let w: f64 =
                DenseOperator::new(m.clone()).log_abs_det(0.0).unwrap().to_scalar().unwrap();
            let g: f64 = got.get(i).unwrap().to_scalar().unwrap();
            assert!((g - w).abs() < 1e-6 * (1.0 + w.abs()), "batch {i}: {g} vs {w}");
        }
    }

    #[test]
    fn an_indefinite_operator_is_refused_by_name() {
        let n = 6;
        let dev = Device::Cpu;
        // diag(3, 3, 3, 3, 3, −1): frankly indefinite.
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            d[i * n + i] = if i == n - 1 { -1.0 } else { 3.0 };
        }
        let a = Tensor::from_vec(d, (n, n), &dev).unwrap();
        let e = slq_logdet_diff(&a, &complete_probes(n), &slq_opts(n)).unwrap_err();
        assert!(e.to_string().contains("SPD"), "{e}");
    }

    // ---- lazy SLQ -----------------------------------------------------------

    #[test]
    fn lazy_slq_matches_dense_on_the_same_probes() {
        let (n, dim) = (10, 2);
        let x = points(n, dim);
        let gamma = Tensor::full(0.8f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let dense = op.to_dense().unwrap();
        let z = complete_probes(n);

        let want: f64 =
            slq_logdet_diff(&dense, &z, &slq_opts(n)).unwrap().to_scalar().unwrap();
        let got: f64 =
            slq_logdet_lazy(Arc::new(op), &z, &slq_opts(n)).unwrap().to_scalar().unwrap();
        assert!((got - want).abs() < 1e-8 * (1.0 + want.abs()), "lazy {got} vs dense {want}");
    }

    /// Leaf gradients of the lazy log-determinant, FD-checked with complete
    /// probes (estimator exact ⇒ FD applicable).
    #[test]
    fn lazy_slq_gradcheck_against_central_differences() {
        let (n, dim) = (6, 2);
        let xvar = Var::from_tensor(&points(n, dim)).unwrap();
        let gvar = Var::from_tensor(&Tensor::full(0.6f64, (1, 1), &Device::Cpu).unwrap()).unwrap();
        let z = complete_probes(n);

        let loss_of = |xt: &Tensor, gt: &Tensor| -> Result<f64> {
            slq_logdet_lazy(Arc::new(rbf_symbolic(xt, gt, 0.5)), &z, &slq_opts(n))?
                .to_scalar::<f64>()
        };

        let op = rbf_symbolic(xvar.as_tensor(), gvar.as_tensor(), 0.5);
        let ld = slq_logdet_lazy(Arc::new(op), &z, &slq_opts(n)).unwrap();
        let grads = ld.backward().unwrap();
        let g_x =
            grads.get(xvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_g =
            grads.get(gvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_x = finite_diff(&xvar, &|t| loss_of(t, gvar.as_tensor()), 1e-5).unwrap();
        let fd_g = finite_diff(&gvar, &|t| loss_of(xvar.as_tensor(), t), 1e-5).unwrap();
        for (i, (ga, gf)) in g_x.iter().zip(&fd_x).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "x grad[{i}]: {ga} vs FD {gf}");
        }
        for (i, (ga, gf)) in g_g.iter().zip(&fd_g).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "γ grad[{i}]: {ga} vs FD {gf}");
        }
    }
    // ---- preconditioned SLQ -------------------------------------------------

    /// Exact `logdet M` for every kind, against a densely-materialised `M`.
    ///
    /// `M` is recovered as `E(Ex)` column by column, so this checks the closed
    /// form against the applies rather than against its own algebra.
    #[test]
    fn log_det_matches_the_dense_determinant_for_every_kind() {
        use crate::operators::precond::CiqPreconditioner;
        let dev = Device::Cpu;
        let n = 12;
        let a = spd(n, 0.5, 31);
        let d = crate::operators::dense_diagonal(&a).unwrap();

        let (q, mu, delta) = {
            let probe = crate::operators::ciq::lanczos_probe(
                |v: &Tensor| a.matmul(v), n, DType::F64, &dev, Some(6),
            )
            .unwrap();
            probe.spectral_parts(0).unwrap()
        };

        for (name, p) in [
            ("jacobi", CiqPreconditioner::jacobi(&d, 1e-12).unwrap()),
            ("pivoted", CiqPreconditioner::pivoted(|v| a.matmul(v), &d, 5).unwrap()),
            ("spectral", CiqPreconditioner::spectral(&q, &mu, delta, 1e-8).unwrap()),
        ] {
            // M, column by column: M e_i = E(E e_i).
            let eye = Tensor::eye(n, DType::F64, &dev).unwrap();
            let m = p.sqrt_apply(&p.sqrt_apply(&eye).unwrap()).unwrap();
            let mv: Vec<f64> = m.flatten_all().unwrap().to_vec1().unwrap();
            let fro = mv.iter().map(|v| v * v).sum::<f64>().sqrt();
            let scaled: Vec<f64> = mv.iter().map(|v| v / fro).collect();
            let (eig, _) = jacobi_eigh(&scaled, n);
            let want: f64 = eig.iter().map(|e| (e * fro).ln()).sum();

            let got = p.log_det().unwrap().to_scalar::<f64>().unwrap();
            assert!(
                (got - want).abs() < 1e-8 * want.abs().max(1.0),
                "{name}: log_det {got} vs dense {want}"
            );
        }
    }

    /// **The capability, and the mutation that proves the correction is live.**
    ///
    /// A preconditioned SLQ is right only if BOTH corrections are applied: the
    /// probes transformed so `E[wwᵀ] = I` still holds for the Lanczos start
    /// vector, and `logdet(M)` added back. Omitting the second shifts the
    /// answer by a large constant — no NaN, no error, just a worse model — so
    /// this test measures the shift explicitly rather than trusting the sum.
    #[test]
    fn preconditioned_slq_matches_the_exact_logdet_and_needs_both_corrections() {
        use crate::operators::precond::CiqPreconditioner;
        let dev = Device::Cpu;
        let n = 24;
        let a = spd(n, 0.5, 77);

        // Exact logdet by eigendecomposition.
        let av: Vec<f64> = a.flatten_all().unwrap().to_vec1().unwrap();
        let fro = av.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scaled: Vec<f64> = av.iter().map(|v| v / fro).collect();
        let (eig, _) = jacobi_eigh(&scaled, n);
        let exact: f64 = eig.iter().map(|e| (e * fro).ln()).sum();

        // A COMPLETE probe basis makes Hutchinson exact, so the only error
        // left is the Lanczos quadrature's — otherwise a 10-probe stochastic
        // band would swamp the effect being measured. `complete_probes` is
        // `√n·I`, which is the scaling the estimator's `/P` normalisation
        // expects (it is written for Rademacher probes, where `‖z‖² = n`).
        let probes = complete_probes(n);
        let opts = slq_opts(n);

        let plain = slq_logdet_diff(&a, &probes, &opts)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        let d = crate::operators::dense_diagonal(&a).unwrap();
        let pre = std::sync::Arc::new(
            CiqPreconditioner::pivoted(|v: &Tensor| a.matmul(v), &d, 6).unwrap(),
        );
        let got = slq_logdet_preconditioned(|v: &Tensor| a.matmul(v), &pre, &probes, &opts).unwrap();

        let log_det_m = pre.log_det().unwrap().to_scalar::<f64>().unwrap();
        println!(
            "exact {exact:.6}  plain-slq {plain:.6}  precond-slq {got:.6}  logdet(M) {log_det_m:.6}"
        );

        assert!(
            (plain - exact).abs() < 1e-4 * exact.abs().max(1.0),
            "the unpreconditioned reference is already off: {plain} vs {exact}"
        );
        assert!(
            (got - exact).abs() < 1e-4 * exact.abs().max(1.0),
            "preconditioned SLQ {got} vs exact {exact}"
        );
        // The mutation: drop `logdet(M)` and the answer must move by far more
        // than the tolerance. If this ever fails, the correction is a no-op and
        // the test above is passing for the wrong reason.
        let without = got - log_det_m;
        assert!(
            (without - exact).abs() > 1e-2 * exact.abs().max(1.0),
            "omitting logdet(M) barely moved the answer ({without} vs {exact}) — the \
             correction is not doing anything and this test proves nothing"
        );
    }

    /// The refusal still fires by default, so an ACCIDENTAL preconditioned
    /// tridiagonal is still a hard error — the opt-in must not weaken it.
    #[test]
    fn a_preconditioned_tridiagonal_is_still_refused_without_the_opt_in() {
        use crate::operators::precond::CiqPreconditioner;
        let n = 10;
        let a = spd(n, 0.5, 3);
        let d = crate::operators::dense_diagonal(&a).unwrap();
        let pre = std::sync::Arc::new(CiqPreconditioner::jacobi(&d, 1e-12).unwrap());
        let probes = complete_probes(n);
        let opts = CgOptions { n_tridiag: n, precond: Some(pre), ..slq_opts(n) };
        let e = linear_cg_tridiag(
            |v: &Tensor| a.matmul(v),
            &probes,
            None,
            NO_PRECOND,
            &opts,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("logdet(M)"), "the refusal must still name the cause: {e}");
        assert!(e.contains("allow_preconditioned_tridiag"), "…and the way out: {e}");
    }
}
