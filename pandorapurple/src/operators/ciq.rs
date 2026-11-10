//! Inferno — Contour-integral quadrature for matrix functions and implicit tensor derivatives.

use candle_core::{DType, Device, Result, Tensor};

use std::sync::Arc;

use super::cg::{linear_cg_shifted, CgOptions, ShiftedCgStats};
use super::minres::linear_minres_shifted;
use super::precond::CiqPreconditioner;
use crate::quadrature::{contour_terms, ContourRule};
use crate::settings;

/// Which multi-shift Krylov solver runs the contour's shifted systems.
///
/// Every contour shift keeps `K + shift·I` SPD, so both are valid here and
/// agree to tolerance (pinned by test). msCG carries 2 N-vectors per shift
/// against msMINRES's 3 and stays the default; msMINRES is the
/// gpytorch-faithful route and the one to reach for if a barely-indefinite
/// seed ever needs to be solved rather than refused.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CiqSolver {
    #[default]
    MsCg,
    MsMinres,
}

/// The one dispatch point for the contour's shifted solves — the driver and
/// `CiqInvSqrtOp::bwd` both come through here, so forward and backward can
/// never disagree on the solver.
fn solve_shifted<F>(
    solver: CiqSolver,
    matmul: F,
    rhs: &Tensor,
    shifts: &[f64],
    opts: &CgOptions,
) -> Result<(Vec<Tensor>, ShiftedCgStats)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match solver {
        CiqSolver::MsCg => linear_cg_shifted(matmul, rhs, shifts, opts),
        CiqSolver::MsMinres => linear_minres_shifted(matmul, rhs, shifts, opts),
    }
}

/// What the CALLER needs from `K^{±1/2}` — which decides whether a
/// preconditioner is sound.
///
/// Not a `bool`. A bare `true` at a call site is invisible in review, and the
/// wrong answer here is silent: a preconditioned CIQ returns the CONJUGATED
/// root `R = E·K̂^{1/2}`, which satisfies `RRᵀ = K` but is a different matrix
/// from the symmetric root.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RootUse {
    /// The downstream result is invariant under `R → RQ` for orthogonal `Q` —
    /// sampling (`Rε` is distributionally identical), quadratic forms,
    /// `‖R⁻¹b‖²`. Preconditioning is SOUND here and the gradient is EXACT,
    /// not approximate: freezing `E` is what FIXES which smooth root family in
    /// `θ` is being differentiated, rather than approximating one.
    AnyRoot,
    /// The caller needs `K^{±1/2}`, the symmetric matrix function itself.
    /// A preconditioner is REFUSED: the conjugated root is a different matrix
    /// and its `E`-dependence is not differentiated. `svgp.rs` whitening is
    /// the recorded example of a consumer in this class.
    SymmetricRoot,
}

/// Everything optional about a CIQ application. `None` fields read
/// [`crate::settings`] at call time, same convention as [`CgOptions`].
#[derive(Clone, Debug)]
pub struct CiqOptions {
    /// Quadrature nodes. `None` ⇒ [`settings::num_contour_quadrature`].
    pub q: Option<usize>,
    /// Which rule places the nodes. Defaults to Elliptic — measured 3.4e-6
    /// worst relative error at κ=1e8 with the default Q=15, where GL at the
    /// same budget is ~3e-2 (pinned by
    /// `elliptic_beats_gauss_legendre_node_for_node`).
    pub rule: ContourRule,
    /// Relative solve tolerance. `None` ⇒ [`settings::minres_tolerance`] —
    /// provisioned for gpytorch's msMINRES; our msCG plays the same role.
    pub tolerance: Option<f64>,
    /// Solver iteration budget. `None` ⇒ [`settings::max_cg_iterations`].
    pub max_iter: Option<usize>,
    /// Lanczos iterations for the spectrum-bound probe. `None` ⇒
    /// [`settings::max_lanczos_quadrature_iterations`].
    pub lanczos_iters: Option<usize>,
    /// Which multi-shift solver runs the shifted systems.
    pub solver: CiqSolver,
        /// Shift-invariant preconditioner (see [`super::precond`]). When set,
    /// the WHOLE pipeline — bounds probe and every shifted solve — runs on
    /// the conjugated `K̂ = E⁻¹KE⁻¹`, and the result is the CONJUGATED root
    /// apply `R^{∓1}·rhs` with `R = E·K̂^{1/2}` (so `RRᵀ = K`), not the
    /// symmetric root. Fine for every root-contract consumer (sampling,
    /// whitening, quadratic forms); wrong if you specifically need
    /// `K^{±1/2}` the symmetric matrix function — leave it `None` there.
    /// Detached driver only: [`ciq_inv_sqrt_diff`] refuses it.
    pub precond: Option<Arc<CiqPreconditioner>>,
    /// Spectrum bounds for the operator the solve ACTUALLY RUNS ON, skipping
    /// the Lanczos probe.
    ///
    /// Read that qualification carefully: with a preconditioner set the
    /// pipeline runs on the conjugated `K̂ = E⁻¹KE⁻¹`, so these must bracket
    /// `K̂`'s spectrum, not `K`'s.
    ///
    /// This exists because the probe is otherwise UNCONDITIONAL
    /// ([`inv_sqrt_core`]), which is what makes a Ritz-basis preconditioner
    /// cost *more* than a pivoted-Cholesky one: it would pay 20 matvecs for the
    /// basis and then 20 conjugated ones anyway. A too-narrow bracket does not
    /// error — it silently returns a slightly wrong root — so supply a bound
    /// you can prove, not one you hope for.
    pub bounds: Option<(f64, f64)>,
}

impl Default for CiqOptions {
    fn default() -> Self {
        Self {
            q: None,
            rule: ContourRule::Elliptic,
            tolerance: None,
            max_iter: None,
            lanczos_iters: None,
            solver: CiqSolver::MsCg,
            precond: None,
            bounds: None,
        }
    }
}

/// Estimated `(λmin_hat, λmax_hat)` bracketing the spectrum of the SPD
/// operator behind `matmul`, from one Lanczos probe.
///
/// Mechanism: a DETERMINISTIC Rademacher probe (hard-coded LCG, not
/// `randn` — determinism is load-bearing for the finite-difference gradient
/// test of the differentiable wrapper, which needs `y(K)` to be a function),
/// run through a direct Lanczos three-term recurrence with full
/// reorthogonalisation, then a host-side Frobenius-pre-scaled `jacobi_eigh`
/// whose min/max are folded without sorting (its output is unsorted by
/// contract).
///
/// Why not `linear_cg_tridiag`'s CG-derived tridiagonal: CG's `safe_div`
/// zeroes any step whose curvature `pᵀAp` is negative-or-tiny, so an
/// indefinite operator never produces a negative Ritz value — it produces a
/// silently corrupted `T` instead (observed: Ritz max 3× the true λmax).
/// This function's whole job includes REFUSING indefinite operators, so it
/// runs the plain recurrence, which represents negative curvature honestly.
///
/// Safeguards, from Ritz interlacing (`ritz_min ≥ λmin`, `ritz_max ≤ λmax`):
/// * `λmax_hat = ritz_max × 1.1` — Ritz UNDERestimates the top, expand.
/// * `λmin_hat = max(ritz_min × 0.25, tridiagonal_jitter × λmax_hat)` — Ritz
///   OVERestimates the bottom; shrinking merely widens the covered interval
///   (the safe direction), while overestimating λmin loses quadrature
///   accuracy where it matters.
/// * `ritz_min` far below zero ⇒ `Err` naming the estimate, mirroring
///   `psd_safe_cholesky`'s report-don't-hide contract.
///
/// Batched operators: the recurrence itself is BATCHED — one shared
/// deterministic probe, per-element Lanczos coefficients, per-element
/// tridiagonals and safeguards — and this function returns the ENVELOPE
/// (min of per-element λmin_hats, max of per-element λmax_hats), so one
/// shared contour covers EVERY batch element's spectrum. Per-element bounds
/// are available from [`spectrum_bounds_lanczos_per_batch`]; per-element
/// CONTOURS (tensor-valued shifts in the multi-shift solvers) remain
/// deferred — the elliptic rule's `O(log κ)` node count makes the envelope
/// cheap even when batch spectra differ by orders of magnitude.
pub fn spectrum_bounds_lanczos<F>(
    matmul: F,
    n: usize,
    dtype: DType,
    device: &Device,
    iters: Option<usize>,
) -> Result<(f64, f64)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let per = spectrum_bounds_lanczos_per_batch(matmul, n, dtype, device, iters)?;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (a, b) in per {
        lo = lo.min(a);
        hi = hi.max(b);
    }
    Ok((lo, hi))
}

/// As [`spectrum_bounds_lanczos`], but one `(λmin_hat, λmax_hat)` PER BATCH
/// ELEMENT, in row-major batch order (an unbatched operator yields one
/// entry).
///
/// One shared probe drives a batched three-term recurrence: every
/// coefficient (`α`, `β`, the reorthogonalisation dots) is a per-element
/// `[.., 1, 1]` tensor, so elements with different spectra build different
/// tridiagonals from the same matvecs. Breakdown (invariant subspace found)
/// is tracked per element: a frozen element's Lanczos vector is zeroed and
/// its `T` stops growing, while the recurrence keeps running for the rest.
/// Everything the bounds probe already computes, instead of only the two
/// numbers it currently returns.
///
/// The Lanczos recurrence below builds an orthonormal basis (twice-repeated
/// DGKS, so it is genuinely orthonormal rather than nominally) and
/// eigendecomposes the tridiagonal — then throws the basis, the eigenvectors
/// and the quadrature weights away. A spectral preconditioner wants all three,
/// and they cost nothing extra.
///
/// Three shapes to know about, all consequences of the per-element freeze:
///
/// * `kept[b] ≤ k_max` — an element whose recurrence hit an invariant subspace
///   stops early, so the number of Ritz pairs DIFFERS across the batch.
/// * `q` and `ritz` are therefore **zero-padded** past `kept[b]`.
/// * `weights` is zero in the padded slots, so a weighted sum over `k_max`
///   silently does the right thing without per-element bookkeeping.
#[derive(Clone)]
pub struct LanczosProbe {
    /// `[..B, n, k_max]` — Ritz vectors as columns, orthonormal, zero-padded.
    pub q: Tensor,
    /// `[..B, k_max]` — Ritz values ASCENDING, zero-padded.
    pub ritz: Tensor,
    /// `[..B, k_max]` — `(s_j)₁²`, the squared first component of each
    /// tridiagonal eigenvector. These are the Gauss-quadrature weights that
    /// make `Σ_j w_j f(μ_j)` the SLQ estimate of `tr f(K)/n`; without them a
    /// mean over Ritz values is biased, because Lanczos clusters them at the
    /// extremes.
    pub weights: Tensor,
    /// Ritz pairs actually resolved per batch element.
    pub kept: Vec<usize>,
    /// Byte-identical to what [`spectrum_bounds_lanczos_per_batch`] returns.
    pub bounds: Vec<(f64, f64)>,
}

impl LanczosProbe {
    /// The `(Q, μ, δ)` a spectral preconditioner needs, for ONE batch element.
    ///
    /// `δ` is the **SLQ-weighted geometric mean**
    /// `exp(Σ_j w_j ln μ_j / Σ_j w_j)` — the Gauss-quadrature estimate of
    /// `exp(tr ln K / n)`, i.e. the geometric mean of the WHOLE spectrum, which
    /// is the right point estimate for an eigenvalue you have no information
    /// about. The weights matter: an unweighted mean of Ritz values is biased,
    /// because Lanczos clusters its Ritz values at the extremes and the weights
    /// are precisely the correction for that.
    ///
    /// Geometric rather than arithmetic because `δ` is a SCALE. The unresolved
    /// bulk contributes `max(λ/δ, δ/λ)` to the conditioning, which the
    /// geometric mean minimises; `pivoted`'s arithmetic mean of the residual
    /// diagonal systematically over-weights the large unresolved modes and so
    /// leaves the small ones worse-conditioned than they need to be.
    ///
    /// Batched operators are refused rather than silently reduced: `δ` is a
    /// scalar in `Kind::Spectral`, so one preconditioner cannot serve a batch
    /// whose elements have different spectra.
    pub fn spectral_parts(&self, elem: usize) -> Result<(Tensor, Tensor, f64)> {
        self.spectral_parts_with(elem, super::ladder::SpectralLadder::Ritz)
    }

    /// As [`Self::spectral_parts`], with the eigenvalue ladder chosen — see
    /// [`super::ladder`] for what a two-parameter shape can and cannot buy.
    ///
    /// `δ` is computed from the MEASURED Ritz values regardless of the ladder:
    /// it estimates the unresolved bulk, which the shape says nothing about.
    pub fn spectral_parts_with(
        &self,
        elem: usize,
        ladder: super::ladder::SpectralLadder,
    ) -> Result<(Tensor, Tensor, f64)> {
        let rank = self.q.rank();
        let (q, ritz, w) = match rank {
            2 => {
                if elem != 0 {
                    return Err(candle_core::Error::Msg(
                        "LanczosProbe::spectral_parts: unbatched probe has only element 0".into(),
                    ));
                }
                (self.q.clone(), self.ritz.clone(), self.weights.clone())
            }
            3 => (
                self.q.narrow(0, elem, 1)?.squeeze(0)?,
                self.ritz.narrow(0, elem, 1)?.squeeze(0)?,
                self.weights.narrow(0, elem, 1)?.squeeze(0)?,
            ),
            r => {
                return Err(candle_core::Error::Msg(format!(
                    "LanczosProbe::spectral_parts: unsupported batch rank {r}; a scalar δ cannot \
                     serve elements with different spectra"
                )))
            }
        };
        let k = *self.kept.get(elem).ok_or_else(|| {
            candle_core::Error::Msg(format!("LanczosProbe::spectral_parts: no element {elem}"))
        })?;
        // Trim the zero padding — a padded μ = 0 would trip the constructor's
        // positivity guard, which is the guard doing its job on junk.
        let q = q.narrow(candle_core::D::Minus1, 0, k)?.contiguous()?;
        let mu = ritz.narrow(candle_core::D::Minus1, 0, k)?.contiguous()?;

        let mu_h: Vec<f64> = mu.to_dtype(DType::F64)?.to_vec1()?;
        let w_h: Vec<f64> =
            w.narrow(candle_core::D::Minus1, 0, k)?.to_dtype(DType::F64)?.to_vec1()?;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (m, wt) in mu_h.iter().zip(&w_h) {
            if *m > 0.0 && *wt > 0.0 {
                num += wt * m.ln();
                den += wt;
            }
        }
        if den <= 0.0 {
            return Err(candle_core::Error::Msg(
                "LanczosProbe::spectral_parts: no positively-weighted Ritz value — the operator \
                 does not look positive definite"
                    .into(),
            ));
        }
        let delta = (num / den).exp();

        // The ladder replaces the VALUES on those directions, never the basis
        // and never δ. `resolve` returns `mu` unchanged for `Ritz`, so this is
        // a no-op on the default path.
        let (lo, hi) = self.bounds[elem];
        let shaped = super::ladder::resolve(ladder, &mu_h, lo, hi);
        let mu = if shaped == mu_h {
            mu
        } else {
            Tensor::from_vec(shaped, mu.shape().clone(), mu.device())?.to_dtype(mu.dtype())?
        };
        Ok((q, mu, delta))
    }
}

/// The bounds only — the pre-existing entry point, now a projection of
/// [`lanczos_probe`] so the two can never disagree.
pub fn spectrum_bounds_lanczos_per_batch<F>(
    matmul: F,
    n: usize,
    dtype: DType,
    device: &Device,
    iters: Option<usize>,
) -> Result<Vec<(f64, f64)>>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    Ok(lanczos_probe(matmul, n, dtype, device, iters)?.bounds)
}

pub fn lanczos_probe<F>(
    matmul: F,
    n: usize,
    dtype: DType,
    device: &Device,
    iters: Option<usize>,
) -> Result<LanczosProbe>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let k = iters
        .unwrap_or_else(settings::max_lanczos_quadrature_iterations::value)
        .min(n)
        .max(2);

    // Deterministic Rademacher probe, unit-normalised.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let probe: Vec<f64> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (s >> 63) == 1 { 1.0 } else { -1.0 }
        })
        .collect();
    let probe = Tensor::from_vec(probe, (n, 1), device)?
        .to_dtype(dtype)?
        .affine(1.0 / (n as f64).sqrt(), 0.0)?;

    // Per-element column dot / norm, `[.., 1, 1]` — the batched analogue of
    // the scalar coefficients. Host copies drive the bookkeeping.
    let cdot = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
        (a.broadcast_mul(b))?.sum_keepdim(candle_core::D::Minus2)
    };
    let host = |t: &Tensor| -> Result<Vec<f64>> {
        t.flatten_all()?.to_dtype(DType::F64)?.to_vec1()
    };

    // Batched Lanczos with full reorthogonalisation (twice — DGKS). The
    // probe starts unbatched [n, 1]; the first matvec broadcasts everything
    // up to the operator's batch shape and it stays there.
    let mut q_cur = probe;
    let mut q_all: Vec<Tensor> = vec![q_cur.clone()];
    let mut q_prev: Option<Tensor> = None;
    let mut beta_prev_t: Option<Tensor> = None;

    // Per-element histories. Sized lazily once the first alpha reveals the
    // batch count.
    let mut alphas_hist: Vec<Vec<f64>> = Vec::new(); // [element][iter]
    let mut betas_hist: Vec<Vec<f64>> = Vec::new();
    let mut frozen: Vec<bool> = Vec::new();
    let mut scale_seen: Vec<f64> = Vec::new();
    // The probe starts UNBATCHED `[n,1]` and only picks up the operator's batch
    // dims at the first matvec, so the batch shape has to be learned rather
    // than assumed — `q_all[0]` will need broadcasting up to it before any cat.
    let mut batch_dims: Vec<usize> = Vec::new();

    for _ in 0..k {
        let mut w = matmul(&q_cur)?;
        if let (Some(qp), Some(bp)) = (&q_prev, &beta_prev_t) {
            w = (&w - qp.broadcast_mul(bp)?)?;
        }
        let alpha_t = cdot(&q_cur, &w)?;
        let alpha_h = host(&alpha_t)?;
        if batch_dims.is_empty() {
            let d = alpha_t.dims();
            batch_dims = d[..d.len() - 2].to_vec();
        }
        if alphas_hist.is_empty() {
            let b = alpha_h.len();
            alphas_hist = vec![Vec::with_capacity(k); b];
            betas_hist = vec![Vec::new(); b];
            frozen = vec![false; b];
            scale_seen = vec![0.0; b];
        }
        if alpha_h.iter().any(|v| !v.is_finite()) {
            return Err(candle_core::Error::Msg(
                "spectrum_bounds_lanczos: non-finite Lanczos coefficient — broken operator".into(),
            ));
        }
        w = (&w - q_cur.broadcast_mul(&alpha_t)?)?;
        for _ in 0..2 {
            for q in &q_all {
                let c = cdot(q, &w)?;
                w = (&w - q.broadcast_mul(&c)?)?;
            }
        }

        let beta_t = cdot(&w, &w)?.sqrt()?;
        let beta_h = host(&beta_t)?;
        if beta_h.iter().any(|v| !v.is_finite()) {
            return Err(candle_core::Error::Msg(
                "spectrum_bounds_lanczos: non-finite Lanczos residual — broken operator".into(),
            ));
        }

        // Record per element; freeze on per-element breakdown (invariant
        // subspace — further rows for THAT element would be noise).
        let mut any_active = false;
        for (b, (&a, &bt)) in alpha_h.iter().zip(&beta_h).enumerate() {
            if frozen[b] {
                continue;
            }
            alphas_hist[b].push(a);
            scale_seen[b] = scale_seen[b].max(a.abs());
            if bt <= 1e-12 * scale_seen[b].max(1.0) {
                frozen[b] = true;
            } else {
                betas_hist[b].push(bt);
                scale_seen[b] = scale_seen[b].max(bt);
                any_active = true;
            }
        }
        if !any_active {
            break;
        }

        // q_{i+1} = w/β, with frozen elements pinned to exactly zero so
        // their (meaningless) later coefficients read 0 rather than junk.
        let thresh = Tensor::from_vec(
            beta_h
                .iter()
                .zip(&frozen)
                .map(|(_, &f)| if f { f64::INFINITY } else { 0.0 })
                .collect::<Vec<_>>(),
            beta_t.shape(),
            device,
        )?
        .to_dtype(beta_t.dtype())?;
        let is_frozen = beta_t.lt(&thresh)?;
        let safe_beta = is_frozen.where_cond(&beta_t.ones_like()?, &beta_t)?;
        let q_next = w.broadcast_div(&safe_beta)?;
        let q_next = is_frozen.broadcast_as(q_next.shape())?.where_cond(&q_next.zeros_like()?, &q_next)?;

        q_prev = Some(std::mem::replace(&mut q_cur, q_next));
        beta_prev_t = Some(beta_t);
        q_all.push(q_cur.clone());
    }

    // Per element: assemble T_b, Frobenius-prescale (jacobi_eigh's
    // convergence test is ABSOLUTE), eig, fold min/max WITHOUT sorting
    // (unsorted by contract), rescale, safeguard.
    let n_elems = alphas_hist.len();
    let k_max = alphas_hist.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::with_capacity(n_elems);
    let mut kept = Vec::with_capacity(n_elems);
    // Zero-padded per element: `kk` differs across the batch whenever one
    // element's recurrence hit an invariant subspace and froze.
    let mut ritz_host = vec![0.0f64; n_elems * k_max];
    let mut w_host = vec![0.0f64; n_elems * k_max];
    let mut v_host = vec![0.0f64; n_elems * k_max * k_max];
    for b in 0..n_elems {
        let alphas = &alphas_hist[b];
        let betas = &betas_hist[b];
        let kk = alphas.len();
        let mut t_host = vec![0.0f64; kk * kk];
        for (i, a) in alphas.iter().enumerate() {
            t_host[i * kk + i] = *a;
        }
        for (i, be) in betas.iter().take(kk.saturating_sub(1)).enumerate() {
            t_host[i * kk + (i + 1)] = *be;
            t_host[(i + 1) * kk + i] = *be;
        }
        let frob = t_host.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(frob.is_finite() && frob > 0.0) {
            return Err(candle_core::Error::Msg(format!(
                "spectrum_bounds_lanczos: batch element {b}: degenerate Lanczos tridiagonal \
                 (‖T‖_F = {frob})"
            )));
        }
        let scaled: Vec<f64> = t_host.iter().map(|v| v / frob).collect();
        let (eigs, vecs) = crate::linalg::jacobi_eigh(&scaled, kk);

        // ASCENDING, with the eigenvector columns permuted to match.
        // `jacobi_eigh` returns them unsorted, and a ladder paired against
        // mis-sorted vectors still yields a perfectly valid SPD preconditioner
        // — it is just useless, and every symmetry/roundtrip invariant passes.
        let mut order: Vec<usize> = (0..kk).collect();
        order.sort_by(|&i, &j| eigs[i].partial_cmp(&eigs[j]).unwrap_or(std::cmp::Ordering::Equal));

        kept.push(kk);
        for (slot, &j) in order.iter().enumerate() {
            let lam = eigs[j] * frob;
            ritz_host[b * k_max + slot] = lam;
            // (s_j)₁² — the squared first component of eigenvector `j`, which
            // sits in COLUMN `j` of the row-major `vecs`. Scale-free, so the
            // Frobenius prescale does not touch it.
            let s1 = vecs[j];
            w_host[b * k_max + slot] = s1 * s1;
            for i in 0..kk {
                v_host[b * k_max * k_max + i * k_max + slot] = vecs[i * kk + j];
            }
        }
        let ritz_min = ritz_host[b * k_max];
        let ritz_max = ritz_host[b * k_max + kk - 1];

        if !(ritz_max.is_finite() && ritz_max > 0.0) {
            return Err(candle_core::Error::Msg(format!(
                "spectrum_bounds_lanczos: batch element {b}: top Ritz value {ritz_max:.3e} is \
                 not positive — the operator does not look positive definite"
            )));
        }
        let lambda_max_hat = ritz_max * 1.1;
        let jitter_floor = settings::tridiagonal_jitter::value() * lambda_max_hat;
        if ritz_min < -100.0 * jitter_floor {
            return Err(candle_core::Error::Msg(format!(
                "spectrum_bounds_lanczos: batch element {b}: estimated λmin ≈ {ritz_min:.3e} \
                 (λmax ≈ {ritz_max:.3e}) — the operator looks genuinely indefinite; \
                 K^{{±1/2}} is undefined"
            )));
        }
        out.push(((ritz_min * 0.25).max(jitter_floor), lambda_max_hat));
    }

    // Ritz vectors = the Lanczos basis times the tridiagonal's eigenvectors.
    // `q_all[0]` is the unbatched probe and the rest carry batch dims, so every
    // column is broadcast to the full shape before the cat. `q_all.len()` is
    // `k_max + 1` when no element broke on the final iteration, so the trailing
    // vector — which has no Ritz value — is dropped by taking only `k_max`.
    let mut col_shape = batch_dims.clone();
    col_shape.extend_from_slice(&[n, 1]);
    let mut cols = Vec::with_capacity(k_max);
    for qi in q_all.iter().take(k_max) {
        cols.push(qi.broadcast_as(col_shape.as_slice())?.contiguous()?);
    }
    let refs: Vec<&Tensor> = cols.iter().collect();
    let q_lanczos = Tensor::cat(&refs, candle_core::D::Minus1)?; // [..B, n, k_max]

    let mut vshape = batch_dims.clone();
    vshape.extend_from_slice(&[k_max, k_max]);
    let v = Tensor::from_vec(v_host, vshape, device)?.to_dtype(dtype)?;
    let q = q_lanczos.broadcast_matmul(&v)?;

    let mut kshape = batch_dims.clone();
    kshape.push(k_max);
    let ritz = Tensor::from_vec(ritz_host, kshape.clone(), device)?.to_dtype(dtype)?;
    let weights = Tensor::from_vec(w_host, kshape, device)?.to_dtype(dtype)?;

    Ok(LanczosProbe { q, ritz, weights, kept, bounds: out })
}

/// `K^{-1/2} rhs` (or, with a preconditioner set, the conjugated-root apply
/// `R⁻¹ rhs` — see [`CiqOptions::precond`]), detached numerics. `matmul`
/// applies `K` (`n × n` per batch element); `rhs` is `[..., N]` or
/// `[..., N, C]`.
pub fn ciq_inv_sqrt_matmul<F>(matmul: F, n: usize, rhs: &Tensor, opts: &CiqOptions) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match &opts.precond {
        // R⁻¹ rhs = K̂^{-1/2} (E⁻¹ rhs) on the conjugated K̂ = E⁻¹KE⁻¹.
        // The inner call goes through a &dyn Fn so the wrapped closure does
        // not recurse in the type system.
        Some(p) => {
            let inner = |x: &Tensor| -> Result<Tensor> {
                p.inv_sqrt_apply(&matmul(&p.inv_sqrt_apply(x)?)?)
            };
            let c = p.inv_sqrt_apply(rhs)?;
            let plain = CiqOptions { precond: None, ..opts.clone() };
            inv_sqrt_core(&inner, n, &c, &plain)
        }
        None => inv_sqrt_core(&matmul, n, rhs, opts),
    }
}

fn inv_sqrt_core(
    matmul: &dyn Fn(&Tensor) -> Result<Tensor>,
    n: usize,
    rhs: &Tensor,
    opts: &CiqOptions,
) -> Result<Tensor> {
    let (lmin, lmax) = match opts.bounds {
        Some(b) => b,
        None => spectrum_bounds_lanczos(matmul, n, rhs.dtype(), rhs.device(), opts.lanczos_iters)?,
    };
    ciq_inv_sqrt_matmul_with_bounds(matmul, rhs, opts, (lmin, lmax))
}

/// As [`ciq_inv_sqrt_matmul`], with the spectrum bounds supplied by the
/// caller — the entry point the differentiable wrapper uses so forward and
/// backward share one quadrature, and callers with known bounds skip the
/// Lanczos probe.
pub fn ciq_inv_sqrt_matmul_with_bounds<F>(
    matmul: F,
    rhs: &Tensor,
    opts: &CiqOptions,
    (lambda_min, lambda_max): (f64, f64),
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let q = opts.q.unwrap_or_else(settings::num_contour_quadrature::value);
    let terms = contour_terms(opts.rule, lambda_min, lambda_max, q);
    let shifts: Vec<f64> = terms.iter().map(|t| t.shift).collect();

    let tol = opts.tolerance.unwrap_or_else(settings::minres_tolerance::value);
    let cg_opts = CgOptions { max_iter: opts.max_iter, ..CgOptions::for_tolerance(tol) };
    let (solves, stats) = solve_shifted(opts.solver, &matmul, rhs, &shifts, &cg_opts)?;
    if !stats.converged && settings::verbose_linalg::value() {
        tracing::warn!(
            iterations = stats.iterations,
            residual = stats.residual_norms.first().copied().unwrap_or(f64::NAN),
            "ciq: shifted solves hit the iteration budget before tolerance"
        );
    }

    // Weighted accumulation in F64: GL weights span orders of magnitude via
    // the (1−u)^{-2} Möbius factor, and an F32 running sum loses digits
    // exactly where Q is large. One downcast at the end.
    let out_dtype = rhs.dtype();
    let mut acc: Option<Tensor> = None;
    for (x, t) in solves.iter().zip(&terms) {
        let term = x.to_dtype(DType::F64)?.affine(t.weight, 0.0)?;
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    acc.expect("q >= 1 guarantees at least one term").to_dtype(out_dtype)
}

/// `K^{+1/2} rhs = K · (K^{-1/2} rhs)` — one extra matvec. With a
/// preconditioner set: `R rhs = E · K̂^{1/2} rhs` (note NO `E⁻¹` on the way
/// in — `R = E·K̂^{1/2}` acts on `rhs` directly).
pub fn ciq_sqrt_matmul<F>(matmul: F, n: usize, rhs: &Tensor, opts: &CiqOptions) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match &opts.precond {
        Some(p) => {
            let inner = |x: &Tensor| -> Result<Tensor> {
                p.inv_sqrt_apply(&matmul(&p.inv_sqrt_apply(x)?)?)
            };
            let plain = CiqOptions { precond: None, ..opts.clone() };
            // K̂^{1/2} rhs = K̂ · (K̂^{-1/2} rhs), then one E on the way out.
            let half = inv_sqrt_core(&inner, n, rhs, &plain)?;
            p.sqrt_apply(&inner(&half)?)
        }
        None => {
            let inv_root = ciq_inv_sqrt_matmul(&matmul, n, rhs, opts)?;
            matmul(&inv_root)
        }
    }
}

// ---------------------------------------------------------------------------
// CiqInvSqrtOp — the differentiable wrapper (dense K only)
// ---------------------------------------------------------------------------

use candle_core::{CpuStorage, CustomOp2, Layout, Shape};
use std::sync::Mutex;

use crate::linalg_diff::{square_batch_dims, storage_f64, storage_from_f64, tensor_f64};
use crate::quadrature::ContourTerm;

/// `y = K^{-1/2} b` with a hand-written analytic VJP, in the
/// [`crate::linalg_diff`] house style: forward and backward both run the
/// detached production driver on host-f64 numerics; candle here is
/// first-order-only, so nothing in `bwd` may build graph.
///
/// The forward's quadrature `(shift, weight)` pairs are cached on the op
/// instance (candle stores the instance in the graph node and hands it back
/// to `bwd`), so backward reuses the forward's EXACT contour — no re-probe,
/// no fwd/bwd shift mismatch.
///
/// # Gradient formulas
///
/// With `y = Σ_j w_j (K + t_j I)^{-1} b`, per-shift solves
/// `v_j = (K + t_j I)^{-1} b` (recomputed — they are not recoverable from
/// `y`) and adjoint solves `u_j = (K + t_j I)^{-1} ḡ`:
///
/// ```text
/// b̄ = Σ_j w_j u_j                        ( = K^{-1/2} ḡ — CIQ of the cotangent)
/// K̄ = −½ Σ_j w_j (u_j v_jᵀ + v_j u_jᵀ)   (symmetrised, per the CholeskyOp
///                                          convention: K is SPD upstream)
/// ```
///
/// Both solve families come from ONE `linear_cg_shifted` on
/// `concat([b, ḡ])`.
///
/// # Documented caveat
///
/// `w_j, t_j` depend on `K` through the estimated spectrum bounds, and the
/// backward treats them as constants. That is exact in the Q→∞ limit (the
/// integral is parameter-free) and matches gpytorch's semantics; at finite Q
/// a finite-difference check sees the discrepancy at the scale of the
/// quadrature error's derivative — hence the deterministic Lanczos probe, a
/// generous Q in the gradcheck, and its slightly loosened tolerance.
struct CiqInvSqrtOp {
    opts: CiqOptions,
    cache: Mutex<Option<Vec<ContourTerm>>>,
}

impl CiqInvSqrtOp {
    fn solve_opts(&self) -> CgOptions {
        let tol = self.opts.tolerance.unwrap_or_else(settings::minres_tolerance::value);
        CgOptions { max_iter: self.opts.max_iter, ..CgOptions::for_tolerance(tol) }
    }
}

impl CustomOp2 for CiqInvSqrtOp {
    fn name(&self) -> &'static str {
        "ciq_inv_sqrt_diff"
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
        let (_, n) = square_batch_dims(l1.shape())?;

        let dev = Device::Cpu;
        let k = Tensor::from_vec(k_data, l1.shape().clone(), &dev)?;
        let b = Tensor::from_vec(b_data, l2.shape().clone(), &dev)?;
        let mm = |x: &Tensor| -> Result<Tensor> { k.broadcast_matmul(x) };

        let (lmin, lmax) =
            spectrum_bounds_lanczos(mm, n, DType::F64, &dev, self.opts.lanczos_iters)?;
        let q = self.opts.q.unwrap_or_else(settings::num_contour_quadrature::value);
        let terms = contour_terms(self.opts.rule, lmin, lmax, q);
        *self.cache.lock().expect("ciq cache poisoned") = Some(terms.clone());

        let y = ciq_inv_sqrt_matmul_with_bounds(mm, &b, &self.opts, (lmin, lmax))?;
        let y_host: Vec<f64> = y.flatten_all()?.to_vec1()?;
        Ok((storage_from_f64(y_host, as_f32), l2.shape().clone()))
    }

    fn bwd(
        &self,
        arg1: &Tensor, // K
        arg2: &Tensor, // b
        _res: &Tensor, // y — the per-shift solves are not recoverable from it
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let terms = self
            .cache
            .lock()
            .expect("ciq cache poisoned")
            .clone()
            .ok_or_else(|| {
                candle_core::Error::Msg("ciq_inv_sqrt_diff: bwd before fwd — no cached contour".into())
            })?;
        let shifts: Vec<f64> = terms.iter().map(|t| t.shift).collect();

        let dev = Device::Cpu;
        let k = Tensor::from_vec(tensor_f64(arg1)?, arg1.shape().clone(), &dev)?;
        let b = Tensor::from_vec(tensor_f64(arg2)?, arg2.shape().clone(), &dev)?;
        let g = Tensor::from_vec(tensor_f64(grad_res)?, grad_res.shape().clone(), &dev)?;
        let c = b.dim(candle_core::D::Minus1)?;

        // One multi-shift solve serves both families: columns [b | ḡ].
        // Routed through the same dispatch as the forward, so fwd and bwd
        // always use the same solver.
        let cat = Tensor::cat(&[&b, &g], candle_core::D::Minus1)?;
        let (solves, _) = solve_shifted(
            self.opts.solver,
            |x| k.broadcast_matmul(x),
            &cat,
            &shifts,
            &self.solve_opts(),
        )?;

        let mut bbar: Option<Tensor> = None;
        let mut kbar: Option<Tensor> = None;
        for (x, t) in solves.iter().zip(&terms) {
            let v = x.narrow(candle_core::D::Minus1, 0, c)?; // (K+tI)⁻¹ b
            let u = x.narrow(candle_core::D::Minus1, c, c)?; // (K+tI)⁻¹ ḡ

            let ub = u.affine(t.weight, 0.0)?;
            bbar = Some(match bbar {
                Some(a) => (a + ub)?,
                None => ub,
            });

            // w_j · u_j v_jᵀ, summed over the C columns by the matmul itself.
            let outer = u.broadcast_matmul(&v.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?.contiguous()?)?;
            let outer = outer.affine(t.weight, 0.0)?;
            kbar = Some(match kbar {
                Some(a) => (a + outer)?,
                None => outer,
            });
        }
        let bbar = bbar.expect("q >= 1");
        let kbar = kbar.expect("q >= 1");
        // K̄ = −½ (Σ + Σᵀ)
        let kbar_t = kbar.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?.contiguous()?;
        let kbar = ((kbar + kbar_t)?).affine(-0.5, 0.0)?;

        Ok((
            Some(kbar.to_dtype(arg1.dtype())?),
            Some(bbar.to_dtype(arg2.dtype())?),
        ))
    }
}

/// Differentiable `K^{-1/2} rhs` for a DENSE SPD `K [..., N, N]`,
/// `rhs [..., N]` or `[..., N, C]`. Gradients flow to both arguments via the
/// analytic VJP above; lazy operators stay on the detached
/// [`ciq_inv_sqrt_matmul`].
///
/// `K` and `rhs` must share their batch dims exactly (or both have none) —
/// gradient reduction across broadcast batch dims is deliberately not
/// implemented.
pub fn ciq_inv_sqrt_diff(k: &Tensor, rhs: &Tensor, opts: &CiqOptions) -> Result<Tensor> {
    if opts.precond.is_some() {
        return Err(candle_core::Error::Msg(
            "ciq_inv_sqrt_diff: preconditioning is detached-driver only — the VJP treats \
             K^{-1/2} as the symmetric root, and differentiating through the conjugated \
             root's E is out of scope. Drop opts.precond or use ciq_inv_sqrt_matmul."
                .into(),
        ));
    }
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(candle_core::D::Minus1)? } else { rhs.clone() };
    let (kb, rb) = (&k.dims()[..k.rank() - 2], &rhs2.dims()[..rhs2.rank() - 2]);
    if kb != rb {
        return Err(candle_core::Error::Msg(format!(
            "ciq_inv_sqrt_diff: K batch dims {kb:?} must equal rhs batch dims {rb:?}"
        )));
    }
    let out = k.contiguous()?.apply_op2(
        &rhs2.contiguous()?,
        CiqInvSqrtOp { opts: opts.clone(), cache: Mutex::new(None) },
    )?;
    if is_vector { out.squeeze(candle_core::D::Minus1) } else { Ok(out) }
}

/// Differentiable `K^{+1/2} rhs = K · (K^{-1/2} rhs)`. The gradient composes
/// through candle's matmul VJP plus [`ciq_inv_sqrt_diff`]'s — no second
/// CustomOp, and automatically consistent with it.
pub fn ciq_sqrt_diff(k: &Tensor, rhs: &Tensor, opts: &CiqOptions) -> Result<Tensor> {
    let is_vector = rhs.rank() == 1;
    let inv = ciq_inv_sqrt_diff(k, rhs, opts)?;
    let inv2 = if is_vector { inv.unsqueeze(candle_core::D::Minus1)? } else { inv };
    let out = k.broadcast_matmul(&inv2)?;
    if is_vector { out.squeeze(candle_core::D::Minus1) } else { Ok(out) }
}

// ---------------------------------------------------------------------------
// Lazy-operator autograd — CIQ gradients WITHOUT densifying K
// ---------------------------------------------------------------------------

/// The seam a lazy/tiled kernel operator implements to get differentiable
/// CIQ — the lazy counterpart of [`CiqInvSqrtOp`]'s dense `K` argument, and
/// the first consumer of the symbolic layer's leaf-substitution primitives
/// (`LazyTensor::leaves` / `with_leaves`).
pub trait DifferentiableKernelOp: Send + Sync {
    /// System size `n` (the operator is `[..., n, n]`, symmetric SPD).
    fn n(&self) -> usize;
    /// Learnable leaf tensors in a STABLE order — the pack gradients return
    /// through. A parameter appearing at two leaf positions appears twice;
    /// `Tensor::cat`'s VJP sums the two contributions back into it, which
    /// is the chain rule for shared parameters (e.g. the same points tensor
    /// behind both `vi` and `vj` of a symmetric kernel).
    fn grad_leaves(&self) -> Vec<Tensor>;
    /// The same operator with its leaves replaced (same order, shapes,
    /// dtypes) — used by the backward to re-track the bilinear form on
    /// fresh `Var`s.
    fn with_grad_leaves(&self, leaves: &[Tensor]) -> Result<Box<dyn DifferentiableKernelOp>>;
    /// Matvec with NO tape — what the Krylov solves and the bounds probe run
    /// on. Must never densify.
    fn matmul_detached(&self, x: &Tensor) -> Result<Tensor>;
    /// Matvec WITH tape — gradients flow to the current leaves. Tiled, so
    /// backward memory is `O(tiles)`, never `O(n²)`.
    fn matmul_tracked(&self, x: &Tensor) -> Result<Tensor>;
}

pub(crate) fn pack_leaves(leaves: &[Tensor]) -> Result<(Tensor, Vec<Vec<usize>>)> {
    let dtype = leaves[0].dtype();
    let mut flats = Vec::with_capacity(leaves.len());
    let mut shapes = Vec::with_capacity(leaves.len());
    for l in leaves {
        if l.dtype() != dtype {
            return Err(candle_core::Error::Msg(
                "ciq_inv_sqrt_lazy: mixed leaf dtypes are not supported".into(),
            ));
        }
        shapes.push(l.dims().to_vec());
        flats.push(l.flatten_all()?);
    }
    Ok((Tensor::cat(&flats, 0)?, shapes))
}

pub(crate) fn unpack_leaves(packed: &Tensor, shapes: &[Vec<usize>]) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(shapes.len());
    let mut off = 0usize;
    for s in shapes {
        let len: usize = s.iter().product::<usize>().max(1);
        out.push(packed.narrow(0, off, len)?.reshape(s.clone())?);
        off += len;
    }
    Ok(out)
}

/// `y = K^{-1/2} rhs` for a LAZY symmetric SPD operator, with gradients to
/// every leaf parameter and to `rhs` — `K` is never densified, forward or
/// backward.
///
/// # How the gradient works (the nested-tape pattern)
///
/// Unrolling the Krylov solves in-graph is not an option — backward depth
/// `O(iterations)` is the exact failure `sinkhorn_coupling_t` had — and a
/// `CustomOp` cannot take an arbitrary parameter set. So:
///
/// * the operator's leaves are PACKED into one tracked tensor, which is the
///   `CustomOp2`'s first argument (the in-repo hyper-packing idiom, cf. the
///   coupling kernel's backend op); `cat`'s VJP fans the packed cotangent
///   back out to the live parameters;
/// * the backward computes the parameter cotangent from the identity
///   `θ̄ = ∂/∂θ [ −Σ_j w_j v_jᵀ K(θ) u_j ]` at frozen `u_j = (K+t_jI)⁻¹b`,
///   `v_j = (K+t_jI)⁻¹ḡ` — a NESTED TAPE over one tracked tiled matvec
///   block (the `Genred`-style bilinear reduction), run on fresh `Var`s
///   rebuilt from the packed values. First-order only, which is all candle
///   has, and all this needs.
///
/// No symmetrisation appears (unlike the dense op's `K̄`): the gradient is
/// taken through the PARAMETRISATION `K(θ)`, which is symmetric-valued by
/// construction, so the chain rule lands directly on `θ`.
///
/// Shares the quadrature-parameter caveat of [`CiqInvSqrtOp`] (bounds
/// treated as constants of `θ`), and refuses `opts.precond` for the same
/// reason [`ciq_inv_sqrt_diff`] does.
pub fn ciq_inv_sqrt_lazy(
    op: Arc<dyn DifferentiableKernelOp>,
    rhs: &Tensor,
    opts: &CiqOptions,
    root_use: RootUse,
) -> Result<Tensor> {
    if opts.precond.is_some() && root_use == RootUse::SymmetricRoot {
        return Err(candle_core::Error::Msg(
            "ciq_inv_sqrt_lazy: a preconditioned CIQ computes the CONJUGATED root \
             R = E·K̂^{1/2}, not the symmetric K^{-1/2}, and E's dependence on θ is not \
             differentiated. Pass RootUse::AnyRoot if your result is invariant under \
             R → RQ (sampling, quadratic forms, ‖R⁻¹b‖²); otherwise drop opts.precond."
                .into(),
        ));
    }
    let leaves = op.grad_leaves();
    if leaves.is_empty() {
        return Err(candle_core::Error::Msg(
            "ciq_inv_sqrt_lazy: operator exposes no gradient leaves — use the detached \
             ciq_inv_sqrt_matmul instead"
                .into(),
        ));
    }
    let (packed, shapes) = pack_leaves(&leaves)?;
    let is_vector = rhs.rank() == 1;
    let rhs2 = if is_vector { rhs.unsqueeze(candle_core::D::Minus1)? } else { rhs.clone() };
    let out = packed.contiguous()?.apply_op2(
        &rhs2.contiguous()?,
        // The preconditioner is CARRIED, not stripped. Deleting the refusal
        // above without changing this line would leave the feature dead while
        // every test still passed — the same class as `tridiag_opts`.
        CiqLazyOp { op, shapes, opts: opts.clone(), cache: Mutex::new(None) },
    )?;
    if is_vector { out.squeeze(candle_core::D::Minus1) } else { Ok(out) }
}

struct CiqLazyOp {
    op: Arc<dyn DifferentiableKernelOp>,
    shapes: Vec<Vec<usize>>,
    opts: CiqOptions,
    cache: Mutex<Option<Vec<ContourTerm>>>,
}

impl CiqLazyOp {
    /// `E⁻¹ K E⁻¹ x` when a preconditioner is set, `K x` otherwise.
    fn conjugated_matvec(&self, x: &Tensor) -> Result<Tensor> {
        match &self.opts.precond {
            Some(p) => p.inv_sqrt_apply(&self.op.matmul_detached(&p.inv_sqrt_apply(x)?)?),
            None => self.op.matmul_detached(x),
        }
    }

    /// `E⁻¹ v`, the identity without a preconditioner.
    fn conjugate(&self, v: &Tensor) -> Result<Tensor> {
        match &self.opts.precond {
            Some(p) => p.inv_sqrt_apply(v),
            None => Ok(v.clone()),
        }
    }

    /// The primal right-hand side becomes `c = E⁻¹b`. The COTANGENT does not —
    /// that asymmetry is the single easiest thing to get wrong here, and
    /// getting it wrong yields a finite, plausible, wrong gradient.
    fn conjugate_rhs(&self, b: &Tensor) -> Result<Tensor> {
        self.conjugate(b)
    }

    fn solve_opts(&self) -> CgOptions {
        let tol = self.opts.tolerance.unwrap_or_else(settings::minres_tolerance::value);
        CgOptions { max_iter: self.opts.max_iter, ..CgOptions::for_tolerance(tol) }
    }
}

impl CustomOp2 for CiqLazyOp {
    fn name(&self) -> &'static str {
        "ciq_inv_sqrt_lazy"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // The packed argument's VALUES are identical to the leaves already
        // inside `self.op` — it exists to give candle a graph edge to route
        // the parameter cotangent through, not to carry data.
        let (b_data, as_f32) = storage_f64(s2, l2)?;
        let dev = Device::Cpu;
        let out_dtype = if as_f32 { DType::F32 } else { DType::F64 };
        let b = Tensor::from_vec(b_data, l2.shape().clone(), &dev)?.to_dtype(out_dtype)?;

        let n = self.op.n();
        // With a preconditioner set the whole pipeline runs on the CONJUGATED
        // operator `K̂ = E⁻¹KE⁻¹` against `c = E⁻¹b`, exactly as the detached
        // driver does — bounds probe included, since `K̂`'s spectrum is the
        // one the contour has to cover.
        let mm = |x: &Tensor| self.conjugated_matvec(x);
        let b = self.conjugate_rhs(&b)?;
        let (lmin, lmax) = spectrum_bounds_lanczos(&mm, n, b.dtype(), &dev, self.opts.lanczos_iters)?;
        let q = self.opts.q.unwrap_or_else(settings::num_contour_quadrature::value);
        let terms = contour_terms(self.opts.rule, lmin, lmax, q);
        *self.cache.lock().expect("ciq lazy cache poisoned") = Some(terms.clone());

        let plain = CiqOptions { precond: None, ..self.opts.clone() };
        let y = ciq_inv_sqrt_matmul_with_bounds(&mm, &b, &plain, (lmin, lmax))?;
        let y_host: Vec<f64> = y.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;
        Ok((storage_from_f64(y_host, as_f32), l2.shape().clone()))
    }

    fn bwd(
        &self,
        packed: &Tensor,
        rhs: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let terms = self
            .cache
            .lock()
            .expect("ciq lazy cache poisoned")
            .clone()
            .ok_or_else(|| {
                candle_core::Error::Msg("ciq_inv_sqrt_lazy: bwd before fwd — no cached contour".into())
            })?;
        let shifts: Vec<f64> = terms.iter().map(|t| t.shift).collect();

        // One multi-shift solve for both families, on the CONJUGATED operator.
        //
        // The two rhs columns are treated DIFFERENTLY and that is the whole
        // subtlety: the primal side is `c = E⁻¹b` because that is what the
        // forward actually solved against, while the cotangent `ḡ` enters
        // unconjugated because it is a cotangent of the OUTPUT. Conjugating
        // both (or neither) gives a finite, plausible, wrong gradient —
        // `preconditioned_lazy_ciq_matches_the_inv_quad_oracle` is the test
        // that catches it, and an invariance test would not.
        let b = self.conjugate_rhs(&rhs.detach())?;
        let g = grad_res.detach().contiguous()?;
        let c = b.dim(candle_core::D::Minus1)?;
        let cat = Tensor::cat(&[&b, &g], candle_core::D::Minus1)?;
        let mm = |x: &Tensor| self.conjugated_matvec(x);
        let (solves, _) = solve_shifted(self.opts.solver, &mm, &cat, &shifts, &self.solve_opts())?;

        // rhs̄ = Σ w_j v_j (CIQ of the cotangent), and the frozen blocks for
        // the bilinear surrogate: U = [u_1 … u_Q], V_w = [w_1 v_1 … w_Q v_Q].
        let mut rhs_bar: Option<Tensor> = None;
        let mut u_blocks = Vec::with_capacity(terms.len());
        let mut vw_blocks = Vec::with_capacity(terms.len());
        for (x, t) in solves.iter().zip(&terms) {
            let u = x.narrow(candle_core::D::Minus1, 0, c)?.detach();
            let v = x.narrow(candle_core::D::Minus1, c, c)?.detach();
            let vw = v.affine(t.weight, 0.0)?;
            rhs_bar = Some(match rhs_bar {
                Some(a) => (a + &vw)?,
                None => vw.clone(),
            });
            u_blocks.push(u);
            vw_blocks.push(vw);
        }
        // b̄ = E⁻¹ Σ_j w_j v̂_j — ONE conjugation after the accumulation, since
        // `E⁻¹` is linear; per-shift would be q times the work for the same
        // answer.
        let rhs_bar = self.conjugate(&rhs_bar.expect("q >= 1"))?;

        // θ̄ via the nested tape: S(θ) = −Σ_j w_j v_jᵀ K(θ) u_j
        //                              = −(V_w ⊙ K(θ) U).sum, one tracked
        // tiled matvec block over Q·C columns.
        let leaf_values = unpack_leaves(&packed.detach(), &self.shapes)?;
        let vars: Vec<candle_core::Var> = leaf_values
            .iter()
            .map(candle_core::Var::from_tensor)
            .collect::<Result<_>>()?;
        let leaf_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
        let tracked_op = self.op.with_grad_leaves(&leaf_tensors)?;

        // ⟨v̂_j, ∂K̂ û_j⟩ = ⟨E⁻¹v̂_j, ∂K E⁻¹û_j⟩ — so the nested tape below is
        // completely unchanged and still differentiates the UNCONJUGATED
        // `K(θ)`. That is what `E` being a detached constant buys.
        let u_all = self.conjugate(&Tensor::cat(&u_blocks, candle_core::D::Minus1)?)?;
        let vw_all = self.conjugate(&Tensor::cat(&vw_blocks, candle_core::D::Minus1)?)?;
        let s = (vw_all * tracked_op.matmul_tracked(&u_all)?)?
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
        Ok((Some(packed_bar), Some(rhs_bar.to_dtype(rhs.dtype())?)))
    }
}

#[cfg(test)]
mod tests {
    //! Verification of contour-integral quadrature for matrix functions and implicit tensor derivatives.

    use super::*;
    use crate::linalg::{sym_inv_sqrt, sym_sqrt};
    use candle_core::Device;

    /// Dense SPD `A = BᵀB + shift·I` in F64 (same construction as cg.rs).
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
            (0..n * c).map(|i| ((i * 31 % 13) as f64) / 6.0 - 1.0).collect::<Vec<_>>(),
            (n, c),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn max_rel(x: &Tensor, y: &Tensor) -> f64 {
        let a: Vec<f64> = x.flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        let b: Vec<f64> = y.flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        let scale = b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-12);
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max) / scale
    }

    // ---- spectral vs pivoted ------------------------------------------------

    /// **The claim, measured.** Pivoted Cholesky picks pivots by
    /// residual-diagonal magnitude, so it only ever finds DOMINANT directions;
    /// Lanczos converges at both extremes first. On a spectrum whose small end
    /// is the bottleneck, the Ritz-basis preconditioner should win on iteration
    /// count and the pivoted one should barely move.
    ///
    /// `cg_ignore_min_iter` ON deliberately: the 10-iteration floor makes every
    /// count below 11 unobservable, and the recorded trap in this repo is a
    /// preconditioner test that passed vacuously at 11 vs 11 because both paths
    /// were sitting on it.
    ///
    /// Measured on this fixture: **plain 9, pivoted 22, spectral 6.**
    ///
    /// Pivoted being WORSE than no preconditioner at all is not a fixture
    /// artefact — it is the point. Flattening the top of a spectrum that was
    /// already clustered there, while leaving the small end untouched, can
    /// raise κ rather than lower it. That is the failure mode a Ritz basis does
    /// not have, because Lanczos resolves the bottom too.
    #[test]
    fn a_spectral_preconditioner_beats_pivoted_when_the_small_end_is_the_bottleneck() {
        let _g = settings::cg_ignore_min_iter::with(true);
        let n = 48;
        let dev = Device::Cpu;

        // A handful of small isolated modes under a tight cluster near 1 — the
        // shape where kappa is set by the bottom and the top is already fine.
        let lams: Vec<f64> = (0..n)
            .map(|i| if i < 4 { 1e-3 * 4f64.powi(i as i32) } else { 1.0 + 0.02 * (i % 5) as f64 })
            .collect();
        let q0 = ortho_basis(n, 404);
        let mut m = vec![0.0f64; n * n];
        for (j, col) in q0.iter().enumerate() {
            for i in 0..n {
                for kk in 0..n {
                    m[i * n + kk] += lams[j] * col[i] * col[kk];
                }
            }
        }
        let a = Tensor::from_vec(m, (n, n), &dev).unwrap();
        let b = rhs(n, 1);
        let opts = CgOptions::for_tolerance(1e-10);


        let plain = crate::operators::cg::linear_cg(
            |v: &Tensor| a.matmul(v),
            &b,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();

        let diag = crate::operators::dense_diagonal(&a).unwrap();
        let piv = crate::operators::precond::CiqPreconditioner::pivoted(
            |v: &Tensor| a.matmul(v),
            &diag,
            8,
        )
        .unwrap();
        let probe =
            lanczos_probe(|v: &Tensor| a.matmul(v), n, DType::F64, &dev, Some(12)).unwrap();
        let (q, mu, delta) = probe.spectral_parts(0).unwrap();
        let spec =
            crate::operators::precond::CiqPreconditioner::spectral(&q, &mu, delta, 1e-8).unwrap();

        let run = |p: crate::operators::precond::CiqPreconditioner| -> usize {
            let o = CgOptions { precond: Some(std::sync::Arc::new(p)), ..opts.clone() };
            crate::operators::cg::linear_cg(
                |v: &Tensor| a.matmul(v),
                &b,
                None,
                None::<fn(&Tensor) -> Result<Tensor>>,
                &o,
            )
            .unwrap()
            .1
            .iterations
        };
        let n_piv = run(piv);
        let n_spec = run(spec);
        println!("plain {}  pivoted {n_piv}  spectral {n_spec}", plain.1.iterations);

        assert!(
            n_spec < n_piv,
            "spectral {n_spec} did not beat pivoted {n_piv} on a small-end-bottlenecked spectrum \
             (plain {})",
            plain.1.iterations
        );
        assert!(n_spec < plain.1.iterations, "spectral did not beat unpreconditioned");
    }

    // ---- the Kumaraswamy ladder ---------------------------------------------

    /// Build the small-end-bottlenecked operator both ladder tests use.
    fn bottleneck_operator(n: usize, seed: u64) -> Tensor {
        let lams: Vec<f64> = (0..n)
            .map(|i| if i < 5 { 1e-3 * 3f64.powi(i as i32) } else { 1.0 + 0.02 * (i % 5) as f64 })
            .collect();
        let q0 = ortho_basis(n, seed);
        let mut m = vec![0.0f64; n * n];
        for (j, col) in q0.iter().enumerate() {
            for i in 0..n {
                for kk in 0..n {
                    m[i * n + kk] += lams[j] * col[i] * col[kk];
                }
            }
        }
        Tensor::from_vec(m, (n, n), &Device::Cpu).unwrap()
    }

    fn ladder_run(
        a: &Tensor,
        b: &Tensor,
        n: usize,
        ladder: crate::operators::ladder::SpectralLadder,
    ) -> (Tensor, usize) {
        let probe = lanczos_probe(|v: &Tensor| a.matmul(v), n, DType::F64, &Device::Cpu, Some(14))
            .unwrap();
        let (q, mu, delta) = probe.spectral_parts_with(0, ladder).unwrap();
        let p = crate::operators::precond::CiqPreconditioner::spectral(&q, &mu, delta, 1e-8)
            .unwrap();
        let o = CgOptions {
            precond: Some(std::sync::Arc::new(p)),
            ..CgOptions::for_tolerance(1e-10)
        };
        let (x, st) = crate::operators::cg::linear_cg(
            |v: &Tensor| a.matmul(v),
            b,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &o,
        )
        .unwrap();
        (x, st.iterations)
    }

    /// **Reachability, not invariance.**
    ///
    /// A supplied ladder must visibly change the SOLVE, not merely leave the
    /// answer alone — "the answer is unchanged" is satisfied by a ladder that
    /// was silently dropped on the floor, which is the trap this repo has
    /// already recorded for preconditioner tests.
    ///
    /// Measured: raw Ritz 7 iterations, `Kumaraswamy{2,2}` 22. Unmistakable.
    /// (An earlier version of this test perturbed `q_a` by 20% and asserted the
    /// count moved; it did not, because every smooth shape is uniformly bad on
    /// this spectrum — see the test below for why. A test whose signal is a
    /// small perturbation of an already-broken configuration proves nothing.)
    #[test]
    fn a_supplied_ladder_is_reached_and_changes_only_the_cost() {
        let _g = settings::cg_ignore_min_iter::with(true);
        use crate::operators::ladder::SpectralLadder;
        let n = 48;
        let a = bottleneck_operator(n, 707);
        let b = rhs(n, 1);

        let (x_r, i_r) = ladder_run(&a, &b, n, SpectralLadder::Ritz);
        let (x_k, i_k) = ladder_run(&a, &b, n, SpectralLadder::Kumaraswamy { q_a: 2.0, q_b: 2.0 });
        println!("ritz {i_r} iters ; supplied Kumaraswamy(2,2) {i_k} iters");

        assert!(max_rel(&x_r, &x_k) < 1e-6, "the ladder changed the ANSWER, not just the cost");
        assert_ne!(i_r, i_k, "the supplied ladder was not reached");
    }

    /// **The measurement that decides whether the ladder earns its place — and
    /// it does not.**
    ///
    /// Measured: raw Ritz **6** iterations, Kumaraswamy-fitted **16**,
    /// unpreconditioned **10**. The fitted shape is worse than no
    /// preconditioner at all.
    ///
    /// The reason is structural, not a tuning failure, and this test pins it
    /// directly rather than only through the iteration count. On this fixture
    /// Lanczos recovers the true eigenvalues exactly — `1e-3·3ⁱ` for the five
    /// small modes, then `1.00, 1.02, 1.04, 1.06` for the cluster. **A monotone
    /// two-parameter family cannot represent a gap**, so the fit lands
    /// `0.20, 0.37, 0.61, 0.90` where the truth is `1.00 … 1.06` — mis-scaling
    /// those directions by up to 5×, which is worse than not deflating them.
    ///
    /// And a gap is precisely the structure that makes deflation worth doing.
    /// So the regime where a smooth ladder could help (a dense, gapless
    /// interior whose Ritz values are noisy) is the opposite of the regime
    /// where a preconditioner pays. That is why [`SpectralLadder::Ritz`] is the
    /// default, and why the ladder should NOT be promoted to a learnable
    /// parameter on this evidence: a shape that cannot beat the raw values with
    /// a good fit will not be rescued by gradient descent.
    #[test]
    fn the_fitted_ladder_cannot_represent_a_gapped_spectrum() {
        let _g = settings::cg_ignore_min_iter::with(true);
        use crate::operators::ladder::SpectralLadder;
        let n = 48;
        let a = bottleneck_operator(n, 909);
        let b = rhs(n, 1);

        let (x_r, i_r) = ladder_run(&a, &b, n, SpectralLadder::Ritz);
        let (x_f, i_f) = ladder_run(&a, &b, n, SpectralLadder::KumaraswamyFitted);
        let plain = crate::operators::cg::linear_cg(
            |v: &Tensor| a.matmul(v),
            &b,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &CgOptions::for_tolerance(1e-10),
        )
        .unwrap();
        println!("ritz {i_r} ; fitted {i_f} ; unpreconditioned {}", plain.1.iterations);

        // Both still SOLVE — a bad preconditioner costs iterations, never
        // correctness, and that is the property worth asserting.
        assert!(max_rel(&x_r, &x_f) < 1e-6, "the two ladders disagree on the answer");
        assert!(max_rel(&x_r, &plain.0) < 1e-6);

        // The raw Ritz values win, and by a wide margin.
        assert!(i_r < i_f, "raw Ritz {i_r} did not beat the fitted ladder {i_f}");

        // The structural claim, pinned directly: the fit mis-scales the
        // clustered end because it cannot put a gap there.
        let probe =
            lanczos_probe(|v: &Tensor| a.matmul(v), n, DType::F64, &Device::Cpu, Some(14)).unwrap();
        let (_, mu_r, _) = probe.spectral_parts_with(0, SpectralLadder::Ritz).unwrap();
        let (_, mu_f, _) =
            probe.spectral_parts_with(0, SpectralLadder::KumaraswamyFitted).unwrap();
        let r: Vec<f64> = mu_r.to_vec1().unwrap();
        let f: Vec<f64> = mu_f.to_vec1().unwrap();
        let worst = r
            .iter()
            .zip(&f)
            .map(|(a1, b1)| (b1 / a1).max(a1 / b1))
            .fold(0.0, f64::max);
        assert!(
            worst > 2.0,
            "the fitted ladder tracked the gapped spectrum to within {worst:.2}× — if a smooth \
             two-parameter family can now represent a gap, this whole finding needs revisiting"
        );
    }

    // ---- the bounds passthrough ---------------------------------------------

    /// **A cost test, because the failure mode is correct values.**
    ///
    /// `inv_sqrt_core` probes the spectrum unconditionally, so a caller who
    /// already knows the bounds — anyone who just ran `lanczos_probe` to build
    /// a spectral preconditioner — pays for a second Lanczos they do not need.
    /// Supplying `bounds` must remove exactly `lanczos_iters` matvecs and
    /// change nothing else.
    ///
    /// Nothing else in the suite can catch this: the answer is identical either
    /// way, so every value assertion passes while the work doubles.
    #[test]
    fn supplying_bounds_skips_the_probe_and_saves_exactly_its_matvecs() {
        use std::cell::Cell;
        let n = 24;
        let a = spd(n, 0.5, 17);
        let b = rhs(n, 1);
        let iters = 12usize;

        let count = |bounds: Option<(f64, f64)>| -> (Tensor, usize) {
            let calls = Cell::new(0usize);
            let mm = |x: &Tensor| {
                calls.set(calls.get() + x.dim(candle_core::D::Minus1).unwrap());
                a.matmul(x)
            };
            let opts = CiqOptions { lanczos_iters: Some(iters), bounds, ..lazy_opts() };
            let y = ciq_inv_sqrt_matmul(&mm, n, &b, &opts).unwrap();
            (y, calls.get())
        };

        let (y_probe, n_probe) = count(None);
        let known =
            spectrum_bounds_lanczos(|x: &Tensor| a.matmul(x), n, DType::F64, &Device::Cpu, Some(iters))
                .unwrap();
        let (y_known, n_known) = count(Some(known));

        assert!(max_rel(&y_known, &y_probe) < 1e-12, "bounds passthrough changed the answer");
        assert_eq!(
            n_probe - n_known,
            iters,
            "expected to save exactly the probe's {iters} matvecs, saved {}",
            n_probe - n_known
        );
    }

    // ---- the Lanczos probe's Ritz pairs -------------------------------------

    /// The refactor must not move a single bound. The probe's LCG is
    /// deterministic, so this is EXACT rather than approximate — anything less
    /// than bit-parity means the recurrence changed.
    #[test]
    fn lanczos_probe_bounds_match_the_bounds_only_entry_point() {
        for (n, seed) in [(16usize, 3u64), (24, 11), (40, 29)] {
            let a = spd(n, 0.5, seed);
            let mm = |x: &Tensor| a.matmul(x);
            let want =
                spectrum_bounds_lanczos_per_batch(&mm, n, DType::F64, &Device::Cpu, None).unwrap();
            let got = lanczos_probe(&mm, n, DType::F64, &Device::Cpu, None).unwrap();
            assert_eq!(got.bounds.len(), want.len());
            for (g, w) in got.bounds.iter().zip(&want) {
                assert_eq!(g.0.to_bits(), w.0.to_bits(), "λmin moved at n={n}");
                assert_eq!(g.1.to_bits(), w.1.to_bits(), "λmax moved at n={n}");
            }
        }
    }

    /// Every closed form in a spectral preconditioner is wrong without an
    /// orthonormal basis, and none of them complains — so the basis is checked
    /// where it is produced.
    #[test]
    fn the_ritz_basis_is_orthonormal() {
        let n = 32;
        let a = spd(n, 0.5, 7);
        let p = lanczos_probe(|x: &Tensor| a.matmul(x), n, DType::F64, &Device::Cpu, None).unwrap();
        let k = p.kept[0];
        let q = p.q.narrow(candle_core::D::Minus1, 0, k).unwrap();
        let g = q.t().unwrap().contiguous().unwrap().matmul(&q).unwrap();
        let eye = Tensor::eye(k, DType::F64, &Device::Cpu).unwrap();
        assert!(max_rel(&g, &eye) < 1e-10, "‖QᵀQ − I‖ too large");
    }

    /// A CLUSTERED spectrum is the case the recurrence's twice-repeated DGKS
    /// exists for: without reorthogonalisation, clustering is what manufactures
    /// ghost Ritz values. So it is also the case worth pinning.
    #[test]
    fn the_ritz_basis_stays_orthonormal_on_a_clustered_spectrum() {
        let n = 30;
        let dev = Device::Cpu;
        // Eigenvalues: three isolated, the rest packed into [1.0, 1.001].
        let lams: Vec<f64> = (0..n)
            .map(|i| match i {
                0 => 1e-3,
                1 => 5e-3,
                2 => 2e-2,
                _ => 1.0 + 1e-3 * (i as f64) / (n as f64),
            })
            .collect();
        let q0 = ortho_basis(n, 91);
        let mut m = vec![0.0f64; n * n];
        for (j, col) in q0.iter().enumerate() {
            for i in 0..n {
                for kk in 0..n {
                    m[i * n + kk] += lams[j] * col[i] * col[kk];
                }
            }
        }
        let a = Tensor::from_vec(m, (n, n), &dev).unwrap();
        let p = lanczos_probe(|x: &Tensor| a.matmul(x), n, DType::F64, &dev, None).unwrap();
        let k = p.kept[0];
        let q = p.q.narrow(candle_core::D::Minus1, 0, k).unwrap();
        let g = q.t().unwrap().contiguous().unwrap().matmul(&q).unwrap();
        assert!(
            max_rel(&g, &Tensor::eye(k, DType::F64, &dev).unwrap()) < 1e-9,
            "clustering broke orthonormality"
        );

        // Ritz values ascending, inside the true spectrum (interlacing).
        let r: Vec<f64> = p.ritz.narrow(candle_core::D::Minus1, 0, k).unwrap().to_vec1().unwrap();
        for w in r.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "Ritz values not ascending: {r:?}");
        }
        let (lo, hi) = (lams[0], lams[n - 1]);
        assert!(r[0] >= lo - 1e-9 && r[k - 1] <= hi + 1e-9, "Ritz outside the spectrum: {r:?}");
    }

    /// A rank-deficient operator freezes the recurrence early — the per-element
    /// path that pins later basis columns to exactly zero. `kept` must report
    /// the truncation and the padding must be genuinely zero, or a weighted sum
    /// over `k_max` picks up junk.
    #[test]
    fn an_invariant_subspace_freezes_and_reports_a_short_basis() {
        let n = 24;
        let rank = 5;
        let dev = Device::Cpu;
        // A = Σ_{j<rank} λ_j q_j q_jᵀ + I — the probe's Krylov space is capped
        // by the rank of the non-identity part.
        let q0 = ortho_basis(n, 55);
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        for (j, col) in q0.iter().enumerate().take(rank) {
            let lam = 2.0 + j as f64;
            for i in 0..n {
                for kk in 0..n {
                    m[i * n + kk] += lam * col[i] * col[kk];
                }
            }
        }
        let a = Tensor::from_vec(m, (n, n), &dev).unwrap();
        let p = lanczos_probe(|x: &Tensor| a.matmul(x), n, DType::F64, &dev, None).unwrap();
        let k = p.kept[0];
        assert!(k <= rank + 1, "expected an early freeze, kept {k}");

        let kmax = p.ritz.dim(candle_core::D::Minus1).unwrap();
        if kmax > k {
            let pad_w: Vec<f64> =
                p.weights.narrow(candle_core::D::Minus1, k, kmax - k).unwrap().to_vec1().unwrap();
            assert!(
                pad_w.iter().all(|v| *v == 0.0),
                "padded quadrature weights must be exactly zero, got {pad_w:?}"
            );
        }
    }

    /// The weights are Gauss-quadrature weights, so they sum to 1 — that is
    /// what makes `Σ_j w_j f(μ_j)` an estimate of `probeᵀf(K)probe` for a unit
    /// probe, and hence of `tr f(K)/n`.
    #[test]
    fn the_quadrature_weights_sum_to_one() {
        let n = 28;
        let a = spd(n, 0.5, 13);
        let p = lanczos_probe(|x: &Tensor| a.matmul(x), n, DType::F64, &Device::Cpu, None).unwrap();
        let w: Vec<f64> = p.weights.to_vec1().unwrap();
        let total: f64 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "weights sum to {total}, not 1");
        assert!(w.iter().all(|v| *v >= 0.0), "a quadrature weight went negative");
    }

    /// Gram–Schmidt on pseudo-random columns.
    fn ortho_basis(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let raw: Vec<f64> = (0..n * n).map(|_| next()).collect();
        let mut q: Vec<Vec<f64>> = Vec::new();
        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|i| raw[i * n + j]).collect();
            for u in &q {
                let d: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    v[i] -= d * u[i];
                }
            }
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            q.push(v);
        }
        q
    }

    #[test]
    fn ciq_inv_sqrt_matches_the_dense_eig_reference() {
        let n = 32;
        let a = spd(n, 1.0, 42);
        let b = rhs(n, 2);
        let want = sym_inv_sqrt(&a).unwrap().matmul(&b).unwrap();

        // F64, generous quadrature and tight solves: the quadrature is the
        // only error source left, and Q=40 GL buries it.
        let opts = CiqOptions {
            q: Some(40),
            tolerance: Some(1e-10),
            max_iter: Some(2000),
            ..Default::default()
        };
        let got = ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &b, &opts).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-6, "F64 CIQ vs dense eig root: rel err {rel:.3e}");

        // F32 at production-ish settings.
        let a32 = a.to_dtype(DType::F32).unwrap();
        let b32 = b.to_dtype(DType::F32).unwrap();
        let opts32 = CiqOptions { q: Some(16), max_iter: Some(2000), ..Default::default() };
        let got32 = ciq_inv_sqrt_matmul(|x| a32.matmul(x), n, &b32, &opts32).unwrap();
        let rel32 = max_rel(&got32, &want);
        assert!(rel32 < 1e-3, "F32 CIQ vs dense eig root: rel err {rel32:.3e}");
    }

    #[test]
    fn ciq_sqrt_roundtrips() {
        let n = 24;
        let a = spd(n, 0.5, 9);
        let b = rhs(n, 1);
        let opts = CiqOptions {
            q: Some(32),
            tolerance: Some(1e-10),
            max_iter: Some(2000),
            ..Default::default()
        };
        // K^{-1/2}(K · K^{-1/2} b) = K^{-1/2} K^{1/2} b = b.
        let half = ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &b, &opts).unwrap();
        let back = a.matmul(&half).unwrap(); // = K^{1/2} b
        let round = ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &back, &opts).unwrap();
        let rel = max_rel(&round, &b);
        assert!(rel < 1e-4, "K^{{-1/2}} K K^{{-1/2}} b ≠ b roundtrip: rel {rel:.3e}");
        // ciq_sqrt_matmul against the dense root on the same operand.
        let want = sym_sqrt(&a).unwrap().matmul(&b).unwrap();
        let got = ciq_sqrt_matmul(|x| a.matmul(x), n, &b, &opts).unwrap();
        let rel2 = max_rel(&got, &want);
        assert!(rel2 < 1e-4, "K^{{1/2}} via CIQ vs dense: {rel2:.3e}");
    }

    #[test]
    fn ciq_handles_batched_operands() {
        let dev = Device::Cpu;
        // Three SPD blocks with similar spectra (the v1 contract: shared
        // envelope bounds cover all of them).
        let blocks: Vec<Tensor> = (0..3).map(|s| spd(16, 1.0 + 0.3 * s as f64, 60 + s as u64)).collect();
        let a = Tensor::stack(&blocks, 0).unwrap(); // [3, 16, 16]
        let b = Tensor::from_vec(
            (0..3 * 16 * 2).map(|i| ((i * 23 % 19) as f64) / 9.0 - 1.0).collect::<Vec<_>>(),
            (3, 16, 2),
            &dev,
        )
        .unwrap();
        let opts = CiqOptions {
            q: Some(40),
            tolerance: Some(1e-10),
            max_iter: Some(2000),
            ..Default::default()
        };
        let got = ciq_inv_sqrt_matmul(|x| a.broadcast_matmul(x), 16, &b, &opts).unwrap();
        assert_eq!(got.dims(), &[3, 16, 2]);
        let want = sym_inv_sqrt(&a).unwrap().broadcast_matmul(&b).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-5, "batched CIQ vs dense eig root: rel err {rel:.3e}");
    }

    #[test]
    fn spectrum_bounds_bracket_the_true_spectrum() {
        let n = 32;
        let a = spd(n, 0.7, 5);
        // True spectrum via the host eig on the full matrix.
        let host: Vec<f64> = a.flatten_all().unwrap().to_vec1().unwrap();
        let frob = host.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scaled: Vec<f64> = host.iter().map(|v| v / frob).collect();
        let (eigs, _) = crate::linalg::jacobi_eigh(&scaled, n);
        let lmin_true = eigs.iter().cloned().fold(f64::INFINITY, f64::min) * frob;
        let lmax_true = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * frob;

        let (lmin_hat, lmax_hat) =
            spectrum_bounds_lanczos(|x| a.matmul(x), n, DType::F64, &Device::Cpu, Some(20)).unwrap();
        assert!(
            lmin_hat <= lmin_true && lmin_hat > 0.0,
            "λmin_hat {lmin_hat:.3e} does not bracket true λmin {lmin_true:.3e} from below"
        );
        assert!(
            lmax_hat >= lmax_true,
            "λmax_hat {lmax_hat:.3e} does not bracket true λmax {lmax_true:.3e} from above"
        );
    }

    // ---- per-batch spectrum bounds -------------------------------------------

    /// True (λmin, λmax) of a dense block via the host eig.
    fn true_bounds(a: &Tensor, n: usize) -> (f64, f64) {
        let host: Vec<f64> = a.flatten_all().unwrap().to_vec1().unwrap();
        let frob = host.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scaled: Vec<f64> = host.iter().map(|v| v / frob).collect();
        let (eigs, _) = crate::linalg::jacobi_eigh(&scaled, n);
        (
            eigs.iter().cloned().fold(f64::INFINITY, f64::min) * frob,
            eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * frob,
        )
    }

    /// The v1 failure mode, closed: batch elements with spectra differing by
    /// orders of magnitude must EACH be bracketed by their own bounds — the
    /// old element-0 collapse could not see element 1's spectrum at all.
    #[test]
    fn spectrum_bounds_bracket_every_batch_element() {
        let n = 24;
        // Element 1's spectrum sits ~100× above element 0's.
        let a0 = spd(n, 0.1, 41);
        let a1 = spd(n, 0.1, 43).affine(100.0, 0.0).unwrap();
        let a = Tensor::stack(&[&a0, &a1], 0).unwrap();
        let per = spectrum_bounds_lanczos_per_batch(
            |x| a.broadcast_matmul(x),
            n,
            DType::F64,
            &Device::Cpu,
            Some(n),
        )
        .unwrap();
        assert_eq!(per.len(), 2);
        for (b, (blk, &(lo, hi))) in [&a0, &a1].iter().zip(&per).enumerate() {
            let (tmin, tmax) = true_bounds(blk, n);
            assert!(
                lo <= tmin && lo > 0.0,
                "element {b}: λmin_hat {lo:.3e} does not bracket true {tmin:.3e}"
            );
            assert!(hi >= tmax, "element {b}: λmax_hat {hi:.3e} does not bracket true {tmax:.3e}");
        }
        // And the per-element bounds must actually DIFFER — if both read the
        // same spectrum, the recurrence is secretly still collapsed.
        assert!(
            per[1].1 > 10.0 * per[0].1,
            "element 1's λmax_hat {:.3e} does not reflect its 100× spectrum (element 0: {:.3e})",
            per[1].1,
            per[0].1
        );

        // The envelope covers both.
        let (lo, hi) = spectrum_bounds_lanczos(
            |x| a.broadcast_matmul(x),
            n,
            DType::F64,
            &Device::Cpu,
            Some(n),
        )
        .unwrap();
        assert!(lo <= per[0].0 && lo <= per[1].0 && hi >= per[0].1 && hi >= per[1].1);
    }

    /// One element breaking down early (rank-deficient: its Krylov space is
    /// exhausted in a few steps) must not stall or corrupt its neighbour.
    #[test]
    fn an_early_breakdown_element_does_not_poison_the_batch() {
        let dev = Device::Cpu;
        let n = 16;
        // Element 0: rank-2 + tiny floor (Lanczos exhausts in ~3 steps).
        let mut a0 = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 * 0.3).sin() * (j as f64 * 0.3).sin();
                let v = (i as f64 * 0.7).cos() * (j as f64 * 0.7).cos();
                a0[i * n + j] = 4.0 * u + 2.0 * v + if i == j { 1e-3 } else { 0.0 };
            }
        }
        let a0 = Tensor::from_vec(a0, (n, n), &dev).unwrap();
        let a1 = spd(n, 0.5, 51);
        let a = Tensor::stack(&[&a0, &a1], 0).unwrap();
        let per = spectrum_bounds_lanczos_per_batch(
            |x| a.broadcast_matmul(x),
            n,
            DType::F64,
            &dev,
            Some(n),
        )
        .unwrap();
        let (tmin1, tmax1) = true_bounds(&a1, n);
        assert!(
            per[1].0 <= tmin1 && per[1].1 >= tmax1,
            "full-rank neighbour lost its bracket next to a breakdown element: \
             hat ({:.3e}, {:.3e}) vs true ({tmin1:.3e}, {tmax1:.3e})",
            per[1].0,
            per[1].1
        );
        assert!(per[0].0 > 0.0 && per[0].1 > 0.0, "breakdown element produced junk bounds");
    }

    #[test]
    fn an_indefinite_batch_element_is_refused_by_name() {
        let dev = Device::Cpu;
        let n = 12;
        let a0 = spd(n, 1.0, 61);
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            d[i * n + i] = if i == 0 { -1.0 } else { 0.5 + i as f64 * 0.3 };
        }
        let a1 = Tensor::from_vec(d, (n, n), &dev).unwrap();
        let a = Tensor::stack(&[&a0, &a1], 0).unwrap();
        let err = spectrum_bounds_lanczos_per_batch(
            |x| a.broadcast_matmul(x),
            n,
            DType::F64,
            &dev,
            Some(n),
        );
        let msg = format!("{err:?}");
        assert!(err.is_err(), "indefinite batch element not refused");
        assert!(msg.contains("batch element 1"), "refusal does not name the element: {msg}");
    }

    /// End-to-end: with heterogeneous batch spectra the envelope contour
    /// must serve every element — accuracy is checked per block against the
    /// dense eig root.
    #[test]
    fn batched_ciq_is_accurate_when_batch_spectra_differ() {
        let n = 16;
        let a0 = spd(n, 1.0, 71);
        let a1 = spd(n, 1.0, 73).affine(100.0, 0.0).unwrap();
        let a = Tensor::stack(&[&a0, &a1], 0).unwrap();
        let b = Tensor::from_vec(
            (0..2 * n * 2).map(|i| ((i * 19 % 11) as f64) / 5.0 - 1.0).collect::<Vec<_>>(),
            (2, n, 2),
            &Device::Cpu,
        )
        .unwrap();
        let opts = CiqOptions {
            q: Some(24),
            tolerance: Some(1e-10),
            max_iter: Some(4000),
            ..Default::default()
        };
        let got = ciq_inv_sqrt_matmul(|x| a.broadcast_matmul(x), n, &b, &opts).unwrap();
        let want = sym_inv_sqrt(&a).unwrap().broadcast_matmul(&b).unwrap();
        for blk in 0..2 {
            let g = got.get(blk).unwrap();
            let w = want.get(blk).unwrap();
            let rel = max_rel(&g, &w);
            assert!(rel < 1e-5, "block {blk}: batched CIQ vs dense eig root rel {rel:.3e}");
        }
    }

    #[test]
    fn a_semidefinite_matrix_gets_the_jitter_floor_not_a_crash() {
        let dev = Device::Cpu;
        // Rank-deficient PSD: A = CᵀC with C [8, 16] → rank ≤ 8, λmin = 0.
        let n = 16;
        let c: Vec<f64> = (0..8 * n).map(|i| (((i * 41) % 23) as f64) / 11.0 - 1.0).collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (0..8).map(|k| c[k * n + i] * c[k * n + j]).sum();
            }
        }
        let a = Tensor::from_vec(a, (n, n), &dev).unwrap();
        let (lmin_hat, lmax_hat) =
            spectrum_bounds_lanczos(|x| a.matmul(x), n, DType::F64, &dev, Some(20)).unwrap();
        let floor = crate::settings::tridiagonal_jitter::value() * lmax_hat;
        assert!(
            lmin_hat >= floor * (1.0 - 1e-12),
            "singular matrix: λmin_hat {lmin_hat:.3e} fell below the jitter floor {floor:.3e}"
        );
    }

    #[test]
    fn a_frankly_indefinite_matrix_is_refused() {
        let dev = Device::Cpu;
        let n = 12;
        let mut d = vec![0.0f64; n * n];
        for i in 0..n {
            d[i * n + i] = if i == 0 { -1.0 } else { 0.5 + i as f64 * 0.3 };
        }
        let a = Tensor::from_vec(d, (n, n), &dev).unwrap();
        let err = spectrum_bounds_lanczos(|x| a.matmul(x), n, DType::F64, &dev, Some(12));
        assert!(err.is_err(), "an indefinite matrix must be refused, got {err:?}");
    }

    // ---- CiqInvSqrtOp (autograd) --------------------------------------------

    /// Central finite difference of `loss(param)` at every element — the
    /// canonical harness from linalg_diff.rs.
    fn finite_diff(
        param: &candle_core::Var,
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

    fn grad_opts() -> CiqOptions {
        // Generous Q + tight solves: the only gradient discrepancy left is
        // the (documented) constant-quadrature-parameters approximation.
        CiqOptions {
            q: Some(64),
            tolerance: Some(1e-11),
            max_iter: Some(4000),
            ..Default::default()
        }
    }

    /// Gradcheck through the symmetric construction `K = BBᵀ + 2I` — the
    /// same route the CholeskyOp gradcheck takes, which keeps every FD
    /// perturbation inside the SPD cone and makes the symmetrised K̄ the
    /// right object automatically (chain rule through BBᵀ).
    ///
    /// Tolerance is one order looser than CholeskyOp's 1e-5: the analytic
    /// VJP treats the quadrature's shifts/weights as constants in K (exact
    /// only as Q→∞), so FD sees the quadrature-error derivative on top.
    #[test]
    fn ciq_gradcheck_against_central_differences() {
        let dev = Device::Cpu;
        let (n, c) = (6, 2);
        let b0 = Tensor::randn(0f64, 1f64, (n, n), &dev).unwrap();
        let bvar = candle_core::Var::from_tensor(&b0).unwrap();
        let rhs0 = Tensor::randn(0f64, 1f64, (n, c), &dev).unwrap();
        let rvar = candle_core::Var::from_tensor(&rhs0).unwrap();
        let w = Tensor::randn(0f64, 1f64, (n, c), &dev).unwrap();

        let make_k = |bt: &Tensor| -> Result<Tensor> {
            bt.matmul(&bt.t()?)?
                .add(&Tensor::eye(n, DType::F64, bt.device())?.affine(2.0, 0.0)?)
        };
        let loss_of = |bt: &Tensor, rt: &Tensor| -> Result<f64> {
            let y = ciq_inv_sqrt_diff(&make_k(bt)?, rt, &grad_opts())?;
            y.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let y = ciq_inv_sqrt_diff(&make_k(bvar.as_tensor()).unwrap(), rvar.as_tensor(), &grad_opts())
            .unwrap();
        let loss = y.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_b = grads.get(bvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_r = grads.get(rvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_b = finite_diff(&bvar, &|bt| loss_of(bt, rvar.as_tensor()), 1e-5).unwrap();
        let fd_r = finite_diff(&rvar, &|rt| loss_of(bvar.as_tensor(), rt), 1e-5).unwrap();

        for (i, (ga, gf)) in g_b.iter().zip(&fd_b).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "K-side grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
        for (i, (ga, gf)) in g_r.iter().zip(&fd_r).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "rhs grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
    }

    /// Pins the b̄ formula in isolation: the gradient w.r.t. `b` of
    /// `Σ (K^{-1/2} b) ⊙ W` is `K^{-1/2} W` — CIQ applied to the cotangent.
    #[test]
    fn ciq_grad_b_equals_ciq_of_the_cotangent() {
        let dev = Device::Cpu;
        let n = 10;
        let a = spd(n, 1.5, 21);
        let rhs0 = rhs(n, 2);
        let rvar = candle_core::Var::from_tensor(&rhs0).unwrap();
        let w = Tensor::randn(0f64, 1f64, (n, 2), &dev).unwrap();

        let y = ciq_inv_sqrt_diff(&a, rvar.as_tensor(), &grad_opts()).unwrap();
        let loss = y.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_r = grads.get(rvar.as_tensor()).unwrap();

        let want = ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &w, &grad_opts()).unwrap();
        let rel = max_rel(g_r, &want);
        assert!(rel < 1e-6, "b̄ vs independent CIQ of the cotangent: rel {rel:.3e}");
    }

    #[test]
    fn ciq_backward_is_finite_at_high_condition_number() {
        let dev = Device::Cpu;
        let n = 16;
        // κ = 1e4 through the symmetric construction (diag of BBᵀ scaled).
        let d: Vec<f64> = (0..n).map(|i| 1e-4f64.powf(1.0 - i as f64 / (n - 1) as f64)).collect();
        let mut kd = vec![0.0f64; n * n];
        for i in 0..n {
            kd[i * n + i] = d[i];
        }
        // small off-diagonal coupling, symmetry preserved, PD kept
        for i in 0..n - 1 {
            let c = 0.05 * (d[i] * d[i + 1]).sqrt();
            kd[i * n + i + 1] = c;
            kd[(i + 1) * n + i] = c;
        }
        let k0 = Tensor::from_vec(kd, (n, n), &dev).unwrap();
        let kvar = candle_core::Var::from_tensor(&k0).unwrap();
        let b = rhs(n, 1);
        let opts = CiqOptions {
            q: Some(24),
            tolerance: Some(1e-8),
            max_iter: Some(10_000),
            lanczos_iters: Some(n),
            ..Default::default()
        };
        let y = ciq_inv_sqrt_diff(kvar.as_tensor(), &b, &opts).unwrap();
        let grads = y.sum_all().unwrap().backward().unwrap();
        let g: Vec<f64> = grads
            .get(kvar.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "non-finite K̄ at κ=1e4");
        assert!(g.iter().any(|v| v.abs() > 0.0), "K̄ identically zero at κ=1e4");
    }

    /// Both multi-shift solvers through the FULL CIQ pipeline (same bounds,
    /// same contour, only the Krylov solver differs) must agree — forward
    /// and gradient. Every contour system is SPD, so any disagreement is a
    /// solver bug, not a modelling choice.
    #[test]
    fn msminres_and_mscg_agree_through_ciq() {
        let n = 20;
        let a = spd(n, 0.8, 33);
        let b = rhs(n, 2);
        let mk = |solver: CiqSolver| CiqOptions {
            q: Some(24),
            tolerance: Some(1e-10),
            max_iter: Some(2000),
            solver,
            ..Default::default()
        };
        let via_cg = ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &b, &mk(CiqSolver::MsCg)).unwrap();
        let via_mr =
            ciq_inv_sqrt_matmul(|x| a.matmul(x), n, &b, &mk(CiqSolver::MsMinres)).unwrap();
        let rel = max_rel(&via_mr, &via_cg);
        assert!(rel < 1e-7, "CIQ forward: msMINRES vs msCG rel {rel:.3e}");

        // Gradients through the CustomOp, per solver.
        let grads_for = |solver: CiqSolver| -> Vec<f64> {
            let kvar = candle_core::Var::from_tensor(&a).unwrap();
            let y = ciq_inv_sqrt_diff(kvar.as_tensor(), &b, &mk(solver)).unwrap();
            let grads = y.sum_all().unwrap().backward().unwrap();
            grads.get(kvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };
        let g_cg = grads_for(CiqSolver::MsCg);
        let g_mr = grads_for(CiqSolver::MsMinres);
        let scale = g_cg.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-12);
        let gdiff = g_cg.iter().zip(&g_mr).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max) / scale;
        assert!(gdiff < 1e-6, "CIQ K̄: msMINRES vs msCG rel {gdiff:.3e}");
    }

    // ---- lazy-operator autograd ----------------------------------------------

    /// Symbolic RBF + δI: `K = exp(−γ‖xᵢ−xⱼ‖²) + δ·I`, SPD for distinct
    /// points, with TWO learnable leaves behind three leaf positions (x sits
    /// behind both `vi` and `vj`).
    fn rbf_symbolic(x: &Tensor, gamma: &Tensor, delta: f64) -> crate::operators::SymbolicKernelOperator {
        use crate::operators::symbolic::LazyTensor;
        let xi = LazyTensor::vi(x);
        let xj = LazyTensor::vj(x);
        let diff = &xi - &xj;
        let d2 = (&diff * &diff).sum_last();
        let k = (&d2 * &LazyTensor::pm(gamma)).affine(-1.0, 0.0).exp();
        let k = &k + &LazyTensor::eye_identity(delta);
        crate::operators::SymbolicKernelOperator::new(k)
    }

    fn lazy_opts() -> CiqOptions {
        CiqOptions { q: Some(64), tolerance: Some(1e-11), max_iter: Some(4000), ..Default::default() }
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
    fn lazy_ciq_matches_the_dense_driver() {
        use crate::operators::LinearOperator;
        let (n, dim) = (12, 2);
        let x = points(n, dim);
        let gamma = Tensor::full(0.8f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let b = rhs(n, 2);

        let dense = op.to_dense().unwrap();
        let want = ciq_inv_sqrt_matmul(|v| dense.matmul(v), n, &b, &lazy_opts()).unwrap();
        let got = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &lazy_opts(), RootUse::SymmetricRoot).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-8, "lazy vs dense CIQ forward: rel {rel:.3e}");
    }

    /// The whole point of the phase: parameter gradients through K^{-1/2}
    /// with K never densified — checked against central finite differences
    /// for BOTH the kernel hyperparameter (γ) and the inputs (x, which sits
    /// behind two leaf positions, so this also pins the cat-VJP summing).
    #[test]
    fn lazy_ciq_gradcheck_against_central_differences() {
        let (n, dim) = (8, 2);
        let x0 = points(n, dim);
        let g0 = Tensor::full(0.6f64, (1, 1), &Device::Cpu).unwrap();
        let xvar = candle_core::Var::from_tensor(&x0).unwrap();
        let gvar = candle_core::Var::from_tensor(&g0).unwrap();
        let b = rhs(n, 2);
        let w = Tensor::from_vec(
            (0..n * 2).map(|i| ((i * 13 % 7) as f64) / 3.0 - 1.0).collect::<Vec<_>>(),
            (n, 2),
            &Device::Cpu,
        )
        .unwrap();

        let loss_of = |xt: &Tensor, gt: &Tensor| -> Result<f64> {
            let op = rbf_symbolic(xt, gt, 0.5);
            let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &lazy_opts(), RootUse::SymmetricRoot)?;
            y.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let op = rbf_symbolic(xvar.as_tensor(), gvar.as_tensor(), 0.5);
        let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &lazy_opts(), RootUse::SymmetricRoot).unwrap();
        let loss = y.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_x = grads.get(xvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_g = grads.get(gvar.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_x = finite_diff(&xvar, &|t| loss_of(t, gvar.as_tensor()), 1e-5).unwrap();
        let fd_g = finite_diff(&gvar, &|t| loss_of(xvar.as_tensor(), t), 1e-5).unwrap();

        for (i, (ga, gf)) in g_x.iter().zip(&fd_x).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "x grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
        for (i, (ga, gf)) in g_g.iter().zip(&fd_g).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "gamma grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
    }

    #[test]
    fn lazy_ciq_rhs_gradient_is_ciq_of_the_cotangent() {
        use crate::operators::LinearOperator;
        let (n, dim) = (10, 2);
        let x = points(n, dim);
        let gamma = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let dense = op.to_dense().unwrap();

        let rvar = candle_core::Var::from_tensor(&rhs(n, 2)).unwrap();
        let w = Tensor::from_vec(
            (0..n * 2).map(|i| ((i * 11 % 9) as f64) / 4.0 - 1.0).collect::<Vec<_>>(),
            (n, 2),
            &Device::Cpu,
        )
        .unwrap();
        let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), rvar.as_tensor(), &lazy_opts(), RootUse::SymmetricRoot).unwrap();
        let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g_r = grads.get(rvar.as_tensor()).unwrap();
        let want = ciq_inv_sqrt_matmul(|v| dense.matmul(v), n, &w, &lazy_opts()).unwrap();
        let rel = max_rel(g_r, &want);
        assert!(rel < 1e-6, "lazy rhs̄ vs CIQ of cotangent: rel {rel:.3e}");
    }

    // ---- the packed-hyper seam (closure kernels) -----------------------------

    /// RBF-with-jitter as a CLOSURE over (x1, x2, hyper=[γ, δ]) — built
    /// entirely from candle ops, with the diagonal term derived from the
    /// zero-distance indicator so it stays correct under blocked evaluation.
    fn hyper_rbf() -> crate::operators::HyperKernelFn {
        std::sync::Arc::new(|a: &Tensor, b: &Tensor, h: &Tensor, diag: bool| {
            let gamma = h.narrow(0, 0, 1)?.reshape(())?;
            let delta = h.narrow(0, 1, 1)?.reshape(())?;
            if diag {
                // k(x, x) = 1 + δ per point.
                let n = a.dim(candle_core::D::Minus2)?;
                let ones = Tensor::ones(n, a.dtype(), a.device())?;
                return ones.broadcast_add(&delta.broadcast_add(&ones.zeros_like()?)?);
            }
            // [.., N, 1, d] − [.., 1, M, d] (candle's D has no Minus3, so
            // the new axes are addressed by rank).
            let d2 = a
                .unsqueeze(a.rank() - 1)?
                .broadcast_sub(&b.unsqueeze(b.rank() - 2)?)?
                .sqr()?
                .sum(candle_core::D::Minus1)?; // [.., N, M]
            let k = d2.broadcast_mul(&gamma)?.neg()?.exp()?;
            // δ on EXACT zero distance — the block-safe identity.
            let mask = d2.le(0.0)?.to_dtype(a.dtype())?;
            k.broadcast_add(&mask.broadcast_mul(&delta)?)
        })
    }

    #[test]
    fn hyper_kernel_matmul_matches_its_dense_form_blocked_and_not() {
        use crate::operators::{HyperKernelOperator, LinearOperator};
        let n = 12;
        let x = points(n, 2);
        let hyper = Tensor::from_vec(vec![0.7f64, 0.4], (2,), &Device::Cpu).unwrap();
        let b = rhs(n, 2);
        let full = HyperKernelOperator::new(x.clone(), x.clone(), hyper.clone(), hyper_rbf());
        let blocked = full.clone().with_block_rows(5);
        let dense = LinearOperator::to_dense(&full).unwrap();
        let want = dense.matmul(&b).unwrap();
        for (name, op) in [("one-shot", &full), ("blocked", &blocked)] {
            let got = LinearOperator::matmul(op, &b).unwrap();
            let rel = max_rel(&got, &want);
            assert!(rel < 1e-12, "{name}: hyper-kernel matmul off by {rel:.3e}");
        }
        // The block-safe diagonal really is global: dense must carry δ on
        // the true diagonal only.
        let diag = LinearOperator::diagonal(&full).unwrap();
        let d0: f64 = diag.get(0).unwrap().to_scalar().unwrap();
        assert!((d0 - 1.4).abs() < 1e-12, "diag(K)[0] = {d0}, want 1 + δ");
    }

    /// The seam's whole purpose: closure-kernel hyperparameters get CIQ
    /// gradients, checked against central finite differences. `x` rides as
    /// BOTH x1 and x2 (two pack entries, one tensor — cat's VJP sums), the
    /// hypers through the explicit packed argument.
    #[test]
    fn hyper_kernel_ciq_gradcheck_against_central_differences() {
        use crate::operators::HyperKernelOperator;
        let (n, dim) = (8, 2);
        let x0 = points(n, dim);
        let h0 = Tensor::from_vec(vec![0.6f64, 0.5], (2,), &Device::Cpu).unwrap();
        let xv = candle_core::Var::from_tensor(&x0).unwrap();
        let hv = candle_core::Var::from_tensor(&h0).unwrap();
        let b = rhs(n, 2);
        let w = Tensor::from_vec(
            (0..n * 2).map(|i| ((i * 7 % 5) as f64) / 2.0 - 1.0).collect::<Vec<_>>(),
            (n, 2),
            &Device::Cpu,
        )
        .unwrap();

        let loss_of = |xt: &Tensor, ht: &Tensor| -> Result<f64> {
            let op = HyperKernelOperator::new(xt.clone(), xt.clone(), ht.clone(), hyper_rbf())
                .with_block_rows(3);
            let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &lazy_opts(), RootUse::SymmetricRoot)?;
            y.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let op = HyperKernelOperator::new(
            xv.as_tensor().clone(),
            xv.as_tensor().clone(),
            hv.as_tensor().clone(),
            hyper_rbf(),
        )
        .with_block_rows(3);
        let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &lazy_opts(), RootUse::SymmetricRoot).unwrap();
        let loss = y.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_x = grads.get(xv.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_h = grads.get(hv.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_x = finite_diff(&xv, &|t| loss_of(t, hv.as_tensor()), 1e-5).unwrap();
        let fd_h = finite_diff(&hv, &|t| loss_of(xv.as_tensor(), t), 1e-5).unwrap();
        for (i, (ga, gf)) in g_x.iter().zip(&fd_x).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "x grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
        for (i, (ga, gf)) in g_h.iter().zip(&fd_h).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-4 * (1.0 + gf.abs()),
                "hyper grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
    }

    /// **The test that proves the conjugation is REACHED and COMPLETE.**
    ///
    /// An invariance test proves nothing here — an IGNORED preconditioner also
    /// leaves the answer unchanged. This test distinguishes the paths
    /// because `L(θ) = ‖ciq_inv_sqrt_lazy(op, b, P)‖²` is root-invariant and
    /// therefore equals `bᵀK(θ)⁻¹b` for EVERY `P`:
    ///
    /// ```text
    /// ‖K̂^{-1/2}E⁻¹b‖² = bᵀE⁻¹K̂⁻¹E⁻¹b = bᵀE⁻¹(E⁻¹KE⁻¹)⁻¹E⁻¹b = bᵀK⁻¹b
    /// ```
    ///
    /// So its gradient must match `inv_quad_lazy`'s — computed by a completely
    /// different code path in `linsolve.rs`, which is why the two cannot drift
    /// in sympathy.
    ///
    /// And crucially it catches PARTIAL conjugation. Omit any one of the three
    /// `E⁻¹` insertions and the `u`/`v` blocks stop being consistent with the
    /// solve, so the gradient moves. An ignored preconditioner passes; a
    /// half-applied one does not.
    ///
    /// Verified by mutation: each of the three insertions was deleted in turn
    /// and this test plus `the_preconditioned_rhs_gradient_is_two_k_inverse_b`
    /// caught all three. Keep BOTH — this one alone missed the `b̄` deletion.
    #[test]
    fn preconditioned_lazy_ciq_matches_the_inv_quad_oracle() {
        use crate::operators::linsolve::inv_quad_lazy;
        use crate::operators::precond::CiqPreconditioner;

        let n = 8;
        let x0 = points(n, 2);
        let g0 = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let b = rhs(n, 1);

        // The gradient the CIQ path must reproduce, from linsolve's own node.
        let gv = candle_core::Var::from_tensor(&g0).unwrap();
        let op = rbf_symbolic(&x0, gv.as_tensor(), 0.5);
        let oracle = inv_quad_lazy(
            std::sync::Arc::new(op),
            &b,
            &CgOptions::for_tolerance(1e-12),
        )
        .unwrap()
        .sum_all()
        .unwrap();
        let want = gv_grad(&gv, &oracle);

        // Four preconditioners, including a deliberately BAD `E` — the result
        // is root-invariant, so quality changes iteration counts and nothing
        // else. Any of them producing a different gradient means the
        // conjugation is inconsistent somewhere.
        let dense = crate::operators::LinearOperator::to_dense(&rbf_symbolic(&x0, &g0, 0.5)).unwrap();
        let diag = crate::operators::dense_diagonal(&dense).unwrap();
        let cands: Vec<(&str, Option<std::sync::Arc<CiqPreconditioner>>)> = vec![
            ("none", None),
            (
                "jacobi",
                Some(std::sync::Arc::new(CiqPreconditioner::jacobi(&diag, 1e-12).unwrap())),
            ),
            (
                "pivoted",
                Some(std::sync::Arc::new(
                    CiqPreconditioner::pivoted(|v| dense.matmul(v), &diag, 4).unwrap(),
                )),
            ),
            (
                "deliberately-bad",
                Some(std::sync::Arc::new(
                    CiqPreconditioner::jacobi(&diag.affine(9.0, 0.0).unwrap(), 1e-12).unwrap(),
                )),
            ),
        ];

        for (name, pre) in cands {
            let gv = candle_core::Var::from_tensor(&g0).unwrap();
            let op = rbf_symbolic(&x0, gv.as_tensor(), 0.5);
            let opts = CiqOptions { precond: pre, ..lazy_opts() };
            let y = ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &b, &opts, RootUse::AnyRoot).unwrap();
            let l = y.sqr().unwrap().sum_all().unwrap();

            // Value first: root-invariance is what makes the gradient claim
            // meaningful, so check it holds before checking the derivative.
            let lv = l.to_scalar::<f64>().unwrap();
            let ov = oracle.to_scalar::<f64>().unwrap();
            assert!(
                (lv - ov).abs() < 1e-6 * (1.0 + ov.abs()),
                "{name}: ‖R⁻¹b‖² = {lv} but bᵀK⁻¹b = {ov}"
            );

            let got = gv_grad(&gv, &l);
            for (i, (a, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - w).abs() < 2e-4 * (1.0 + w.abs()),
                    "{name}: grad[{i}] = {a} vs inv_quad oracle {w} — the conjugation is \
                     inconsistent (a partially-applied E⁻¹ looks exactly like this)"
                );
            }
        }
    }

    /// The RHS half of the same claim, and it needs its own test.
    ///
    /// A mutation check found the gap: dropping the `E⁻¹` on `b̄` leaves
    /// `preconditioned_lazy_ciq_matches_the_inv_quad_oracle` GREEN, because
    /// that test differentiates a kernel hyperparameter and holds `b` fixed.
    /// Three `E⁻¹` insertions, and only two of them were covered.
    ///
    /// Oracle: `L = ‖R⁻¹b‖² = bᵀK⁻¹b`, so `∂L/∂b = 2K⁻¹b` — independent of the
    /// preconditioner, and computable densely.
    #[test]
    fn the_preconditioned_rhs_gradient_is_two_k_inverse_b() {
        use crate::operators::precond::CiqPreconditioner;

        let n = 8;
        let x0 = points(n, 2);
        let g0 = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let b0 = rhs(n, 1);

        let dense = crate::operators::LinearOperator::to_dense(&rbf_symbolic(&x0, &g0, 0.5)).unwrap();
        let diag = crate::operators::dense_diagonal(&dense).unwrap();
        // 2 K⁻¹ b, by a dense route that shares no code with CIQ.
        let want: Vec<f64> = {
            let (x, _) = crate::operators::cg::linear_cg(
                |v: &Tensor| dense.matmul(v),
                &b0,
                None,
                None::<fn(&Tensor) -> Result<Tensor>>,
                &CgOptions::for_tolerance(1e-13),
            )
            .unwrap();
            x.affine(2.0, 0.0).unwrap().flatten_all().unwrap().to_vec1().unwrap()
        };

        for (name, pre) in [
            ("none", None),
            (
                "jacobi",
                Some(std::sync::Arc::new(CiqPreconditioner::jacobi(&diag, 1e-12).unwrap())),
            ),
            (
                "pivoted",
                Some(std::sync::Arc::new(
                    CiqPreconditioner::pivoted(|v| dense.matmul(v), &diag, 4).unwrap(),
                )),
            ),
        ] {
            let bv = candle_core::Var::from_tensor(&b0).unwrap();
            let op = rbf_symbolic(&x0, &g0, 0.5);
            let opts = CiqOptions { precond: pre, ..lazy_opts() };
            let y = ciq_inv_sqrt_lazy(
                std::sync::Arc::new(op),
                bv.as_tensor(),
                &opts,
                RootUse::AnyRoot,
            )
            .unwrap();
            let got = gv_grad(&bv, &y.sqr().unwrap().sum_all().unwrap());
            for (i, (a, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - w).abs() < 2e-4 * (1.0 + w.abs()),
                    "{name}: rhs grad[{i}] = {a} vs 2K⁻¹b = {w}"
                );
            }
        }
    }

    /// `SymmetricRoot` still refuses, and the message says what to do instead.
    #[test]
    fn a_symmetric_root_consumer_still_refuses_a_preconditioner() {
        let x = points(6, 2);
        let gamma = Tensor::full(0.5f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let d = crate::operators::dense_diagonal(
            &crate::operators::LinearOperator::to_dense(&op).unwrap(),
        )
        .unwrap();
        let pre = crate::operators::precond::CiqPreconditioner::jacobi(&d, 1e-12).unwrap();
        let opts = CiqOptions { precond: Some(std::sync::Arc::new(pre)), ..Default::default() };
        let e = ciq_inv_sqrt_lazy(
            std::sync::Arc::new(op),
            &rhs(6, 1),
            &opts,
            RootUse::SymmetricRoot,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("AnyRoot"), "the refusal must name the way out: {e}");
    }

    fn gv_grad(v: &candle_core::Var, loss: &Tensor) -> Vec<f64> {
        loss.backward()
            .unwrap()
            .get(v.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
    }

    #[test]
    fn lazy_ciq_refuses_preconditioners_and_leafless_operators() {
        let x = points(6, 2);
        let gamma = Tensor::full(0.5f64, (1, 1), &Device::Cpu).unwrap();
        let op = rbf_symbolic(&x, &gamma, 0.5);
        let d = crate::operators::dense_diagonal(&crate::operators::LinearOperator::to_dense(&op).unwrap()).unwrap();
        let pre = crate::operators::precond::CiqPreconditioner::jacobi(&d, 1e-12).unwrap();
        let opts = CiqOptions { precond: Some(std::sync::Arc::new(pre)), ..Default::default() };
        assert!(ciq_inv_sqrt_lazy(std::sync::Arc::new(op), &rhs(6, 1), &opts, RootUse::SymmetricRoot).is_err());
    }

    // ---- shift-invariant preconditioning ------------------------------------

    /// Low-rank-plus-diagonal SPD — the ill-conditioned shape the pivoted
    /// preconditioner is built for.
    fn lowrank_spd(n: usize, p: usize, delta: f64, seed: u64) -> Tensor {
        let dev = Device::Cpu;
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let l: Vec<f64> = (0..n * p).map(|_| 3.0 * next()).collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = if i == j { delta } else { 0.0 };
                for k in 0..p {
                    acc += l[i * p + k] * l[j * p + k];
                }
                a[i * n + j] = acc;
            }
        }
        Tensor::from_vec(a, (n, n), &dev).unwrap()
    }

    /// Per-column `xᵀK⁻¹x` through the dense host Cholesky — the oracle both
    /// root-validity invariants reduce to.
    fn inv_quad_cols(a: &Tensor, x: &Tensor) -> Vec<f64> {
        let solve = crate::linalg::cholesky_solve(a, x, 0.0).unwrap();
        let q = (x * &solve).unwrap().sum(candle_core::D::Minus2).unwrap();
        q.flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap()
    }

    /// With a preconditioner, CIQ applies the CONJUGATED root
    /// `R = E·K̂^{1/2}` — different entries from the symmetric root, but
    /// `RRᵀ = K`, which is what these two invariants test (they hold for ANY
    /// valid root):  `‖R⁻¹x‖² = xᵀK⁻¹x`  and  `(Rx)ᵀK⁻¹(Rx) = ‖x‖²`.
    #[test]
    fn preconditioned_ciq_applies_a_valid_root() {
        let n = 40;
        let a = lowrank_spd(n, 3, 1e-4, 11); // κ ~ 1e6
        let d = crate::operators::dense_diagonal(&a).unwrap();
        let pre = crate::operators::precond::CiqPreconditioner::pivoted(|x| a.matmul(x), &d, 5)
            .unwrap();
        let opts = CiqOptions {
            q: Some(24),
            tolerance: Some(1e-10),
            max_iter: Some(2000),
            precond: Some(std::sync::Arc::new(pre)),
            ..Default::default()
        };
        let x = rhs(n, 2);

        // ‖R⁻¹x‖² = xᵀK⁻¹x, per column.
        let y = ciq_inv_sqrt_matmul(|v| a.matmul(v), n, &x, &opts).unwrap();
        let got: Vec<f64> = y
            .sqr().unwrap().sum(candle_core::D::Minus2).unwrap()
            .flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        for (g, w) in got.iter().zip(inv_quad_cols(&a, &x)) {
            assert!((g - w).abs() < 1e-5 * w.max(1.0), "‖R⁻¹x‖² {g} vs xᵀK⁻¹x {w}");
        }

        // (Rx)ᵀK⁻¹(Rx) = ‖x‖², per column.
        let z = ciq_sqrt_matmul(|v| a.matmul(v), n, &x, &opts).unwrap();
        let want_norms: Vec<f64> = x
            .sqr().unwrap().sum(candle_core::D::Minus2).unwrap()
            .flatten_all().unwrap().to_dtype(DType::F64).unwrap().to_vec1().unwrap();
        for (g, w) in inv_quad_cols(&a, &z).iter().zip(want_norms) {
            assert!((g - w).abs() < 1e-5 * w.max(1.0), "(Rx)ᵀK⁻¹(Rx) {g} vs ‖x‖² {w}");
        }

        // R(R⁻¹x) = x.
        let round = ciq_sqrt_matmul(|v| a.matmul(v), n, &y, &opts).unwrap();
        let rel = max_rel(&round, &x);
        assert!(rel < 1e-5, "R R⁻¹ x ≠ x: rel {rel:.3e}");
    }

    /// The point of the whole phase: at a FIXED small iteration budget on a
    /// κ~1e6 operator, the conjugated pipeline (κ(K̂) ~ 1) satisfies the
    /// root invariant while the unpreconditioned run — needing O(√κ)
    /// iterations it doesn't have — visibly does not. Both arms are judged
    /// by the SAME invariant, which any valid root must satisfy.
    #[test]
    fn preconditioning_wins_at_a_fixed_iteration_budget() {
        let n = 48;
        let a = lowrank_spd(n, 3, 1e-4, 17);
        let d = crate::operators::dense_diagonal(&a).unwrap();
        let x = rhs(n, 1);
        let want = inv_quad_cols(&a, &x)[0];

        let budget = 25usize;
        let base = CiqOptions {
            q: Some(20),
            tolerance: Some(1e-12),
            max_iter: Some(budget),
            lanczos_iters: Some(20),
            ..Default::default()
        };
        let err_of = |opts: &CiqOptions| -> f64 {
            let y = ciq_inv_sqrt_matmul(|v| a.matmul(v), n, &x, opts).unwrap();
            let got: f64 = y
                .sqr().unwrap().sum_all().unwrap().to_dtype(DType::F64).unwrap().to_scalar().unwrap();
            (got - want).abs() / want
        };

        let pre = crate::operators::precond::CiqPreconditioner::pivoted(|v| a.matmul(v), &d, 5)
            .unwrap();
        let with = err_of(&CiqOptions {
            precond: Some(std::sync::Arc::new(pre)),
            ..base.clone()
        });
        let without = err_of(&base);
        assert!(with < 1e-4, "preconditioned run inaccurate at budget {budget}: {with:.3e}");
        assert!(
            without > 10.0 * with.max(1e-12) && without > 1e-3,
            "unpreconditioned run unexpectedly fine at budget {budget}: {without:.3e} \
             (preconditioned: {with:.3e}) — the fixture no longer stresses κ"
        );
    }

    #[test]
    fn the_differentiable_wrapper_refuses_a_preconditioner() {
        let n = 8;
        let a = spd(n, 1.0, 3);
        let d = crate::operators::dense_diagonal(&a).unwrap();
        let pre = crate::operators::precond::CiqPreconditioner::jacobi(&d, 1e-12).unwrap();
        let opts = CiqOptions { precond: Some(std::sync::Arc::new(pre)), ..Default::default() };
        assert!(ciq_inv_sqrt_diff(&a, &rhs(n, 1), &opts).is_err());
    }

    /// The κ sweep on matrices, with Q from the Phase-1 scalar table. κ=1e6
    /// is deliberately absent: unpreconditioned CG needs O(√κ) iterations
    /// and the scalar table already pins the quadrature there — this test
    /// pins the plumbing, not the rule.
    #[test]
    fn ciq_error_tracks_the_scalar_table_across_condition_numbers() {
        let dev = Device::Cpu;
        for &(kappa, q) in &[(1e2f64, 12usize), (1e4, 16)] {
            let n = 24;
            // Diagonal with a geometric spectrum in [1/κ, 1].
            let d: Vec<f64> =
                (0..n).map(|i| (1.0 / kappa).powf(1.0 - i as f64 / (n - 1) as f64)).collect();
            let diag = Tensor::from_vec(d.clone(), (n, 1), &dev).unwrap();
            let dmat = |x: &Tensor| -> Result<Tensor> { x.broadcast_mul(&diag) };
            let b = rhs(n, 1);
            let opts = CiqOptions {
                q: Some(q),
                tolerance: Some(1e-10),
                max_iter: Some(20_000),
                lanczos_iters: Some(n),
                ..Default::default()
            };
            let got = ciq_inv_sqrt_matmul(dmat, n, &b, &opts).unwrap();
            // Exact answer elementwise: b_i / √d_i.
            let want = b.broadcast_div(&diag.sqrt().unwrap()).unwrap();
            let rel = max_rel(&got, &want);
            assert!(rel < 1e-3, "κ={kappa:.0e} Q={q}: rel err {rel:.3e}");
        }
    }
}
