//! Inferno — Preconditioners for iterative tensor linear solvers.

use candle_core::{DType, Result, Tensor, D};

/// Rank-`p` pivoted (partial) Cholesky of an SPD operator given only
/// matvec access and the diagonal: returns `(L [.., n, p'], resid_diag
/// [.., n])` with `K ≈ L Lᵀ` and `p' ≤ rank` (early exit once the largest
/// residual diagonal falls below `rel_tol ×` its starting value).
///
/// Each step costs ONE matvec (the pivot column is `K e_i`). Batched
/// operators share one pivot order, chosen from the batch-MEAN residual
/// diagonal — a shared-pivot `L` preconditions some batch elements less
/// sharply, but any SPD `M` yields a valid conjugated root, so this trades
/// preconditioning quality, never correctness.
pub fn pivoted_cholesky<F>(
    matmul: F,
    diag: &Tensor,
    rank: usize,
    rel_tol: f64,
) -> Result<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let dims = diag.dims().to_vec();
    let n = dims[dims.len() - 1];
    let rank = rank.min(n).max(1);
    let dev = diag.device();
    let dtype = diag.dtype();

    let mut d = diag.clone(); // residual diagonal [.., n]
    let mut cols: Vec<Tensor> = Vec::with_capacity(rank);
    let mut used = vec![false; n];
    let mut initial_best: Option<f64> = None;

    for _ in 0..rank {
        // Pivot from the batch-mean residual diagonal, on the host.
        let mut dm = d.clone();
        while dm.rank() > 1 {
            dm = dm.mean(0)?;
        }
        let host: Vec<f64> = dm.to_dtype(DType::F64)?.to_vec1()?;
        let (mut best_i, mut best_v) = (usize::MAX, f64::NEG_INFINITY);
        for (i, &v) in host.iter().enumerate() {
            if !used[i] && v > best_v {
                best_i = i;
                best_v = v;
            }
        }
        let start = *initial_best.get_or_insert(best_v.max(f64::MIN_POSITIVE));
        if best_i == usize::MAX || best_v <= rel_tol * start {
            break;
        }
        used[best_i] = true;

        // col = K e_i  −  L (L[i, :])ᵀ, then scale by 1/√d_i per batch.
        let mut e = vec![0.0f64; n];
        e[best_i] = 1.0;
        let e = Tensor::from_vec(e, (n, 1), dev)?.to_dtype(dtype)?;
        let mut col = matmul(&e)?; // [.., n, 1]
        if !cols.is_empty() {
            let l = Tensor::cat(&cols, D::Minus1)?; // [.., n, k]
            let row = l.narrow(D::Minus2, best_i, 1)?; // [.., 1, k]
            col =
                (col - l.broadcast_matmul(&row.transpose(D::Minus2, D::Minus1)?.contiguous()?)?)?;
        }
        let d_i = d.narrow(D::Minus1, best_i, 1)?.unsqueeze(D::Minus1)?; // [.., 1, 1]
                                                                         // Shared pivots can leave a batch element's d_i near zero — clamp
                                                                         // rather than divide into junk; that element simply gains a ~zero
                                                                         // column (weaker preconditioning, still a valid M).
        let floor = Tensor::full(1e-12 * start, d_i.shape(), dev)?.to_dtype(dtype)?;
        let denom = d_i.maximum(&floor)?.sqrt()?;
        let lk = col.broadcast_div(&denom)?;

        d = (d - lk.squeeze(D::Minus1)?.sqr()?)?;
        cols.push(lk);
    }

    if cols.is_empty() {
        return Err(candle_core::Error::Msg(
            "pivoted_cholesky: no usable pivot — diagonal is non-positive?".into(),
        ));
    }
    Ok((Tensor::cat(&cols, D::Minus1)?, d))
}

enum Kind {
    /// `M = diag(d)`; `E = diag(√d)`.
    Jacobi { e: Tensor, e_inv: Tensor }, // both [.., n]
    /// `M = δI + L Lᵀ` via the rank-p eigen identity.
    LowRank {
        l_tilde: Tensor, // [.., n, p]  (= L/√δ)
        w: Tensor,       // [.., p, p]  eigenvectors of L̃ᵀL̃ (column j = vec j)
        h_plus: Tensor,  // [.., p]     h_{+1/2}(λ)
        h_minus: Tensor, // [.., p]     h_{−1/2}(λ)
        h_inv: Tensor,   // [.., p]     h_{−1}(λ) — for `M⁻¹`, the CG form
        sqrt_delta: f64,
    },
    /// `M = δI + Q diag(μ − δ) Qᵀ` for an ORTHONORMAL `Q` — a spectral
    /// preconditioner over a Krylov (Ritz) basis.
    ///
    /// Why this is not just `LowRank` with a different `L`: because `Q` is
    /// already orthonormal there is no Gram to form and no `jacobi_eigh` to
    /// run, so the `p ≤ max_preconditioner_size` ceiling — which exists purely
    /// to bound that host eigensolve — does not apply. And where pivoted
    /// Cholesky picks its directions by residual-diagonal magnitude and so only
    /// ever finds DOMINANT ones, Lanczos converges at both extremes first: this
    /// is the variant that can deflate the small end, which is the end `√κ`
    /// is built from.
    Spectral {
        q: Tensor,       // [.., n, k]  orthonormal columns
        h_plus: Tensor,  // [.., k]     (μ/δ)^{+1/2} − 1
        h_minus: Tensor, // [.., k]     (μ/δ)^{−1/2} − 1
        h_inv: Tensor,   // [.., k]     (μ/δ)^{−1}   − 1
        sqrt_delta: f64,
    },
}

/// Which of the three `M` constructions a [`CiqPreconditioner`] holds.
///
/// A DISCRIMINANT, deliberately, rather than making `Kind` itself public:
/// publishing `Kind` would put `l_tilde`/`w`/`h_plus`/`h_minus`/`h_inv`/
/// `sqrt_delta`/`q` on the public API and freeze the rank-p eigen identity's
/// internal layout. Callers that need to record, cache or select a
/// preconditioner by kind want the label, not the payload.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrecondKind {
    /// `M = diag(d)` — the free baseline.
    Jacobi,
    /// `M = δI + LLᵀ` from pivoted Cholesky. Finds DOMINANT directions only,
    /// and its rank is capped by `settings::max_preconditioner_size` (the
    /// host `jacobi_eigh` bound).
    LowRank,
    /// `M = δI + Q diag(μ − δ) Qᵀ` over an orthonormal Krylov basis. The
    /// variant that can deflate the SMALL end, and the one no rank ceiling
    /// applies to.
    Spectral,
}

/// A symmetric positive-definite `E = M^{1/2}` with `O(np)` `E·x` and
/// `E⁻¹·x` — everything CIQ needs to run on the conjugated operator
/// `K̂ = E⁻¹KE⁻¹`. Built from detached numerics and treated as a CONSTANT by
/// any surrounding autograd, like every preconditioner.
pub struct CiqPreconditioner {
    kind: Kind,
}

impl CiqPreconditioner {
    /// Which construction this is — for cache keys, bake-off tables and
    /// telemetry. See [`PrecondKind`] for why this is a label and not the
    /// payload.
    pub fn kind(&self) -> PrecondKind {
        match &self.kind {
            Kind::Jacobi { .. } => PrecondKind::Jacobi,
            Kind::LowRank { .. } => PrecondKind::LowRank,
            Kind::Spectral { .. } => PrecondKind::Spectral,
        }
    }

    /// The rank of the correction — `0` for [`PrecondKind::Jacobi`], `p` or
    /// `k` otherwise.
    ///
    /// Worth checking at a call site: `pivoted_cholesky` early-exits when it
    /// runs out of usable pivots and `RecycleSubspace::as_preconditioner`
    /// declines below rank 2, so a `LowRank` arm can silently be built at a
    /// rank far below the one requested — which reads on a bake-off table as
    /// "preconditioning doesn't help".
    pub fn rank(&self) -> usize {
        match &self.kind {
            Kind::Jacobi { .. } => 0,
            Kind::LowRank { h_inv, .. } => h_inv.dims().last().copied().unwrap_or(0),
            Kind::Spectral { h_inv, .. } => h_inv.dims().last().copied().unwrap_or(0),
        }
    }
}

impl std::fmt::Debug for CiqPreconditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            Kind::Jacobi { e, .. } => write!(
                f,
                "CiqPreconditioner::Jacobi(n={})",
                e.dims().last().unwrap()
            ),
            Kind::LowRank {
                l_tilde,
                sqrt_delta,
                ..
            } => write!(
                f,
                "CiqPreconditioner::LowRank(rank={}, delta={:.3e})",
                l_tilde.dims().last().unwrap(),
                sqrt_delta * sqrt_delta
            ),
            // The μ range is the first thing to look at when it is not
            // helping: all-equal μ means `M ≈ δI` and the preconditioner is
            // silently the identity.
            Kind::Spectral {
                q,
                h_plus,
                sqrt_delta,
                ..
            } => {
                let d2 = sqrt_delta * sqrt_delta;
                let span = h_plus
                    .flatten_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::F64))
                    .and_then(|t| t.to_vec1::<f64>())
                    .map(|v| {
                        // h_plus = (μ/δ)^{1/2} − 1  ⇒  μ = δ(1+h)²
                        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
                        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        (d2 * (1.0 + lo).powi(2), d2 * (1.0 + hi).powi(2))
                    })
                    .unwrap_or((f64::NAN, f64::NAN));
                write!(
                    f,
                    "CiqPreconditioner::Spectral(k={}, delta={:.3e}, mu=[{:.3e}, {:.3e}])",
                    q.dims().last().unwrap(),
                    d2,
                    span.0,
                    span.1
                )
            }
        }
    }
}

impl CiqPreconditioner {
    /// Diagonal (Jacobi) preconditioner from a diagonal `[.., n]`, floored.
    /// Useful when `K` has genuinely heterogeneous scale on its diagonal; a
    /// constant-diagonal kernel makes this a no-op.
    pub fn jacobi(diag: &Tensor, floor: f64) -> Result<Self> {
        let fl = Tensor::full(floor.max(f64::MIN_POSITIVE), diag.shape(), diag.device())?
            .to_dtype(diag.dtype())?;
        let e = diag.maximum(&fl)?.sqrt()?;
        Ok(Self {
            kind: Kind::Jacobi {
                e_inv: e.recip()?,
                e,
            },
        })
    }

    /// Pivoted-Cholesky preconditioner: `M = δI + L Lᵀ` with `L` from
    /// [`pivoted_cholesky`] and `δ` = the mean of the (clamped-positive)
    /// residual diagonal — the Nyström-style choice that makes `M ≈ K` tight
    /// exactly when `K` is low-rank-plus-diagonal.
    pub fn pivoted<F>(matmul: F, diag: &Tensor, rank: usize) -> Result<Self>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let (l, resid) = pivoted_cholesky(matmul, diag, rank, 1e-8)?;
        let mean_resid: f64 = resid
            .relu()?
            .flatten_all()?
            .to_dtype(DType::F64)?
            .mean(0)?
            .to_scalar()?;
        let mean_diag: f64 = diag
            .flatten_all()?
            .to_dtype(DType::F64)?
            .mean(0)?
            .to_scalar()?;
        let delta = mean_resid.max(1e-8 * mean_diag.max(f64::MIN_POSITIVE));
        Self::low_rank(&l, delta)
    }

    /// `M = δI + Q diag(μ − δ) Qᵀ` for ORTHONORMAL `Q [.., n, k]`.
    ///
    /// `mu [.., k]` are the eigenvalues on `Q`'s columns — Ritz values from
    /// [`super::ciq::lanczos_probe`], or a fitted ladder over them — and `δ` is
    /// the eigenvalue assigned to the whole orthogonal complement.
    ///
    /// Every guard below is for a failure that is otherwise SILENT:
    ///
    /// * **Non-orthonormal `Q`.** Every closed form here assumes `QᵀQ = I`;
    ///   without it `M^α` is simply not `M` to the `α`, and no apply complains.
    ///   `sim::primitives::MassMatrix::LowRank` makes orthonormality "the
    ///   constructor's contract" for the same reason and `amy::mass` discharges
    ///   it with the same check.
    /// * **`μ_j ≤ 0`.** `M` is then indefinite or singular, and `inv_apply`
    ///   returns `inf` with no error.
    /// * **`μ_j ≈ δ` for every `j`.** `M ≈ δI` and the preconditioner is a
    ///   scalar — it "works", costs matvecs, and does nothing. Refused, because
    ///   `ciq_route_opts` already degrades a failed build to unpreconditioned
    ///   CIQ, so `Err` is the wired-up safe branch.
    pub fn spectral(q: &Tensor, mu: &Tensor, delta: f64, ortho_tol: f64) -> Result<Self> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::spectral: delta must be finite and positive, got {delta}"
            )));
        }
        let k = q.dim(D::Minus1)?;
        if mu.dim(D::Minus1)? != k {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::spectral: Q has {k} columns but mu has {}",
                mu.dim(D::Minus1)?
            )));
        }

        // QᵀQ = I, verified rather than assumed.
        let gram = q
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(q)?;
        let eye = Tensor::eye(k, gram.dtype(), gram.device())?;
        let defect = gram
            .broadcast_sub(&eye)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?;
        if !(defect <= ortho_tol) {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::spectral: ‖QᵀQ − I‖_∞ = {defect:.3e} exceeds {ortho_tol:.3e} \
                 — the basis is not orthonormal, and every closed form here is wrong without \
                 it (silently: the applies would still return finite, symmetric-looking values)"
            )));
        }

        let mu_h: Vec<f64> = mu.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
        if let Some(bad) = mu_h.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::spectral: eigenvalue {bad:.3e} is not finite and positive — \
                 M would be singular or indefinite and inv_apply would return inf"
            )));
        }
        let spread = mu_h
            .iter()
            .map(|m| (m / delta).ln().abs())
            .fold(0.0f64, f64::max);
        if spread < 1e-6 {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::spectral: every mu is within {spread:.3e} (log) of delta, so \
                 M is effectively delta*I and this preconditioner would cost matvecs and do \
                 nothing"
            )));
        }

        // h_α = (μ/δ)^α − 1. `expm1(α·ln r)` rather than `r.powf(α) − 1` for
        // the same reason `low_rank` uses it: near r = 1 the subtraction is
        // catastrophic cancellation and `expm1` is exact there.
        let ratio = mu.affine(1.0 / delta, 0.0)?;
        let ln_r = ratio.log()?;
        let h_of = |a: f64| -> Result<Tensor> { ln_r.affine(a, 0.0)?.exp()?.affine(1.0, -1.0) };

        Ok(Self {
            kind: Kind::Spectral {
                q: q.clone(),
                h_plus: h_of(0.5)?,
                h_minus: h_of(-0.5)?,
                h_inv: h_of(-1.0)?,
                sqrt_delta: delta.sqrt(),
            },
        })
    }

    /// `M = δI + L Lᵀ` from an explicit root `L [.., n, p]`, `δ > 0`.
    pub fn low_rank(l: &Tensor, delta: f64) -> Result<Self> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(candle_core::Error::Msg(format!(
                "CiqPreconditioner::low_rank: need delta > 0, got {delta}"
            )));
        }
        let sqrt_delta = delta.sqrt();
        let l_tilde = l.affine(1.0 / sqrt_delta, 0.0)?;
        let p = *l_tilde.dims().last().unwrap();

        // Gram [.., p, p] → per-batch host jacobi_eigh (Frobenius-prescaled;
        // its convergence test is absolute).
        let gram = l_tilde
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(&l_tilde)?;
        let gdims = gram.dims().to_vec();
        let bsz: usize = gdims[..gdims.len() - 2].iter().product::<usize>().max(1);
        let host: Vec<f64> = gram.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;

        let mut w_all = Vec::with_capacity(bsz * p * p);
        let mut hp_all = Vec::with_capacity(bsz * p);
        let mut hm_all = Vec::with_capacity(bsz * p);
        let mut hi_all = Vec::with_capacity(bsz * p);
        for b in 0..bsz {
            let g = &host[b * p * p..(b + 1) * p * p];
            let frob = g
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(f64::MIN_POSITIVE);
            let scaled: Vec<f64> = g.iter().map(|v| v / frob).collect();
            let (eigs, vecs) = crate::linalg::jacobi_eigh(&scaled, p);
            // h_α(λ) = expm1(α·ln1p(λ))/λ, stabilised at λ → 0 (→ α). The
            // Gram is PSD; tiny negative eigenvalues are jacobi noise.
            let h = |lam: f64, alpha: f64| -> f64 {
                let lam = lam.max(0.0);
                if lam < 1e-14 {
                    alpha
                } else {
                    ((alpha * lam.ln_1p()).exp_m1()) / lam
                }
            };
            for e in &eigs {
                let lam = e * frob;
                hp_all.push(h(lam, 0.5));
                hm_all.push(h(lam, -0.5));
                hi_all.push(h(lam, -1.0)); // = −1/(1+λ), exactly
            }
            w_all.extend_from_slice(&vecs); // eigenvector j in COLUMN j
        }

        let dev = l.device();
        let dtype = l.dtype();
        let mut wshape: Vec<usize> = gdims.clone();
        let mut hshape: Vec<usize> = gdims[..gdims.len() - 1].to_vec();
        wshape.truncate(gdims.len());
        hshape[gdims.len() - 2] = p;
        let w = Tensor::from_vec(w_all, wshape, dev)?.to_dtype(dtype)?;
        let h_plus = Tensor::from_vec(hp_all, hshape.clone(), dev)?.to_dtype(dtype)?;
        let h_minus = Tensor::from_vec(hm_all, hshape.clone(), dev)?.to_dtype(dtype)?;
        let h_inv = Tensor::from_vec(hi_all, hshape, dev)?.to_dtype(dtype)?;
        Ok(Self {
            kind: Kind::LowRank {
                l_tilde,
                w,
                h_plus,
                h_minus,
                h_inv,
                sqrt_delta,
            },
        })
    }

    fn low_rank_apply(
        l_tilde: &Tensor,
        w: &Tensor,
        h: &Tensor,
        scale: f64,
        x: &Tensor,
    ) -> Result<Tensor> {
        // scale · (x + L̃ W diag(h) Wᵀ L̃ᵀ x)
        let t = l_tilde
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(x)?;
        let t = w
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(&t)?;
        let t = t.broadcast_mul(&h.unsqueeze(D::Minus1)?)?;
        let t = w.broadcast_matmul(&t)?;
        let core = l_tilde.broadcast_matmul(&t)?;
        (x + core)?.affine(scale, 0.0)
    }

    /// `scale · (x + Q diag(h) Qᵀ x)` — three matmuls, no `W`.
    ///
    /// Deliberately NOT `low_rank_apply` with `W = I`: the identity matmuls
    /// would run, cost real time, and hide the fact that no eigendecomposition
    /// happened here. `spectral_apply_matches_low_rank_with_identity_w` pins
    /// the two together so this stays a specialisation rather than a divergence.
    ///
    /// Note the `h` here is `(μ/δ)^α − 1`, with **no `/λ` division and no
    /// `λ → 0` limit** — unlike `low_rank`'s `h_α(λ) = expm1(α·ln1p(λ))/λ`,
    /// whose correction is multiplicative in `λ` and needs the limit to stay
    /// finite. Do not go looking for the missing stabiliser; there is nothing
    /// to stabilise.
    fn spectral_apply(q: &Tensor, h: &Tensor, scale: f64, x: &Tensor) -> Result<Tensor> {
        let t = q
            .transpose(D::Minus2, D::Minus1)?
            .contiguous()?
            .broadcast_matmul(x)?;
        let t = t.broadcast_mul(&h.unsqueeze(D::Minus1)?)?;
        let core = q.broadcast_matmul(&t)?;
        (x + core)?.affine(scale, 0.0)
    }

    /// `E x = M^{1/2} x`, `x: [.., n, C]`.
    pub fn sqrt_apply(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            Kind::Jacobi { e, .. } => x.broadcast_mul(&e.unsqueeze(D::Minus1)?),
            Kind::LowRank {
                l_tilde,
                w,
                h_plus,
                sqrt_delta,
                ..
            } => Self::low_rank_apply(l_tilde, w, h_plus, *sqrt_delta, x),
            Kind::Spectral {
                q,
                h_plus,
                sqrt_delta,
                ..
            } => Self::spectral_apply(q, h_plus, *sqrt_delta, x),
        }
    }

    /// `E⁻¹ x = M^{-1/2} x`, `x: [.., n, C]`.
    pub fn inv_sqrt_apply(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            Kind::Jacobi { e_inv, .. } => x.broadcast_mul(&e_inv.unsqueeze(D::Minus1)?),
            Kind::LowRank {
                l_tilde,
                w,
                h_minus,
                sqrt_delta,
                ..
            } => Self::low_rank_apply(l_tilde, w, h_minus, 1.0 / sqrt_delta, x),
            Kind::Spectral {
                q,
                h_minus,
                sqrt_delta,
                ..
            } => Self::spectral_apply(q, h_minus, 1.0 / sqrt_delta, x),
        }
    }

    /// `log det M`, exactly and in closed form — `[..]` (batch shape).
    ///
    /// This is what turns the preconditioned-SLQ refusal into a capability. A
    /// preconditioned CG tridiagonalises `M^{-1/2}AM^{-1/2}`, so the quadrature
    /// returns `logdet(A) − logdet(M)`; the correction was "unavailable" only
    /// because nothing computed it.
    ///
    /// Nothing new is stored — both structured kinds carry `λ` implicitly in
    /// the coefficients they already hold:
    ///
    /// ```text
    /// LowRank:  h_{−1}(λ) = −1/(1+λ)          ⇒ ln(1+λ) = −ln(−h_inv)
    ///           logdet = n·ln δ − Σ_j ln(−h_inv_j)
    /// Spectral: h_{+1/2}(μ) = (μ/δ)^{1/2} − 1 ⇒ ln μ = ln δ + 2·ln(1+h_plus)
    ///           logdet = n·ln δ + 2·Σ_j ln(1 + h_plus_j)
    /// ```
    ///
    /// Both land on `n·ln δ` plus a rank-sized correction, which is the same
    /// statement as "`δ` is the eigenvalue on the whole orthogonal complement".
    pub fn log_det(&self) -> Result<Tensor> {
        match &self.kind {
            // M = diag(e²).
            Kind::Jacobi { e, .. } => e.log()?.sum(D::Minus1)?.affine(2.0, 0.0),
            Kind::LowRank {
                l_tilde,
                h_inv,
                sqrt_delta,
                ..
            } => {
                let n = l_tilde.dim(D::Minus2)? as f64;
                let n_ln_delta = n * (sqrt_delta * sqrt_delta).ln();
                // −Σ ln(−h_inv). `h_inv` is strictly negative for λ ≥ 0.
                let corr = h_inv.neg()?.log()?.sum(D::Minus1)?;
                corr.affine(-1.0, n_ln_delta)
            }
            Kind::Spectral {
                q,
                h_plus,
                sqrt_delta,
                ..
            } => {
                let n = q.dim(D::Minus2)? as f64;
                let n_ln_delta = n * (sqrt_delta * sqrt_delta).ln();
                let corr = h_plus.affine(1.0, 1.0)?.log()?.sum(D::Minus1)?;
                corr.affine(2.0, n_ln_delta)
            }
        }
    }

    /// `M⁻¹ x`, `x: [.., n, C]` — the preconditioner a PLAIN CG wants.
    ///
    /// Distinct from the `E`/`E⁻¹` pair above, and the distinction is the
    /// whole reason this module exists: a multi-shift solve cannot take
    /// `M⁻¹` at all (see the module header), so it CONJUGATES with
    /// `E = M^{1/2}`. Single-shift [`super::cg::linear_cg`] has no such
    /// objection and takes `M⁻¹` directly, which is one apply per iteration
    /// instead of two.
    ///
    /// Same rank-`p` identity at `α = −1`, where `h_{−1}(λ) = −1/(1+λ)`
    /// exactly — no cancellation to stabilise, unlike the `±1/2` cases.
    pub fn inv_apply(&self, x: &Tensor) -> Result<Tensor> {
        match &self.kind {
            // M = diag(e²) ⇒ M⁻¹ = diag(e_inv²).
            Kind::Jacobi { e_inv, .. } => x.broadcast_mul(&e_inv.sqr()?.unsqueeze(D::Minus1)?),
            Kind::LowRank {
                l_tilde,
                w,
                h_inv,
                sqrt_delta,
                ..
            } => Self::low_rank_apply(l_tilde, w, h_inv, 1.0 / (sqrt_delta * sqrt_delta), x),
            Kind::Spectral {
                q,
                h_inv,
                sqrt_delta,
                ..
            } => Self::spectral_apply(q, h_inv, 1.0 / (sqrt_delta * sqrt_delta), x),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Verification of preconditioners for iterative tensor linear solvers.

    use super::*;
    use candle_core::Device;

    /// Low-rank-plus-diagonal SPD: the shape pivoted Cholesky is built for.
    fn lowrank_spd(n: usize, p: usize, delta: f64, seed: u64) -> (Tensor, Tensor) {
        let dev = Device::Cpu;
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        let a = Tensor::from_vec(a, (n, n), &dev).unwrap();
        let d = crate::operators::dense_diagonal(&a).unwrap();
        (a, d)
    }

    fn frob(t: &Tensor) -> f64 {
        t.sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
            .sqrt()
    }

    #[test]
    fn pivoted_cholesky_error_decreases_with_rank_and_nails_the_true_rank() {
        let (a, d) = lowrank_spd(40, 3, 1e-4, 5);
        let mm = |x: &Tensor| a.matmul(x);
        let mut prev = f64::INFINITY;
        for rank in [1usize, 2, 3] {
            let (l, _) = pivoted_cholesky(mm, &d, rank, 0.0).unwrap();
            let err = frob(&(&a - l.matmul(&l.t().unwrap()).unwrap()).unwrap());
            assert!(
                err < prev,
                "rank {rank}: error {err:.3e} did not decrease from {prev:.3e}"
            );
            prev = err;
        }
        // At the true rank the residual is the δI floor plus rounding.
        assert!(
            prev < 1e-2 * frob(&a),
            "rank-3 residual {prev:.3e} too large"
        );
    }

    /// `inv_apply` must be the true `M⁻¹`, i.e. the square of `E⁻¹` — checked
    /// against the composition rather than re-deriving the identity, since
    /// the point is that the `α = −1` coefficients agree with the `α = −1/2`
    /// ones already trusted by CIQ. Covers both kinds: a `Jacobi` bug here
    /// (squaring `e_inv` vs not) is invisible when the diagonal is near 1.
    #[test]
    fn inv_apply_is_the_square_of_inv_sqrt_apply() {
        let dev = Device::Cpu;
        let (a, d) = lowrank_spd(24, 3, 1e-2, 11);
        let x = Tensor::from_vec(
            (0..24 * 2)
                .map(|i| ((i * 23 % 7) as f64) / 3.0 - 1.0)
                .collect::<Vec<_>>(),
            (24, 2),
            &dev,
        )
        .unwrap();

        // Extended rather than duplicated for `Spectral`, so symmetry,
        // roundtrip and the inverse identity cannot drift apart between the
        // three kinds.
        for (name, p) in [
            (
                "pivoted",
                CiqPreconditioner::pivoted(|v| a.matmul(v), &d, 5).unwrap(),
            ),
            ("jacobi", CiqPreconditioner::jacobi(&d, 1e-12).unwrap()),
            ("spectral", spectral_from_dense(&a, 6)),
        ] {
            let twice = p.inv_sqrt_apply(&p.inv_sqrt_apply(&x).unwrap()).unwrap();
            let once = p.inv_apply(&x).unwrap();
            let err = frob(&(&once - &twice).unwrap()) / frob(&twice).max(1.0);
            assert!(err < 1e-9, "{name}: inv_apply vs (inv_sqrt)²: {err:.3e}");

            // …and M(M⁻¹x) = x, so it really is the inverse and not just
            // self-consistent with a shared sign error.
            let back = p.sqrt_apply(&p.sqrt_apply(&once).unwrap()).unwrap();
            let err = frob(&(&back - &x).unwrap()) / frob(&x).max(1.0);
            assert!(err < 1e-8, "{name}: M M⁻¹ x ≠ x: {err:.3e}");
        }
    }

    #[test]
    fn preconditioner_roundtrips_matches_m_and_is_symmetric() {
        let dev = Device::Cpu;
        let (a, d) = lowrank_spd(30, 3, 1e-3, 9);
        let x = Tensor::from_vec(
            (0..30 * 2)
                .map(|i| ((i * 17 % 13) as f64) / 5.0 - 1.0)
                .collect::<Vec<_>>(),
            (30, 2),
            &dev,
        )
        .unwrap();
        for (kind, p) in [
            (
                "pivoted",
                CiqPreconditioner::pivoted(|x| a.matmul(x), &d, 5).unwrap(),
            ),
            ("spectral", spectral_from_dense(&a, 8)),
        ] {
            let _ = kind;

            // E(E⁻¹x) = x
            let rt = p.sqrt_apply(&p.inv_sqrt_apply(&x).unwrap()).unwrap();
            assert!(
                frob(&(&rt - &x).unwrap()) < 1e-9 * frob(&x).max(1.0),
                "E E⁻¹ x ≠ x"
            );

            // Symmetry: ⟨Ex, y⟩ = ⟨x, Ey⟩
            let y = Tensor::from_vec(
                (0..30 * 2)
                    .map(|i| ((i * 29 % 11) as f64) / 4.0 - 1.0)
                    .collect::<Vec<_>>(),
                (30, 2),
                &dev,
            )
            .unwrap();
            let lhs: f64 = (p.sqrt_apply(&x).unwrap() * &y)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            let rhs: f64 = (&x * p.sqrt_apply(&y).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
                "E not symmetric: {lhs} vs {rhs}"
            );

            // E(Ex) = Mx = δx + L(Lᵀx): verify M ≈ A where the rank covers it.
            let mm_x = p.sqrt_apply(&p.sqrt_apply(&x).unwrap()).unwrap();
            let ax = a.matmul(&x).unwrap();
            let rel = frob(&(&mm_x - &ax).unwrap()) / frob(&ax);
            assert!(
                rel < 0.05,
                "{kind}: M ≉ K on a low-rank-plus-diagonal K: rel {rel:.3e}"
            );
        }
    }

    // ---- Spectral -----------------------------------------------------------

    /// Dense symmetric eigendecomposition → the top-`k` eigenpairs, so the
    /// `Spectral` tests have an exactly-known `(Q, μ)` and do not depend on
    /// Lanczos converging.
    fn spectral_from_dense(a: &Tensor, k: usize) -> CiqPreconditioner {
        let (q, mu, delta) = top_eigenpairs(a, k);
        CiqPreconditioner::spectral(&q, &mu, delta, 1e-10).unwrap()
    }

    /// `(Q [n,k], mu [k], delta)` — the `k` LARGEST eigenpairs, with `delta`
    /// the geometric mean of the rest.
    fn top_eigenpairs(a: &Tensor, k: usize) -> (Tensor, Tensor, f64) {
        let dev = Device::Cpu;
        let n = a.dim(D::Minus1).unwrap();
        let av: Vec<f64> = a
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec1()
            .unwrap();
        let fro = av.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scaled: Vec<f64> = av.iter().map(|v| v / fro).collect();
        let (eig, vecs) = crate::linalg::jacobi_eigh(&scaled, n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| eig[j].partial_cmp(&eig[i]).unwrap());

        let mut qh = vec![0.0f64; n * k];
        let mut mh = vec![0.0f64; k];
        for (slot, &j) in order.iter().take(k).enumerate() {
            mh[slot] = eig[j] * fro;
            for i in 0..n {
                qh[i * k + slot] = vecs[i * n + j];
            }
        }
        // delta = geometric mean of the UNRESOLVED eigenvalues.
        let rest: Vec<f64> = order
            .iter()
            .skip(k)
            .map(|&j| (eig[j] * fro).max(1e-300))
            .collect();
        let delta = if rest.is_empty() {
            mh[k - 1]
        } else {
            (rest.iter().map(|v| v.ln()).sum::<f64>() / rest.len() as f64).exp()
        };
        (
            Tensor::from_vec(qh, (n, k), &dev).unwrap(),
            Tensor::from_vec(mh, (k,), &dev).unwrap(),
            delta,
        )
    }

    /// **The test that proves the eigenvalues are used at all.**
    ///
    /// At `k = n` the basis is complete, so `M = K` exactly and `E(Ex)` must
    /// reproduce `Kx` to rounding. Every other invariant here — symmetry,
    /// roundtrip, `inv_apply = (inv_sqrt)²` — is satisfied by an `M` that
    /// silently replaced every `μ` with `δ`, i.e. by a scalar.
    #[test]
    fn a_complete_spectral_basis_reproduces_the_operator_exactly() {
        let n = 16;
        let (a, _) = lowrank_spd(n, 4, 1e-2, 3);
        let p = spectral_from_dense(&a, n);
        let x = Tensor::from_vec(
            (0..n * 2)
                .map(|i| ((i * 13 % 9) as f64) / 3.0 - 1.0)
                .collect::<Vec<_>>(),
            (n, 2),
            &Device::Cpu,
        )
        .unwrap();
        let mm_x = p.sqrt_apply(&p.sqrt_apply(&x).unwrap()).unwrap();
        let ax = a.matmul(&x).unwrap();
        let rel = frob(&(&mm_x - &ax).unwrap()) / frob(&ax);
        assert!(
            rel < 1e-9,
            "complete basis must give M = K exactly, rel {rel:.3e}"
        );
    }

    /// `spectral_apply` is `low_rank_apply` at `W = I` — pinned so the
    /// three-matmul specialisation stays a specialisation.
    #[test]
    fn spectral_apply_matches_low_rank_with_identity_w() {
        let n = 12;
        let k = 5;
        let (a, _) = lowrank_spd(n, 3, 1e-2, 21);
        let (q, mu, delta) = top_eigenpairs(&a, k);
        let x = Tensor::from_vec(
            (0..n).map(|i| (i as f64) / 4.0 - 1.0).collect::<Vec<_>>(),
            (n, 1),
            &Device::Cpu,
        )
        .unwrap();

        let h = mu
            .affine(1.0 / delta, 0.0)
            .unwrap()
            .sqrt()
            .unwrap()
            .affine(1.0, -1.0)
            .unwrap();
        let scale = delta.sqrt();
        let spec = CiqPreconditioner::spectral_apply(&q, &h, scale, &x).unwrap();
        let eye = Tensor::eye(k, DType::F64, &Device::Cpu).unwrap();
        let low = CiqPreconditioner::low_rank_apply(&q, &eye, &h, scale, &x).unwrap();
        assert!(
            frob(&(&spec - &low).unwrap()) < 1e-12,
            "the two applies diverged"
        );
    }

    /// Each guard exists for a failure that is otherwise silent, so each is
    /// tested for the NAMED refusal rather than merely for `is_err`.
    #[test]
    fn the_spectral_constructor_refuses_every_silent_failure() {
        let dev = Device::Cpu;
        let n = 8;
        let (a, _) = lowrank_spd(n, 3, 1e-2, 5);
        let (q, mu, delta) = top_eigenpairs(&a, 4);

        // A basis that is not orthonormal — every apply would still return
        // finite, symmetric-looking numbers.
        let skew = q.affine(1.7, 0.0).unwrap();
        let e = CiqPreconditioner::spectral(&skew, &mu, delta, 1e-10)
            .unwrap_err()
            .to_string();
        assert!(e.contains("orthonormal"), "unhelpful refusal: {e}");

        // A non-positive eigenvalue — inv_apply would return inf.
        let bad_mu = mu
            .affine(1.0, -mu.get(0).unwrap().to_scalar::<f64>().unwrap() - 1.0)
            .unwrap();
        let e = CiqPreconditioner::spectral(&q, &bad_mu, delta, 1e-10)
            .unwrap_err()
            .to_string();
        assert!(e.contains("finite and positive"), "unhelpful refusal: {e}");

        // Every mu equal to delta — M is a scalar; it "works" and does nothing.
        let flat = Tensor::full(delta, (4,), &dev).unwrap();
        let e = CiqPreconditioner::spectral(&q, &flat, delta, 1e-10)
            .unwrap_err()
            .to_string();
        assert!(e.contains("delta*I"), "unhelpful refusal: {e}");

        assert!(CiqPreconditioner::spectral(&q, &mu, -1.0, 1e-10).is_err());
        assert!(CiqPreconditioner::spectral(&q, &mu, f64::NAN, 1e-10).is_err());
    }

    #[test]
    fn jacobi_preconditioner_roundtrips() {
        let dev = Device::Cpu;
        let d = Tensor::from_vec(vec![4.0f64, 0.25, 9.0, 1.0, 16.0], (5,), &dev).unwrap();
        let p = CiqPreconditioner::jacobi(&d, 1e-12).unwrap();
        let x = Tensor::from_vec(
            (0..5).map(|i| i as f64 - 2.0).collect::<Vec<_>>(),
            (5, 1),
            &dev,
        )
        .unwrap();
        let rt = p.inv_sqrt_apply(&p.sqrt_apply(&x).unwrap()).unwrap();
        assert!(frob(&(&rt - &x).unwrap()) < 1e-12);
        // E² x = diag(d) x
        let dd = p.sqrt_apply(&p.sqrt_apply(&x).unwrap()).unwrap();
        let want = x.broadcast_mul(&d.unsqueeze(D::Minus1).unwrap()).unwrap();
        assert!(frob(&(&dd - &want).unwrap()) < 1e-12);
    }

    #[test]
    fn batched_operators_share_pivots_but_keep_valid_roots() {
        let dev = Device::Cpu;
        let (a0, _) = lowrank_spd(20, 3, 1e-3, 21);
        let (a1, _) = lowrank_spd(20, 3, 1e-3, 22);
        let a = Tensor::stack(&[&a0, &a1], 0).unwrap(); // [2, 20, 20]
        let d = crate::operators::dense_diagonal(&a).unwrap(); // [2, 20]
        let p = CiqPreconditioner::pivoted(|x| a.broadcast_matmul(x), &d, 5).unwrap();
        let x = Tensor::from_vec(
            (0..2 * 20)
                .map(|i| ((i * 7 % 5) as f64) - 2.0)
                .collect::<Vec<_>>(),
            (2, 20, 1),
            &dev,
        )
        .unwrap();
        let rt = p.sqrt_apply(&p.inv_sqrt_apply(&x).unwrap()).unwrap();
        assert_eq!(rt.dims(), &[2, 20, 1]);
        assert!(frob(&(&rt - &x).unwrap()) < 1e-9 * frob(&x).max(1.0));
    }
}
