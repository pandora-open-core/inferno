//! Inferno — Conjugate-gradient solves with configurable convergence and preconditioning.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};

use crate::settings;

/// Outcome of a CG solve, beyond the solution itself.
#[derive(Clone, Debug, PartialEq)]
pub struct CgStats {
    /// Iterations actually run.
    pub iterations: usize,
    /// Mean residual norm at exit, over columns.
    pub residual_norm: f64,
    /// Whether the tolerance was met before the iteration budget ran out.
    ///
    /// `false` means the answer is whatever CG had reached when it stopped.
    /// The Python emits a `NumericalWarning` here; this returns it as data,
    /// so a caller can decide rather than parse a log line.
    pub converged: bool,
}

/// Everything optional about a solve. `Default` matches the Python's
/// defaults, which read from [`crate::settings`] at call time.
#[derive(Clone, Debug)]
pub struct CgOptions {
    /// `None` ⇒ [`settings::max_cg_iterations`].
    pub max_iter: Option<usize>,
    /// `None` ⇒ [`settings::cg_tolerance`].
    pub tolerance: Option<f64>,
    /// Guard against division by zero.
    ///
    /// THIS BOUNDS THE ACHIEVABLE RESIDUAL. `beta`'s denominator is
    /// `r_{k-1}ᵀ z_{k-1}`, i.e. `‖r‖²` unpreconditioned, so once
    /// `‖r‖ < √eps` the guard fires, `beta` is forced to 0, the conjugate
    /// direction resets to the raw residual and CG silently degenerates
    /// into steepest descent — which stagnates.
    ///
    /// At the inherited default of `1e-10` that floor is `‖r‖ ≈ 1e-5`.
    /// gpytorch never meets it because its `cg_tolerance` default is `1`,
    /// but anyone asking for a genuinely tight solve will, and the symptom
    /// is a solve that runs its full iteration budget and returns a
    /// plausible wrong answer. Use [`CgOptions::for_tolerance`] rather than
    /// setting this by hand.
    pub eps: f64,
    /// A column stops updating once its residual norm falls below this.
    pub stop_updating_after: f64,
    /// Number of leading columns to build a Lanczos tridiagonal for. 0
    /// disables it. See [`linear_cg_tridiag`].
    pub n_tridiag: usize,
    /// `None` ⇒ [`settings::max_lanczos_quadrature_iterations`].
    pub max_tridiag_iter: Option<usize>,
    /// `M⁻¹` for the solve, applied via
    /// [`CiqPreconditioner::inv_apply`](super::precond::CiqPreconditioner::inv_apply).
    ///
    /// This field is pure TRANSPORT. [`linear_cg`] keeps its generic
    /// `preconditioner` argument — this exists because the `linsolve` entry
    /// points take only a `&CgOptions`, so there is otherwise no way to hand
    /// them one. `run` reads whichever arrives; setting both is a caller
    /// error and the explicit argument wins.
    ///
    /// **Only the pure solves honour it.** Anything producing a Lanczos
    /// tridiagonal (`slq_logdet_*`'s forward) is refused outright in `run` —
    /// a preconditioned `T` describes `M^{-1/2}AM^{-1/2}` and every number
    /// built on it is silently wrong. `A⁻¹b` and `bᵀA⁻¹b` are
    /// preconditioning-invariant, so those are safe.
    pub precond: Option<Arc<super::precond::CiqPreconditioner>>,
    /// Permit a preconditioner ALONGSIDE a Lanczos tridiagonal.
    ///
    /// Off by default and refused loudly, because both halves of that
    /// combination are silent: the tridiagonal describes `M^{-1/2}AM^{-1/2}`,
    /// so an SLQ built on it is short by `logdet(M)`; and the Euclidean probe
    /// norms are the wrong quadrature weights once the Lanczos start vector is
    /// `M^{-1/2}z` rather than `z`.
    ///
    /// A caller who sets this is asserting they have handled BOTH — see
    /// [`super::linsolve::slq_logdet_preconditioned`], which transforms its
    /// probes so the weights stay right and adds
    /// [`super::precond::CiqPreconditioner::log_det`] back. Setting it without
    /// doing that returns a plausible, wrong number.
    pub allow_preconditioned_tridiag: bool,
}

impl CgOptions {
    /// Options for a target residual tolerance, with `eps` set low enough
    /// that the tolerance is actually reachable.
    ///
    /// The stagnation floor is `‖r‖ ≈ √eps` (see [`CgOptions::eps`]), so
    /// `eps` must sit well below `tolerance²`. Two orders of margin keeps
    /// the guard doing its real job — catching an exactly-zero denominator
    /// — without it truncating the solve.
    ///
    /// This has no gpytorch counterpart. There, the interaction is left for
    /// the caller to discover.
    pub fn for_tolerance(tolerance: f64) -> Self {
        let eps = (tolerance * tolerance * 1e-2).min(1e-10).max(f64::MIN_POSITIVE);
        Self {
            tolerance: Some(tolerance),
            eps,
            stop_updating_after: tolerance * 1e-2,
            ..Default::default()
        }
    }
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            max_iter: None,
            tolerance: None,
            eps: 1e-10,
            stop_updating_after: 1e-10,
            n_tridiag: 0,
            max_tridiag_iter: None,
            precond: None,
            allow_preconditioned_tridiag: false,
        }
    }
}

/// Solve `A x = rhs`. `matmul` applies `A`; `rhs` is `[..., N]` or
/// `[..., N, C]`.
///
/// `preconditioner` applies `M⁻¹`. `None` is the identity, which is the
/// unpreconditioned method.
pub fn linear_cg<F, P>(
    matmul: F,
    rhs: &Tensor,
    initial_guess: Option<&Tensor>,
    preconditioner: Option<P>,
    opts: &CgOptions,
) -> Result<(Tensor, CgStats)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    P: Fn(&Tensor) -> Result<Tensor>,
{
    let (x, stats, _) = run(matmul, rhs, initial_guess, preconditioner, opts, false, None)?;
    Ok((x, stats))
}

/// As [`linear_cg`], but also returns the Lanczos tridiagonal matrices
/// implied by the CG coefficients, shaped `[n_tridiag, k, k]`.
///
/// CG and Lanczos generate the same Krylov space, so the `α`/`β` sequences
/// already contain the tridiagonalisation — it costs nothing beyond the
/// bookkeeping. Its eigenvalues are the Ritz values used for stochastic
/// Lanczos quadrature (log-determinant estimation), which is why this
/// exists at all.
///
/// Requires `opts.n_tridiag > 0`.
pub fn linear_cg_tridiag<F, P>(
    matmul: F,
    rhs: &Tensor,
    initial_guess: Option<&Tensor>,
    preconditioner: Option<P>,
    opts: &CgOptions,
) -> Result<(Tensor, CgStats, Tensor)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    P: Fn(&Tensor) -> Result<Tensor>,
{
    if opts.n_tridiag == 0 {
        return Err(candle_core::Error::Msg(
            "linear_cg_tridiag needs opts.n_tridiag > 0".into(),
        ));
    }
    let (x, stats, t) = run(matmul, rhs, initial_guess, preconditioner, opts, true, None)?;
    Ok((x, stats, t.expect("requested tridiag")))
}

/// Outcome of a multi-shift solve.
#[derive(Clone, Debug, PartialEq)]
pub struct ShiftedCgStats {
    /// Iterations actually run (shared — one Krylov space serves every shift).
    pub iterations: usize,
    /// Mean residual norm per shift at exit, in the CALLER's shift order.
    pub residual_norms: Vec<f64>,
    /// Whether the seed system met the tolerance. Because every shifted
    /// residual is `ζ_k ≤ 1` times the seed's, seed convergence bounds them
    /// all.
    pub converged: bool,
}

/// Solve `(A + σ_j I) x_j = rhs` for every `σ_j` in `shifts`, from ONE Krylov
/// space (Frommer–Maass / Jegerlehner CG-M).
///
/// `matmul` applies the UNSHIFTED `A`; the shifts are added internally. One
/// matrix-vector product per iteration serves every shift — the whole point
/// over looping [`linear_cg`] per shift.
///
/// # Seed choice
///
/// The seed system is `A + σ_min I` (the smallest shift) and the remaining
/// systems are reached through incremental shifts `δ_j = σ_j − σ_min ≥ 0`.
/// With an SPD-or-better seed and non-negative increments, the shift
/// coefficients satisfy `ζ_k ∈ (0, 1]` monotonically non-increasing, so every
/// shifted residual is bounded by the seed's — the classic multi-shift
/// breakdown (seed converging before a shifted system) is structurally
/// impossible in this configuration. It also means a `K` whose true λmin is
/// slightly negative (fp noise) still yields an SPD seed as long as
/// `σ_min` clears the noise floor.
///
/// # Restrictions, by construction
///
/// * **Unpreconditioned.** `M⁻¹(A + σI)` is not a shifted family in a shared
///   Krylov space unless `M` commutes with `A` (i.e. is a polynomial in `A`),
///   so generic preconditioning is incompatible with the single-recurrence
///   method. A shift-invariant preconditioner is future work.
/// * **Zero initial guess.** Krylov shift invariance holds for `x_0 = 0`
///   only; a warm start would need a different seed residual per shift.
/// * `shifts` must be finite and `≥ 0`.
///
/// Callers wanting a genuinely tight solve must pass
/// [`CgOptions::for_tolerance`] — the raw default `cg_tolerance = 1.0` is
/// gpytorch's normalized-residual convention for stochastic trace estimation,
/// not for solutions that are read directly.
pub fn linear_cg_shifted<F>(
    matmul: F,
    rhs: &Tensor,
    shifts: &[f64],
    opts: &CgOptions,
) -> Result<(Vec<Tensor>, ShiftedCgStats)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if opts.precond.is_some() {
        return Err(candle_core::Error::Msg(
            "linear_cg_shifted: a preconditioner was supplied and this solver cannot honour it. \
             No nontrivial M keeps {{M⁻¹(A + σI)}} a shifted family, so one Krylov space \
             cannot serve every shift under preconditioning — the multi-shift identity \
             the solver is built on would silently stop holding. Conjugate the operator \
             instead (`K̂ = E⁻¹KE⁻¹`, see `precond.rs`), or drop `opts.precond`."
                .to_string(),
        ));
    }
    if shifts.is_empty() {
        return Err(candle_core::Error::Msg("linear_cg_shifted: needs at least one shift".into()));
    }
    if shifts.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return Err(candle_core::Error::Msg(format!(
            "linear_cg_shifted: shifts must be finite and >= 0, got {shifts:?}"
        )));
    }

    let is_vector = rhs.rank() == 1;
    let rhs = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };

    let n_rows = rhs.dim(D::Minus2)?;
    let max_iter = opts.max_iter.unwrap_or_else(settings::max_cg_iterations::value);
    let tolerance = opts.tolerance.unwrap_or_else(settings::cg_tolerance::value);
    let n_iter = if settings::terminate_cg_by_size::value() {
        max_iter.min(n_rows)
    } else {
        max_iter
    };

    // Seed on the smallest shift; every other shift is an increment δ ≥ 0.
    // `order[j]` is the caller index of the j-th smallest shift.
    let mut order: Vec<usize> = (0..shifts.len()).collect();
    order.sort_by(|&a, &b| shifts[a].total_cmp(&shifts[b]));
    let sigma_min = shifts[order[0]];
    let deltas: Vec<f64> = order.iter().map(|&i| shifts[i] - sigma_min).collect();
    let matmul_seed = |x: &Tensor| -> Result<Tensor> {
        let ax = matmul(x)?;
        if sigma_min == 0.0 { Ok(ax) } else { ax + x.affine(sigma_min, 0.0)? }
    };

    // -- normalise the RHS (same contract as `run`) ---------------------------
    let rhs_norm = col_norm(&rhs)?;
    let eps_t = Tensor::full(opts.eps, rhs_norm.shape(), rhs_norm.device())?
        .to_dtype(rhs_norm.dtype())?;
    let rhs_is_zero = rhs_norm.lt(&eps_t)?;
    let rhs_norm = rhs_is_zero.where_cond(&rhs_norm.ones_like()?, &rhs_norm)?;
    let rhs = rhs.broadcast_div(&rhs_norm)?;

    // x_0 = 0 for every system, so r_0 = b for every system.
    let mut residual = rhs.clone();
    if !scalar_is_finite(&residual)? {
        return Err(candle_core::Error::Msg(
            "linear_cg_shifted: non-finite right-hand side".into(),
        ));
    }
    let mut result = rhs.zeros_like()?; // seed solution
    let mut conj = residual.clone();
    let mut residual_inner_prod = col_dot(&residual, &residual)?;

    let mut residual_norm = col_norm(&residual)?;
    let stop_t = Tensor::full(opts.stop_updating_after, residual_norm.shape(), residual_norm.device())?
        .to_dtype(residual_norm.dtype())?;
    let mut has_converged = residual_norm.lt(&stop_t)?;

    // Per-shift Jegerlehner state, indexed like `deltas` (ascending shifts).
    // The seed (deltas[0] == 0) is served by the seed iterates directly; the
    // recurrence with δ = 0 reduces to the identity, so tracking it would be
    // pure duplicate work.
    struct ShiftState {
        x: Tensor,      // [..., N, C]
        p: Tensor,      // [..., N, C]
        z_prev: Tensor, // ζ_{k-1}  [..., 1, C]
        z_cur: Tensor,  // ζ_k      [..., 1, C]
    }
    let ones_coeff = residual_inner_prod.ones_like()?;
    let mut shifted: Vec<ShiftState> = deltas[1..]
        .iter()
        .map(|_| {
            Ok(ShiftState {
                x: rhs.zeros_like()?,
                p: residual.clone(),
                z_prev: ones_coeff.clone(),
                z_cur: ones_coeff.clone(),
            })
        })
        .collect::<Result<_>>()?;

    // α_{k-1} and the β that formed the CURRENT direction p_k (i.e. last
    // iteration's fresh β) — the exact bookkeeping the ζ denominator needs.
    let mut alpha_prev = ones_coeff.clone();
    let mut beta_prev = ones_coeff.zeros_like()?;

    let mut tolerance_reached = false;
    let mut k_done = 0usize;

    for k in 0..n_iter {
        k_done = k + 1;
        let mvms = matmul_seed(&conj)?;

        let denom = col_dot(&conj, &mvms)?;
        let mut alpha = safe_div(&residual_inner_prod, &denom, opts.eps)?;
        // Freeze converged seed columns. ζ ≤ 1 makes every shifted residual
        // for that column at least as small, so the freeze is globally safe.
        alpha = has_converged.where_cond(&alpha.zeros_like()?, &alpha)?;

        // ---- shifted coefficient updates that need PRE-update state --------
        // ζ_{k+1} = ζ_k ζ_{k-1} α_{k-1} /
        //           ( α_k β_k (ζ_{k-1} − ζ_k) + ζ_{k-1} α_{k-1} (1 + δ α_k) )
        // where β_k is the β that formed p_k — last iteration's fresh β.
        let mut z_next_all: Vec<Tensor> = Vec::with_capacity(shifted.len());
        for (st, &delta) in shifted.iter_mut().zip(&deltas[1..]) {
            let one_plus = alpha.affine(delta, 1.0)?; // 1 + δ·α_k
            let term_a = (&st.z_prev * &alpha_prev)?.mul(&one_plus)?;
            let term_b = (&alpha * &beta_prev)?.mul(&(&st.z_prev - &st.z_cur)?)?;
            let denom_s = (term_a + term_b)?;
            let numer = (&st.z_cur * &st.z_prev)?.mul(&alpha_prev)?;
            let z_next = safe_div(&numer, &denom_s, opts.eps)?;

            // α_δ = α_k ζ_{k+1}/ζ_k ; a frozen (safe_div-zeroed) ζ yields 0
            // and the shifted iterate simply stops moving.
            let ratio = safe_div(&z_next, &st.z_cur, opts.eps)?;
            let alpha_s = (&alpha * &ratio)?;
            st.x = (&st.x + st.p.broadcast_mul(&alpha_s)?)?;
            z_next_all.push(z_next);
        }

        // ---- seed updates ---------------------------------------------------
        residual = (&residual - mvms.broadcast_mul(&alpha)?)?;
        result = (&result + conj.broadcast_mul(&alpha)?)?;

        let prev_inner = residual_inner_prod.clone();
        residual_inner_prod = col_dot(&residual, &residual)?;
        let beta = safe_div(&residual_inner_prod, &prev_inner, opts.eps)?;
        conj = (&residual + conj.broadcast_mul(&beta)?)?;

        // ---- shifted direction updates that need the FRESH β and r_{k+1} ---
        // β_{k+1}^δ = β_{k+1} (ζ_{k+1}/ζ_k)² ;  p^δ ← ζ_{k+1} r_{k+1} + β^δ p^δ
        for (st, z_next) in shifted.iter_mut().zip(z_next_all) {
            let ratio = safe_div(&z_next, &st.z_cur, opts.eps)?;
            let beta_s = (&beta * &ratio)?.mul(&ratio)?;
            st.p = (residual.broadcast_mul(&z_next)? + st.p.broadcast_mul(&beta_s)?)?;
            st.z_prev = std::mem::replace(&mut st.z_cur, z_next);
        }
        alpha_prev = alpha;
        beta_prev = beta;

        residual_norm = col_norm(&residual)?;
        residual_norm = rhs_is_zero.where_cond(&residual_norm.zeros_like()?, &residual_norm)?;
        has_converged = residual_norm.lt(&stop_t)?;

        // Same floor as `run`: don't let a fluke-small early residual end the
        // solve before a real Krylov space exists.
        if k >= 10.min(max_iter.saturating_sub(1)) && mean_scalar(&residual_norm)? < tolerance {
            tolerance_reached = true;
            break;
        }
    }

    // Un-normalise and hand back in the CALLER's shift order. Per-shift
    // residual norms come free: ‖r^δ‖ = ζ ‖r_seed‖ column-wise.
    let mut solutions: Vec<Option<Tensor>> = vec![None; shifts.len()];
    let mut norms = vec![0.0f64; shifts.len()];
    let finish = |x: &Tensor| -> Result<Tensor> {
        let x = x.broadcast_mul(&rhs_norm)?;
        Ok(if is_vector { x.squeeze(D::Minus1)? } else { x })
    };
    solutions[order[0]] = Some(finish(&result)?);
    norms[order[0]] = mean_scalar(&residual_norm)?;
    for (j, st) in shifted.iter().enumerate() {
        let caller = order[j + 1];
        solutions[caller] = Some(finish(&st.x)?);
        norms[caller] = mean_scalar(&residual_norm.broadcast_mul(&st.z_cur)?.abs()?)?;
    }

    let converged = tolerance_reached || n_iter == 0;
    if !converged && settings::verbose_linalg::value() {
        tracing::warn!(
            iterations = k_done, residual_norm = norms[order[0]], tolerance,
            "linear_cg_shifted hit its iteration budget without reaching tolerance"
        );
    }
    let stats = ShiftedCgStats { iterations: k_done, residual_norms: norms, converged };
    Ok((solutions.into_iter().map(|x| x.expect("every shift filled")).collect(), stats))
}

// ---------------------------------------------------------------------------

/// Column-wise L2 norm over the row axis, keeping dims: `[..., 1, C]`.
pub(crate) fn col_norm(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.sum_keepdim(D::Minus2)?.sqrt()
}

/// `Σ_rows a ⊙ b`, keeping dims: `[..., 1, C]`.
pub(crate) fn col_dot(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a * b)?.sum_keepdim(D::Minus2)
}

/// `num / den`, yielding exactly 0 wherever `den < eps`.
///
/// The Python's three-step dance (`masked_fill_(is_zero, 1)`, divide,
/// `masked_fill_(is_zero, 0)`) rather than a plain guarded divide, and the
/// order matters: dividing first and masking after would already have
/// produced an inf/NaN, and NaN propagates through the subsequent
/// `curr_conjugate_vec` update into every column via the shared
/// convergence test.
pub(crate) fn safe_div(num: &Tensor, den: &Tensor, eps: f64) -> Result<Tensor> {
    let eps_t = Tensor::full(eps, den.shape(), den.device())?.to_dtype(den.dtype())?;
    let is_small = den.lt(&eps_t)?; // u8 mask
    let ones = den.ones_like()?;
    let safe_den = is_small.where_cond(&ones, den)?;
    let q = (num / &safe_den)?;
    is_small.where_cond(&q.zeros_like()?, &q)
}

/// Search directions harvested from a completed solve, for
/// [`super::recycle`].
///
/// These come free: `p_k` and `A p_k` are computed by the iteration anyway,
/// and `p_kᵀ A p_k` is already formed as `alpha`'s denominator. They are also
/// mutually A-CONJUGATE by construction, which is what makes `UᵀAU` diagonal
/// and the Galerkin projection an `O(r)` divide instead of an `r × r` solve.
///
/// (Proper GCRO-DR would want harmonic Ritz vectors, which need the Lanczos
/// basis `V` — this loop does not store it. Thick restart from the CG
/// directions is the variant that costs nothing.)
#[derive(Clone, Debug, Default)]
pub struct CgDirections {
    /// `p_k`, in the RHS-NORMALISED space the iteration runs in. Only the
    /// SPAN matters downstream and Galerkin projection is scale-covariant, so
    /// this is not a hazard — but it is why they must never be compared
    /// against vectors from the caller's space.
    pub p: Vec<Tensor>,
    /// `A p_k`.
    pub ap: Vec<Tensor>,
    /// `p_kᵀ A p_k`, per column, `[..., 1, C]`.
    pub pap: Vec<Tensor>,
}

/// [`linear_cg`], additionally returning the search directions it built.
pub fn linear_cg_harvest<F, P>(
    matmul: F,
    rhs: &Tensor,
    initial_guess: Option<&Tensor>,
    preconditioner: Option<P>,
    opts: &CgOptions,
) -> Result<(Tensor, CgStats, CgDirections)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    P: Fn(&Tensor) -> Result<Tensor>,
{
    let mut dirs = CgDirections::default();
    let (x, stats, _) =
        run(matmul, rhs, initial_guess, preconditioner, opts, false, Some(&mut dirs))?;
    Ok((x, stats, dirs))
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn run<F, P>(
    matmul: F,
    rhs_in: &Tensor,
    initial_guess: Option<&Tensor>,
    preconditioner: Option<P>,
    opts: &CgOptions,
    want_tridiag: bool,
    mut harvest: Option<&mut CgDirections>,
) -> Result<(Tensor, CgStats, Option<Tensor>)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    P: Fn(&Tensor) -> Result<Tensor>,
{
    // A bare `[..., N]` is one column; restore the shape on the way out.
    let is_vector = rhs_in.rank() == 1;
    let rhs = if is_vector { rhs_in.unsqueeze(D::Minus1)? } else { rhs_in.clone() };

    let n_rows = rhs.dim(D::Minus2)?;
    let max_iter = opts.max_iter.unwrap_or_else(settings::max_cg_iterations::value);
    let tolerance = opts.tolerance.unwrap_or_else(settings::cg_tolerance::value);
    let max_tridiag_iter = opts
        .max_tridiag_iter
        .unwrap_or_else(settings::max_lanczos_quadrature_iterations::value);

    if want_tridiag && max_tridiag_iter > max_iter {
        return Err(candle_core::Error::Msg(format!(
            "max_tridiag_iter ({max_tridiag_iter}) > max_iter ({max_iter}): cannot get more \
             Lanczos coefficients than CG iterations run"
        )));
    }

    // The tridiagonal and a preconditioner are mutually exclusive, and the
    // failure is SILENT in both of its parts — hence a refusal rather than a
    // caveat.
    //
    // `alpha`/`beta` below come from the PRECONDITIONED recurrence, so `T`
    // tridiagonalises `M^{-1/2} A M^{-1/2}`, not `A`. Two consequences, and
    // fixing only the first still leaves a wrong number:
    //
    //   1. SLQ reads `logdet(A) − logdet(M)`. Correctable in closed form
    //      (determinant lemma on `M = δI + LLᵀ`), but only if someone
    //      remembers to.
    //   2. The quadrature weight is `zᵀMz` while
    //      `linsolve::host_col_norms_sq` supplies the EUCLIDEAN `‖z‖²`, so
    //      every per-probe term is mis-scaled as well.
    //
    // A preconditioned SOLVE is unaffected — `A⁻¹b` and `bᵀA⁻¹b` are
    // preconditioning-invariant — so callers wanting both should run the
    // solve preconditioned and the tridiagonal unpreconditioned, or conjugate
    // (`precond`'s `K̂ = E⁻¹KE⁻¹`) and add `logdet M` back explicitly.
    if want_tridiag
        && (preconditioner.is_some() || opts.precond.is_some())
        && !opts.allow_preconditioned_tridiag
    {
        return Err(candle_core::Error::Msg(
            "linear_cg_tridiag: a preconditioner would tridiagonalise \
             M^{-1/2}AM^{-1/2}, so the Lanczos coefficients describe the \
             preconditioned operator and any SLQ built on them is wrong by \
             logdet(M) — with a second error from the Euclidean probe norms. \
             Run the tridiagonal unpreconditioned, or set \
             `allow_preconditioned_tridiag` if you are correcting BOTH (see \
             linsolve::slq_logdet_preconditioned)."
                .into(),
        ));
    }

    let mut n_iter = if settings::terminate_cg_by_size::value() {
        max_iter.min(n_rows)
    } else {
        max_iter
    };
    let n_tridiag_iter = max_tridiag_iter.min(n_rows);

    // -- normalise the RHS ---------------------------------------------------
    // Columns with ~zero norm are remembered and their norm forced to 1, so
    // the division below is defined. Their residual is then pinned to 0:
    // the solution to `A x = 0` is `x = 0` and there is nothing to iterate.
    let rhs_norm = col_norm(&rhs)?;
    let eps_t = Tensor::full(opts.eps, rhs_norm.shape(), rhs_norm.device())?
        .to_dtype(rhs_norm.dtype())?;
    let rhs_is_zero = rhs_norm.lt(&eps_t)?;
    let rhs_norm = rhs_is_zero.where_cond(&rhs_norm.ones_like()?, &rhs_norm)?;
    let rhs = rhs.broadcast_div(&rhs_norm)?;

    let x0 = match initial_guess {
        Some(g) => {
            let g = if g.rank() == 1 { g.unsqueeze(D::Minus1)? } else { g.clone() };
            g.broadcast_div(&rhs_norm)?
        }
        None => rhs.zeros_like()?,
    };

    // residual_0 = b - A x_0
    let mut residual = (&rhs - matmul(&x0)?)?;
    let mut result = x0;

    if settings::verbose_linalg::value() {
        tracing::debug!(
            rows = n_rows, cols = rhs.dim(D::Minus1)?, n_iter, tolerance,
            "linear_cg starting"
        );
    }

    // A NaN here means the operator itself is broken; iterating would only
    // spread it, and the residual test would never fire because NaN
    // compares false against the tolerance.
    if !scalar_is_finite(&residual)? {
        return Err(candle_core::Error::Msg(
            "linear_cg: matrix-vector product produced a non-finite residual".into(),
        ));
    }

    let mut residual_norm = col_norm(&residual)?;
    let stop_t = Tensor::full(opts.stop_updating_after, residual_norm.shape(), residual_norm.device())?
        .to_dtype(residual_norm.dtype())?;
    let mut has_converged = residual_norm.lt(&stop_t)?;

    // Lucky: the initial guess (or the preconditioner) already solved it.
    if !want_tridiag && all_true(&has_converged)? {
        n_iter = 0;
    }

    // One `M⁻¹` from either source — the explicit generic argument or the
    // options field carried in for callers who only hold a `&CgOptions`.
    // The explicit argument wins; supplying both is a caller error.
    let apply_precond = |r: &Tensor| -> Result<Tensor> {
        match (&preconditioner, &opts.precond) {
            (Some(p), _) => p(r),
            (None, Some(m)) => m.inv_apply(r),
            (None, None) => Ok(r.clone()),
        }
    };

    let mut precond_residual = apply_precond(&residual)?;
    let mut conj = precond_residual.clone();
    let mut residual_inner_prod = col_dot(&precond_residual, &residual)?;

    // Tridiagonal state. `[k][k]` entries are per-column `[..., n_tridiag]`.
    let mut t_diag: Vec<Tensor> = Vec::new();
    let mut t_offdiag: Vec<Tensor> = Vec::new();
    let mut prev_alpha_recip: Option<Tensor> = None;
    let mut prev_beta: Option<Tensor> = None;
    let mut update_tridiag = true;
    let mut last_tridiag_iter = 0usize;

    let mut tolerance_reached = false;
    let mut k_done = 0usize;

    for k in 0..n_iter {
        k_done = k + 1;
        let mvms = matmul(&conj)?;

        // alpha_k = (r_{k-1}ᵀ z_{k-1}) / (p_{k-1}ᵀ A p_{k-1})
        let denom = col_dot(&conj, &mvms)?;
        // Harvest BEFORE the update, so `p`, `Ap` and `pᵀAp` are the matching
        // triple for this iteration rather than a mix of two.
        if let Some(h) = harvest.as_mut() {
            h.p.push(conj.clone());
            h.ap.push(mvms.clone());
            h.pap.push(denom.clone());
        }
        let mut alpha = safe_div(&residual_inner_prod, &denom, opts.eps)?;
        // Freeze columns that have already converged: a zero alpha makes
        // every subsequent update a no-op for them, which is how they stay
        // in the batch without being disturbed.
        alpha = has_converged.where_cond(&alpha.zeros_like()?, &alpha)?;

        // r_k = r_{k-1} - alpha_k A p_{k-1}
        residual = (&residual - mvms.broadcast_mul(&alpha)?)?;
        // x_k = x_{k-1} + alpha_k p_{k-1}
        result = (&result + conj.broadcast_mul(&alpha)?)?;

        precond_residual = apply_precond(&residual)?;

        // beta_k = (r_kᵀ z_k) / (r_{k-1}ᵀ z_{k-1})
        let prev_inner = residual_inner_prod.clone();
        residual_inner_prod = col_dot(&residual, &precond_residual)?;
        let beta = safe_div(&residual_inner_prod, &prev_inner, opts.eps)?;

        // p_k = z_k + beta_k p_{k-1}
        conj = (&precond_residual + conj.broadcast_mul(&beta)?)?;

        residual_norm = col_norm(&residual)?;
        // A zero-RHS column has residual 0 by definition; without this its
        // normalised residual is meaningless and would hold the mean above
        // tolerance forever.
        residual_norm = rhs_is_zero.where_cond(&residual_norm.zeros_like()?, &residual_norm)?;
        has_converged = residual_norm.lt(&stop_t)?;

        // Tridiagonal update, before the early exit — the Python checks
        // tolerance first but guards it with `not (n_tridiag and k <
        // n_tridiag_iter)`, i.e. it will not exit early while tridiag rows
        // are still owed. Doing the update first expresses the same rule
        // more directly.
        if want_tridiag && k < n_tridiag_iter && update_tridiag {
            let n_t = opts.n_tridiag;
            let alpha_t = narrow_cols(&alpha, n_t)?;
            let beta_t = narrow_cols(&beta, n_t)?;

            // 1/alpha, defined as 0 where alpha is 0 (a frozen column).
            let zero = alpha_t.zeros_like()?;
            let is_zero = alpha_t.eq(&zero)?;
            let safe_alpha = is_zero.where_cond(&alpha_t.ones_like()?, &alpha_t)?;
            let alpha_recip = (alpha_t.ones_like()? / &safe_alpha)?;
            let alpha_recip = is_zero.where_cond(&zero, &alpha_recip)?;

            match (&prev_alpha_recip, &prev_beta) {
                (Some(par), Some(pb)) => {
                    t_diag.push((&alpha_recip + (pb * par)?)?);
                    let off = (pb.sqrt()? * par)?;
                    // Once the off-diagonal collapses, the Lanczos vectors
                    // have lost independence and further rows are noise.
                    if max_abs(&off)? < 1e-6 {
                        update_tridiag = false;
                    }
                    t_offdiag.push(off);
                }
                _ => t_diag.push(alpha_recip.clone()),
            }
            last_tridiag_iter = k;
            prev_alpha_recip = Some(alpha_recip);
            prev_beta = Some(beta_t);
        }

        // The `k >= min(10, max_iter-1)` floor stops a fluke-small early
        // residual from ending the solve before CG has built any Krylov
        // space worth the name. `cg_ignore_min_iter` drops it to 0 — see
        // that setting for when the ten wasted matvecs are worth reclaiming
        // (a well-preconditioned solve converges on step one and pays the
        // floor in full).
        let min_iter = if settings::cg_ignore_min_iter::value() { 0 } else { 10 };
        let owed_tridiag = want_tridiag && k < n_tridiag_iter.min(max_iter.saturating_sub(1));
        if k >= min_iter.min(max_iter.saturating_sub(1))
            && mean_scalar(&residual_norm)? < tolerance
            && !owed_tridiag
        {
            tolerance_reached = true;
            break;
        }
    }

    // Un-normalise.
    let result = result.broadcast_mul(&rhs_norm)?;
    let result = if is_vector { result.squeeze(D::Minus1)? } else { result };

    let final_norm = mean_scalar(&residual_norm)?;
    let converged = tolerance_reached || n_iter == 0;
    if !converged && settings::verbose_linalg::value() {
        tracing::warn!(
            iterations = k_done, residual_norm = final_norm, tolerance,
            "linear_cg hit its iteration budget without reaching tolerance"
        );
    }
    let stats = CgStats { iterations: k_done, residual_norm: final_norm, converged };

    // Charge this solve's work against the spectral cache's odometer. ONCE, at
    // exit, and in matvec-COLUMNS: inside the loop it would be both slower and
    // — worse — would age entries by an amount depending on where the loop
    // happened to break. Work rather than wall clock is what makes a test and
    // a training run age entries identically.
    let n_cols: u64 = rhs.dims().last().copied().unwrap_or(1) as u64;
    super::cgcache::tick((k_done as u64).saturating_mul(n_cols.max(1)));

    let tridiag = if want_tridiag {
        Some(assemble_tridiag(&t_diag, &t_offdiag, last_tridiag_iter + 1, opts.n_tridiag)?)
    } else {
        None
    };
    Ok((result, stats, tridiag))
}

/// Keep the first `n` columns of a `[..., 1, C]` coefficient tensor and drop
/// the singleton row axis, giving `[..., n]`.
fn narrow_cols(x: &Tensor, n: usize) -> Result<Tensor> {
    x.squeeze(D::Minus2)?.narrow(D::Minus1, 0, n)
}

/// Stack the per-iteration coefficients into `[n_tridiag, k, k]`.
fn assemble_tridiag(
    diag: &[Tensor],
    offdiag: &[Tensor],
    k: usize,
    n_tridiag: usize,
) -> Result<Tensor> {
    let k = k.min(diag.len());
    if k == 0 {
        return Err(candle_core::Error::Msg("no tridiagonal rows were built".into()));
    }
    let dev = diag[0].device();
    let dtype = diag[0].dtype();
    // Host-side assembly: k ≤ max_lanczos_quadrature_iterations (20 by
    // default), so this is a handful of scalars per column and the tensor
    // round-trip costs less than the indexing gymnastics would. Assembled
    // in f64 — an f64 solve must not have its Lanczos coefficients rounded
    // through f32 on the way to the Ritz eigendecomposition (SLQ consumes
    // these, and FD gradchecks amplify any value noise by 1/(2h)).
    let mut out = vec![0f64; n_tridiag * k * k];
    for i in 0..k {
        let d: Vec<f64> = diag[i].flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
        for (c, v) in d.iter().take(n_tridiag).enumerate() {
            out[c * k * k + i * k + i] = *v;
        }
        if i > 0 && i - 1 < offdiag.len() {
            let o: Vec<f64> = offdiag[i - 1].flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
            for (c, v) in o.iter().take(n_tridiag).enumerate() {
                out[c * k * k + i * k + (i - 1)] = *v;
                out[c * k * k + (i - 1) * k + i] = *v; // symmetric
            }
        }
    }
    Tensor::from_vec(out, (n_tridiag, k, k), dev)?.to_dtype(dtype)
}

pub(crate) fn mean_scalar(x: &Tensor) -> Result<f64> {
    Ok(x.flatten_all()?.mean(0)?.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

fn max_abs(x: &Tensor) -> Result<f64> {
    Ok(x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

fn all_true(mask: &Tensor) -> Result<bool> {
    let s: f64 = mask.to_dtype(DType::F64)?.flatten_all()?.sum(0)?.to_scalar()?;
    Ok(s as usize == mask.elem_count())
}

pub(crate) fn scalar_is_finite(x: &Tensor) -> Result<bool> {
    let v: Vec<f64> = x.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
    Ok(v.iter().all(|q| q.is_finite()))
}

#[cfg(test)]
mod tests {
    //! Verification of conjugate-gradient solves with configurable convergence and preconditioning.

    use super::*;
    use crate::settings::{cg_tolerance, max_cg_iterations, terminate_cg_by_size};
    use candle_core::Device;

    /// Dense SPD system: `A = Bᵀ B + shift·I`.
    fn spd(n: usize, shift: f64, seed: u64) -> (Tensor, Vec<f64>) {
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
        let t = Tensor::from_vec(a.clone(), (n, n), &dev).unwrap();
        (t, a)
    }

    fn op(a: &Tensor) -> impl Fn(&Tensor) -> Result<Tensor> + '_ {
        move |x: &Tensor| a.matmul(x)
    }

    fn max_err(x: &Tensor, y: &Tensor) -> f64 {
        let a: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f64> = y.flatten_all().unwrap().to_vec1().unwrap();
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
    }

    /// The load-bearing property: recover a known solution.
    #[test]
    fn solves_an_spd_system() {
        let n = 24;
        let (a, _) = spd(n, 1.0, 7);
        let dev = Device::Cpu;
        let x_true = Tensor::from_vec(
            (0..n).map(|i| ((i % 7) as f64) - 3.0).collect::<Vec<f64>>(),
            (n, 1),
            &dev,
        )
        .unwrap();
        let b = a.matmul(&x_true).unwrap();

        let opts = CgOptions::for_tolerance(1e-12);
        let (x, stats) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        assert!(max_err(&x, &x_true) < 1e-6, "err {}", max_err(&x, &x_true));
        assert!(stats.iterations <= n + 5, "took {} iterations", stats.iterations);
    }

    /// A `[N]` rhs must come back as `[N]`, not `[N, 1]`.
    #[test]
    fn a_vector_rhs_round_trips_its_shape() {
        let n = 12;
        let (a, _) = spd(n, 2.0, 3);
        let dev = Device::Cpu;
        let x_true =
            Tensor::from_vec((0..n).map(|i| i as f64 * 0.1).collect::<Vec<f64>>(), (n,), &dev)
                .unwrap();
        let b = a.matmul(&x_true.reshape((n, 1)).unwrap()).unwrap().reshape((n,)).unwrap();

        let opts = CgOptions::for_tolerance(1e-12);
        let (x, _) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        assert_eq!(x.dims(), &[n]);
        assert!(max_err(&x, &x_true) < 1e-6);
    }

    /// Each column is an independent system and must be solved as one.
    #[test]
    fn multiple_right_hand_sides_are_solved_independently() {
        let n = 16;
        let (a, _) = spd(n, 1.5, 11);
        let dev = Device::Cpu;
        let x_true = Tensor::from_vec(
            (0..n * 3).map(|i| ((i % 5) as f64) - 2.0).collect::<Vec<f64>>(),
            (n, 3),
            &dev,
        )
        .unwrap();
        let b = a.matmul(&x_true).unwrap();

        let opts = CgOptions::for_tolerance(1e-12);
        let (x, _) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        assert!(max_err(&x, &x_true) < 1e-6);
    }

    /// The case the normalisation guard exists for. A zero column has
    /// solution zero; without the `rhs_is_zero` handling its normalised
    /// residual is 0/0 and the mean never falls below tolerance.
    #[test]
    fn a_zero_rhs_column_yields_zero_without_poisoning_its_neighbours() {
        let n = 10;
        let (a, _) = spd(n, 1.0, 5);
        let dev = Device::Cpu;
        let mut cols = vec![0.0f64; n * 2];
        for i in 0..n {
            cols[i * 2] = 1.0; // column 0 nonzero, column 1 all zeros
        }
        let b = Tensor::from_vec(cols, (n, 2), &dev).unwrap();

        let opts = CgOptions::for_tolerance(1e-12);
        let (x, _) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        let v: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..n {
            assert_eq!(v[i * 2 + 1], 0.0, "zero column should solve to exactly zero");
        }
        // and the real column was actually solved
        let recon = a.matmul(&x).unwrap();
        let rv: Vec<f64> = recon.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..n {
            assert!((rv[i * 2] - 1.0).abs() < 1e-6, "column 0 unsolved: {}", rv[i * 2]);
        }
    }

    /// An exact initial guess should cost no work at all.
    #[test]
    fn an_exact_initial_guess_terminates_immediately() {
        let n = 8;
        let (a, _) = spd(n, 1.0, 2);
        let dev = Device::Cpu;
        let x_true =
            Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();
        let b = a.matmul(&x_true).unwrap();

        let (x, stats) =
            linear_cg(op(&a), &b, Some(&x_true), NO_PRECOND, &CgOptions::default()).unwrap();
        assert_eq!(stats.iterations, 0, "should not have iterated");
        assert!(stats.converged);
        assert!(max_err(&x, &x_true) < 1e-9);
    }

    /// A Jacobi preconditioner must not change the answer — only the path
    /// to it. This is the check that the preconditioned branch computes the
    /// same thing as the plain one.
    #[test]
    fn preconditioning_preserves_the_solution() {
        let n = 20;
        let (a, dense) = spd(n, 0.5, 13);
        let dev = Device::Cpu;
        let x_true = Tensor::from_vec(
            (0..n).map(|i| ((i % 3) as f64) - 1.0).collect::<Vec<f64>>(),
            (n, 1),
            &dev,
        )
        .unwrap();
        let b = a.matmul(&x_true).unwrap();

        let inv_diag: Vec<f64> = (0..n).map(|i| 1.0 / dense[i * n + i]).collect();
        let inv_diag = Tensor::from_vec(inv_diag, (n, 1), &dev).unwrap();
        let precond = |r: &Tensor| r.broadcast_mul(&inv_diag);

        let opts = CgOptions::for_tolerance(1e-12);
        let (x_pre, _) = linear_cg(op(&a), &b, None, Some(precond), &opts).unwrap();
        let (x_raw, _) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();

        assert!(max_err(&x_pre, &x_true) < 1e-6, "preconditioned solve is wrong");
        assert!(max_err(&x_pre, &x_raw) < 1e-5, "the two paths disagree");
    }

    /// The `CgOptions.precond` transport must actually be REACHED — proved by
    /// iteration count, because agreement on the answer proves nothing here
    /// (an ignored preconditioner also agrees).
    ///
    /// The fixture is a DIAGONAL operator with a spectrum spread over six
    /// orders. Jacobi then reproduces `M = A` exactly, so `M⁻¹A = I` and the
    /// preconditioned solve must land in ONE iteration while the plain one
    /// grinds through the spread. An unreachable transport field cannot fake
    /// that; a merely-clustered fixture can (both paths hit the same floor,
    /// which is how an earlier version of this test passed vacuously).
    #[test]
    fn the_options_preconditioner_is_reached_and_cuts_iterations() {
        let dev = Device::Cpu;
        let n = 40usize;
        let d: Vec<f64> = (0..n).map(|i| 10f64.powf(-3.0 + 6.0 * (i as f64) / (n as f64 - 1.0))).collect();
        let mut dense = vec![0.0f64; n * n];
        for i in 0..n {
            dense[i * n + i] = d[i];
        }
        let a = Tensor::from_vec(dense, (n, n), &dev).unwrap();
        let diag = Tensor::from_vec(d, n, &dev).unwrap();
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();

        let m = std::sync::Arc::new(
            crate::operators::precond::CiqPreconditioner::jacobi(&diag, 1e-30).unwrap(),
        );
        let plain = CgOptions::for_tolerance(1e-10);
        let pre = CgOptions { precond: Some(m), ..CgOptions::for_tolerance(1e-10) };

        let (x_plain, s_plain) = linear_cg(op(&a), &b, None, NO_PRECOND, &plain).unwrap();
        // NO_PRECOND as the explicit argument — the ONLY source here is the
        // options field, which is exactly what is under test.
        let (x_pre, s_pre) = linear_cg(op(&a), &b, None, NO_PRECOND, &pre).unwrap();

        // `M⁻¹A = I` exactly, so the Krylov space is one-dimensional and the
        // solve is done after the first step. It still reports 11 because of
        // the `k >= 10` floor above — the guard against a fluke-small early
        // residual. So hitting exactly the floor is the strongest evidence
        // available that the preconditioner was applied: nothing converges
        // sooner than the floor allows.
        const ITER_FLOOR: usize = 11;
        assert_eq!(
            s_pre.iterations, ITER_FLOOR,
            "options preconditioner never reached the solve — an exact M⁻¹ \
             should bottom out at the {ITER_FLOOR}-iteration floor (plain took {})",
            s_plain.iterations
        );
        assert!(
            s_plain.iterations > 5 * ITER_FLOOR,
            "fixture is too easy to prove anything: plain CG took only {}",
            s_plain.iterations
        );
        assert!(max_err(&x_pre, &x_plain) < 1e-6, "the two paths disagree");

        // With the floor lifted, the same solve shows its TRUE cost: `M⁻¹A = I`
        // means a one-dimensional Krylov space, so one iteration. This is the
        // assertion the floor was hiding, and it is the sharpest possible
        // statement that the preconditioner was applied.
        {
            let _g = settings::cg_ignore_min_iter::with(true);
            let (x, s) = linear_cg(op(&a), &b, None, NO_PRECOND, &pre).unwrap();
            assert_eq!(
                s.iterations, 1,
                "an exact M⁻¹ must converge in one iteration once the floor is lifted"
            );
            assert!(max_err(&x, &x_plain) < 1e-6, "early exit changed the answer");
        }
    }

    /// The floor is a SAFEGUARD, so lifting it must not change any answer —
    /// only how long the solve takes to reach it.
    #[test]
    fn cg_ignore_min_iter_saves_iterations_without_moving_the_answer() {
        let n = 20;
        let (a, _) = spd(n, 0.5, 77);
        let dev = Device::Cpu;
        let x_true = Tensor::from_vec(
            (0..n).map(|i| ((i % 5) as f64) - 2.0).collect::<Vec<f64>>(),
            (n, 1),
            &dev,
        )
        .unwrap();
        let b = a.matmul(&x_true).unwrap();
        let opts = CgOptions::for_tolerance(1e-10);

        let (x_floor, s_floor) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        let (x_free, s_free) = {
            let _g = settings::cg_ignore_min_iter::with(true);
            linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap()
        };

        assert!(max_err(&x_free, &x_true) < 1e-6, "lifting the floor broke the solve");
        assert!(max_err(&x_free, &x_floor) < 1e-6, "the two paths disagree");
        assert!(
            s_free.iterations <= s_floor.iterations,
            "lifting the floor cannot make a solve longer: {} vs {}",
            s_free.iterations,
            s_floor.iterations
        );
        assert!(s_floor.converged && s_free.converged);
    }

    /// …but the SAME preconditioner must be REFUSED when a tridiagonal is
    /// wanted, because there the two paths genuinely do not agree and nothing
    /// downstream would notice.
    ///
    /// `alpha`/`beta` come from the preconditioned recurrence, so `T`
    /// describes `M^{-1/2}AM^{-1/2}` and any SLQ built on it is off by
    /// `logdet(M)` — plus a second error from `host_col_norms_sq`'s Euclidean
    /// probe norms. Both are silent: a wrong logdet is a plausible number.
    #[test]
    fn a_preconditioner_is_refused_when_the_tridiagonal_is_wanted() {
        let n = 12;
        let (a, dense) = spd(n, 0.5, 5);
        let dev = Device::Cpu;
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();

        let inv_diag: Vec<f64> = (0..n).map(|i| 1.0 / dense[i * n + i]).collect();
        let inv_diag = Tensor::from_vec(inv_diag, (n, 1), &dev).unwrap();
        let precond = |r: &Tensor| r.broadcast_mul(&inv_diag);

        let opts = CgOptions { n_tridiag: 1, ..CgOptions::for_tolerance(1e-10) };
        let err = linear_cg_tridiag(op(&a), &b, None, Some(precond), &opts)
            .expect_err("a preconditioned tridiagonal must be refused, not silently wrong");
        let msg = err.to_string();
        assert!(
            msg.contains("logdet"),
            "the refusal should name WHY (logdet(M)), got: {msg}"
        );

        // The unpreconditioned tridiagonal is unaffected.
        assert!(linear_cg_tridiag(op(&a), &b, None, NO_PRECOND, &opts).is_ok());
    }

    /// Exhausting the budget must be reported, not hidden. The Python
    /// raises a warning; we return it so the caller can act on it.
    #[test]
    fn a_truncated_solve_reports_that_it_did_not_converge() {
        let n = 40;
        let (a, _) = spd(n, 1e-3, 17); // deliberately ill-conditioned
        let dev = Device::Cpu;
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();

        let opts = CgOptions { max_iter: Some(2), ..CgOptions::for_tolerance(1e-14) };
        let (_, stats) = linear_cg(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        assert!(!stats.converged, "should not claim convergence after 2 iterations");
        assert_eq!(stats.iterations, 2);
    }

    /// The settings must actually govern the solve.
    #[test]
    fn settings_drive_the_defaults() {
        let n = 30;
        let (a, _) = spd(n, 1e-3, 19);
        let dev = Device::Cpu;
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();

        let _t = cg_tolerance::with(1e-14);
        {
            let _m = max_cg_iterations::with(3);
            let (_, stats) = linear_cg(op(&a), &b, None, NO_PRECOND, &CgOptions::default()).unwrap();
            assert_eq!(stats.iterations, 3);
        }
        {
            // terminate_cg_by_size caps at n for an n x n system
            let _m = max_cg_iterations::with(10_000);
            let _s = terminate_cg_by_size::with(true);
            let (_, stats) = linear_cg(op(&a), &b, None, NO_PRECOND, &CgOptions::default()).unwrap();
            assert!(stats.iterations <= n, "ran {} > n = {n}", stats.iterations);
        }
    }

    /// CG and Lanczos span the same Krylov space, so the tridiagonal built
    /// from the CG coefficients must have eigenvalues that INTERLACE the
    /// true spectrum — in particular its extremes lie inside `[λmin, λmax]`.
    /// That is the property stochastic Lanczos quadrature relies on, and a
    /// sign error in the alpha/beta recurrence breaks it immediately.
    #[test]
    fn the_tridiagonal_ritz_values_lie_inside_the_true_spectrum() {
        let n = 20;
        let (a, dense) = spd(n, 1.0, 23);
        let dev = Device::Cpu;
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();

        let opts = CgOptions {
            n_tridiag: 1,
            max_tridiag_iter: Some(10),
            max_iter: Some(60),
            ..CgOptions::for_tolerance(1e-14)
        };
        let (_, _, t) = linear_cg_tridiag(op(&a), &b, None, NO_PRECOND, &opts).unwrap();
        assert_eq!(t.dims()[0], 1, "one tridiagonal per requested column");
        let k = t.dims()[1];
        assert!(k > 1, "expected several Lanczos rows, got {k}");

        // True spectrum bounds via a few power iterations on A and on
        // (cI - A) for the smallest.
        let (lo, hi) = spectrum_bounds(&dense, n);

        let tv: Vec<f64> = t.flatten_all().unwrap().to_vec1().unwrap();
        let (ritz, _) = crate::linalg::jacobi_eigh(&tv, k);
        for r in &ritz {
            assert!(
                *r >= lo - 1e-6 * hi.abs() && *r <= hi + 1e-6 * hi.abs(),
                "Ritz value {r} outside the true spectrum [{lo}, {hi}]"
            );
        }
    }

    /// EXACT spectral bounds, via the same Jacobi routine the rest of the
    /// crate uses.
    ///
    /// Power iteration was the obvious choice and the wrong one: its
    /// estimate of the SMALLEST eigenvalue converges slowly, and a bound
    /// that is itself off by a percent cannot adjudicate whether a Ritz
    /// value sits inside the spectrum. At n = 20 the exact answer is free.
    ///
    /// Note the pre-scaling: `jacobi_eigh`'s convergence test is absolute
    /// (`off.sqrt() < 1e-14`), so a matrix with entries of order 100 would
    /// burn all 100 sweeps chasing a threshold it cannot express. Scale to
    /// unit Frobenius norm, decompose, scale the eigenvalues back.
    fn spectrum_bounds(a: &[f64], n: usize) -> (f64, f64) {
        let fro = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scaled: Vec<f64> = a.iter().map(|x| x / fro).collect();
        let (eig, _) = crate::linalg::jacobi_eigh(&scaled, n);
        let lo = eig.iter().cloned().fold(f64::INFINITY, f64::min) * fro;
        let hi = eig.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * fro;
        (lo, hi)
    }

    #[test]
    fn requesting_a_tridiag_larger_than_the_iteration_budget_is_refused() {
        let n = 6;
        let (a, _) = spd(n, 1.0, 1);
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &Device::Cpu).unwrap();
        let opts = CgOptions {
            n_tridiag: 1,
            max_iter: Some(3),
            max_tridiag_iter: Some(10),
            ..Default::default()
        };
        let e = linear_cg_tridiag(op(&a), &b, None, NO_PRECOND, &opts).unwrap_err();
        assert!(e.to_string().contains("Lanczos coefficients"), "{e}");
    }

    #[test]
    fn a_non_finite_operator_is_reported_not_iterated() {
        let n = 4;
        let dev = Device::Cpu;
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();
        let bad = |x: &Tensor| x.affine(f64::NAN, 0.0);
        let e = linear_cg(bad, &b, None, NO_PRECOND, &CgOptions::default()).unwrap_err();
        assert!(e.to_string().contains("non-finite"), "{e}");
    }

    /// The trap `for_tolerance` exists to avoid, pinned so it stays
    /// visible.
    ///
    /// `beta`'s denominator is `‖r‖²`, so the `den < eps` guard fires once
    /// `‖r‖ < √eps`, zeroing beta and collapsing CG into steepest descent.
    /// Asking for a tolerance below that floor with the inherited
    /// `eps = 1e-10` does not error — it burns the entire iteration budget
    /// and returns a plausible wrong answer.
    #[test]
    fn a_tolerance_below_sqrt_eps_stagnates_unless_eps_follows_it() {
        let n = 24;
        let (a, _) = spd(n, 1.0, 7);
        let dev = Device::Cpu;
        let x_true = Tensor::from_vec(
            (0..n).map(|i| ((i % 7) as f64) - 3.0).collect::<Vec<f64>>(),
            (n, 1),
            &dev,
        )
        .unwrap();
        let b = a.matmul(&x_true).unwrap();

        // Inherited eps, tolerance far below its √eps ≈ 1e-5 floor.
        let naive = CgOptions { tolerance: Some(1e-12), ..Default::default() };
        let (x_bad, stats_bad) = linear_cg(op(&a), &b, None, NO_PRECOND, &naive).unwrap();
        assert!(!stats_bad.converged, "stagnation should not be reported as success");
        assert!(
            stats_bad.iterations > n * 4,
            "expected the full budget to burn, took {}",
            stats_bad.iterations
        );
        assert!(
            max_err(&x_bad, &x_true) > 1e-6,
            "if this no longer stagnates the guard has changed and the doc is stale"
        );

        // Same tolerance, eps chosen to match it.
        let (x_ok, stats_ok) =
            linear_cg(op(&a), &b, None, NO_PRECOND, &CgOptions::for_tolerance(1e-12)).unwrap();
        assert!(stats_ok.converged);
        assert!(stats_ok.iterations <= n + 5, "took {}", stats_ok.iterations);
        assert!(max_err(&x_ok, &x_true) < 1e-6);
    }

    /// `safe_div` must yield exactly zero, not inf or NaN, on a zero
    /// denominator — the property the whole iteration's NaN-freedom rests on.
    #[test]
    fn safe_div_returns_zero_rather_than_infinity() {
        let dev = Device::Cpu;
        let num = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], (1, 3), &dev).unwrap();
        let den = Tensor::from_vec(vec![0.0f64, 1e-30, 2.0], (1, 3), &dev).unwrap();
        let q = safe_div(&num, &den, 1e-10).unwrap();
        let v: Vec<f64> = q.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert!((v[2] - 1.5).abs() < 1e-12);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// Typed `None` for the preconditioner parameter.
    const NO_PRECOND: Option<fn(&Tensor) -> Result<Tensor>> = None;

    // ---- multi-shift CG ----------------------------------------------------

    const SHIFTS: [f64; 5] = [1e-4, 1e-2, 1.0, 10.0, 1e3];

    /// The oracle every msCG test diffs against: independent `linear_cg`
    /// solves of the explicitly shifted systems.
    fn per_shift_reference(a: &Tensor, b: &Tensor, shifts: &[f64], opts: &CgOptions) -> Vec<Tensor> {
        shifts
            .iter()
            .map(|&s| {
                let shifted_op = |x: &Tensor| -> Result<Tensor> {
                    a.broadcast_matmul(x)? + x.affine(s, 0.0)?
                };
                linear_cg(shifted_op, b, None, NO_PRECOND, opts).unwrap().0
            })
            .collect()
    }

    #[test]
    fn shifted_cg_matches_the_per_shift_reference() {
        let dev = Device::Cpu;
        let (a, _) = spd(24, 1.0, 7);
        let b = Tensor::from_vec((0..24).map(|i| ((i * 37 % 11) as f64) - 5.0).collect::<Vec<_>>(), (24, 1), &dev).unwrap();
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_cg_shifted(op(&a), &b, &SHIFTS, &opts).unwrap();
        assert!(stats.converged, "seed did not converge: {stats:?}");
        let refs = per_shift_reference(&a, &b, &SHIFTS, &opts);
        for ((x, r), s) in xs.iter().zip(&refs).zip(&SHIFTS) {
            let err = max_err(x, r);
            assert!(err < 1e-6, "shift {s}: msCG vs per-shift reference differ by {err:.3e}");
        }
    }

    #[test]
    fn shifted_cg_matches_a_dense_solve() {
        let dev = Device::Cpu;
        let (a, _) = spd(16, 0.5, 3);
        let b = Tensor::from_vec((0..16).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>(), (16, 1), &dev).unwrap();
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xs, _) = linear_cg_shifted(op(&a), &b, &SHIFTS, &opts).unwrap();
        let eye = Tensor::eye(16, candle_core::DType::F64, &dev).unwrap();
        for (x, &s) in xs.iter().zip(&SHIFTS) {
            let a_s = (&a + eye.affine(s, 0.0).unwrap()).unwrap();
            let x_dense = crate::linalg::cholesky_solve(&a_s, &b, 0.0).unwrap();
            let err = max_err(x, &x_dense);
            assert!(err < 1e-6, "shift {s}: msCG vs dense solve differ by {err:.3e}");
        }
    }

    /// Pins the seed-choice rationale: a matrix whose λmin is slightly
    /// NEGATIVE (fp noise scale) must still solve cleanly, because the seed
    /// system is `A + σ_min I`, never `A` itself. Diagonal, so the exact
    /// per-shift solutions are `b_i / (d_i + σ)`.
    #[test]
    fn the_seed_is_the_smallest_shift_so_a_barely_indefinite_matrix_still_solves() {
        let dev = Device::Cpu;
        let d: Vec<f64> = vec![-1e-8, 0.3, 0.7, 1.1, 1.9, 2.5, 3.0, 4.2];
        let n = d.len();
        let diag = Tensor::from_vec(d.clone(), (n, 1), &dev).unwrap();
        let dmat = |x: &Tensor| -> Result<Tensor> { x.broadcast_mul(&diag) };
        let b = Tensor::from_vec((0..n).map(|i| 1.0 + i as f64).collect::<Vec<_>>(), (n, 1), &dev).unwrap();
        let shifts = [1e-4, 1.0];
        let opts = CgOptions { max_iter: Some(200), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_cg_shifted(dmat, &b, &shifts, &opts).unwrap();
        assert!(stats.converged);
        for (x, &s) in xs.iter().zip(&shifts) {
            let got: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
            for (i, (gi, di)) in got.iter().zip(&d).enumerate() {
                let want = (1.0 + i as f64) / (di + s);
                assert!(
                    (gi - want).abs() < 1e-6 * want.abs().max(1.0),
                    "shift {s} row {i}: {gi} vs {want}"
                );
            }
        }
    }

    /// ζ ∈ (0, 1] means every shifted residual is bounded by the seed's —
    /// the no-breakdown property the seed choice buys.
    #[test]
    fn shifted_residuals_never_exceed_the_seed_residual() {
        let dev = Device::Cpu;
        let (a, _) = spd(20, 0.1, 11);
        let b = Tensor::from_vec((0..20).map(|i| ((i * 13 % 7) as f64) - 3.0).collect::<Vec<_>>(), (20, 1), &dev).unwrap();
        // A budget small enough that nothing fully converges, so the
        // residuals are genuinely nonzero and the ordering is informative.
        let opts = CgOptions { max_iter: Some(6), ..CgOptions::for_tolerance(1e-14) };
        let (_, stats) = linear_cg_shifted(op(&a), &b, &SHIFTS, &opts).unwrap();
        let seed_norm = stats.residual_norms[0]; // SHIFTS[0] is the smallest
        for (&s, &rn) in SHIFTS.iter().zip(&stats.residual_norms) {
            assert!(
                rn <= seed_norm * (1.0 + 1e-9),
                "shift {s}: residual {rn:.3e} exceeds the seed's {seed_norm:.3e}"
            );
        }
    }

    #[test]
    fn shifted_cg_solves_batched_multi_rhs() {
        let dev = Device::Cpu;
        // Three different SPD blocks, two RHS columns each.
        let blocks: Vec<Tensor> = (0..3).map(|s| spd(16, 0.5 + s as f64, 20 + s as u64).0).collect();
        let a = Tensor::stack(&blocks, 0).unwrap(); // [3, 16, 16]
        let b = Tensor::from_vec(
            (0..3 * 16 * 2).map(|i| ((i * 29 % 17) as f64) / 7.0 - 1.0).collect::<Vec<_>>(),
            (3, 16, 2),
            &dev,
        )
        .unwrap();
        let bop = |x: &Tensor| -> Result<Tensor> { a.broadcast_matmul(x) };
        let shifts = [1e-2, 1.0, 50.0];
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_cg_shifted(bop, &b, &shifts, &opts).unwrap();
        assert!(stats.converged);
        let refs = per_shift_reference(&a, &b, &shifts, &opts);
        for ((x, r), s) in xs.iter().zip(&refs).zip(&shifts) {
            assert_eq!(x.dims(), &[3, 16, 2]);
            let err = max_err(x, r);
            assert!(err < 1e-6, "shift {s}: batched msCG off by {err:.3e}");
        }
    }

    #[test]
    fn an_ill_conditioned_seed_reports_nonconvergence_honestly() {
        let dev = Device::Cpu;
        // κ = 1e6 diagonal; a 5-iteration budget cannot solve it.
        let n = 32;
        let d: Vec<f64> = (0..n).map(|i| 1e-6f64.powf(1.0 - i as f64 / (n - 1) as f64)).collect();
        let diag = Tensor::from_vec(d, (n, 1), &dev).unwrap();
        let dmat = |x: &Tensor| -> Result<Tensor> { x.broadcast_mul(&diag) };
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();
        let opts = CgOptions { max_iter: Some(5), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_cg_shifted(dmat, &b, &[1e-8, 1e-2], &opts).unwrap();
        assert!(!stats.converged, "cannot have converged in 5 iterations at κ=1e6");
        for x in &xs {
            let v: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
            assert!(v.iter().all(|q| q.is_finite()), "nonconverged solve leaked non-finites");
        }
    }

    #[test]
    fn shifted_cg_refuses_negative_or_empty_shifts() {
        let dev = Device::Cpu;
        let (a, _) = spd(8, 1.0, 5);
        let b = Tensor::from_vec(vec![1.0f64; 8], (8, 1), &dev).unwrap();
        assert!(linear_cg_shifted(op(&a), &b, &[], &CgOptions::default()).is_err());
        assert!(linear_cg_shifted(op(&a), &b, &[1.0, -0.5], &CgOptions::default()).is_err());
    }

    /// The module doc of `precond.rs` has always said the shifted solvers
    /// "refuse preconditioners outright". They did not — both took a
    /// `&CgOptions` and silently DROPPED `opts.precond`, so a caller who set
    /// it got an unpreconditioned solve and no way to know. A documented
    /// restriction that the code does not enforce is the same as no
    /// restriction, except it also misleads whoever reads the doc.
    ///
    /// **Vacuous version to refuse:** asserting only that the call errors.
    /// It must error for THIS reason, so the message is asserted too.
    #[test]
    fn a_shifted_solve_refuses_a_preconditioner() {
        use crate::operators::precond::CiqPreconditioner;
        let dev = Device::Cpu;
        let n = 6;
        let diag = Tensor::full(2.0f64, n, &dev).unwrap();
        let p = std::sync::Arc::new(CiqPreconditioner::jacobi(&diag, 1e-8).unwrap());
        let a = Tensor::eye(n, DType::F64, &dev).unwrap().affine(2.0, 0.0).unwrap();
        let rhs = Tensor::ones((n, 1), DType::F64, &dev).unwrap();
        let matmul = |x: &Tensor| a.matmul(x);
        let opts = CgOptions { precond: Some(p), ..CgOptions::for_tolerance(1e-8) };

        for e in [
            linear_cg_shifted(&matmul, &rhs, &[0.0, 1.0], &opts).unwrap_err().to_string(),
            crate::operators::minres::linear_minres_shifted(&matmul, &rhs, &[0.0, 1.0], &opts)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(e.contains("preconditioner"), "must name the cause: {e}");
            assert!(e.contains("shifted family"), "must say WHY it cannot: {e}");
        }

        // …and without one, both still solve. Otherwise the guard could be
        // rejecting everything and this test would not notice.
        let plain = CgOptions::for_tolerance(1e-8);
        assert!(linear_cg_shifted(&matmul, &rhs, &[0.0, 1.0], &plain).is_ok());
        assert!(
            crate::operators::minres::linear_minres_shifted(&matmul, &rhs, &[0.0, 1.0], &plain)
                .is_ok()
        );
    }

    /// `cg::run` must charge the spectral cache's odometer once per solve, in
    /// matvec-columns. Without it every entry lives forever and the cache
    /// silently becomes "never refresh".
    ///
    /// **Vacuous version to refuse:** asserting the odometer merely moved. It
    /// has to move by `iterations × columns`, because a per-ITERATION tick
    /// (the obvious wrong placement) also moves it and would age a 1-column
    /// solve and a 64-column solve identically.
    #[test]
    fn a_solve_charges_the_cache_odometer_in_columns() {
        use crate::operators::cgcache;
        let dev = Device::Cpu;
        let n = 8;
        let (a, _) = spd(n, 1.0, 5);
        let opts = CgOptions::for_tolerance(1e-10);

        for cols in [1usize, 4] {
            cgcache::with_cache(|c| c.clear());
            let rhs = Tensor::ones((n, cols), DType::F64, &dev).unwrap();
            let (_, stats) =
                linear_cg(op(&a), &rhs, None, NO_PRECOND, &opts).unwrap();
            let charged = cgcache::with_cache(|c| c.odometer());
            assert!(stats.iterations > 1, "fixture converged too fast to distinguish");
            assert_eq!(
                charged,
                stats.iterations as u64 * cols as u64,
                "odometer charged {charged} for {} iterations × {cols} columns",
                stats.iterations
            );
        }
        cgcache::with_cache(|c| c.clear());
    }
}
