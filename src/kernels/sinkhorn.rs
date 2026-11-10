//! Inferno — Log-stabilised entropic transport divergence and dual potentials on Candle tensors.
//! Gradients replay the final 16 iterations rather than the entire solve.

use super::sq_pdist;
use candle_core::shape::Dim;
use candle_core::{DType, Device, Result as CResult, Tensor, D};

#[derive(Clone, Debug)]
pub enum SinkhornCost {
    /// C_ij = ½ ||x_i - y_j||²
    SquaredEuclidean,
    /// Caller supplies the cost matrix [n, m] directly.
    Custom(Tensor),
}

#[derive(Copy, Clone, Debug)]
pub struct SinkhornOptions {
    /// Entropic regularization ε. Lower = sharper transport plan, more
    /// iterations needed.
    pub eps: f64,
    pub max_iters: usize,
    /// Stop when both potentials' max-abs update falls below this.
    pub tol: f64,
    /// If true, return the *debiased* Sinkhorn divergence
    /// `S_ε = OT_ε(a,b) - ½ OT_ε(a,a) - ½ OT_ε(b,b)`. If false, return
    /// the raw `OT_ε(a, b)`.
    pub debias: bool,
}

impl Default for SinkhornOptions {
    fn default() -> Self {
        Self {
            eps: 0.05,
            max_iters: 200,
            tol: 1e-6,
            debias: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SinkhornOutput {
    /// Scalar divergence (or raw OT cost if `debias=false`).
    pub divergence: Tensor,
    /// Dual potential on x: shape `[n]`.
    pub f: Tensor,
    /// Dual potential on y: shape `[m]`.
    pub g: Tensor,
    /// Number of Sinkhorn iterations actually run.
    pub iterations: usize,
}

/// Sinkhorn divergence between empirical distributions over `x [n, d]`
/// and `y [m, d]`. Uniform marginals are used unless the caller supplies
/// `(a, b)` via [`sinkhorn_divergence_weighted`].
pub fn sinkhorn_divergence(
    x: &Tensor,
    y: &Tensor,
    cost: SinkhornCost,
    opts: SinkhornOptions,
) -> CResult<SinkhornOutput> {
    let n = x.dim(0)?;
    let m = y.dim(0)?;
    let device = x.device();
    let dtype = x.dtype();
    let a = uniform_marginal(n, dtype, device)?;
    let b = uniform_marginal(m, dtype, device)?;
    sinkhorn_divergence_weighted(x, y, &a, &b, cost, opts)
}

pub fn sinkhorn_divergence_weighted(
    x: &Tensor,
    y: &Tensor,
    a: &Tensor,
    b: &Tensor,
    cost: SinkhornCost,
    opts: SinkhornOptions,
) -> CResult<SinkhornOutput> {
    let c_xy = build_cost(x, y, &cost)?;
    let (ot_xy, f, g, iters) = log_sinkhorn(&c_xy, a, b, opts)?;

    if !opts.debias {
        return Ok(SinkhornOutput {
            divergence: ot_xy,
            f,
            g,
            iterations: iters,
        });
    }

    // Debiasing: OT_ε(x,x) and OT_ε(y,y) — same opts, symmetric marginals.
    //
    // REFUSAL: the self-legs below are built with SquaredEuclidean, so under
    // a Custom cost they would measure a DIFFERENT geometry than the (x,y)
    // leg. The debias then fails to cancel and S(x,x) is not 0 — a plausible
    // finite number rather than an error, which is the failure mode worth a
    // hard stop. A correct Custom debias needs c_xx and c_yy supplied too
    // (they are [N,N] and [M,M]; the caller's Custom is [N,M], so they
    // cannot be derived from it). Until SinkhornCost carries all three,
    // refuse rather than return a silently wrong divergence.
    if matches!(cost, SinkhornCost::Custom(_)) {
        return Err(candle_core::Error::Msg(
            "sinkhorn_divergence: debias=true with SinkhornCost::Custom would debias against \
             a squared-Euclidean self-geometry, so S(x,x) != 0. Pass debias=false, or use \
             SinkhornCost::SquaredEuclidean."
                .into(),
        ));
    }
    let c_xx = build_cost(x, x, &SinkhornCost::SquaredEuclidean)?;
    let (ot_xx, _, _, _) = log_sinkhorn(&c_xx, a, a, opts)?;
    let c_yy = build_cost(y, y, &SinkhornCost::SquaredEuclidean)?;
    let (ot_yy, _, _, _) = log_sinkhorn(&c_yy, b, b, opts)?;

    let half = (ot_xx + ot_yy)?.affine(0.5, 0.0)?;
    let div = ot_xy.sub(&half)?;
    Ok(SinkhornOutput {
        divergence: div,
        f,
        g,
        iterations: iters,
    })
}

/// Return just the potentials `(f, g)` and number of iterations. Useful
/// when you want to plug them into a forward / backward SDE drift
/// (matches `bridge.py`'s Sinkhorn-correction term).
pub fn sinkhorn_potentials(
    x: &Tensor,
    y: &Tensor,
    cost: SinkhornCost,
    opts: SinkhornOptions,
) -> CResult<(Tensor, Tensor, usize)> {
    let n = x.dim(0)?;
    let m = y.dim(0)?;
    let a = uniform_marginal(n, x.dtype(), x.device())?;
    let b = uniform_marginal(m, y.dtype(), y.device())?;
    let c = build_cost(x, y, &cost)?;
    let (_, f, g, iters) = log_sinkhorn(&c, &a, &b, opts)?;
    Ok((f, g, iters))
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn uniform_marginal(n: usize, dtype: DType, device: &Device) -> CResult<Tensor> {
    let w = 1.0_f64 / n as f64;
    Tensor::full(w, (n,), device)?.to_dtype(dtype)
}

fn build_cost(x: &Tensor, y: &Tensor, cost: &SinkhornCost) -> CResult<Tensor> {
    match cost {
        SinkhornCost::SquaredEuclidean => {
            // C = ½ ||x - y||²
            Ok((sq_pdist(x, y)? * 0.5_f64)?)
        }
        SinkhornCost::Custom(c) => Ok(c.clone()),
    }
}

/// Log-stabilized Sinkhorn. Updates dual potentials f ∈ ℝⁿ, g ∈ ℝᵐ via
///
/// ```text
/// f_i ← -ε · LSE_j [ (g_j - C_ij) / ε  +  log b_j ]
/// g_j ← -ε · LSE_i [ (f_i - C_ij) / ε  +  log a_i ]
/// ```
///
/// Returns `(OT_ε, f, g, iters_run)`.
fn log_sinkhorn(
    c: &Tensor,
    a: &Tensor,
    b: &Tensor,
    opts: SinkhornOptions,
) -> CResult<(Tensor, Tensor, Tensor, usize)> {
    let device = c.device();
    let dtype = c.dtype();
    let n = c.dim(0)?;
    let m = c.dim(1)?;
    let eps = opts.eps;
    let log_a = a.log()?; // [n]
    let log_b = b.log()?; // [m]

    let mut f = Tensor::zeros((n,), dtype, device)?;
    let mut g = Tensor::zeros((m,), dtype, device)?;

    let mut iters = 0;
    for _ in 0..opts.max_iters {
        iters += 1;

        // M_ij = (g_j - C_ij) / ε + log b_j        [n, m]
        // f_i = -ε · logsumexp_j M_ij
        let g_b = (g.add(&log_b.affine(eps, 0.0)?)?.unsqueeze(0)?).broadcast_as((n, m))?; // [n, m]
        let m_ij = g_b.sub(c)?.affine(1.0 / eps, 0.0)?;
        let new_f = lse(&m_ij, D::Minus1)?.affine(-eps, 0.0)?;

        // M'_ji = (f_i - C_ij) / ε + log a_i       [m, n]
        // g_j = -ε · logsumexp_i M'
        let f_a = (new_f.add(&log_a.affine(eps, 0.0)?)?.unsqueeze(1)?).broadcast_as((n, m))?;
        let m_ji_t = f_a.sub(c)?.affine(1.0 / eps, 0.0)?;
        let new_g = lse(&m_ji_t, 0_usize)?.affine(-eps, 0.0)?;

        // Convergence check (max-abs update).
        let df = (new_f.sub(&f)?.abs()?).max(0_usize)?;
        let dg = (new_g.sub(&g)?.abs()?).max(0_usize)?;
        let stop_f = df.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        let stop_g = dg.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        f = new_f;
        g = new_g;
        if stop_f.max(stop_g) < opts.tol {
            break;
        }
    }

    // OT_ε = ⟨a, f⟩ + ⟨b, g⟩
    let inner_a = (a * &f)?.sum_all()?;
    let inner_b = (b * &g)?.sum_all()?;
    let ot = inner_a.add(&inner_b)?;
    Ok((ot, f, g, iters))
}

/// Log-sum-exp along an axis. Uses the standard `max + log(sum exp(x-max))`
/// trick to stay finite for large negative entries.
fn lse<Dm: Dim + Copy>(t: &Tensor, axis: Dm) -> CResult<Tensor> {
    let m = t.max_keepdim(axis)?; // [..., 1, ...]
    let shifted = t.broadcast_sub(&m)?;
    let summed = shifted.exp()?.sum(axis)?; // axis reduced
    let m_red = m.squeeze(axis)?;
    summed.log()?.add(&m_red)
}

/// Graph-preserving weighted coupling used by the GMM bridge and latent drift.
pub use pandorapurple::kernels::sinkhorn::sinkhorn_coupling_t;
