//! Fused pairwise kernel primitives: forward and closed-form VJP.
//!
//! WHY THESE ARE FUSED OPS RATHER THAN COMPOSITIONS
//!
//! Every kernel here maps `x1 [M,D]`, `x2 [N,D]` to `[M,N]` through a
//! contraction over the feature axis. Composed from elementwise primitives
//! each one materialises an `[M,N,D]` intermediate — the blow-up KeOps
//! exists to avoid — and its backward stores that intermediate too. Fused,
//! the forward writes `[M,N]` directly and the backward accumulates into
//! `[M,D]` / `[N,D]` without ever forming the rank-3 tensor.
//!
//! For `softmin` the argument is stronger still: the closed-form VJP needs
//! only the transport plan `P = softmax_j(A)`, so the backward is O(1) graph
//! depth instead of unrolling a logsumexp.
//!
//! CONSTANTS AND CONVENTIONS are taken from the existing Rust kernel
//! (`pandorapurple::gp::coupling_kernel::pair_covar_lazy`) so the two agree
//! numerically, NOT from the textbook forms:
//!   * `MAT52_SQRT5 = 2.236` and `MAT52_FIVE_THIRDS = 1.6666667` are the
//!     literals that implementation uses; the derivatives below differentiate
//!     what is actually computed, so gradcheck matches exactly.
//!   * RBF is `exp(-d²)`, with NO factor of 1/2.
//!   * `scale_52` and `inv_bandwidth` are INVERSE lengthscales (they
//!     multiply distances).
//!
//! Shapes are 2-D. A leading batch axis is a caller-side loop for now; the
//! batched form arrives with the GPU path, where it is a grid dimension
//! rather than extra index arithmetic.

use crate::error::{Error, Result};

/// `+1e-12` inside the Matérn `sqrt`. Not defensive rounding: `d/dx √x` is
/// infinite at 0 and self-pairs sit exactly there, so without it the
/// diagonal gradient is NaN. Same value and same reason as
/// `pair_covar_lazy`.
pub const SQRT_EPS: f64 = 1e-12;

/// The literals `pair_covar_lazy` uses (not `5f64.sqrt()` / `5.0/3.0`).
pub const MAT52_SQRT5: f64 = 2.236;
pub const MAT52_FIVE_THIRDS: f64 = 1.6666667;

/// `[M,D] x [N,D] -> [M,N]` pairwise problem shape.
#[derive(Copy, Clone, Debug)]
pub struct PairShape {
    pub m: usize,
    pub n: usize,
    pub d: usize,
}

impl PairShape {
    pub fn check(&self, x1: &[f64], x2: &[f64], w: &[f64]) -> Result<()> {
        if x1.len() != self.m * self.d || x2.len() != self.n * self.d || w.len() != self.d {
            return Err(Error::Msg(format!(
                "pair kernel: expected x1 {}x{}, x2 {}x{}, w {} — got {}, {}, {}",
                self.m,
                self.d,
                self.n,
                self.d,
                self.d,
                x1.len(),
                x2.len(),
                w.len()
            )));
        }
        Ok(())
    }
}

/// `d2[i,j] = Σ_d ((x1[i,d] − x2[j,d])·w[d])²` — the scaled squared distance
/// every stationary kernel here is a function of.
pub fn sqdist_scaled(x1: &[f64], x2: &[f64], w: &[f64], s: PairShape) -> Vec<f64> {
    let mut out = vec![0.0; s.m * s.n];
    for i in 0..s.m {
        for j in 0..s.n {
            let mut acc = 0.0;
            for d in 0..s.d {
                let t = (x1[i * s.d + d] - x2[j * s.d + d]) * w[d];
                acc += t * t;
            }
            out[i * s.n + j] = acc;
        }
    }
    out
}

/// Shared backward for anything that factors through `sqdist_scaled`.
///
/// `outer[i,j] = v[i,j] · ∂k[i,j]/∂d²[i,j]` — the only kernel-specific part.
/// With `Δ = x1[i,d] − x2[j,d]`:
///   ∂d²/∂x1 = 2Δw²,  ∂d²/∂x2 = −2Δw²,  ∂d²/∂w = 2Δ²w
pub fn sqdist_scaled_backward(
    x1: &[f64],
    x2: &[f64],
    w: &[f64],
    outer: &[f64],
    s: PairShape,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut gx1 = vec![0.0; s.m * s.d];
    let mut gx2 = vec![0.0; s.n * s.d];
    let mut gw = vec![0.0; s.d];
    for i in 0..s.m {
        for j in 0..s.n {
            let o = outer[i * s.n + j];
            if o == 0.0 {
                continue;
            }
            for d in 0..s.d {
                let delta = x1[i * s.d + d] - x2[j * s.d + d];
                let w_d = w[d];
                let common = 2.0 * o * delta * w_d;
                gx1[i * s.d + d] += common * w_d;
                gx2[j * s.d + d] -= common * w_d;
                gw[d] += common * delta;
            }
        }
    }
    (gx1, gx2, gw)
}

// ---------------------------------------------------------------------------
// Matern 5/2
// ---------------------------------------------------------------------------

/// `k = (1 + a + (5/3)·d²·s²)·exp(−a)` with `a = √5·√(d²+τ)·s`.
/// `s` is an INVERSE lengthscale.
pub fn matern52(x1: &[f64], x2: &[f64], w: &[f64], scale: f64, sh: PairShape) -> Result<Vec<f64>> {
    sh.check(x1, x2, w)?;
    let d2 = sqdist_scaled(x1, x2, w, sh);
    Ok(d2
        .iter()
        .map(|&q| {
            let r = (q + SQRT_EPS).sqrt();
            let a = MAT52_SQRT5 * r * scale;
            (1.0 + a + MAT52_FIVE_THIRDS * q * scale * scale) * (-a).exp()
        })
        .collect())
}

/// VJP of [`matern52`] -> (dx1, dx2, dw, dscale).
pub fn matern52_backward(
    x1: &[f64],
    x2: &[f64],
    w: &[f64],
    scale: f64,
    v: &[f64],
    sh: PairShape,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    sh.check(x1, x2, w)?;
    let d2 = sqdist_scaled(x1, x2, w, sh);
    let mut outer = vec![0.0; d2.len()];
    let mut gscale = 0.0;
    for (idx, &q) in d2.iter().enumerate() {
        let r = (q + SQRT_EPS).sqrt();
        let a = MAT52_SQRT5 * r * scale;
        let poly = 1.0 + a + MAT52_FIVE_THIRDS * q * scale * scale;
        let e = (-a).exp();
        // d/dq: a depends on q through r = √(q+τ), dr/dq = 1/(2r)
        let da_dq = MAT52_SQRT5 * scale / (2.0 * r);
        let dpoly_dq = da_dq + MAT52_FIVE_THIRDS * scale * scale;
        outer[idx] = v[idx] * e * (dpoly_dq - poly * da_dq);
        // d/dscale at fixed q
        let da_ds = MAT52_SQRT5 * r;
        let dpoly_ds = da_ds + 2.0 * MAT52_FIVE_THIRDS * q * scale;
        gscale += v[idx] * e * (dpoly_ds - poly * da_ds);
    }
    let (gx1, gx2, gw) = sqdist_scaled_backward(x1, x2, w, &outer, sh);
    Ok((gx1, gx2, gw, gscale))
}

// ---------------------------------------------------------------------------
// RBF  —  k = exp(-d²)   (no 1/2; matches pair_covar_lazy)
// ---------------------------------------------------------------------------

pub fn rbf(x1: &[f64], x2: &[f64], w: &[f64], sh: PairShape) -> Result<Vec<f64>> {
    sh.check(x1, x2, w)?;
    Ok(sqdist_scaled(x1, x2, w, sh)
        .into_iter()
        .map(|q| (-q).exp())
        .collect())
}

pub fn rbf_backward(
    x1: &[f64],
    x2: &[f64],
    w: &[f64],
    v: &[f64],
    sh: PairShape,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    sh.check(x1, x2, w)?;
    let d2 = sqdist_scaled(x1, x2, w, sh);
    // ∂k/∂d² = −k
    let outer: Vec<f64> = d2
        .iter()
        .enumerate()
        .map(|(idx, &q)| -v[idx] * (-q).exp())
        .collect();
    Ok(sqdist_scaled_backward(x1, x2, w, &outer, sh))
}

// ---------------------------------------------------------------------------
// Rational quadratic  —  k = (1 + d²/(2α))^(−α), TENSOR exponent
// ---------------------------------------------------------------------------

pub fn rational_quadratic(
    x1: &[f64],
    x2: &[f64],
    w: &[f64],
    alpha: f64,
    sh: PairShape,
) -> Result<Vec<f64>> {
    sh.check(x1, x2, w)?;
    if alpha <= 0.0 {
        return Err(Error::Msg(format!(
            "rational_quadratic: alpha must be > 0, got {alpha}"
        )));
    }
    Ok(sqdist_scaled(x1, x2, w, sh)
        .into_iter()
        .map(|q| (1.0 + q / (2.0 * alpha)).powf(-alpha))
        .collect())
}

/// VJP -> (dx1, dx2, dw, dalpha).
///
/// `u = 1 + d²/(2α)`; `∂k/∂d² = −u^(−α−1)/2`;
/// `∂k/∂α = k·(−ln u + d²/(2αu))` — the second term is the α inside `u`,
/// which a naive "treat the exponent as constant" derivative drops.
pub fn rational_quadratic_backward(
    x1: &[f64],
    x2: &[f64],
    w: &[f64],
    alpha: f64,
    v: &[f64],
    sh: PairShape,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    sh.check(x1, x2, w)?;
    let d2 = sqdist_scaled(x1, x2, w, sh);
    let mut outer = vec![0.0; d2.len()];
    let mut galpha = 0.0;
    for (idx, &q) in d2.iter().enumerate() {
        let u = 1.0 + q / (2.0 * alpha);
        let k = u.powf(-alpha);
        outer[idx] = -0.5 * v[idx] * u.powf(-alpha - 1.0);
        galpha += v[idx] * k * (-u.ln() + q / (2.0 * alpha * u));
    }
    let (gx1, gx2, gw) = sqdist_scaled_backward(x1, x2, w, &outer, sh);
    Ok((gx1, gx2, gw, galpha))
}

// ---------------------------------------------------------------------------
// Gibbs  —  the NON-STATIONARY one (Gibbs 1997; Paciorek & Schervish 2004)
// ---------------------------------------------------------------------------
//
// Note the name collision inside this repo: `stochastics::sinkhorn`'s
// "Gibbs formula" is the OT kernel exp(−C/ε), and `gibbs_alpha_prior` is a
// loss term over a Gibbs measure. This is neither — it is the covariance
// with input-dependent lengthscales:
//
//   k(x,x') = ∏_d [2 ℓ_d(x) ℓ_d(x') / (ℓ_d(x)²+ℓ_d(x)'²)]^{1/2}
//             · exp( −Σ_d (x_d−x'_d)² / (ℓ_d(x)²+ℓ_d(x')²) )
//
// Lengthscales are INPUTS (`l1 [M,D]`, `l2 [N,D]`), not parameters: the
// whole point is that a network emits them per point, so the kernel must be
// differentiable w.r.t. them. It reduces to RBF when ℓ is constant, which is
// the property the tests pin.
//
// Evaluated in logs — the prefactor is a product over D of square roots, and
// multiplying those directly underflows for even modest D.

fn gibbs_check(l1: &[f64], l2: &[f64], sh: PairShape) -> Result<()> {
    if l1.len() != sh.m * sh.d || l2.len() != sh.n * sh.d {
        return Err(Error::Msg(format!(
            "gibbs: lengthscales must match inputs ({}x{}, {}x{}), got {} and {}",
            sh.m,
            sh.d,
            sh.n,
            sh.d,
            l1.len(),
            l2.len()
        )));
    }
    if let Some(bad) = l1.iter().chain(l2.iter()).find(|v| **v <= 0.0) {
        return Err(Error::Msg(format!(
            "gibbs: lengthscales must be strictly positive, found {bad}"
        )));
    }
    Ok(())
}

pub fn gibbs(
    x1: &[f64],
    x2: &[f64],
    l1: &[f64],
    l2: &[f64],
    sh: PairShape,
) -> Result<Vec<f64>> {
    gibbs_check(l1, l2, sh)?;
    let mut out = vec![0.0; sh.m * sh.n];
    for i in 0..sh.m {
        for j in 0..sh.n {
            let mut log_k = 0.0;
            for d in 0..sh.d {
                let a = l1[i * sh.d + d];
                let b = l2[j * sh.d + d];
                let s = a * a + b * b;
                let delta = x1[i * sh.d + d] - x2[j * sh.d + d];
                log_k += 0.5 * (2.0f64.ln() + a.ln() + b.ln() - s.ln());
                log_k -= delta * delta / s;
            }
            out[i * sh.n + j] = log_k.exp();
        }
    }
    Ok(out)
}

/// VJP -> (dx1, dx2, dl1, dl2).
///
/// With `s = ℓ1²+ℓ2²`, `Δ = x1−x2`:
///   ∂ln k/∂Δ  = −2Δ/s
///   ∂ln k/∂ℓ1 = 1/(2ℓ1) − ℓ1/s + 2ℓ1Δ²/s²
/// (the middle term is the prefactor's normaliser — dropping it is the
/// classic Gibbs-kernel gradient bug, and it is what makes the kernel PSD.)
pub fn gibbs_backward(
    x1: &[f64],
    x2: &[f64],
    l1: &[f64],
    l2: &[f64],
    v: &[f64],
    sh: PairShape,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    gibbs_check(l1, l2, sh)?;
    let k = gibbs(x1, x2, l1, l2, sh)?;
    let mut gx1 = vec![0.0; sh.m * sh.d];
    let mut gx2 = vec![0.0; sh.n * sh.d];
    let mut gl1 = vec![0.0; sh.m * sh.d];
    let mut gl2 = vec![0.0; sh.n * sh.d];
    for i in 0..sh.m {
        for j in 0..sh.n {
            let vk = v[i * sh.n + j] * k[i * sh.n + j];
            if vk == 0.0 {
                continue;
            }
            for d in 0..sh.d {
                let a = l1[i * sh.d + d];
                let b = l2[j * sh.d + d];
                let s = a * a + b * b;
                let delta = x1[i * sh.d + d] - x2[j * sh.d + d];
                let dx = vk * (-2.0 * delta / s);
                gx1[i * sh.d + d] += dx;
                gx2[j * sh.d + d] -= dx;
                let d2s = delta * delta / (s * s);
                gl1[i * sh.d + d] += vk * (0.5 / a - a / s + 2.0 * a * d2s);
                gl2[j * sh.d + d] += vk * (0.5 / b - b / s + 2.0 * b * d2s);
            }
        }
    }
    Ok((gx1, gx2, gl1, gl2))
}

// ---------------------------------------------------------------------------
// softmin  —  the Sinkhorn potential update
// ---------------------------------------------------------------------------
//
//   A[i,j]   = (g[j] − C[i,j])/ε + log_b[j]
//   out[i]   = −ε · logsumexp_j A[i,j]
//
// Matches `coupling_kernel::softmin_lazy` and the pykeops `_keops_softmin`
// it was ported from.

/// Row-stable forward. Returns `out [M]`.
pub fn softmin(c: &[f64], log_b: &[f64], g: &[f64], eps: f64, m: usize, n: usize) -> Result<Vec<f64>> {
    softmin_check(c, log_b, g, eps, m, n)?;
    let mut out = vec![0.0; m];
    for i in 0..m {
        let (lse, _) = row_lse(c, log_b, g, eps, i, n);
        out[i] = -eps * lse;
    }
    Ok(out)
}

fn softmin_check(
    c: &[f64],
    log_b: &[f64],
    g: &[f64],
    eps: f64,
    m: usize,
    n: usize,
) -> Result<()> {
    if c.len() != m * n || log_b.len() != n || g.len() != n {
        return Err(Error::Msg(format!(
            "softmin: expected C {m}x{n}, log_b {n}, g {n} — got {}, {}, {}",
            c.len(),
            log_b.len(),
            g.len()
        )));
    }
    if !(eps > 0.0) {
        return Err(Error::Msg(format!("softmin: eps must be > 0, got {eps}")));
    }
    Ok(())
}

/// `(logsumexp_j A[i,j], P[i,·])` where `P = softmax_j A` — the transport
/// plan row. The max-shift is what keeps `exp` in range; note the shift
/// cancels exactly in both the value and the softmax, so it introduces no
/// error (unlike an approximation, it is an identity).
fn row_lse(c: &[f64], log_b: &[f64], g: &[f64], eps: f64, i: usize, n: usize) -> (f64, Vec<f64>) {
    let mut a = vec![0.0; n];
    let mut max = f64::NEG_INFINITY;
    for j in 0..n {
        let v = (g[j] - c[i * n + j]) / eps + log_b[j];
        a[j] = v;
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0;
    for j in 0..n {
        a[j] = (a[j] - max).exp();
        sum += a[j];
    }
    for j in 0..n {
        a[j] /= sum;
    }
    (max + sum.ln(), a)
}

/// VJP -> (dC, dlog_b, dg, deps).
///
/// The closed form. With `P = softmax_j A` and cotangent `v`:
///   ∂out_i/∂A_ij = −ε·P_ij
///   dC_ij      =  v_i·P_ij                  (∂A/∂C = −1/ε)
///   dg_j       = −Σ_i v_i·P_ij              (∂A/∂g  = +1/ε)
///   dlog_b_j   =  ε·dg_j                    (∂A/∂log_b = 1)
///   dε         =  Σ_i v_i·[out_i/ε + (1/ε)Σ_j P_ij(g_j − C_ij)]
///
/// The ε term uses `out_i = −ε·LSE` to eliminate a second LSE evaluation.
/// Note `dlog_b = ε·dg` exactly — a cheap invariant the tests assert, and a
/// good tripwire for a sign error in either.
pub fn softmin_backward(
    c: &[f64],
    log_b: &[f64],
    g: &[f64],
    eps: f64,
    v: &[f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
    softmin_check(c, log_b, g, eps, m, n)?;
    let mut dc = vec![0.0; m * n];
    let mut dg = vec![0.0; n];
    let mut deps = 0.0;
    for i in 0..m {
        let (lse, p) = row_lse(c, log_b, g, eps, i, n);
        let vi = v[i];
        let mut weighted = 0.0;
        for j in 0..n {
            dc[i * n + j] = vi * p[j];
            dg[j] -= vi * p[j];
            weighted += p[j] * (g[j] - c[i * n + j]);
        }
        // ∂out_i/∂ε = −LSE + (1/ε)·Σ_j P_ij(g_j − C_ij)
        deps += vi * (-lse + weighted / eps);
    }
    let dlog_b: Vec<f64> = dg.iter().map(|x| x * eps).collect();
    Ok((dc, dlog_b, dg, deps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(m: usize, n: usize, d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, PairShape) {
        let x1: Vec<f64> = (0..m * d).map(|i| ((i * 7 + 1) as f64 * 0.31).sin()).collect();
        let x2: Vec<f64> = (0..n * d).map(|i| ((i * 5 + 2) as f64 * 0.27).cos()).collect();
        let w: Vec<f64> = (0..d).map(|i| 0.6 + 0.2 * i as f64).collect();
        (x1, x2, w, PairShape { m, n, d })
    }

    /// Every stationary kernel must be exactly 1 at zero distance — the
    /// property the NKN gate products rely on.
    #[test]
    fn stationary_kernels_are_one_at_zero_distance() {
        let x = vec![0.4, -1.1, 0.7];
        let w = vec![1.0, 1.0, 1.0];
        let sh = PairShape { m: 1, n: 1, d: 3 };
        assert!((matern52(&x, &x, &w, 1.3, sh).unwrap()[0] - 1.0).abs() < 1e-9);
        assert!((rbf(&x, &x, &w, sh).unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((rational_quadratic(&x, &x, &w, 2.0, sh).unwrap()[0] - 1.0).abs() < 1e-12);
    }

    /// The SQRT_EPS guard: without it the Matérn self-pair gradient is NaN.
    #[test]
    fn matern_self_pair_gradient_is_finite() {
        let x = vec![0.3, 0.9];
        let w = vec![1.0, 1.0];
        let sh = PairShape { m: 1, n: 1, d: 2 };
        let (gx1, gx2, gw, gs) = matern52_backward(&x, &x, &w, 1.0, &[1.0], sh).unwrap();
        for v in gx1.iter().chain(&gx2).chain(&gw).chain(std::iter::once(&gs)) {
            assert!(v.is_finite(), "non-finite gradient at zero distance: {v}");
        }
    }

    /// Gibbs collapses to RBF when the lengthscale is constant. With
    /// ℓ ≡ 1/(w√2) the exponent −Δ²/(ℓ²+ℓ²) equals −(Δw)², and the
    /// prefactor is exactly 1. This is the check that catches a wrong
    /// normaliser, which a self-consistency test alone would not.
    #[test]
    fn gibbs_reduces_to_rbf_for_constant_lengthscales() {
        let (x1, x2, w, sh) = grid(4, 3, 2);
        let l1: Vec<f64> = (0..sh.m * sh.d)
            .map(|i| 1.0 / (w[i % sh.d] * 2.0f64.sqrt()))
            .collect();
        let l2: Vec<f64> = (0..sh.n * sh.d)
            .map(|i| 1.0 / (w[i % sh.d] * 2.0f64.sqrt()))
            .collect();
        let g = gibbs(&x1, &x2, &l1, &l2, sh).unwrap();
        let r = rbf(&x1, &x2, &w, sh).unwrap();
        for (a, b) in g.iter().zip(&r) {
            assert!((a - b).abs() < 1e-12, "gibbs {a} vs rbf {b}");
        }
    }

    #[test]
    fn gibbs_rejects_nonpositive_lengthscales() {
        let (x1, x2, _, sh) = grid(2, 2, 2);
        let l = vec![1.0; 4];
        let mut bad = l.clone();
        bad[1] = 0.0;
        assert!(gibbs(&x1, &x2, &bad, &l, sh).is_err());
    }

    /// softmin's defining identity, independent of the implementation:
    /// with a uniform cost the potential is C + the log-partition, so
    /// out_i = C - eps*log(sum_j b_j exp(g_j/eps)).
    #[test]
    fn softmin_matches_closed_form_for_uniform_cost() {
        let (m, n) = (3, 4);
        let cost = 0.85;
        let c = vec![cost; m * n];
        let log_b: Vec<f64> = (0..n).map(|j| -(n as f64).ln() + 0.1 * j as f64).collect();
        let g: Vec<f64> = (0..n).map(|j| 0.3 * j as f64 - 0.2).collect();
        let eps = 0.4;
        let out = softmin(&c, &log_b, &g, eps, m, n).unwrap();
        let z: f64 = (0..n).map(|j| (log_b[j] + g[j] / eps).exp()).sum();
        let want = cost - eps * z.ln();
        for o in out {
            assert!((o - want).abs() < 1e-9, "{o} vs {want}");
        }
    }

    /// Overflow guard: a large |g|/eps must not produce inf/NaN. Without
    /// the max-shift in row_lse this is exp(1e3) = inf.
    #[test]
    fn softmin_is_stable_for_extreme_ratios() {
        let (m, n) = (2, 3);
        let c = vec![0.0; m * n];
        let log_b = vec![0.0; n];
        let g = vec![1000.0, -1000.0, 0.0];
        let out = softmin(&c, &log_b, &g, 1e-3, m, n).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        // dominated by the largest g_j: out ≈ −g_max
        assert!((out[0] + 1000.0).abs() < 1e-6, "{}", out[0]);
    }

    /// `dlog_b = eps * dg` falls out of the two paths through A; asserting
    /// it catches a sign slip in either without needing a second oracle.
    #[test]
    fn softmin_backward_respects_the_logb_dg_identity() {
        let (m, n) = (3, 4);
        let c: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.37).sin() + 1.0).collect();
        let log_b: Vec<f64> = (0..n).map(|j| -(n as f64).ln() + 0.05 * j as f64).collect();
        let g: Vec<f64> = (0..n).map(|j| 0.2 * j as f64).collect();
        let v: Vec<f64> = (0..m).map(|i| 0.5 + i as f64).collect();
        let eps = 0.3;
        let (_, dlog_b, dg, _) = softmin_backward(&c, &log_b, &g, eps, &v, m, n).unwrap();
        for (a, b) in dlog_b.iter().zip(&dg) {
            assert!((a - eps * b).abs() < 1e-12, "{a} vs {}", eps * b);
        }
    }

    /// dC rows sum to v_i because P is a probability distribution — the
    /// conservation law that a mis-normalised softmax would break.
    #[test]
    fn softmin_dc_rows_sum_to_the_cotangent() {
        let (m, n) = (3, 5);
        let c: Vec<f64> = (0..m * n).map(|i| (i as f64 * 0.61).cos()).collect();
        let log_b = vec![-(5.0f64).ln(); n];
        let g: Vec<f64> = (0..n).map(|j| 0.11 * j as f64).collect();
        let v: Vec<f64> = (0..m).map(|i| 1.0 + 0.5 * i as f64).collect();
        let (dc, _, _, _) = softmin_backward(&c, &log_b, &g, 0.25, &v, m, n).unwrap();
        for i in 0..m {
            let s: f64 = (0..n).map(|j| dc[i * n + j]).sum();
            assert!((s - v[i]).abs() < 1e-12, "row {i}: {s} vs {}", v[i]);
        }
    }
}
