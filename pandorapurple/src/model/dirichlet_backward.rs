//! Inferno — Explicit vector-Jacobian products for Dirichlet and coregionalisation transforms.

use candle_core::{Result, Tensor, D};

use super::utils::sigmoid;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// `x^e` for tensor exponent via exp(e·log x). Callers guarantee `x > 0`
/// (everything routed here is clamped to [eps, 1] first). Broadcasts.
fn tpow(x: &Tensor, e: &Tensor) -> Result<Tensor> {
    x.log()?.broadcast_mul(e)?.exp()
}

/// Sum a broadcasted grad back to `reference`'s shape (batch/sample axes
/// summed) — the Python `_reduce_to`.
fn reduce_to(g: &Tensor, reference: &Tensor) -> Result<Tensor> {
    let mut g = g.clone();
    while g.rank() > reference.rank() {
        g = g.sum(0)?;
    }
    let (gd, rd) = (g.dims().to_vec(), reference.dims().to_vec());
    for (i, (gs, rs)) in gd.iter().zip(rd.iter()).enumerate() {
        if *rs == 1 && *gs != 1 {
            g = g.sum_keepdim(i)?;
        }
    }
    Ok(g)
}

// ---------------------------------------------------------------------------
// 1. Kumaraswamy inverse CDF and its analytic VJP
// ---------------------------------------------------------------------------

/// `v = (1 − (1−u)^{1/q_b})^{1/q_a}` — the Kumaraswamy inverse CDF used by the
/// Dirichlet's stick breaking. Shapes broadcast (`u` `[.., K−1]`, `q_*` `[K−1]`).
/// `q_a`, `q_b` are the CONSTRAINED (positive) parameters, not the raws.
pub fn kumaraswamy_icdf(u: &Tensor, qa: &Tensor, qb: &Tensor, eps: f64) -> Result<Tensor> {
    let om_u = u.affine(-1.0, 1.0)?.clamp(eps, 1.0)?; // 1 − u
    let w = tpow(&om_u, &qb.recip()?)?; // (1−u)^{1/q_b}
    let s = w.affine(-1.0, 1.0)?.clamp(eps, 1.0)?; // 1 − w
    tpow(&s, &qa.recip()?)
}

/// VJP of the Kumaraswamy inverse CDF: given ∂L/∂v, return (∂L/∂q_a, ∂L/∂q_b).
///
/// With `w = (1−u)^{1/q_b}`, `s = 1−w`, `v = s^{1/q_a}`:
/// ```text
///   ∂v/∂q_a = −v·log(s)/q_a²
///   ∂v/∂q_b = (v/(q_a·s))·w·log(1−u)/q_b²
/// ```
/// Grads are REDUCED to the shape of `q_a`/`q_b` (summed over broadcast/sample
/// axes), so per-stick parameters get the batch-summed gradient.
pub fn kumaraswamy_icdf_vjp(
    u: &Tensor,
    qa: &Tensor,
    qb: &Tensor,
    dv: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let om_u = u.affine(-1.0, 1.0)?.clamp(eps, 1.0)?;
    let log_om_u = om_u.log()?;
    let w = tpow(&om_u, &qb.recip()?)?;
    let s = w.affine(-1.0, 1.0)?.clamp(eps, 1.0)?;
    let v = tpow(&s, &qa.recip()?)?;

    // ∂v/∂q_a = −v·log(s)/q_a²
    let dv_dqa = (v.clone() * s.log()?)?.broadcast_div(&qa.sqr()?)?.neg()?;
    // ∂v/∂q_b = (v/(q_a·s))·w·log(1−u)/q_b²
    let qa_s = s.broadcast_mul(qa)?;
    let dv_dqb = ((v.div(&qa_s)? * w)? * log_om_u)?.broadcast_div(&qb.sqr()?)?;

    let ga = reduce_to(&(dv.clone() * dv_dqa)?, qa)?;
    let gb = reduce_to(&(dv.clone() * dv_dqb)?, qb)?;
    Ok((ga, gb))
}

// ---------------------------------------------------------------------------
// 2. Stick-breaking π_k = v_k ∏_{j<k}(1 − v_j) and its VJP
// ---------------------------------------------------------------------------

/// `[.., K−1]` sticks → `[.., K]` simplex. `π_k = v_k·∏_{j<k}(1−v_j)`,
/// `π_{K−1} = ∏_j(1−v_j)` (the remainder atom).
///
/// No `cumprod` in candle — since `1−v` is clamped to `[eps, 1]` the product
/// is computed as `exp(cumsum(log(1−v)))`.
pub fn stick_break(v: &Tensor, eps: f64) -> Result<Tensor> {
    let k1 = v.dim(D::Minus1)?; // K−1 sticks
    let om = v.affine(-1.0, 1.0)?.clamp(eps, 1.0)?; // 1 − v
    let cprod = om.log()?.cumsum(D::Minus1)?.exp()?; // ∏_{j≤k}(1−v_j)
    let lead = cprod.narrow(D::Minus1, 0, 1)?.ones_like()?;
    let prefix = if k1 == 1 {
        lead
    } else {
        Tensor::cat(&[&lead, &cprod.narrow(D::Minus1, 0, k1 - 1)?], D::Minus1)?
    }; // ∏_{j<k}(1−v_j)
    let pi_head = (v.clone() * prefix)?; // [.., K−1]
    let pi_last = cprod.narrow(D::Minus1, k1 - 1, 1)?; // remainder
    Tensor::cat(&[&pi_head, &pi_last], D::Minus1)
}

/// VJP of [`stick_break`]: given ∂L/∂π `[.., K]`, return ∂L/∂v `[.., K−1]`.
///
/// Exact closed form (`∂π_k/∂v_i`: `+prefix` at `i=k`, `−π_k/(1−v_i)` for
/// `i<k`), assembled without densifying the `[K, K−1]` Jacobian. The suffix
/// sum `Σ_{k>i} dπ_k·π_k` is computed as `total − cumsum` (candle has no
/// `flip`), aligned to the stick axis.
pub fn stick_break_vjp(v: &Tensor, dpi: &Tensor, eps: f64) -> Result<Tensor> {
    let k1 = v.dim(D::Minus1)?; // K−1 sticks
    let k = dpi.dim(D::Minus1)?; // K atoms
    debug_assert_eq!(k, k1 + 1, "dpi must have one more atom than v has sticks");

    // recompute the forward intermediates (cheap; keeps the fn stateless)
    let om = v.affine(-1.0, 1.0)?.clamp(eps, 1.0)?;
    let cprod = om.log()?.cumsum(D::Minus1)?.exp()?;
    let lead = cprod.narrow(D::Minus1, 0, 1)?.ones_like()?;
    let prefix = if k1 == 1 {
        lead
    } else {
        Tensor::cat(&[&lead, &cprod.narrow(D::Minus1, 0, k1 - 1)?], D::Minus1)?
    };
    let pi_head = (v.clone() * prefix.clone())?;
    let pi_last = cprod.narrow(D::Minus1, k1 - 1, 1)?;
    let pi = Tensor::cat(&[&pi_head, &pi_last], D::Minus1)?; // [.., K]

    // ∂L/∂v_i = dπ_i·prefix_i − (Σ_{k>i} dπ_k·π_k)/(1−v_i)
    let dpi_pi = (dpi.clone() * pi)?; // [.., K]
    let csum = dpi_pi.cumsum(D::Minus1)?; // Σ_{k≤i}
    let total = csum.narrow(D::Minus1, k - 1, 1)?; // Σ over all atoms
    let suffix_gt = total
        .broadcast_sub(&csum)? // Σ_{k>i}
        .narrow(D::Minus1, 0, k1)?; // aligned to sticks
    (dpi.narrow(D::Minus1, 0, k1)? * prefix)?.sub(&suffix_gt.div(&om)?)
}

// ---------------------------------------------------------------------------
// 3. LMC coregionalisation B = (LLᵀ)⊙(ππᵀ) + diag(lmc_var + jitter) and its VJP
// ---------------------------------------------------------------------------
// Mirrors `AmortisedDirichlet::coregionalisation_matrix` (L = lmc_matrix
// [K, num_latents], lmc_var [K], pi [B, K] — CONSTRAINED values, not raws).

/// `B [B,K,K] = (L Lᵀ) ⊙ (π πᵀ) + diag(lmc_var + jitter)`.
pub fn coreg_matrix(l: &Tensor, lmc_var: &Tensor, pi: &Tensor, jitter: f64) -> Result<Tensor> {
    let k = lmc_var.dim(0)?;
    let b_base = l.matmul(&l.t()?)?; // [K, K]
    let b = b_base
        .unsqueeze(0)? // [1, K, K]
        .broadcast_mul(&pi.unsqueeze(2)?)? // × π_i (rows)
        .broadcast_mul(&pi.unsqueeze(1)?)?; // × π_j (cols)
    let eye = Tensor::eye(k, l.dtype(), l.device())?;
    let diag = eye.broadcast_mul(&lmc_var.affine(1.0, jitter)?.unsqueeze(0)?)?;
    b.broadcast_add(&diag)
}

/// VJP of [`coreg_matrix`]: given ∂L/∂B `[B,K,K]`, return
/// `(dL [K,num_latents], dvar [K], dpi [B,K])`. Analytic:
/// ```text
///   Bπ = dB ⊙ (π πᵀ)                        (the base sees the π-mask)
///   dL      = (G + Gᵀ) L,  G = Σ_b Bπ_b     (from B_base = L Lᵀ)
///   dvar_k  = Σ_b dB_b[k,k]
///   dπ_b,k  = Σ_j dB_b[k,j]·B_base[k,j]·π_b,j + Σ_i dB_b[i,k]·B_base[i,k]·π_b,i
/// ```
/// (row and column π-factors contracted SEPARATELY, so it is exact for an
/// asymmetric dB, not just the symmetric case).
pub fn coreg_vjp(
    l: &Tensor,
    lmc_var: &Tensor,
    pi: &Tensor,
    db: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let k = lmc_var.dim(0)?;
    let b_base = l.matmul(&l.t()?)?; // [K, K]

    // dL from B_base = L Lᵀ: grad = (G + Gᵀ) L with G = Σ_b dB_b ⊙ (π πᵀ)
    let ppt = pi.unsqueeze(2)?.broadcast_mul(&pi.unsqueeze(1)?)?; // [B, K, K]
    let bpi = (db.clone() * ppt)?;
    let g = bpi.sum(0)?; // [K, K]
    let dl = g.add(&g.t()?)?.matmul(l)?; // [K, num_latents]

    // dvar from the diagonal add
    let eye = Tensor::eye(k, l.dtype(), l.device())?;
    let dvar = db
        .broadcast_mul(&eye.unsqueeze(0)?)?
        .sum(D::Minus1)? // pick the diagonal per (b, k)
        .sum(0)?; // [K]

    // dpi: both π factors, contracted separately
    let t = db.broadcast_mul(&b_base.unsqueeze(0)?)?; // [B, K, K]
    let dpi_row = t.broadcast_mul(&pi.unsqueeze(1)?)?.sum(D::Minus1)?; // bkj,kj,bj → bk
    let dpi_col = t.broadcast_mul(&pi.unsqueeze(2)?)?.sum(1)?; // bik,ik,bi → bk
    let dpi = dpi_row.add(&dpi_col)?;

    Ok((dl, dvar, dpi))
}

// ---------------------------------------------------------------------------
// 4. Constraint VJPs — closed forms of utils::{positive, greater_than, interval}
// ---------------------------------------------------------------------------
// The Python `_constraint_vjp` runs one-shot autograd through the gpytorch
// transform; here the three constraints are known closed forms, so their
// Jacobians are one-liners. Keep in lockstep with `utils.rs`.

/// VJP of `q = softplus(raw) + shift` (also `greater_than`: the additive `lo`
/// has zero derivative): `∂q/∂raw = σ(raw)`.
pub fn softplus_vjp(raw: &Tensor, g: &Tensor) -> Result<Tensor> {
    g.mul(&sigmoid(raw)?)
}

/// VJP of `q = lo + (hi−lo)·σ(raw)` (`interval`): `∂q/∂raw = (hi−lo)·σ(1−σ)`.
pub fn interval_vjp(raw: &Tensor, g: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    let s = sigmoid(raw)?;
    let ds = (s.clone() * s.affine(-1.0, 1.0)?)?.affine(hi - lo, 0.0)?;
    g.mul(&ds)
}

// ---------------------------------------------------------------------------
// Grads container
// ---------------------------------------------------------------------------

/// Closed-form gradients w.r.t. the Dirichlet's raw parameters, assembled by
/// `AmortisedDirichlet::analytic_backward`. `None` = branch not requested
/// (e.g. no ∂L/∂B supplied → no LMC grads).
#[derive(Debug)]
pub struct DirichletGrads {
    pub raw_q_a: Option<Tensor>,
    pub raw_q_b: Option<Tensor>,
    pub raw_lmc_matrix: Option<Tensor>,
    pub raw_lmc_var: Option<Tensor>,
    /// combined ∂L/∂π (direct + via B) — useful as a probe / for chaining
    pub dpi_total: Option<Tensor>,
}

// ===========================================================================
// tests — every VJP against central finite differences in f64 (the Python
// file checks against torch autograd; fd is the autograd-independent twin)
// ===========================================================================
#[cfg(test)]
mod tests {
    //! Verification of explicit vector-jacobian products for dirichlet and coregionalisation transforms.

    use super::*;
    use candle_core::Device;

    const EPS: f64 = 1e-12;
    const H: f64 = 1e-5; // central-diff step (f64 → error ~H²)
    const TOL: f64 = 1e-7;

    fn dev() -> Device {
        Device::Cpu
    }

    fn t1(v: &[f64]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &dev()).unwrap()
    }

    fn t2(v: &[f64], r: usize, c: usize) -> Tensor {
        Tensor::from_vec(v.to_vec(), (r, c), &dev()).unwrap()
    }

    /// Central finite-difference gradient of scalar-valued `f` w.r.t. `x`.
    fn fd_grad(f: impl Fn(&Tensor) -> f64, x: &Tensor) -> Tensor {
        let flat: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
        let dims = x.dims().to_vec();
        let mut g = vec![0f64; flat.len()];
        for i in 0..flat.len() {
            let mut p = flat.clone();
            let mut m = flat.clone();
            p[i] += H;
            m[i] -= H;
            let xp = Tensor::from_vec(p, dims.clone(), x.device()).unwrap();
            let xm = Tensor::from_vec(m, dims.clone(), x.device()).unwrap();
            g[i] = (f(&xp) - f(&xm)) / (2.0 * H);
        }
        Tensor::from_vec(g, dims, x.device()).unwrap()
    }

    fn weighted_sum(out: &Tensor, w: &Tensor) -> f64 {
        (out.clone() * w.clone()).unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap()
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f64, what: &str) {
        let d = a
            .sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        assert!(d < tol, "{what}: max |Δ| = {d:e} (tol {tol:e})");
    }

    // fixtures: B=2 samples, K=4 atoms (3 sticks), num_latents=3
    fn fix_u() -> Tensor {
        t2(&[0.30, 0.60, 0.20, 0.70, 0.40, 0.55], 2, 3)
    }
    fn fix_qa() -> Tensor {
        t1(&[1.3, 2.1, 1.7])
    }
    fn fix_qb() -> Tensor {
        t1(&[1.9, 1.2, 2.4])
    }

    #[test]
    fn kumaraswamy_vjp_matches_fd() {
        let (u, qa, qb) = (fix_u(), fix_qa(), fix_qb());
        let dv = t2(&[0.5, -1.2, 0.8, 1.1, 0.3, -0.7], 2, 3);

        let (ga, gb) = kumaraswamy_icdf_vjp(&u, &qa, &qb, &dv, EPS).unwrap();
        let ga_fd =
            fd_grad(|q| weighted_sum(&kumaraswamy_icdf(&u, q, &qb, EPS).unwrap(), &dv), &qa);
        let gb_fd =
            fd_grad(|q| weighted_sum(&kumaraswamy_icdf(&u, &qa, q, EPS).unwrap(), &dv), &qb);
        assert_close(&ga, &ga_fd, TOL, "dq_a");
        assert_close(&gb, &gb_fd, TOL, "dq_b");
    }

    #[test]
    fn stick_break_sums_to_one_and_vjp_matches_fd() {
        let v = fix_u(); // any values in (0,1) are valid sticks
        let pi = stick_break(&v, EPS).unwrap();
        let sums: Vec<f64> = pi.sum(D::Minus1).unwrap().to_vec1().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-12, "simplex sum = {s}");
        }

        let dpi = t2(&[0.4, -0.9, 1.3, 0.2, -0.6, 0.8, -1.1, 0.5], 2, 4);
        let dv = stick_break_vjp(&v, &dpi, EPS).unwrap();
        let dv_fd = fd_grad(|x| weighted_sum(&stick_break(x, EPS).unwrap(), &dpi), &v);
        assert_close(&dv, &dv_fd, TOL, "dv");
    }

    #[test]
    fn coreg_vjp_matches_fd() {
        let l = t2(&[0.5, -0.3, 0.8, 1.1, 0.2, -0.7, -0.4, 0.9, 0.6, 0.1, -1.2, 0.3], 4, 3);
        let lmc_var = t1(&[0.4, 0.9, 0.2, 1.3]);
        let pi = t2(&[0.10, 0.20, 0.30, 0.40, 0.25, 0.25, 0.30, 0.20], 2, 4);
        // deliberately ASYMMETRIC dB — exercises the separate row/col π terms
        let db_vals: Vec<f64> =
            (0..32).map(|i| 0.1 * (i as f64) - 1.7 + ((i * i) % 5) as f64 * 0.3).collect();
        let db = Tensor::from_vec(db_vals, (2, 4, 4), &dev()).unwrap();
        let jit = 1e-5;

        let (dl, dvar, dpi_g) = coreg_vjp(&l, &lmc_var, &pi, &db).unwrap();
        let dl_fd =
            fd_grad(|x| weighted_sum(&coreg_matrix(x, &lmc_var, &pi, jit).unwrap(), &db), &l);
        let dvar_fd =
            fd_grad(|x| weighted_sum(&coreg_matrix(&l, x, &pi, jit).unwrap(), &db), &lmc_var);
        let dpi_fd =
            fd_grad(|x| weighted_sum(&coreg_matrix(&l, &lmc_var, x, jit).unwrap(), &db), &pi);
        assert_close(&dl, &dl_fd, TOL, "dL");
        assert_close(&dvar, &dvar_fd, TOL, "dvar");
        assert_close(&dpi_g, &dpi_fd, TOL, "dpi");
    }

    #[test]
    fn end_to_end_raw_qa_chain_matches_fd() {
        // raw_q_a → softplus+shift → icdf → stick_break → weighted loss,
        // gradient assembled by chaining the three VJPs (the analytic_backward
        // composition) vs one finite difference over the raw parameter.
        let (u, qb) = (fix_u(), fix_qb());
        let raw_qa = t1(&[-0.4, 0.7, 0.1]);
        let shift = 1.035;
        let dpi = t2(&[0.4, -0.9, 1.3, 0.2, -0.6, 0.8, -1.1, 0.5], 2, 4);

        let forward = |raw: &Tensor| -> f64 {
            let qa = super::super::utils::softplus(raw).unwrap().affine(1.0, shift).unwrap();
            let v = kumaraswamy_icdf(&u, &qa, &qb, EPS).unwrap();
            weighted_sum(&stick_break(&v, EPS).unwrap(), &dpi)
        };

        // analytic chain
        let qa = super::super::utils::softplus(&raw_qa).unwrap().affine(1.0, shift).unwrap();
        let v = kumaraswamy_icdf(&u, &qa, &qb, EPS).unwrap();
        let dv = stick_break_vjp(&v, &dpi, EPS).unwrap();
        let (ga, _gb) = kumaraswamy_icdf_vjp(&u, &qa, &qb, &dv, EPS).unwrap();
        let g_raw = softplus_vjp(&raw_qa, &ga).unwrap();

        let g_fd = fd_grad(forward, &raw_qa);
        assert_close(&g_raw, &g_fd, TOL, "raw_q_a end-to-end");
    }

    #[test]
    fn constraint_vjps_match_fd() {
        let raw = t1(&[-1.1, 0.3, 2.0]);
        let g = t1(&[0.7, -0.4, 1.2]);
        let (lo, hi) = (-2.0, 3.5);

        let sp = softplus_vjp(&raw, &g).unwrap();
        let sp_fd = fd_grad(
            |r| weighted_sum(&super::super::utils::greater_than(r, 0.123).unwrap(), &g),
            &raw,
        );
        assert_close(&sp, &sp_fd, TOL, "softplus/greater_than vjp");

        let iv = interval_vjp(&raw, &g, lo, hi).unwrap();
        let iv_fd = fd_grad(
            |r| weighted_sum(&super::super::utils::interval(r, lo, hi).unwrap(), &g),
            &raw,
        );
        assert_close(&iv, &iv_fd, TOL, "interval vjp");
    }
}
