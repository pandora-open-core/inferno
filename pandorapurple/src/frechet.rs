//! Inferno — Wasserstein Fréchet regression, entropic barycentres and simplex-valued prediction.

use candle_core::{DType, Result, Tensor, D};

/// Below this, a projected weight vector is treated as having lost real mass.
pub const NEG_MASS_WARN: f64 = 0.05;
/// Chunk budget for the `[q, n, K, K]` IBP intermediate. 2 GiB.
pub const MEM_BUDGET_BYTES: usize = 2 << 30;

fn clamp_min(t: &Tensor, lo: f64) -> Result<Tensor> {
    t.maximum(&t.affine(0.0, lo)?)
}

fn logsumexp(t: &Tensor, dim: D) -> Result<Tensor> {
    let m = t.max_keepdim(dim)?;
    let s = t.broadcast_sub(&m)?.exp()?.sum_keepdim(dim)?;
    s.log()?.add(&m)
}

/// `log(K @ exp(lx))` for `K = exp(-C/eps)`, in logs throughout — never
/// forms `K` (it underflows on exactly the well-separated atom pairs whose
/// transport matters most). `lx: [.., K]`, `neg_c_eps: [K, K]` → `[.., K]`.
fn log_k(lx: &Tensor, neg_c_eps: &Tensor) -> Result<Tensor> {
    let arg = neg_c_eps.broadcast_add(&lx.unsqueeze(D::Minus2)?)?;
    logsumexp(&arg, D::Minus1)?.squeeze(D::Minus1)
}

/// Weighted entropic barycentre of `[n, K]` measures — `[Q, K]`
/// log-simplices for `Q` weight rows (Benamou et al. 2015, IBP).
///
/// `log_p [n, K]`, `w [Q, n]` (NON-NEGATIVE, rows summing to 1),
/// `c [K, K]`. The barycentre is a weighted GEOMETRIC mean of transported
/// marginals — exactly why `w` must be non-negative (trap 1).
pub fn log_ibp_barycentre(
    log_p: &Tensor,
    w: &Tensor,
    c: &Tensor,
    eps: f64,
    n_iter: usize,
) -> Result<Tensor> {
    let w_min = w.min_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    if w_min < 0.0 {
        candle_core::bail!(
            "log_ibp_barycentre needs w >= 0; project first \
             (WassersteinFrechetRegression does this and reports how much mass it moved)"
        );
    }
    let neg_c = c.affine(-1.0 / eps, 0.0)?;
    let (q_total, n) = w.dims2()?;
    let k = log_p.dim(D::Minus1)?;
    // Memory is O(q·n·K²); chunk Q to a fixed budget rather than trusting
    // the caller not to ask for every training point at once.
    let elem = log_p.dtype().size_in_bytes();
    let per_row = n * k * k * elem;
    let qc = (MEM_BUDGET_BYTES / per_row.max(1)).clamp(1, q_total.max(1));

    let mut outs = Vec::new();
    let mut lo = 0usize;
    while lo < q_total {
        let take = qc.min(q_total - lo);
        let wq = w.narrow(0, lo, take)?; // [q, n]
        let lp = log_p.unsqueeze(0)?; // [1, n, K]
        let mut lv = Tensor::zeros((take, n, k), log_p.dtype(), log_p.device())?;
        let mut lb = Tensor::zeros((take, k), log_p.dtype(), log_p.device())?;
        for _ in 0..n_iter {
            let lu = lp.broadcast_sub(&log_k(&lv, &neg_c)?)?; // [q, n, K]
            let lktu = log_k(&lu, &neg_c)?; // C symmetric
            lb = wq.unsqueeze(D::Minus1)?.broadcast_mul(&lktu)?.sum(1)?; // [q, K]
            lv = lb.unsqueeze(1)?.broadcast_sub(&lktu)?;
        }
        let z = logsumexp(&lb, D::Minus1)?;
        outs.push(lb.broadcast_sub(&z)?);
        lo += take;
    }
    if outs.len() == 1 {
        Ok(outs.pop().unwrap())
    } else {
        Tensor::cat(&outs, 0)
    }
}

/// Petersen–Müller GLOBAL weights — the linear-regression analogue:
/// `s_i(x) = 1 + (x − x̄)ᵀ Σ⁻¹ (X_i − x̄)`, rows summing to 1.
/// THESE CAN BE NEGATIVE — they are the OLS hat weights; the caller decides
/// (the estimator projects and reports).
pub fn frechet_weights_global(x: &Tensor, xq: &Tensor, ridge: f64) -> Result<Tensor> {
    let n = x.dim(0)?;
    let xbar = x.mean_keepdim(0)?;
    let xc = x.broadcast_sub(&xbar)?;
    let sig = xc.t()?.matmul(&xc)?.affine(1.0 / (n.max(2) - 1) as f64, 0.0)?;
    let p = sig.dim(0)?;
    let diag_mean = {
        let v = sig.to_dtype(DType::F64)?.to_vec2::<f64>()?;
        ((0..p).map(|i| v[i][i]).sum::<f64>() / p as f64).max(1e-12)
    };
    let eye = Tensor::eye(p, sig.dtype(), sig.device())?;
    let sig = sig.add(&eye.affine(ridge * diag_mean, 0.0)?)?;
    // solve Σ · Z = (xq − x̄)ᵀ  →  s = 1 + Zᵀ Xcᵀ
    let rhs = xq.broadcast_sub(&xbar)?.t()?.contiguous()?; // [p, Q]
    let sol = crate::linalg::cholesky_solve(&sig, &rhs, 0.0)?; // [p, Q]
    let s = sol.t()?.matmul(&xc.t()?)?.affine(1.0, 1.0)?; // [Q, n]
    let z = clamp_min(&s.sum_keepdim(D::Minus1)?, 1e-12)?;
    s.broadcast_div(&z)
}

/// Kernel (Nadaraya–Watson) weights — NON-NEGATIVE by construction, which
/// is what makes them safe for IBP. `[Q, n]`, rows summing to 1.
pub fn frechet_weights_local(
    x: &Tensor,
    xq: &Tensor,
    h: f64,
    kernel: &str,
) -> Result<Tensor> {
    // cdist², expanded (the crate has no cdist).
    let x2 = x.sqr()?.sum(D::Minus1)?; // [n]
    let q2 = xq.sqr()?.sum_keepdim(D::Minus1)?; // [Q, 1]
    let cross = xq.matmul(&x.t()?)?; // [Q, n]
    let d2 = clamp_min(
        &q2.broadcast_add(&x2)?.broadcast_sub(&cross.affine(2.0, 0.0)?)?,
        0.0,
    )?;
    let h = h.max(1e-12);
    match kernel {
        "gaussian" => {
            let lw = d2.affine(-0.5 / (h * h), 0.0)?;
            let z = logsumexp(&lw, D::Minus1)?;
            lw.broadcast_sub(&z)?.exp()
        }
        "epanechnikov" => {
            let u = d2.sqrt()?.affine(1.0 / h, 0.0)?.minimum(&d2.affine(0.0, 1.0)?)?;
            let w = clamp_min(&u.sqr()?.affine(-1.0, 1.0)?, 0.0)?;
            let z = clamp_min(&w.sum_keepdim(D::Minus1)?, 1e-12)?;
            w.broadcast_div(&z)
        }
        other => candle_core::bail!("unknown kernel {other:?}"),
    }
}

/// MEDIAN pairwise distance × n^(−1/5). Deliberately the median — the mean
/// is tail-dragged and the diagonal zeros bias it down (the same defect
/// `kernels/mmd::silverman_bandwidth` carried before its fix).
pub fn silverman_bandwidth(x: &Tensor) -> Result<f64> {
    let n = x.dim(0)?;
    if n < 3 {
        return Ok(1.0);
    }
    let x2 = x.sqr()?.sum(D::Minus1)?;
    let cross = x.matmul(&x.t()?)?;
    let d2 = clamp_min(
        &x2.unsqueeze(1)?
            .broadcast_add(&x2.unsqueeze(0)?)?
            .broadcast_sub(&cross.affine(2.0, 0.0)?)?,
        0.0,
    )?;
    let m = d2.sqrt()?.to_dtype(DType::F64)?.to_vec2::<f64>()?;
    let mut off: Vec<f64> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .map(|(i, j)| m[i][j])
        .collect();
    off.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if off.len() % 2 == 1 {
        off[off.len() / 2]
    } else {
        0.5 * (off[off.len() / 2 - 1] + off[off.len() / 2])
    }
    .max(1e-9);
    Ok(med * (n as f64).powf(-0.2))
}

// ---------------------------------------------------------------------------
// The per-row simplex divergence (flow/GEOC_drift.py:74-102, unrolled)
// ---------------------------------------------------------------------------

/// Balanced entropic-OT value `⟨f,a⟩ + ⟨g,b⟩` for `[.., K]` simplices over a
/// shared `[K, K]` cost, in the log domain — per row, unrolled Sinkhorn.
pub fn entropic_ot_cost(
    log_a: &Tensor,
    log_b: &Tensor,
    c: &Tensor,
    eps: f64,
    n_iter: usize,
) -> Result<Tensor> {
    let mut f = log_a.zeros_like()?;
    let mut g = log_b.zeros_like()?;
    for _ in 0..n_iter {
        // f = −ε·LSE_j[(g_j − C_ij)/ε + log b_j]
        let arg = g
            .unsqueeze(D::Minus2)?
            .broadcast_sub(c)?
            .affine(1.0 / eps, 0.0)?
            .broadcast_add(&log_b.unsqueeze(D::Minus2)?)?;
        f = logsumexp(&arg, D::Minus1)?.squeeze(D::Minus1)?.affine(-eps, 0.0)?;
        let arg = f
            .unsqueeze(D::Minus1)?
            .broadcast_sub(c)?
            .affine(1.0 / eps, 0.0)?
            .broadcast_add(&log_a.unsqueeze(D::Minus1)?)?;
        g = logsumexp(&arg, D::Minus2)?.squeeze(D::Minus2)?.affine(-eps, 0.0)?;
    }
    let a = log_a.exp()?;
    let b = log_b.exp()?;
    f.mul(&a)?.sum(D::Minus1)?.add(&g.mul(&b)?.sum(D::Minus1)?)
}

/// Debiased `S_ε = OT(a,b) − ½OT(a,a) − ½OT(b,b)` per `[.., K]` row.
/// Returns `(ot_ab, s)`.
pub fn sinkhorn_divergence_simplex(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    eps: f64,
    n_iter: usize,
) -> Result<(Tensor, Tensor)> {
    let la = clamp_min(a, 1e-30)?.log()?;
    let lb = clamp_min(b, 1e-30)?.log()?;
    let ot_ab = entropic_ot_cost(&la, &lb, c, eps, n_iter)?;
    let ot_aa = entropic_ot_cost(&la, &la, c, eps, n_iter)?;
    let ot_bb = entropic_ot_cost(&lb, &lb, c, eps, n_iter)?;
    let s = ot_ab.sub(&ot_aa.add(&ot_bb)?.affine(0.5, 0.0)?)?;
    Ok((ot_ab, s))
}

// ---------------------------------------------------------------------------
// The estimator
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FrechetFit {
    /// `[Q, K]` predicted densities.
    pub pred: Tensor,
    /// `[K]` unconditional barycentre.
    pub null: Tensor,
    /// `1 − SS_res/SS_null`, in `S_ε`. Negative is a REAL outcome: the
    /// conditional mean is worse than ignoring the covariate.
    pub r2: f64,
    pub ss_res: f64,
    pub ss_null: f64,
    pub neg_weight_mass: f64,
    pub eps: f64,
    pub mode: String,
    pub bandwidth: Option<f64>,
}

pub struct WassersteinFrechetRegression {
    pub c: Tensor,
    pub eps: f64,
    pub mode: String,
    pub bandwidth: Option<f64>,
    pub n_iter: usize,
    pub div_iter: usize,
    x: Option<Tensor>,
    log_p: Option<Tensor>,
    pub neg_weight_mass: f64,
}

impl WassersteinFrechetRegression {
    pub fn new(
        c: Tensor,
        eps: f64,
        mode: &str,
        bandwidth: Option<f64>,
        n_iter: usize,
        div_iter: usize,
    ) -> Result<Self> {
        if mode != "local" && mode != "global" {
            candle_core::bail!("mode must be 'local' or 'global'");
        }
        Ok(Self {
            c,
            eps,
            mode: mode.to_string(),
            bandwidth,
            n_iter,
            div_iter,
            x: None,
            log_p: None,
            neg_weight_mass: 0.0,
        })
    }

    pub fn fit(&mut self, x: &Tensor, p: &Tensor) -> Result<()> {
        if x.dim(0)? != p.dim(0)? {
            candle_core::bail!("{} covariates vs {} measures", x.dim(0)?, p.dim(0)?);
        }
        self.x = Some(x.clone());
        self.log_p = Some(clamp_min(p, 1e-30)?.log()?);
        if self.mode == "local" && self.bandwidth.is_none() {
            let h = silverman_bandwidth(x)?;
            tracing::info!(bandwidth = h, "bandwidth (median-Silverman)");
            self.bandwidth = Some(h);
        }
        Ok(())
    }

    fn weights(&mut self, xq: &Tensor) -> Result<Tensor> {
        let x = self.x.as_ref().expect("fit first");
        if self.mode == "local" {
            return frechet_weights_local(x, xq, self.bandwidth.unwrap(), "gaussian");
        }
        let w = frechet_weights_global(x, xq, 1e-8)?;
        let neg = clamp_min(&w.neg()?, 0.0)?.sum(D::Minus1)?;
        self.neg_weight_mass = neg.max_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        if self.neg_weight_mass > 0.0 {
            if self.neg_weight_mass > NEG_MASS_WARN {
                tracing::warn!(
                    neg_mass = self.neg_weight_mass,
                    "global Frechet weights carry negative mass; IBP needs w >= 0 so it is \
                     being projected onto the simplex. Above {NEG_MASS_WARN} the global fit \
                     is NOT trustworthy — use mode='local'."
                );
            }
            let w = clamp_min(&w, 0.0)?;
            let z = clamp_min(&w.sum_keepdim(D::Minus1)?, 1e-12)?;
            return w.broadcast_div(&z);
        }
        Ok(w)
    }

    /// `[Q, K]` densities.
    pub fn predict(&mut self, xq: &Tensor) -> Result<Tensor> {
        let w = self.weights(xq)?;
        let log_p = self.log_p.clone().expect("fit first");
        log_ibp_barycentre(&log_p, &w, &self.c, self.eps, self.n_iter)?.exp()
    }

    /// The unconditional Fréchet mean — the baseline that must be beaten.
    pub fn null_barycentre(&self) -> Result<Tensor> {
        let log_p = self.log_p.as_ref().expect("fit first");
        let n = log_p.dim(0)?;
        let w = Tensor::full(1.0 / n as f64, (1, n), log_p.device())?
            .to_dtype(log_p.dtype())?;
        let b = log_ibp_barycentre(log_p, &w, &self.c, self.eps, self.n_iter)?.exp()?;
        b.get(0)
    }

    /// Per-row `S_ε` between matched `[.., K]` rows. NOT W2 — trap 2.
    pub fn divergence(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (_, s) = sinkhorn_divergence_simplex(a, b, &self.c, self.eps, self.div_iter)?;
        Ok(s)
    }

    /// Fit, predict at the training covariates (or `xq`), score against the
    /// unconditional barycentre. Deduplicates the training covariates before
    /// predicting (a discrete covariate turns `Q = n` into `Q = #levels` —
    /// the difference between a 166 GB IBP intermediate and a trivial one).
    /// No backward is ever run on this path — a measurement, like Python's
    /// `@torch.no_grad` counterpart.
    pub fn fit_predict(
        &mut self,
        x: &Tensor,
        p: &Tensor,
        xq: Option<&Tensor>,
    ) -> Result<FrechetFit> {
        self.fit(x, p)?;
        // unique rows + inverse index (host-side; covariates are small).
        let xv = x.to_dtype(DType::F64)?.to_vec2::<f64>()?;
        let mut uniq: Vec<Vec<f64>> = Vec::new();
        let mut inv = Vec::with_capacity(xv.len());
        for row in &xv {
            match uniq.iter().position(|u| {
                u.len() == row.len() && u.iter().zip(row).all(|(a, b)| a == b)
            }) {
                Some(i) => inv.push(i as u32),
                None => {
                    uniq.push(row.clone());
                    inv.push((uniq.len() - 1) as u32);
                }
            }
        }
        let pdim = x.dim(1)?;
        let uniq_t = Tensor::from_vec(
            uniq.iter().flatten().copied().collect::<Vec<f64>>(),
            (uniq.len(), pdim),
            x.device(),
        )?
        .to_dtype(x.dtype())?;
        let pred_u = self.predict(&uniq_t)?;
        let inv_t = Tensor::from_vec(inv, xv.len(), x.device())?;
        let pred_tr = pred_u.index_select(&inv_t, 0)?;

        let null = self.null_barycentre()?;
        let null_b = null.unsqueeze(0)?.broadcast_as(p.dims())?.contiguous()?;
        let sumpos = |t: &Tensor| -> Result<f64> {
            clamp_min(t, 0.0)?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()
        };
        let ss_res = sumpos(&self.divergence(p, &pred_tr)?)?;
        let ss_null = sumpos(&self.divergence(p, &null_b)?)?;
        let pred = match xq {
            None => pred_tr,
            Some(q) => self.predict(q)?,
        };
        Ok(FrechetFit {
            pred,
            null,
            r2: if ss_null > 0.0 { 1.0 - ss_res / ss_null } else { f64::NAN },
            ss_res,
            ss_null,
            neg_weight_mass: self.neg_weight_mass,
            eps: self.eps,
            mode: self.mode.clone(),
            bandwidth: self.bandwidth,
        })
    }
}

/// `1 − Σ d²(P_i, m(X_i)) / Σ d²(P_i, null)` in `S_ε` — negative is real.
pub fn frechet_r2(
    p: &Tensor,
    pred: &Tensor,
    null: &Tensor,
    c: &Tensor,
    eps: f64,
    n_iter: usize,
) -> Result<f64> {
    let null_b = null.unsqueeze(0)?.broadcast_as(p.dims())?.contiguous()?;
    let (_, sr) = sinkhorn_divergence_simplex(p, pred, c, eps, n_iter)?;
    let (_, sn) = sinkhorn_divergence_simplex(p, &null_b, c, eps, n_iter)?;
    let sum = |t: &Tensor| -> Result<f64> {
        clamp_min(t, 0.0)?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()
    };
    let den = sum(&sn)?;
    Ok(if den > 0.0 { 1.0 - sum(&sr)? / den } else { f64::NAN })
}

/// The data `plot_densities` would draw: `(support, rows)` where `support`
/// is the (sorted) atom coordinate per column and each row is a `[K]`
/// density in that order. Rendering is the caller's business.
pub fn densities_table(
    pred: &Tensor,
    atom_coords: Option<&Tensor>,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let rows = pred.to_dtype(DType::F64)?.to_vec2::<f64>()?;
    let k = rows.first().map_or(0, Vec::len);
    let (supp, order): (Vec<f64>, Vec<usize>) = match atom_coords {
        Some(cds) if cds.elem_count() == k => {
            let s = cds.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?;
            let mut idx: Vec<usize> = (0..k).collect();
            idx.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap());
            (idx.iter().map(|&i| s[i]).collect(), idx)
        }
        _ => ((0..k).map(|i| i as f64).collect(), (0..k).collect()),
    };
    let rows = rows
        .into_iter()
        .map(|r| order.iter().map(|&i| r[i]).collect())
        .collect();
    Ok((supp, rows))
}

#[cfg(test)]
mod tests {
    //! Verification of wasserstein fréchet regression, entropic barycentres and simplex-valued prediction.

    use super::*;
    use candle_core::Device;

    fn grid_cost(k: usize, dev: &Device) -> (Tensor, Tensor) {
        let supp: Vec<f64> = (0..k)
            .map(|i| -3.0 + 6.0 * i as f64 / (k - 1) as f64)
            .collect();
        let s = Tensor::from_vec(supp, k, dev).unwrap();
        let c = s
            .unsqueeze(1).unwrap()
            .broadcast_sub(&s.unsqueeze(0).unwrap()).unwrap()
            .sqr().unwrap();
        (s, c)
    }

    fn density(supp: &Tensor, centre: &Tensor, width: f64) -> Tensor {
        // [n, K] Gaussian bumps at `centre [n, 1]`.
        let z = supp
            .unsqueeze(0).unwrap()
            .broadcast_sub(centre).unwrap()
            .affine(1.0 / width, 0.0).unwrap();
        let d = z.sqr().unwrap().affine(-0.5, 0.0).unwrap().exp().unwrap()
            .affine(1.0, 1e-9).unwrap();
        let s = d.sum_keepdim(D::Minus1).unwrap();
        d.broadcast_div(&s).unwrap()
    }

    #[test]
    fn barycentre_preserves_mass_and_uniform_weights_recover_the_centre() {
        let dev = Device::Cpu;
        let (supp, c) = grid_cost(15, &dev);
        // Two symmetric bumps: their barycentre must sit between them.
        let centres = Tensor::from_vec(vec![-1.0f64, 1.0], (2, 1), &dev).unwrap();
        let p = density(&supp, &centres, 0.6);
        let log_p = p.log().unwrap();
        let w = Tensor::full(0.5f64, (1, 2), &dev).unwrap();
        let lb = log_ibp_barycentre(&log_p, &w, &c, 0.1, 60).unwrap();
        let b = lb.exp().unwrap();
        let mass = b.sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!((mass - 1.0).abs() < 1e-6, "barycentre mass {mass}");
        // Mean of the barycentre ≈ 0 (midpoint of ±1).
        let mean = b
            .broadcast_mul(&supp.unsqueeze(0).unwrap()).unwrap()
            .sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(mean.abs() < 0.15, "barycentre mean {mean}");
        // Negative weights refuse.
        let wneg = Tensor::from_vec(vec![1.5f64, -0.5], (1, 2), &dev).unwrap();
        assert!(log_ibp_barycentre(&log_p, &wneg, &c, 0.1, 10).is_err());
    }

    #[test]
    fn global_weights_are_the_ols_hat_weights() {
        let dev = Device::Cpu;
        // 1-D covariate: s_i(x) = 1 + (x − x̄)(X_i − x̄)/var — hand-checkable.
        let x = Tensor::from_vec(vec![-1.0f64, 0.0, 1.0], (3, 1), &dev).unwrap();
        let xq = Tensor::from_vec(vec![1.0f64], (1, 1), &dev).unwrap();
        let w = frechet_weights_global(&x, &xq, 1e-12).unwrap();
        let v = w.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        // var = 1 (n−1 denominator), so raw s = 1 + 1·(X_i) = (0, 1, 2);
        // normalised → (0, 1/3, 2/3).
        assert!((v[0] - 0.0).abs() < 1e-9, "{v:?}");
        assert!((v[1] - 1.0 / 3.0).abs() < 1e-9, "{v:?}");
        assert!((v[2] - 2.0 / 3.0).abs() < 1e-9, "{v:?}");
        let sum: f64 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn divergence_is_zero_iff_equal_and_positive_otherwise() {
        let dev = Device::Cpu;
        let (supp, c) = grid_cost(12, &dev);
        let centres = Tensor::from_vec(vec![-1.0f64, 0.5], (2, 1), &dev).unwrap();
        let p = density(&supp, &centres, 0.5);
        let (_, s_same) = sinkhorn_divergence_simplex(&p, &p, &c, 0.05, 60).unwrap();
        let v = s_same.to_vec1::<f64>().unwrap();
        assert!(v.iter().all(|x| x.abs() < 1e-8), "S(p,p) = {v:?}");
        let q = density(&supp, &centres.affine(1.0, 0.7).unwrap(), 0.5);
        let (_, s_diff) = sinkhorn_divergence_simplex(&p, &q, &c, 0.05, 60).unwrap();
        let v = s_diff.to_vec1::<f64>().unwrap();
        assert!(v.iter().all(|x| *x > 1e-4), "S(p,q) = {v:?}");
    }

    #[test]
    fn signal_beats_the_null_and_the_null_family_scores_near_zero() {
        // The Python synthetic self-test, shrunk: a location family whose
        // mean tracks x must give r² >> 0; an x-independent family ~ 0.
        let dev = Device::Cpu;
        let (supp, c) = grid_cost(15, &dev);
        let n = 40;
        // deterministic "uniform" x in [-1, 1]
        let xs: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * ((i * 7919) % 1000) as f64 / 999.0)
            .collect();
        let x = Tensor::from_vec(xs.clone(), (n, 1), &dev).unwrap();
        let centres = x.affine(1.8, 0.0).unwrap();
        let p_sig = density(&supp, &centres, 0.6);
        // x-independent centres (a shuffled copy of xs)
        let mut sh = xs.clone();
        sh.rotate_left(17);
        let cen_null = Tensor::from_vec(sh, (n, 1), &dev).unwrap().affine(1.8, 0.0).unwrap();
        let p_null = density(&supp, &cen_null, 0.6);

        let mut fr = WassersteinFrechetRegression::new(c.clone(), 0.05, "local", None, 40, 40).unwrap();
        let fit_sig = fr.fit_predict(&x, &p_sig, None).unwrap();
        assert!(fit_sig.r2 > 0.5, "signal r² {} should be >> 0", fit_sig.r2);

        let mut fr2 = WassersteinFrechetRegression::new(c, 0.05, "local", None, 40, 40).unwrap();
        let fit_null = fr2.fit_predict(&x, &p_null, None).unwrap();
        assert!(
            fit_null.r2 < 0.5 * fit_sig.r2,
            "null r² {} must be well below signal {}",
            fit_null.r2,
            fit_sig.r2
        );
    }

    #[test]
    fn global_mode_reports_negative_mass_and_densities_table_orders() {
        let dev = Device::Cpu;
        let (supp, c) = grid_cost(10, &dev);
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 2.0 - 1.0).collect();
        let x = Tensor::from_vec(xs, (n, 1), &dev).unwrap();
        let p = density(&supp, &x.affine(1.5, 0.0).unwrap(), 0.6);
        let mut fr = WassersteinFrechetRegression::new(c, 0.05, "global", None, 30, 30).unwrap();
        let fit = fr.fit_predict(&x, &p, None).unwrap();
        assert!(fit.neg_weight_mass >= 0.0);
        assert_eq!(fit.mode, "global");

        let (s, rows) = densities_table(&fit.pred, Some(&supp)).unwrap();
        assert_eq!(s.len(), 10);
        assert!(s.windows(2).all(|w| w[0] <= w[1]), "support must sort");
        assert_eq!(rows.len(), n);
    }
}
