//! Inferno — Gaussian-mixture bridge drift fields and stochastic integration settings.

use candle_core::{Result, Tensor};

use crate::flows::{Direction, DriftField};

/// Diagonal-covariance Gaussian mixture: `weights [K]`, `means [K, D]`,
/// `vars [K, D]` (per-dimension variances).
#[derive(Clone, Debug)]
pub struct GaussianMixture {
    pub weights: Tensor,
    pub means: Tensor,
    pub vars: Tensor,
}

impl GaussianMixture {
    /// Aggregate a batch of diagonal mixtures into one — the canonical
    /// `_aggregate_mixture` replacement (`stochastics/utils.py from_batched`).
    /// `weights_b [B, K]`, `means_b [B, K, D]`, `vars_b [B, K, D]`.
    /// Modes: `"first"` (batch element 0), `"concat"` (pool B·K components,
    /// weights ÷ B), `"average"` (moment-match: the between-batch mean spread
    /// folds into the variance, elementwise-exact for diagonals).
    pub fn from_batched(
        weights_b: &Tensor,
        means_b: &Tensor,
        vars_b: &Tensor,
        mode: &str,
    ) -> Result<Self> {
        let (b, k, d) = means_b.dims3()?;
        match mode {
            "first" => Ok(Self::new(
                weights_b.narrow(0, 0, 1)?.squeeze(0)?,
                means_b.narrow(0, 0, 1)?.squeeze(0)?,
                vars_b.narrow(0, 0, 1)?.squeeze(0)?,
            )),
            "concat" => Ok(Self::new(
                weights_b.affine(1.0 / b as f64, 0.0)?.reshape(b * k)?,
                means_b.reshape((b * k, d))?,
                vars_b.reshape((b * k, d))?,
            )),
            "average" => {
                let w = weights_b.mean(0)?;
                let m = means_b.mean(0)?; // [K, D]
                let spread = means_b.broadcast_sub(&m.unsqueeze(0)?)?.sqr()?.mean(0)?;
                Ok(Self::new(w, m, vars_b.mean(0)?.add(&spread)?))
            }
            other => Err(candle_core::Error::Msg(format!("unknown aggregate mode {other}"))),
        }
    }

    pub fn new(weights: Tensor, means: Tensor, vars: Tensor) -> Self {
        Self { weights, means, vars }
    }

    pub fn k(&self) -> Result<usize> {
        self.means.dim(0)
    }
    pub fn d(&self) -> Result<usize> {
        self.means.dim(1)
    }
    /// Mixture mean `Σ_k w_k μ_k` → `[D]`.
    pub fn mean(&self) -> Result<Tensor> {
        self.weights.unsqueeze(1)?.broadcast_mul(&self.means)?.sum(0)
    }
}

#[derive(Clone, Debug)]
pub struct SdeConfig {
    pub t: f64,
    pub sigma: f64,
    pub eps_t: f64,
    pub sinkhorn_eps: f64,
    pub sinkhorn_iters: usize,
}

impl Default for SdeConfig {
    fn default() -> Self {
        Self { t: 1.0, sigma: 0.5, eps_t: 1e-3, sinkhorn_eps: 0.05, sinkhorn_iters: 400 }
    }
}

/// A fitted GMM Schrödinger bridge. Build once per policy; `drift` is then
/// closed form. Implements [`DriftField`] so it drops into [`crate::flows::Flow`].
pub struct GmmSchrodingerBridge {
    policy: GaussianMixture,
    expert: GaussianMixture,
    /// Diagonal cross-covariance `C*_{ij,d} = √(aᵢ_d · bⱼ_d)` → `[K0, K1, D]`.
    cstar: Tensor,
    /// Sinkhorn coupling over components → `[K0, K1]`.
    w: Tensor,
    cfg: SdeConfig,
}

impl GmmSchrodingerBridge {
    /// Fit: diagonal Bures cross-cov + diagonal Bures-W² cost + Sinkhorn `W`,
    /// with the config's constant `sinkhorn_eps`. On any training path prefer
    /// [`fit_with_eps`](Self::fit_with_eps) with the learnable temperature.
    pub fn fit(policy: GaussianMixture, expert: GaussianMixture, cfg: SdeConfig) -> Result<Self> {
        let eps = policy.weights.zeros_like()?.affine(0.0, cfg.sinkhorn_eps)?.narrow(0, 0, 1)?.reshape(())?;
        Self::fit_with_eps(policy, expert, cfg, &eps)
    }

    /// Tensor-valued temperature retains coupling gradients.
    pub fn fit_with_eps(
        policy: GaussianMixture,
        expert: GaussianMixture,
        cfg: SdeConfig,
        eps: &Tensor,
    ) -> Result<Self> {
        if !cfg.t.is_finite() || cfg.t <= 0.0 || !cfg.eps_t.is_finite() || cfg.eps_t <= 0.0 || cfg.eps_t >= cfg.t / 2.0 || !cfg.sigma.is_finite() || cfg.sigma < 0.0 {
            candle_core::bail!("invalid bridge horizon, endpoint cutoff or diffusion");
        }
        for mixture in [&policy, &expert] {
            let (k, d) = mixture.means.dims2()?;
            if k == 0 || d == 0 || mixture.weights.dims() != [k] || mixture.vars.dims() != [k, d] { candle_core::bail!("invalid mixture dimensions"); }
            for tensor in [&mixture.means, &mixture.vars] {
                let values = tensor.to_dtype(candle_core::DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
                if values.iter().any(|v| !v.is_finite()) { candle_core::bail!("nonfinite mixture"); }
            }
            if mixture.vars.min_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()? <= 0.0 { candle_core::bail!("mixture variances must be positive"); }
        }
        if policy.d()? != expert.d()? { candle_core::bail!("mixture feature widths differ"); }
        let ai = policy.vars.unsqueeze(1)?; // [K0, 1, D]
        let bj = expert.vars.unsqueeze(0)?; // [1, K1, D]
        let cstar = ai.broadcast_mul(&bj)?.sqrt()?; // [K0, K1, D]

        // cost = ||μi − μj||² + Σ_d (√aᵢ_d − √bⱼ_d)²  (diagonal Bures-W²)
        let mi = policy.means.unsqueeze(1)?; // [K0, 1, D]
        let mj = expert.means.unsqueeze(0)?; // [1, K1, D]
        let mdiff = mi.broadcast_sub(&mj)?.sqr()?.sum(2)?; // [K0, K1]
        let sdiff = ai.sqrt()?.broadcast_sub(&bj.sqrt()?)?.sqr()?.sum(2)?; // [K0, K1]
        let cost = (mdiff + sdiff)?;

        let w = crate::kernels::sinkhorn::sinkhorn_coupling_t(
            &policy.weights,
            &expert.weights,
            &cost,
            eps,
            cfg.sinkhorn_iters,
            1e-9,
        )?;

        Ok(Self { policy, expert, cstar, w, cfg })
    }

    fn s_of(&self, t_phys: f64) -> f64 {
        let r = self.cfg.eps_t / self.cfg.t;
        (t_phys / self.cfg.t).clamp(r, 1.0 - r)
    }

    /// Per-pair Brownian marginal `(m [K0,K1,D], v [K0,K1,D])` at `t_phys`.
    fn per_pair_marginal(&self, t_phys: f64) -> Result<(Tensor, Tensor)> {
        if !t_phys.is_finite() || t_phys < 0.0 || t_phys > self.cfg.t { candle_core::bail!("bridge time outside [0,T]"); }
        let t_phys = t_phys.clamp(self.cfg.eps_t, self.cfg.t - self.cfg.eps_t);
        let s = self.s_of(t_phys);
        let (t, sigma) = (self.cfg.t, self.cfg.sigma);
        let mi = self.policy.means.unsqueeze(1)?; // [K0,1,D]
        let mj = self.expert.means.unsqueeze(0)?; // [1,K1,D]
        let ai = self.policy.vars.unsqueeze(1)?;
        let bj = self.expert.vars.unsqueeze(0)?;

        let m = mi.affine(1.0 - s, 0.0)?.broadcast_add(&mj.affine(s, 0.0)?)?;
        let bridge_var = sigma * sigma * t_phys * (t - t_phys) / t;
        let v = ai
            .affine((1.0 - s) * (1.0 - s), 0.0)?
            .broadcast_add(&bj.affine(s * s, 0.0)?)?
            .broadcast_add(&self.cstar.affine(2.0 * s * (1.0 - s), 0.0)?)?
            .affine(1.0, bridge_var)?;
        Ok((m, v))
    }

    /// Closed-form drift `b(x, t)` for `x: [B, D]` → `[B, D]`.
    /// The Sinkhorn coupling over components, `[K_policy, K_expert]`.
    ///
    /// Exposed because it IS a deliverable, not just an intermediate: the
    /// transport plan between the two endpoint mixtures is what a
    /// customer buying the training-free rung is buying.
    pub fn coupling(&self) -> &Tensor {
        &self.w
    }

    /// `log ρ_t(x)` — log-density of the bridge marginal at `t_phys`
    /// (`x: [B, D]` → `[B]`): a `[K0·K1]`-component diagonal Gaussian
    /// mixture with the (normalised) Sinkhorn coupling as weights. The
    /// Python `GaussianSubBridges.log_prob_marginal`, consumed by the
    /// likelihood network's bridge-support gate.
    pub fn log_prob_marginal(&self, x: &Tensor, t_phys: f64) -> Result<Tensor> {
        let (m, v) = self.per_pair_marginal(t_phys)?; // [K0,K1,D]
        let d = self.policy.d()?;
        let x4 = x.unsqueeze(1)?.unsqueeze(1)?; // [B,1,1,D]
        let m4 = m.unsqueeze(0)?;
        let v4 = v.unsqueeze(0)?;
        let diff = x4.broadcast_sub(&m4)?;
        let quad = diff.sqr()?.broadcast_div(&v4)?.sum(3)?; // [B,K0,K1]
        let logdet = v.log()?.sum(2)?.unsqueeze(0)?; // [1,K0,K1]
        let two_pi = (2.0 * std::f64::consts::PI).ln() * d as f64;
        let log_n = quad
            .broadcast_add(&logdet)?
            .affine(-0.5, -0.5 * two_pi)?; // [B,K0,K1]
        let wsum = self.w.sum_all()?.clamp(1e-12, f64::INFINITY)?;
        let log_w = self
            .w
            .broadcast_div(&wsum)?
            .clamp(1e-30, f64::INFINITY)?
            .log()?
            .unsqueeze(0)?; // [1,K0,K1]
        let arg = log_n.broadcast_add(&log_w)?;
        let (b, k0, k1) = arg.dims3()?;
        let flat = arg.reshape((b, k0 * k1))?;
        // logsumexp over components, max-shifted for stability.
        let mx = flat.max_keepdim(candle_core::D::Minus1)?;
        let lse = flat
            .broadcast_sub(&mx)?
            .exp()?
            .sum_keepdim(candle_core::D::Minus1)?
            .clamp(1e-30, f64::INFINITY)?
            .log()?
            .broadcast_add(&mx)?;
        lse.squeeze(candle_core::D::Minus1)
    }

    /// Per-dimension marginal spread of the bridge at `t_phys` — the
    /// responsibility-free mean over the per-pair diagonal covariances
    /// (`[D]`). A scale, not a per-sample quantity: the resolution the
    /// likelihood network's drift anchor wants.
    pub fn marginal_diag_spread(&self, t_phys: f64) -> Result<Tensor> {
        let (_m, v) = self.per_pair_marginal(t_phys)?; // [K0,K1,D]
        let d = self.policy.d()?;
        v.reshape(((), d))?.mean(0)
    }

    /// The endpoint mixtures this bridge was fit between.
    pub fn policy(&self) -> &GaussianMixture {
        &self.policy
    }

    pub fn expert(&self) -> &GaussianMixture {
        &self.expert
    }

    pub fn drift(&self, x: &Tensor, t_phys: f64, direction: Direction) -> Result<Tensor> {
        if !t_phys.is_finite() || t_phys < 0.0 || t_phys > self.cfg.t { candle_core::bail!("bridge time outside [0,T]"); }
        let t_phys = t_phys.clamp(self.cfg.eps_t, self.cfg.t - self.cfg.eps_t);
        let s = self.s_of(t_phys);
        let (m, v) = self.per_pair_marginal(t_phys)?; // [K0,K1,D]
        let d = self.policy.d()?;

        // ---- responsibilities q_{ij}(x) ∝ W_{ij} · N(x; m, diag v) ----
        let x4 = x.unsqueeze(1)?.unsqueeze(1)?; // [B,1,1,D]
        let m4 = m.unsqueeze(0)?; // [1,K0,K1,D]
        let v4 = v.unsqueeze(0)?;
        let diff = x4.broadcast_sub(&m4)?;
        let quad = diff.sqr()?.broadcast_div(&v4)?.sum(3)?; // [B,K0,K1]
        let logdet = v.log()?.sum(2)?.unsqueeze(0)?; // [1,K0,K1]
        let two_pi = (2.0 * std::f64::consts::PI).ln() * d as f64;
        let log_n = quad
            .broadcast_add(&logdet)?
            .affine(1.0, two_pi)?
            .affine(-0.5, 0.0)?; // [B,K0,K1]
        let log_w = self.w.maximum(&self.w.affine(0.0, 1e-30)?)?.log()?.unsqueeze(0)?; // [1,K0,K1]
        let log_q = log_n.broadcast_add(&log_w)?; // [B,K0,K1]
        let q = softmax_2d(&log_q)?; // [B,K0,K1], normalised over (K0,K1)

        // ---- per-pair conditional drift ----
        let ai = self.policy.vars.unsqueeze(1)?; // [K0,1,D]
        let bj = self.expert.vars.unsqueeze(0)?; // [1,K1,D]
        let (cross, anchor, denom) = match direction {
            Direction::Forward => {
                // Cov(X_T, X_t) = (1−s)C*ᵀ + sΣⱼ ; anchor = μⱼ ; denom = T−t
                let cross = self
                    .cstar
                    .affine(1.0 - s, 0.0)?
                    .broadcast_add(&bj.affine(s, 0.0)?)?; // [K0,K1,D]
                let anchor = self.expert.means.unsqueeze(0)?; // [1,K1,D]
                (cross, anchor, (self.cfg.t - t_phys).max(self.cfg.eps_t))
            }
            Direction::Backward => {
                // Cov(X_0, X_t) = (1−s)Σᵢ + sC* ; anchor = μᵢ ; denom = t
                let cross = ai
                    .affine(1.0 - s, 0.0)?
                    .broadcast_add(&self.cstar.affine(s, 0.0)?)?; // [K0,K1,D]
                let anchor = self.policy.means.unsqueeze(1)?; // [K0,1,D]
                (cross, anchor, t_phys.max(self.cfg.eps_t))
            }
        };

        let ratio = diff.broadcast_div(&v4)?; // [B,K0,K1,D]  (x − m)/v
        let cond = cross
            .unsqueeze(0)?
            .broadcast_mul(&ratio)?
            .broadcast_add(&anchor.unsqueeze(0)?)?; // [B,K0,K1,D] = anchor + cross·(x−m)/v
        let sign = if direction == Direction::Forward { 1.0 } else { -1.0 };
        let per_pair = cond.broadcast_sub(&x4)?.affine(sign / denom, 0.0)?; // [B,K0,K1,D]

        // ---- mix by responsibilities ----
        let weighted = q.unsqueeze(3)?.broadcast_mul(&per_pair)?; // [B,K0,K1,D]
        weighted.sum(1)?.sum(1) // sum over K0 then K1 → [B, D]
    }


}

impl DriftField for GmmSchrodingerBridge {
    fn drift(&self, x: &Tensor, t: &Tensor, direction: Direction) -> Result<Tensor> {
        if t.elem_count() != 1 { candle_core::bail!("bridge drift requires one shared physical time"); }
        let t_phys = t.reshape(())?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
        GmmSchrodingerBridge::drift(self, x, t_phys, direction)
    }
}

/// Softmax over the last two axes of a `[B, K0, K1]` tensor.
fn softmax_2d(log_q: &Tensor) -> Result<Tensor> {
    let (b, k0, k1) = log_q.dims3()?;
    let flat = log_q.reshape((b, k0 * k1))?;
    let sm = candle_nn::ops::softmax(&flat, 1)?;
    sm.reshape((b, k0, k1))
}
