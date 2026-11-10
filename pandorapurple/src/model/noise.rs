//! Inferno — Noise parameterisations and sampling helpers for probabilistic models.

use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Deterministic per-site seed
// ---------------------------------------------------------------------------

/// Deterministic per-site seed from the site name (FNV-1a; the reference uses
/// blake2b — the exact value differs but determinism per site name holds).
/// `salt` sweeps noise realisations (e.g. for ensembling) without changing the
/// site manifest.
pub fn site_seed(site: &str, salt: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in site.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h ^ salt) & 0x7FFF_FFFF
}

// ---------------------------------------------------------------------------
// Quasi-random normal draw (Owen-scrambled Sobol → Φ⁻¹)
// ---------------------------------------------------------------------------

/// `sobol_burley`'s direction tables cover 256 dimensions; wider draws chunk
/// into independent blocks with a derived (golden-ratio-mixed) seed each.
const SOBOL_MAX_DIMS: usize = 256;

#[inline]
fn sobol_u(index: usize, dim: usize, seed: u64) -> f64 {
    let block = dim / SOBOL_MAX_DIMS;
    let block_seed = (seed as u32).wrapping_add((block as u32).wrapping_mul(0x9E37_79B9));
    sobol_burley::sample(index as u32, (dim % SOBOL_MAX_DIMS) as u32, block_seed) as f64
}

/// Owen-scrambled Sobol points as a `[samples, dimensions]` tensor in
/// `[0, 1)` — the raw engine (the `sobol_burley` wrapper); `seed` keys the
/// scrambling, so distinct sites/salts get independent randomisations of the
/// same low-discrepancy structure.
pub fn sobol_engine(
    samples: usize,
    dimensions: usize,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(samples * dimensions);
    for i in 0..samples {
        for d in 0..dimensions {
            data.push(sobol_u(i, d, seed) as f32);
        }
    }
    Tensor::from_vec(data, (samples, dimensions), device)
}

/// SOFT Dirichlet boundary: affine squash `u ↦ ε + (1−2ε)·u` into
/// `(ε, 1−ε)`. Unlike a hard clamp — which piles probability mass at the
/// rails and hands `log`/`Φ⁻¹` consumers exactly the −∞-adjacent values that
/// blow up stick-breaking near 0 — the squash is smooth, order-preserving,
/// and keeps the low-discrepancy structure intact.
#[inline]
fn soft_unit(u: f64, eps: f64) -> f64 {
    eps + (1.0 - 2.0 * eps) * u
}

/// `n × d` standard-normal QMC draws (row-major `[n][d]`), deterministic in
/// `seed`. `scramble=false` fixes the Owen key to 0 (the shared unscrambled
/// sequence); `true` keys it by `seed`. `u → Φ⁻¹(u) = √2·erf⁻¹(2u−1)`
/// (statrs), after the soft boundary squash.
fn qmc_randn(n: usize, d: usize, seed: u64, scramble: bool) -> Vec<f32> {
    let key = if scramble { seed } else { 0 };
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let u = soft_unit(sobol_u(i, j, key), 1e-7);
            out[i * d + j] =
                (std::f64::consts::SQRT_2 * statrs::function::erf::erf_inv(2.0 * u - 1.0)) as f32;
        }
    }
    out
}

/// Owen-scrambled Sobol → standard normals as a `[samples, dimensions]`
/// tensor — the pure-Rust `sobol_normal_engine`.
pub fn sobol_normal_engine(
    samples: usize,
    dimensions: usize,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    Tensor::from_vec(qmc_randn(samples, dimensions, seed, true), (samples, dimensions), device)
}

/// Scramble key for column `j` of a per-dimension draw. Golden-ratio mix, so
/// adjacent columns land far apart in key space.
#[inline]
fn per_dim_key(seed: u64, j: usize) -> u64 {
    seed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
}

/// PER-DIMENSION Owen scrambling — every column an INDEPENDENT scramble of the
/// same 1-D Sobol sequence, rather than one scramble of a `d`-dimensional net.
///
/// # Why weight matrices want this and integrators do not
///
/// [`sobol_randn_like`] draws `n` points of a `d`-dimensional net, which is the
/// right object for integration: the whole point is JOINT equidistribution over
/// the `d` axes. A weight matrix is not an integration problem. Its columns are
/// separate fan-in directions, and what a net's joint structure buys there is a
/// pairwise-projection artefact — measured `max|corr|` between columns of
/// **0.66** at `[512, 512]`, against **0.195** for an iid draw, and it does NOT
/// shrink with `n` because it is structural rather than sampling noise. (The
/// MEAN correlation is meanwhile better than iid, which is exactly why this
/// hides: the aggregate looks healthy while one pair is badly aliased.)
///
/// Scrambling each dimension under its own key gives up the joint property and
/// keeps the one that matters here — each column is still a perfectly
/// stratified, unbiased 1-D sequence, so the marginals stay tighter than iid
/// while the columns become genuinely independent.
pub fn sobol_randn_like_per_dim(shape: &[usize], device: &Device, seed: u64) -> Result<Tensor> {
    let n = shape.first().copied().unwrap_or(1);
    let d: usize = shape.iter().skip(1).product::<usize>().max(1);
    let mut out = vec![0f32; n * d];
    for j in 0..d {
        let key = per_dim_key(seed, j);
        for i in 0..n {
            let u = soft_unit(sobol_u(i, 0, key), 1e-7);
            out[i * d + j] =
                (std::f64::consts::SQRT_2 * statrs::function::erf::erf_inv(2.0 * u - 1.0)) as f32;
        }
    }
    Tensor::from_vec(out, shape.to_vec(), device)
}

/// [`sobol_randn_like_per_dim`]'s bounded twin — uniform `[0, 1)`.
pub fn sobol_uniform_like_per_dim(shape: &[usize], device: &Device, seed: u64) -> Result<Tensor> {
    let n = shape.first().copied().unwrap_or(1);
    let d: usize = shape.iter().skip(1).product::<usize>().max(1);
    let mut out = vec![0f32; n * d];
    for j in 0..d {
        let key = per_dim_key(seed, j);
        for i in 0..n {
            out[i * d + j] = sobol_u(i, 0, key) as f32;
        }
    }
    Tensor::from_vec(out, shape.to_vec(), device)
}

/// UNIFORM `[0, 1)` draw matching `shape` via the raw QMC sequence — the
/// pre-`Φ⁻¹` half of [`sobol_randn_like`], for parameters whose support is
/// bounded and whose init must therefore not have Gaussian tails.
pub fn sobol_uniform_like(shape: &[usize], device: &Device, seed: u64) -> Result<Tensor> {
    let n = shape.first().copied().unwrap_or(1);
    let d: usize = shape.iter().skip(1).product::<usize>().max(1);
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            data.push(sobol_u(i, j, seed) as f32);
        }
    }
    Tensor::from_vec(data, shape.to_vec(), device)
}

/// Gaussian draw matching `shape` via the QMC sequence → `Φ⁻¹`. First dim is
/// the sample axis; the rest is the per-sample dimensionality `d`.
pub fn sobol_randn_like(shape: &[usize], device: &Device, seed: u64) -> Result<Tensor> {
    let n = shape.first().copied().unwrap_or(1);
    let d: usize = shape.iter().skip(1).product::<usize>().max(1);
    let data = qmc_randn(n, d, seed, true);
    Tensor::from_vec(data, shape.to_vec(), device)
}

// ---------------------------------------------------------------------------
// NoiseSource — the one place a reproducible draw comes from
// ---------------------------------------------------------------------------

/// Where a sampling path gets its randomness, seeded or not, chosen once and
/// threaded rather than reached for ad hoc.
///
/// WHY THIS EXISTS. Reproducibility kept failing for a reason that was hard
/// to see from any single site: `Device::set_seed` is refused by the
/// vendored candle's CPU backend, so `Tensor::randn`/`randn_like` ignore a
/// requested seed there, and `rand::random()` never had one to ignore. A
/// run could therefore be "seeded" in its config, in its logs and in its
/// report, and still differ every time. Making the source an explicit
/// argument is what turns that from a property nobody can check into one
/// the type system carries.
///
/// The seeded variant draws from the SAME Owen-scrambled Sobol sequence the
/// rest of this module uses, so a low-discrepancy draw is the default rather
/// than an opt-in — the point of quasi-random sampling is that it covers the
/// space better per draw, which matters most exactly where these are used
/// (small chain counts, small mixture-component counts).
///
/// # NOT for secrets
///
/// This is a DETERMINISTIC engine and must never back a session token, an
/// API key, a CSP nonce or an HMAC nonce. Those live in `deepkernelsapi`
/// and correctly use `rand::thread_rng().fill_bytes` — a CSPRNG seeded from
/// the OS. "Normalising" them onto this would make every one of them
/// predictable from a seed, which is a vulnerability rather than a
/// consistency win. The distinction is between randomness that must be
/// REPRODUCIBLE and randomness that must be UNGUESSABLE, and no single
/// engine can be both.
#[derive(Debug)]
pub struct NoiseSource {
    seed: Option<u64>,
    /// Advanced per draw so successive calls differ while the whole
    /// sequence stays a function of `seed`.
    counter: std::sync::atomic::AtomicU64,
}

impl NoiseSource {
    /// Reproducible: every draw is a function of `seed` and the call order.
    pub fn seeded(seed: u64) -> Self {
        Self { seed: Some(seed), counter: std::sync::atomic::AtomicU64::new(0) }
    }

    /// Non-reproducible, and named so at the call site. Use where a fresh
    /// draw per run is wanted; do NOT use where a secret is wanted.
    pub fn entropy() -> Self {
        Self { seed: None, counter: std::sync::atomic::AtomicU64::new(0) }
    }

    pub fn is_seeded(&self) -> bool {
        self.seed.is_some()
    }

    /// Next per-draw key. Mixed with a golden-ratio odd constant so
    /// consecutive draws get well-separated Owen scramblings rather than
    /// adjacent ones.
    fn next_key(&self) -> Option<u64> {
        let n = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.seed.map(|s| s ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// Standard normals of `shape`.
    pub fn randn(&self, shape: &[usize], device: &Device, dtype: DType) -> Result<Tensor> {
        match self.next_key() {
            Some(k) => sobol_randn_like(shape, device, k)?.to_dtype(dtype),
            None => Tensor::randn(0f32, 1f32, shape, device)?.to_dtype(dtype),
        }
    }

    /// `n` uniforms in `[0, 1)`.
    pub fn uniform(&self, n: usize) -> Vec<f64> {
        match self.next_key() {
            Some(k) => (0..n).map(|i| sobol_u(i, 0, k)).collect(),
            None => (0..n).map(|_| rand::random::<f64>()).collect(),
        }
    }

    /// `n` independent categorical draws over `weights`, by inverse CDF.
    ///
    /// Weights need not be normalised. A non-positive total is degenerate
    /// rather than an error — every draw returns component 0, which is what
    /// a mixture with no mass anywhere can honestly say.
    pub fn categorical(&self, weights: &[f32], n: usize) -> Vec<usize> {
        if weights.is_empty() {
            return vec![0; n];
        }
        let total: f32 = weights.iter().sum();
        if !(total > 0.0) {
            return vec![0; n];
        }
        self.uniform(n)
            .into_iter()
            .map(|u| {
                let r = u as f32 * total;
                let mut acc = 0f32;
                for (i, w) in weights.iter().enumerate() {
                    acc += *w;
                    if r <= acc {
                        return i;
                    }
                }
                weights.len() - 1
            })
            .collect()
    }
}

impl Default for NoiseSource {
    /// Entropy, matching what every un-threaded call site did before this
    /// type existed — so adding it changes no behaviour until a caller opts
    /// into a seed.
    fn default() -> Self {
        Self::entropy()
    }
}

// ---------------------------------------------------------------------------
// NoiseBundle
// ---------------------------------------------------------------------------

/// Per-site reparameterisation noise, each tensor carrying a leading `S`
/// (sample) axis. Mirrors the Python `NoiseBundle`'s dict-style access.
#[derive(Clone, Debug, Default)]
pub struct NoiseBundle(pub HashMap<String, Tensor>);

impl NoiseBundle {
    pub fn from_map(m: HashMap<String, Tensor>) -> Self {
        Self(m)
    }
    pub fn get(&self, site: &str) -> Option<&Tensor> {
        self.0.get(site)
    }
    pub fn insert(&mut self, site: impl Into<String>, t: Tensor) {
        self.0.insert(site.into(), t);
    }
    pub fn sites(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Shared `S` sample axis.
    pub n_samples: usize,
    /// Per-site stratification count (averaged within each `S` slot).
    pub per_site_strata: HashMap<String, usize>,
    /// Sites that get antithetic pairing (free 2× variance reduction).
    pub antithetic_sites: HashSet<String>,
    /// Quasi-Monte-Carlo (Halton) vs plain randn.
    pub qmc: bool,
    pub iwae: bool,
    pub dreg: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        let per_site_strata = HashMap::from([
            ("local_evidence_d".to_string(), 4),
            ("local_evidence_f".to_string(), 4),
            ("latent_z".to_string(), 2),
            ("decoder_alpha".to_string(), 4),
        ]);
        let antithetic_sites = HashSet::from([
            "latent_z".to_string(),
            "local_evidence_d".to_string(),
            "local_evidence_f".to_string(),
            "decoder_alpha".to_string(),
        ]);
        Self { n_samples: 8, per_site_strata, antithetic_sites, qmc: true, iwae: false, dreg: false }
    }
}

/// Draws deterministic, stratified, (optionally) quasi-random noise bundles.
pub struct BayesianPosteriorSampler {
    pub cfg: SamplerConfig,
}

impl BayesianPosteriorSampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        Self { cfg }
    }

    /// Build one [`NoiseBundle`] with a leading `S` axis. `site_shapes` are the
    /// per-sample shapes (without `S`). `salt` sweeps realisations.
    pub fn draw_bundle(
        &self,
        site_shapes: &HashMap<String, Vec<usize>>,
        device: &Device,
        salt: u64,
    ) -> Result<NoiseBundle> {
        let s = self.cfg.n_samples;
        let mut sites: HashMap<String, Tensor> = HashMap::new();

        for (site, shape) in site_shapes {
            let strata = self.cfg.per_site_strata.get(site).copied().unwrap_or(1);
            let n_eff = s * strata;

            let mut full = vec![n_eff];
            full.extend_from_slice(shape);

            let mut eps = if self.cfg.qmc {
                sobol_randn_like(&full, device, site_seed(site, salt))?
            } else {
                Tensor::randn(0f32, 1f32, full.clone(), device)?
            };

            // Antithetic pairing: [η, −η].
            if self.cfg.antithetic_sites.contains(site) && n_eff % 2 == 0 {
                let half = eps.narrow(0, 0, n_eff / 2)?;
                eps = Tensor::cat(&[&half, &half.affine(-1.0, 0.0)?], 0)?;
            }

            // Stratify within each S slot: average the `strata` draws.
            if strata > 1 {
                let mut vshape = vec![s, strata];
                vshape.extend_from_slice(shape);
                eps = eps.reshape(vshape)?.mean(1)?;
            }

            sites.insert(site.clone(), eps);
        }
        Ok(NoiseBundle::from_map(sites))
    }

    /// One noise realisation per site WITHOUT the `S` axis — the Python
    /// `draw_single` the refinement loop calls once per Itô step
    /// (`salt = step_idx`): deterministic per site, distinct per step,
    /// common-random-numbers across forwards. Per-site strata still average
    /// within the single slot (matching `draw_bundle`'s stratification);
    /// antithetic pairing is skipped — a lone sample paired with its own
    /// negation would cancel to exactly zero.
    pub fn draw_single(
        &self,
        site_shapes: &HashMap<String, Vec<usize>>,
        device: &Device,
        salt: u64,
    ) -> Result<NoiseBundle> {
        let mut sites: HashMap<String, Tensor> = HashMap::new();
        for (site, shape) in site_shapes {
            let strata = self.cfg.per_site_strata.get(site).copied().unwrap_or(1);
            let mut full = vec![strata];
            full.extend_from_slice(shape);
            let eps = if self.cfg.qmc {
                sobol_randn_like(&full, device, site_seed(site, salt))?
            } else {
                Tensor::randn(0f32, 1f32, full, device)?
            };
            // mean over the strata axis also drops it: [strata, ..] → [..]
            sites.insert(site.clone(), eps.mean(0)?);
        }
        Ok(NoiseBundle::from_map(sites))
    }
}

// ---------------------------------------------------------------------------
// ELBO aggregation across the S axis (IWAE / DReG / vanilla) + diagnostics
// ---------------------------------------------------------------------------

fn logsumexp0(t: &Tensor) -> Result<Tensor> {
    // logsumexp over the leading (S) axis → reduces dim 0.
    let m = t.max_keepdim(0)?;
    let lse = t.broadcast_sub(&m)?.exp()?.sum(0)?.log()?;
    lse.add(&m.squeeze(0)?)
}

/// `−(data_term − analytic_kl)` aggregated over `S`. `log_*` are `[S, B]`.
pub fn elbo_loss(
    log_p_xz: &Tensor,
    log_p_z: &Tensor,
    log_q_z: &Tensor,
    analytic_kl: f64,
    iwae: bool,
    dreg: bool,
) -> Result<Tensor> {
    let log_w = ((log_p_xz + log_p_z)? - log_q_z)?; // [S, B]
    let s = log_w.dim(0)? as f64;

    let data_term = if iwae && dreg {
        // DReG: w_norm² · log_w, summed over S (kills the score-function term).
        let log_w_d = log_w.detach();
        let w_norm = log_w_d.broadcast_sub(&logsumexp0(&log_w_d)?)?.exp()?;
        w_norm.sqr()?.mul(&log_w)?.sum(0)?
    } else if iwae {
        logsumexp0(&log_w)?.affine(1.0, -s.ln())?
    } else {
        log_w.mean(0)? // vanilla SVI
    };

    data_term.mean_all()?.affine(-1.0, analytic_kl)
}

/// Effective sample size per batch element from importance weights `[S, B] → [B]`.
pub fn effective_sample_size(log_w: &Tensor) -> Result<Tensor> {
    let w = log_w.broadcast_sub(&logsumexp0(log_w)?)?.exp()?; // [S, B]
    let denom = w.sqr()?.sum(0)?.affine(1.0, 1e-12)?; // [B]
    denom.recip()
}

/// Self-normalised importance-sampling estimate with its Monte-Carlo
/// standard error — the Owen (2013) SNIS variance estimator as used in the
/// PSIS paper (Vehtari et al., eq. 5/7):
///
/// ```text
///   w̃_s   = w_s / Σ w_s'                       (normalised weights)
///   μ̃     = Σ w̃_s h(θ_s)                       (SNIS estimate)
///   V̂ar   = Σ w̃_s² (h(θ_s) − μ̃)² / R_eff       (R_eff = 1 for indep. draws)
///   MCSE  = √V̂ar
///   ESS_h = [ (1/S) Σ (h − h̄)² ] / V̂ar         (h-specific effective size)
/// ```
///
/// `log_w`/`h`: `[S, B]`; `r_eff` is the MCMC relative efficiency
/// (`S_eff/S`, split-chain R̂ machinery — pass `None` ⇒ 1.0 for independent
/// draws, which is what this sampler produces). Returns `(μ̃, mcse, ess_h)`,
/// each `[B]`.
pub fn snis_estimate(
    log_w: &Tensor,
    h: &Tensor,
    r_eff: Option<f64>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let s = log_w.dim(0)? as f64;
    let r_eff = r_eff.unwrap_or(1.0).max(1e-12);
    let w = log_w.broadcast_sub(&logsumexp0(log_w)?)?.exp()?; // w̃ [S, B]
    let mu = w.mul(h)?.sum(0)?; // μ̃ [B]
    let dev2 = h.broadcast_sub(&mu.unsqueeze(0)?)?.sqr()?; // (h−μ̃)² [S, B]
    let var_is = w.sqr()?.mul(&dev2)?.sum(0)?.affine(1.0 / r_eff, 0.0)?; // [B]
    let mcse = var_is.sqrt()?;
    // plain sample variance of h over S (eq. 7 numerator)
    let h_bar = h.mean(0)?;
    let h_var = h.broadcast_sub(&h_bar.unsqueeze(0)?)?.sqr()?.sum(0)?.affine(1.0 / s, 0.0)?;
    let ess_h = h_var.div(&var_is.affine(1.0, 1e-30)?)?;
    Ok((mu, mcse, ess_h))
}

/// PSIS Pareto k̂ per batch element (`[S, B] → [B]`) — Vehtari et al.'s
/// importance-weight tail diagnostic via the Zhang–Stephens (2009) GPD fit
/// (the arviz `gpdfit` formulation, incl. the n/(n+10) prior smoothing).
/// Tail = the largest `min(0.2·S, 3√S)` weights' exceedances over the cutoff
/// order statistic. `k̂ > 0.7` ⇒ the importance estimate is unreliable.
/// Columns with a degenerate tail (fewer than 5 exceedances) return NaN.
pub fn pareto_khat(log_w: &Tensor) -> Result<Tensor> {
    let (s, b) = log_w.dims2()?;
    let cols: Vec<Vec<f32>> = log_w
        .t()?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec2()?; // [B][S]
    let tail_len = ((0.2 * s as f64).min(3.0 * (s as f64).sqrt()).ceil() as usize).min(s - 1);
    let mut khats = Vec::with_capacity(b);
    for col in cols {
        let mut lw: Vec<f64> = col.iter().map(|x| *x as f64).collect();
        lw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // exceedances over the cutoff, in the (stabilised) weight domain
        let cutoff = lw[s - 1 - tail_len];
        let max_lw = lw[s - 1];
        let x: Vec<f64> = lw[s - tail_len..]
            .iter()
            .map(|l| (l - max_lw).exp() - (cutoff - max_lw).exp())
            .filter(|v| *v > 0.0)
            .collect();
        let n = x.len();
        if n < 5 {
            khats.push(f32::NAN);
            continue;
        }
        // Zhang–Stephens profile-posterior fit
        let m = 30 + (n as f64).sqrt() as usize;
        let q1 = x[(n as f64 / 4.0 + 0.5) as usize - 1];
        let xn = x[n - 1];
        let bs: Vec<f64> = (1..=m)
            .map(|j| (1.0 - (m as f64 / (j as f64 - 0.5)).sqrt()) / (3.0 * q1) + 1.0 / xn)
            .collect();
        let ks: Vec<f64> = bs
            .iter()
            .map(|bj| x.iter().map(|xi| (-bj * xi).ln_1p()).sum::<f64>() / n as f64)
            .collect();
        let lens: Vec<f64> = bs
            .iter()
            .zip(&ks)
            .map(|(bj, kj)| n as f64 * ((-(bj / kj)).ln() - kj - 1.0))
            .collect();
        let lmax = lens.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = lens.iter().map(|l| (l - lmax).exp()).collect();
        let wsum: f64 = ws.iter().sum();
        let b_post: f64 = bs.iter().zip(&ws).map(|(bj, wj)| bj * wj).sum::<f64>() / wsum;
        let k_post = x.iter().map(|xi| (-b_post * xi).ln_1p()).sum::<f64>() / n as f64;
        // prior smoothing toward 0.5 (a = 10, the arviz `prior_k`)
        let k_hat = (n as f64 * k_post + 10.0 * 0.5) / (n as f64 + 10.0);
        khats.push(k_hat as f32);
    }
    Tensor::from_vec(khats, b, log_w.device())
}

#[cfg(test)]
mod tests {
    //! Verification of noise parameterisations and sampling helpers for probabilistic models.

    use super::*;

    #[test]
    fn a_seeded_source_repeats_and_an_entropy_one_does_not() {
        let dev = Device::Cpu;
        let f = |src: &NoiseSource| {
            src.randn(&[4, 3], &dev, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        assert_eq!(f(&NoiseSource::seeded(9)), f(&NoiseSource::seeded(9)));
        assert_ne!(f(&NoiseSource::seeded(9)), f(&NoiseSource::seeded(10)));
        assert_ne!(f(&NoiseSource::entropy()), f(&NoiseSource::entropy()));
        assert!(NoiseSource::seeded(1).is_seeded());
        assert!(!NoiseSource::entropy().is_seeded());
    }

    /// Successive draws from ONE seeded source must differ, or every step of
    /// a chain would get identical noise — reproducible and useless.
    #[test]
    fn successive_draws_from_one_seeded_source_differ_but_the_sequence_repeats() {
        let dev = Device::Cpu;
        let seq = || {
            let s = NoiseSource::seeded(4);
            (0..3)
                .map(|_| s.randn(&[2, 2], &dev, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap())
                .collect::<Vec<_>>()
        };
        let a = seq();
        assert_ne!(a[0], a[1], "consecutive draws must not repeat");
        assert_ne!(a[1], a[2]);
        assert_eq!(a, seq(), "…but the whole sequence must be a function of the seed");
    }

    #[test]
    fn categorical_respects_the_weights_and_survives_degenerate_ones() {
        let s = NoiseSource::seeded(11);
        // All mass on component 2.
        assert!(s.categorical(&[0.0, 0.0, 1.0], 32).iter().all(|i| *i == 2));
        // Rough proportionality on a 3:1 split.
        let picks = NoiseSource::seeded(3).categorical(&[3.0, 1.0], 400);
        let zeros = picks.iter().filter(|i| **i == 0).count();
        assert!((250..=350).contains(&zeros), "3:1 split gave {zeros}/400 on component 0");
        // Degenerate inputs are answerable, not panics.
        assert_eq!(NoiseSource::seeded(1).categorical(&[], 3), vec![0, 0, 0]);
        assert_eq!(NoiseSource::seeded(1).categorical(&[0.0, 0.0], 2), vec![0, 0]);
    }

    /// The whole point: a seeded source makes a mixture draw reproducible,
    /// where the un-threaded path could not be — `Device::set_seed` is
    /// refused on CPU, so seeding it was previously a no-op.

    #[test]
    fn sobol_engine_range_determinism_and_chunking() {
        let dev = Device::Cpu;
        // raw uniform engine: [0, 1), deterministic in seed, seed-sensitive
        let a = sobol_engine(64, 3, 7, &dev).unwrap();
        let b = sobol_engine(64, 3, 7, &dev).unwrap();
        let c = sobol_engine(64, 3, 8, &dev).unwrap();
        assert_eq!(a.dims(), &[64, 3]);
        let d_ab = a.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let d_ac = a.sub(&c).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d_ab, 0.0, "same seed must reproduce");
        assert!(d_ac > 0.0, "different Owen key must rescramble");
        let v: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| (0.0..1.0).contains(x)));

        // past the 256-dim tables: chunked blocks stay in range and vary
        let wide = sobol_engine(4, 300, 1, &dev).unwrap();
        let w: Vec<f32> = wide.flatten_all().unwrap().to_vec1().unwrap();
        assert!(w.iter().all(|x| (0.0..1.0).contains(x)));
        let col256: Vec<f32> = wide.narrow(1, 256, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let col0: Vec<f32> = wide.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(col0, col256, "chunk blocks must be independently scrambled");
    }

    #[test]
    fn sobol_is_deterministic_and_unit_variance() {
        let dev = Device::Cpu;
        let a = sobol_randn_like(&[256, 4], &dev, site_seed("encoder_z", 0)).unwrap();
        let b = sobol_randn_like(&[256, 4], &dev, site_seed("encoder_z", 0)).unwrap();
        // deterministic in seed
        let da = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let db = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(da, db);
        // ~ standard normal
        let mean = a.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let var = a.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() - mean * mean;
        assert!(mean.abs() < 0.1, "QMC mean {mean}");
        assert!((var - 1.0).abs() < 0.2, "QMC var {var}");
        // different salt → different draw
        let c = sobol_randn_like(&[256, 4], &dev, site_seed("encoder_z", 7)).unwrap();
        let dc = c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(da, dc);
    }

    #[test]
    fn draw_bundle_shapes_and_determinism() {
        let dev = Device::Cpu;
        let cfg = SamplerConfig::default();
        let sampler = BayesianPosteriorSampler::new(cfg.clone());
        let shapes = HashMap::from([
            ("encoder_z".to_string(), vec![16]),       // antithetic + strata 2
            ("global_stick".to_string(), vec![29]),    // strata 1, no antithetic
        ]);
        let bundle = sampler.draw_bundle(&shapes, &dev, 0).unwrap();
        // leading S axis after stratification.
        assert_eq!(bundle.get("encoder_z").unwrap().dims(), &[cfg.n_samples, 16]);
        assert_eq!(bundle.get("global_stick").unwrap().dims(), &[cfg.n_samples, 29]);
        // deterministic
        let b2 = sampler.draw_bundle(&shapes, &dev, 0).unwrap();
        let v1 = bundle.get("encoder_z").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v2 = b2.get("encoder_z").unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn draw_single_shapes_determinism_and_salt() {
        let dev = Device::Cpu;
        let sampler = BayesianPosteriorSampler::new(SamplerConfig::default());
        let shapes = HashMap::from([
            ("encoder_z".to_string(), vec![3usize, 16]), // strata 2
            ("global_stick".to_string(), vec![3usize, 29]),
        ]);
        let b0 = sampler.draw_single(&shapes, &dev, 0).unwrap();
        // No S axis — per-sample shapes come back verbatim.
        assert_eq!(b0.get("encoder_z").unwrap().dims(), &[3, 16]);
        assert_eq!(b0.get("global_stick").unwrap().dims(), &[3, 29]);
        // Deterministic in salt; distinct across salts (the per-step draw).
        let b0b = sampler.draw_single(&shapes, &dev, 0).unwrap();
        let b1 = sampler.draw_single(&shapes, &dev, 1).unwrap();
        let v0: Vec<f32> = b0.get("encoder_z").unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let v0b: Vec<f32> = b0b.get("encoder_z").unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let v1: Vec<f32> = b1.get("encoder_z").unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v0, v0b, "same salt must reproduce");
        assert_ne!(v0, v1, "distinct salt must redraw");
        // Not degenerate (the lone-sample antithetic-cancellation trap).
        assert!(v0.iter().any(|x| x.abs() > 1e-3), "draw must be non-zero: {v0:?}");
    }

    #[test]
    fn snis_uniform_weights_recover_plain_monte_carlo() {
        // With uniform weights, SNIS must collapse to plain MC:
        // μ̃ = h̄, V̂ar = Var(h)/S, ESS_h ≈ S.
        let dev = Device::Cpu;
        let s = 512usize;
        let h = sobol_randn_like(&[s, 1], &dev, 42).unwrap().reshape((s, 1)).unwrap();
        let log_w = Tensor::zeros((s, 1), candle_core::DType::F32, &dev).unwrap();
        let (mu, mcse, ess) = snis_estimate(&log_w, &h, None).unwrap();
        let mu = mu.to_vec1::<f32>().unwrap()[0];
        let mcse = mcse.to_vec1::<f32>().unwrap()[0] as f64;
        let ess = ess.to_vec1::<f32>().unwrap()[0] as f64;
        let hbar = h.mean(0).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((mu - hbar).abs() < 1e-6, "SNIS mean must equal h̄");
        let sd = (h.broadcast_sub(&h.mean(0).unwrap().unsqueeze(0).unwrap()).unwrap()
            .sqr().unwrap().mean(0).unwrap().to_vec1::<f32>().unwrap()[0] as f64)
            .sqrt();
        let want_mcse = sd / (s as f64).sqrt();
        assert!((mcse - want_mcse).abs() / want_mcse < 1e-3, "MCSE {mcse} vs {want_mcse}");
        assert!((ess - s as f64).abs() / (s as f64) < 1e-3, "ESS_h {ess} vs S={s}");
        // R_eff = 0.5 halves the effective draws → doubles the variance
        let (_, mcse2, ess2) = snis_estimate(&log_w, &h, Some(0.5)).unwrap();
        let mcse2 = mcse2.to_vec1::<f32>().unwrap()[0] as f64;
        let ess2 = ess2.to_vec1::<f32>().unwrap()[0] as f64;
        assert!((mcse2 / mcse - 2f64.sqrt()).abs() < 1e-3);
        assert!((ess2 / ess - 0.5).abs() < 1e-3);
    }

    #[test]
    fn pareto_khat_flags_heavy_tails() {
        let dev = Device::Cpu;
        let s = 64usize;
        // column 0: near-uniform log-weights (benign); column 1: one dominant
        // outlier + spread tail (pathological importance distribution).
        let mut lw = vec![0f32; s * 2];
        for i in 0..s {
            lw[i * 2] = (i as f32) * 1e-3; // benign
            lw[i * 2 + 1] = (i as f32) * 0.6; // exponentially exploding weights
        }
        let log_w = Tensor::from_vec(lw, (s, 2), &dev).unwrap();
        let k = pareto_khat(&log_w).unwrap().to_vec1::<f32>().unwrap();
        assert!(k[0].is_finite() && k[1].is_finite(), "khat must be finite: {k:?}");
        assert!(k[1] > k[0], "heavy tail must score higher: {k:?}");
        assert!(k[0] < 0.7, "benign column must pass the PSIS threshold: {k:?}");
    }

    #[test]
    fn ess_in_range() {
        let dev = Device::Cpu;
        let log_w = Tensor::zeros((8, 5), candle_core::DType::F32, &dev).unwrap(); // uniform weights
        let ess = effective_sample_size(&log_w).unwrap();
        // uniform log_w ⇒ ESS == S for every batch element.
        let v = ess.to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|e| (e - 8.0).abs() < 1e-3), "ess {v:?}");
    }
}

#[cfg(test)]
mod qmc_init_tests {
    //! Verification of noise parameterisations and sampling helpers for probabilistic models.

    use super::*;
    use candle_core::Device;

    fn stats(t: &Tensor) -> (f64, f64, f64) {
        let (n, d) = t.dims2().unwrap();
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let col = |j: usize| -> Vec<f64> { (0..n).map(|i| v[i * d + j] as f64).collect() };
        let corr = |a: &[f64], b: &[f64]| -> f64 {
            let m = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
            let (ma, mb) = (m(a), m(b));
            let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
            for k in 0..a.len() {
                let (x, y) = (a[k] - ma, b[k] - mb);
                num += x * y;
                da += x * x;
                db += y * y;
            }
            num / (da.sqrt() * db.sqrt() + 1e-12)
        };
        let cols: Vec<Vec<f64>> = (0..d).map(col).collect();
        let mut mx = 0.0f64;
        for i in 0..d {
            for j in (i + 1)..d {
                mx = mx.max(corr(&cols[i], &cols[j]).abs());
            }
        }
        let nn = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / nn;
        let sd = (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / nn).sqrt();
        (mx, mean, sd)
    }

    /// PER-DIMENSION scrambling exists to kill a pairwise-projection artefact,
    /// and this is the measurement that says so.
    ///
    /// A `d`-dimensional Sobol NET is the right object for integration and the
    /// wrong one for a weight matrix: some column pair is badly aliased, and
    /// because the aliasing is structural rather than sampling noise it does
    /// NOT improve with `n` — measured `max|corr| = 0.66` at `[512, 512]`
    /// against `0.20` for iid. The MEAN correlation is meanwhile better than
    /// iid, which is exactly how it hides.
    ///
    /// Scrambling each column under its own key trades the joint property for
    /// independence and brings the worst pair back near iid, while keeping the
    /// marginals tighter than iid — which is the half worth having.
    #[test]
    fn per_dimension_scrambling_kills_the_pairwise_projection_artefact() {
        let dev = Device::Cpu;
        let shape = [512usize, 512];
        let (net_mx, _, _) = stats(&sobol_randn_like(&shape, &dev, 0xDEAD_BEEF).unwrap());
        let (pd_mx, pd_mean, pd_sd) =
            stats(&sobol_randn_like_per_dim(&shape, &dev, 0xDEAD_BEEF).unwrap());

        assert!(net_mx > 0.5, "the artefact this exists for is gone: net max|corr| {net_mx}");
        assert!(pd_mx < net_mx / 2.0, "per-dim {pd_mx} vs net {net_mx}");
        assert!(pd_mx < 0.35, "per-dim max|corr| {pd_mx} is still far above iid (~0.20)");
        // The marginals are the half we keep: tighter than an iid draw's.
        assert!(pd_mean.abs() < 1e-3, "mean {pd_mean}");
        assert!((pd_sd - 1.0).abs() < 5e-3, "sd {pd_sd}");
    }

    /// Same seed, same draw — the whole reason any of this is QMC.
    #[test]
    fn the_per_dimension_draw_is_deterministic_in_its_seed() {
        let dev = Device::Cpu;
        let a = sobol_randn_like_per_dim(&[32, 16], &dev, 7).unwrap();
        let b = sobol_randn_like_per_dim(&[32, 16], &dev, 7).unwrap();
        let c = sobol_randn_like_per_dim(&[32, 16], &dev, 8).unwrap();
        let v = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
        assert_eq!(v(&a), v(&b));
        assert_ne!(v(&a), v(&c), "a different seed must give a different draw");
    }
}
