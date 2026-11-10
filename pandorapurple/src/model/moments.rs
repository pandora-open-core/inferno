//! Inferno — Routing-distribution statistics and Gaussian moments for latent neural models.

use candle_core::{Result, Tensor, D};
fn clamp_min(t: &Tensor, v: f64) -> Result<Tensor> { t.maximum(&t.affine(0.0, v)?) }
#[derive(Clone, Debug)]
pub struct RoutingMoments {
    pub mean: Tensor,
    pub variance: Tensor,
    pub skew: Tensor,
    pub excess_kurtosis: Tensor,
    pub entropy: Tensor,
}

pub fn routing_moments(routing: &Tensor) -> Result<RoutingMoments> {
    let last = routing.rank() - 1;
    let k = routing.dim(last)?;
    let idx = Tensor::arange(0f32, k as f32, routing.device())?.to_dtype(routing.dtype())?; // [K]

    let p = clamp_min(routing, 1e-12)?;
    let p = p.broadcast_div(&p.sum_keepdim(last)?)?; // normalise → [.., K]

    let mean = p.broadcast_mul(&idx)?.sum(last)?; // [..]
    let centred = idx.broadcast_sub(&mean.unsqueeze(last)?)?; // [.., K]
    let c2 = centred.sqr()?;
    let c3 = c2.mul(&centred)?;
    let c4 = c2.sqr()?;

    let variance = p.broadcast_mul(&c2)?.sum(last)?;
    let std = clamp_min(&variance, 1e-12)?.sqrt()?;
    let skew = p.broadcast_mul(&c3)?.sum(last)?.div(&std.powf(3.0)?)?;
    let excess_kurtosis = p
        .broadcast_mul(&c4)?
        .sum(last)?
        .div(&clamp_min(&variance, 1e-12)?.sqr()?)?
        .affine(1.0, -3.0)?;
    let entropy = p.broadcast_mul(&p.log()?)?.sum(last)?.affine(-1.0, 0.0)?;

    // Bound the STANDARDISED moments: entmax15 drives routing toward one-hot,
    // so var → 0 and skew (÷std³) / excess-kurtosis (÷var²) diverge — the
    // "moments accumulating to infinity". Clamp to the range a genuinely
    // spread K-support distribution can occupy (|skew| ≤ K, kurtosis ∈
    // [−2, K²]) so a degenerate routing can't poison the running-mean
    // accumulation.
    let kf = k as f64;
    let skew = skew
        .maximum(&skew.affine(0.0, -kf)?)?
        .minimum(&skew.affine(0.0, kf)?)?;
    let excess_kurtosis = excess_kurtosis
        .maximum(&excess_kurtosis.affine(0.0, -2.0)?)?
        .minimum(&excess_kurtosis.affine(0.0, kf * kf)?)?;

    Ok(RoutingMoments { mean, variance, skew, excess_kurtosis, entropy })
}

// ---------------------------------------------------------------------------
// GaussianMoments — count-weighted running accumulation across refinement steps
// ---------------------------------------------------------------------------

/// First four standardised moments + entropy of a routing/simplex
/// distribution, accumulated across the VAE refinement steps as a
/// count-weighted running mean (`models/state.py::GaussianMoments`).
#[derive(Clone, Debug)]
pub struct GaussianMoments {
    pub mean: Tensor,
    pub variance: Tensor,
    pub skew: Tensor,
    pub excess_kurtosis: Tensor,
    pub entropy: Tensor,
    pub count: usize,
}

impl GaussianMoments {
    pub fn from_routing_moments(m: RoutingMoments) -> Self {
        Self {
            mean: m.mean,
            variance: m.variance,
            skew: m.skew,
            excess_kurtosis: m.excess_kurtosis,
            entropy: m.entropy,
            count: 1,
        }
    }

    /// One refinement step's moments from a `counterfactual_routing` simplex
    /// (`[B, K, E]`, moments over the expert axis).
    pub fn from_routing(routing: &Tensor) -> Result<Self> {
        Ok(Self::from_routing_moments(routing_moments(routing)?))
    }

    /// Count-weighted running-mean merge — accumulates one refinement step's
    /// moments into the running estimate. A fresh accumulator (`None`) adopts
    /// the first real step's moments verbatim.
    pub fn accumulate(running: Option<Self>, step: Option<Self>) -> Result<Option<Self>> {
        match (running, step) {
            (r, None) => Ok(r),
            (None, s) => Ok(s),
            (Some(r), Some(s)) => {
                let n = r.count + s.count;
                let w1 = s.count as f64 / n as f64;
                let w0 = 1.0 - w1;
                let merge = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
                    a.affine(w0, 0.0)?.add(&b.affine(w1, 0.0)?)
                };
                Ok(Some(Self {
                    mean: merge(&r.mean, &s.mean)?,
                    variance: merge(&r.variance, &s.variance)?,
                    skew: merge(&r.skew, &s.skew)?,
                    excess_kurtosis: merge(&r.excess_kurtosis, &s.excess_kurtosis)?,
                    entropy: merge(&r.entropy, &s.entropy)?,
                    count: n,
                }))
            }
        }
    }
}

