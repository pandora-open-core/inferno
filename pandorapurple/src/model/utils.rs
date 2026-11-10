//! Inferno — Shared numerical and tensor utilities for library components.

use candle_core::{Tensor, D, Result, bail};                 // dropped `Result` to avoid clash with anyhow's
use ndarray::{ArrayView1};    // ArrayView1 was missing
use crate::gp::{KernelHyper, RawHyper};

/// Numerically-stable softplus on a Candle tensor:
/// `log(1 + exp(x)) = max(0, x) + log(1 + exp(-|x|))`.
///
/// `max(0, x)` is written as `½(x + |x|)` — algebraically identical, but it
/// FIXES the gradient at exactly `x = 0`.
///
/// With the literal `x.maximum(&zeros)` form the derivative at the tie is
/// **0 instead of ½**: candle's `maximum` gives the tie to neither branch
/// and `sign(0) = 0` kills the `|x|` term, so the two contributions cancel.
/// That is not academic — raw parameters here are routinely initialised at
/// exactly `0.0` and constrained through `softplus`/[`positive`]/
/// [`greater_than`], so such a parameter would sit at a spurious stationary
/// point and never receive its first update.
///
/// In the `½(x + |x|)` form the explicit `½x` supplies the missing half and
/// the `sign(0) = 0` symmetry cancels the rest, giving exactly `σ(0) = ½`;
/// away from zero both forms agree to the last bit. Regression-tested in
/// `softplus_gradient_is_sigmoid_everywhere_including_zero`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let neg_abs = abs_x.neg()?;
    let log_part = (neg_abs.exp()? + 1.0)?.log()?;
    // max(0, x) ≡ ½(x + |x|), but differentiable at the tie.
    let max_part = x.add(&abs_x)?.affine(0.5, 0.0)?;
    Ok(max_part.add(&log_part)?)
}

pub fn tpow(x: &Tensor, e: &Tensor) -> Result<Tensor> {
    x.log()?.broadcast_mul(e)?.exp()
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(x) // Now the types match perfectly
}

pub fn interval(x: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    let s = sigmoid(x)?;
    s.affine(hi - lo, lo)
}

/// `R → (0, 1)` with ALGEBRAIC tails — `½(1 + x/√(1+x²))`.
///
/// The non-saturating half of Python's `nonsaturating_interval`
/// (`models/dirichlet.py:143`). A logistic squash saturates, and a saturated
/// squash passes no gradient, so a parameter pinned against its box has
/// stopped learning; this one decays as `1/x²` and stays live at the limit.
pub fn algebraic_squash(x: &Tensor) -> Result<Tensor> {
    let denom = x.sqr()?.affine(1.0, 1.0)?.sqrt()?; // √(1 + x²)
    x.div(&denom)?.affine(0.5, 0.5)
}

/// Inverse of [`algebraic_squash`]: `u = 2p − 1 ⇒ x = u/√(1 − u²)`.
///
/// The clamp is the guard gpytorch's own `inv_sigmoid` needs — a value sitting
/// exactly ON a bound has no finite preimage. `±(1 − 1e-7)` puts it at
/// `|raw| ≈ 2.2e3`: extreme, and (the point of the exercise) differentiable.
pub fn algebraic_squash_inv(p: f64) -> f64 {
    let u = (2.0 * p - 1.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
    u / (1.0 - u * u).max(1e-14).sqrt()
}

/// `interval` with algebraic rather than logistic tails — the transform half
/// of Python's `nonsaturating_interval(lo, hi)`.
///
/// NOTE this is NOT interchangeable with [`interval`]. Both agree at `raw = 0`
/// (the box midpoint) and diverge everywhere else: at `raw = 1` the logistic
/// form gives `0.731` of the box and this one `0.854`. A raw value trained
/// under one and decoded under the other lands somewhere its author never
/// chose.
pub fn interval_ns(x: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    algebraic_squash(x)?.affine(hi - lo, lo)
}

/// The raw value that [`interval_ns`] decodes to exactly `target`.
pub fn interval_ns_inv(target: f64, lo: f64, hi: f64) -> f64 {
    algebraic_squash_inv((target - lo) / (hi - lo))
}

pub fn greater_than(x: &Tensor, lo: f64) -> Result<Tensor> {
    let sp = softplus(x)?;
    sp.affine(1.0, lo)
}

pub fn positive(x: &Tensor) -> Result<Tensor> {
    softplus(x)
}


pub fn softsign(x: &Tensor) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let denom = abs_x.affine(1.0, 1.0)?; // |x| + 1
    x.div(&denom)
}

// `smooth_clamp` / `affine_unit` moved to the canonical `crate::numerics`
// module (the `deepsecrets.numerics` port); re-exported here so every
// existing `crate::model::{smooth_clamp, affine_unit}` caller keeps working.
pub use crate::numerics::{affine_unit, smooth_clamp};

/// Standard-normal CDF `Φ(η) = ½(1 + erf(η/√2))` — maps the sampler's
/// Gaussian bundle sites onto the Uniform(0,1) draws the Kumaraswamy
/// inverse-CDF sticks need. Since the bundle is `Φ⁻¹(sobol_u)`, this
/// recovers the low-discrepancy uniform exactly.
pub fn normal_cdf(eta: &Tensor) -> Result<Tensor> {
    eta.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0)?
        .erf()?
        .affine(0.5, 0.5)
}

/// `logsumexp` over the last dim, keepdim — stable log-softmax denominator.
pub fn logsumexp_keepdim(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?;
    let lse = x.broadcast_sub(&m)?.exp()?.sum_keepdim(D::Minus1)?.log()?;
    lse.broadcast_add(&m)
}


/// Forward substitution for a batched lower-triangular solve `L · X = B`.
/// `L`: `[N, M, M]`, `B`: `[N, M, K]`  →  `X`: `[N, M, K]`.
///
/// Naive O(M²) — acceptable for serving-time M (a few hundred to a few
/// thousand). If you ever push M past ~10k, replace with a blocked solve
/// or move the chain off Candle entirely.
pub fn forward_substitute(l: &Tensor, b: &Tensor) -> Result<Tensor> {
    let m = l.dim(D::Minus1)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(m);
    for i in 0..m {
        let mut accum = b.narrow(1, i, 1)?; // [N, 1, K]
        for j in 0..i {
            let lij = l.narrow(1, i, 1)?.narrow(D::Minus1, j, 1)?; // [N, 1, 1]
            let xj = &rows[j]; // [N, 1, K]
            accum = accum.sub(&lij.broadcast_mul(xj)?)?;
        }
        let lii = l.narrow(1, i, 1)?.narrow(D::Minus1, i, 1)?;
        rows.push(accum.broadcast_div(&lii)?);
    }
    Ok(Tensor::cat(&rows, 1)?)
}

// ---------------------------------------------------------------------------
// θ → KernelHyper hook (SWAG / MCMC consumer)
// ---------------------------------------------------------------------------

/// Overwrite `hp` from a flat θ vector. Layout must match whatever the
/// SWAG approximator / MCMC oracle was fit on — keep in sync with
/// `autograd::score`'s θ-block map.
///
/// Current layout (86 entries):
///   [ 0..  6) — six scalar raw hyperparameters
///   [ 6.. 14) — raw_outputscale       (8)
///   [14.. 22) — raw_latent_amplitude  (8)
///   [22.. 54) — raw_inv_bandwidth     (32)
///   [54.. 86) — raw_nkn_weights       (4×8 row-major)
/// pub fn apply_theta_to_hyper(theta: &ArrayView1<f32>, hp: &mut KernelHyper) -> Result<()> {
    
pub fn apply_theta_to_hyper(theta: &ArrayView1<f32>, hp: &mut KernelHyper) -> Result<()> {
    if theta.len() < 86 {
        bail!("θ too short: {} < 86", theta.len());
    }
    let mut raw = RawHyper {
        raw_scale_12:    theta[0],
        raw_scale_32:    theta[1],
        raw_scale_52:    theta[2],
        raw_rq_alpha:    theta[3],
        raw_poly_offset: theta[4],
        raw_linear_scale: theta[5],
        raw_outputscale:     [0.0; 8],
        raw_latent_amplitude: [0.0; 8],
        raw_inv_bandwidth:   [0.0; 32],
        raw_nkn_weights:    [[0.0; 8]; 4],
    };
    for i in 0..8 {
        raw.raw_outputscale[i]     = theta[6 + i];
        raw.raw_latent_amplitude[i] = theta[14 + i];
    }
    for d in 0..32 {
        raw.raw_inv_bandwidth[d] = theta[22 + d];
    }
    for n in 0..4 {
        for p in 0..8 {
            raw.raw_nkn_weights[n][p] = theta[54 + n * 8 + p];
        }
    }
    *hp = KernelHyper::from_raw(&raw);
    Ok(())
}
#[cfg(test)]
mod tests {
    //! Verification of shared numerical and tensor utilities for library components.

    use candle_core::{Device, Tensor, Var};

    /// `d/dx softplus = σ(x)` must hold EVERYWHERE — most importantly at
    /// exactly `x = 0`, where raw parameters are initialised.
    ///
    /// Regression test for a real bug: the previous
    /// `x.maximum(&zeros) + log1p(exp(-|x|))` form returned a gradient of
    /// 0 at the tie (both branch selectors have zero subgradient there and
    /// cancelled), so every zero-initialised softplus-constrained
    /// parameter was stuck at a spurious stationary point.
    #[test]
    fn softplus_gradient_is_sigmoid_everywhere_including_zero() {
        let dev = Device::Cpu;
        for x0 in [0.0f32, -0.0, 1e-6, -1e-6, 0.5, -0.5, 3.0, -3.0, 20.0, -20.0] {
            let v = Var::from_tensor(&Tensor::new(&[x0], &dev).unwrap()).unwrap();
            let y = super::softplus(v.as_tensor()).unwrap().sum_all().unwrap();
            let g = y.backward().unwrap();
            let got = g.get(&v).unwrap().to_vec1::<f32>().unwrap()[0];
            let want = 1.0 / (1.0 + (-x0).exp());
            assert!(
                (got - want).abs() < 1e-5,
                "d softplus/dx at {x0}: got {got}, want σ({x0}) = {want}"
            );
        }
    }

    /// The value must be unchanged by the reformulation, and stay finite
    /// in both tails.
    #[test]
    fn softplus_value_matches_log1p_exp_and_stays_finite() {
        let dev = Device::Cpu;
        let xs = [0.0f32, 1e-6, 0.5, -0.5, 3.0, -3.0, 40.0, -40.0, 88.0, -88.0];
        let x = Tensor::new(&xs[..], &dev).unwrap();
        let got: Vec<f32> = super::softplus(&x).unwrap().to_vec1().unwrap();
        for (i, &xi) in xs.iter().enumerate() {
            // stable reference: max(x,0) + ln(1+e^-|x|)
            let want = xi.max(0.0) + (1.0 + (-xi.abs()).exp()).ln();
            assert!(got[i].is_finite(), "softplus({xi}) not finite");
            assert!(
                (got[i] - want).abs() < 1e-4,
                "softplus({xi}): {} vs {want}",
                got[i]
            );
        }
        // softplus(0) = ln 2
        assert!((got[0] - std::f32::consts::LN_2).abs() < 1e-6);
    }
}

#[cfg(test)]
mod squash_tests {
    //! Verification of shared numerical and tensor utilities for library components.

    use super::*;
    use candle_core::{DType, Device};

    /// `interval_ns_inv` is the exact inverse of `interval_ns`, which is what
    /// lets a parameter be seeded at a chosen CONSTRAINED value.
    #[test]
    fn the_algebraic_box_round_trips() {
        let dev = Device::Cpu;
        for (lo, hi) in [(0.05, 4.0), (-5.0, 5.0), (0.01, 20.0)] {
            for target in [lo + 0.1 * (hi - lo), 0.5 * (lo + hi), hi - 0.1 * (hi - lo)] {
                let raw = interval_ns_inv(target, lo, hi);
                let t = Tensor::new(&[raw as f32][..], &dev).unwrap();
                let got = interval_ns(&t, lo, hi).unwrap().to_vec1::<f32>().unwrap()[0];
                assert!(
                    (got as f64 - target).abs() < 1e-4,
                    "box ({lo}, {hi}): {target} → raw {raw} → {got}",
                );
            }
        }
    }

    /// The two boxes are NOT interchangeable, and the gap is large enough to
    /// matter — a raw value trained under one and decoded under the other lands
    /// somewhere its author never chose.
    ///
    /// They agree at raw 0 (both give the midpoint) and diverge everywhere
    /// else, which is exactly why the disagreement is easy to miss: a
    /// freshly-initialised model looks identical under either rule.
    #[test]
    fn the_logistic_and_algebraic_boxes_agree_only_at_the_midpoint() {
        let dev = Device::Cpu;
        let (lo, hi) = (0.05, 4.0);
        let zero = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let a = interval(&zero, lo, hi).unwrap().to_vec1::<f32>().unwrap()[0];
        let b = interval_ns(&zero, lo, hi).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((a - b).abs() < 1e-6, "both must give the midpoint at raw 0");

        let one = Tensor::new(&[1.0f32][..], &dev).unwrap();
        let a = interval(&one, lo, hi).unwrap().to_vec1::<f32>().unwrap()[0];
        let b = interval_ns(&one, lo, hi).unwrap().to_vec1::<f32>().unwrap()[0];
        // logistic σ(1) = 0.731 of the box; algebraic ½(1 + 1/√2) = 0.854.
        assert!((a - 2.938).abs() < 1e-2, "logistic gave {a}");
        assert!((b - 3.424).abs() < 1e-2, "algebraic gave {b}");
        assert!((a - b).abs() > 0.4, "the two boxes must not be interchangeable");
    }

    /// The algebraic tail does NOT saturate: it keeps passing gradient far out,
    /// where the logistic one has underflowed to nothing. A saturated squash
    /// means a parameter pinned against its box has stopped learning.
    #[test]
    fn the_algebraic_tail_still_passes_gradient_where_the_logistic_one_has_died() {
        let dev = Device::Cpu;
        let h = 1e-3f64;
        let at = |f: &dyn Fn(&Tensor) -> Tensor, x: f64| -> f64 {
            let up = f(&Tensor::new(&[(x + h) as f32][..], &dev).unwrap())
                .to_vec1::<f32>().unwrap()[0] as f64;
            let dn = f(&Tensor::new(&[(x - h) as f32][..], &dev).unwrap())
                .to_vec1::<f32>().unwrap()[0] as f64;
            (up - dn) / (2.0 * h)
        };
        let log = |t: &Tensor| sigmoid(t).unwrap();
        let alg = |t: &Tensor| algebraic_squash(t).unwrap();
        let (dl, da) = (at(&log, 20.0).abs(), at(&alg, 20.0).abs());
        assert!(dl < 1e-6, "logistic should be dead at 20, got {dl}");
        assert!(da > 1e-5, "algebraic must stay live at 20, got {da}");
    }
}
