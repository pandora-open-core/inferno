//! Inferno — Iterative solver selection and escalation for structured linear systems.

/// How to choose the eigenvalue ladder placed on the Ritz basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectralLadder {
    /// The Ritz values themselves — the REFERENCE, and the default.
    ///
    /// A first-class variant rather than a fallback, because without it
    /// "did the shape help?" is unanswerable.
    Ritz,
    /// Fit a Kumaraswamy quantile to the Ritz values in log λ.
    KumaraswamyFitted,
    /// A supplied shape, e.g. from a trained bridge.
    Kumaraswamy { q_a: f64, q_b: f64 },
}

impl Default for SpectralLadder {
    fn default() -> Self {
        Self::Ritz
    }
}

/// Kumaraswamy inverse CDF, `v = (1 − (1−u)^{1/b})^{1/a}`, clamped away from
/// the boundaries so `ln` and the fractional powers stay finite.
pub fn kuma_icdf(u: f64, a: f64, b: f64, eps: f64) -> f64 {
    let om_u = (1.0 - u).clamp(eps, 1.0);
    let w = om_u.powf(1.0 / b);
    let s = (1.0 - w).clamp(eps, 1.0);
    s.powf(1.0 / a)
}

/// `μ̃_j = λmin·(λmax/λmin)^{F⁻¹(u_j; a, b)}` on a midpoint grid — a GEOMETRIC
/// interpolation between the bounds, monotone increasing in `j`.
pub fn kuma_ladder(lambda_min: f64, lambda_max: f64, k: usize, a: f64, b: f64) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    let (lo, hi) = (lambda_min.max(f64::MIN_POSITIVE).ln(), lambda_max.max(f64::MIN_POSITIVE).ln());
    (0..k)
        .map(|j| {
            let u = (j as f64 + 0.5) / k as f64;
            let v = kuma_icdf(u, a, b, 1e-12);
            (lo + v * (hi - lo)).exp()
        })
        .collect()
}

/// Least-squares fit of `(q_a, q_b)` to `mu` (ASCENDING) in log λ.
///
/// Exploits a closed form rather than a 2-D descent. With
/// `v_j* = (ln μ_j − ln λmin)/(ln λmax − ln λmin)` the target on `[0,1]`, and
///
/// ```text
/// ln v_j = (1/a)·ln(1 − (1−u_j)^{1/b}) = (1/a)·L_j(b)
/// ```
///
/// `a` is available in closed form for any fixed `b`:
/// `1/a = Σ L_j ln v_j* / Σ L_j²`. So the fit is a ONE-dimensional search over
/// `b` with `a` solved exactly at each point — no derivatives, no Jacobian, and
/// no chance of the 2-D descent stalling in the corner where `a → 0` (which is
/// the "rogue alpha" regime `model/dirichlet.rs` records as producing infinite
/// gradients).
///
/// Returns `None` when the Ritz values carry no usable shape — a degenerate
/// (flat, or single-valued) spectrum — so the caller keeps [`SpectralLadder::Ritz`]
/// rather than fitting noise.
pub fn fit_kuma(mu: &[f64], lambda_min: f64, lambda_max: f64) -> Option<(f64, f64)> {
    let k = mu.len();
    if k < 3 {
        return None;
    }
    if !(lambda_min > 0.0 && lambda_max > lambda_min) {
        return None;
    }
    let span = lambda_max.ln() - lambda_min.ln();
    if span < 1e-9 {
        return None; // flat: nothing to shape
    }
    if mu[k - 1].ln() - mu[0].ln() < 1e-9 {
        return None; // the Ritz values themselves carry no shape
    }

    // Targets on (0,1). The bounds come from the CALLER — the probe's
    // safeguarded `[λmin, λmax]`, which bracket the Ritz values with margin
    // (`ritz_min·0.25`, `ritz_max·1.1`). Deriving them from `mu`'s own ends
    // instead would force `v* = 0` and `v* = 1` exactly at the endpoints; those
    // then clamp, and `ln(1e-9) = −20.7` is an outlier that dominates the least
    // squares and drags the fit to the `a`-clamp. That was the first version of
    // this function, and `the_fit_recovers_the_shape_that_generated_the_ladder`
    // is what caught it.
    let eps = 1e-12;
    let targets: Vec<f64> = mu
        .iter()
        .map(|m| (((m.max(f64::MIN_POSITIVE).ln() - lambda_min.ln()) / span).clamp(eps, 1.0 - eps)))
        .collect();
    let ln_v: Vec<f64> = targets.iter().map(|v| v.ln()).collect();
    let us: Vec<f64> = (0..k).map(|j| (j as f64 + 0.5) / k as f64).collect();

    // For a given b: the exact best `a`, and the residual it achieves.
    let eval = |b: f64| -> Option<(f64, f64)> {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut ls = Vec::with_capacity(k);
        for &u in &us {
            let om_u = (1.0 - u).clamp(1e-12, 1.0);
            let w = om_u.powf(1.0 / b);
            let s = (1.0 - w).clamp(1e-12, 1.0);
            let l = s.ln();
            ls.push(l);
        }
        for (l, lv) in ls.iter().zip(&ln_v) {
            num += l * lv;
            den += l * l;
        }
        if !(den > 0.0) || !num.is_finite() {
            return None;
        }
        let inv_a = num / den;
        if !(inv_a.is_finite() && inv_a > 0.0) {
            return None;
        }
        let a = 1.0 / inv_a;
        let sse: f64 = ls.iter().zip(&ln_v).map(|(l, lv)| (inv_a * l - lv).powi(2)).sum();
        Some((a, sse))
    };

    // Coarse log-spaced sweep over b, then a bisection refinement. 40 + 20
    // evaluations of three transcendentals each — nil against a single matvec.
    let mut best: Option<(f64, f64, f64)> = None; // (b, a, sse)
    for i in 0..40 {
        let b = 10f64.powf(-1.0 + 3.0 * (i as f64) / 39.0); // 0.1 … 100
        if let Some((a, sse)) = eval(b) {
            if best.is_none_or(|(_, _, s)| sse < s) {
                best = Some((b, a, sse));
            }
        }
    }
    let (mut b0, _, _) = best?;
    let mut step = b0 * 0.5;
    for _ in 0..20 {
        let mut improved = false;
        for cand in [b0 - step, b0 + step] {
            if cand <= 0.0 {
                continue;
            }
            if let (Some((_, s_new)), Some((_, s_cur))) = (eval(cand), eval(b0)) {
                if s_new < s_cur {
                    b0 = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    let (a, _) = eval(b0)?;
    // The floor `model/dirichlet.rs:869-875` records: `a` below ~1 makes the
    // `1/a` exponent enormous and the shape explodes. Clamped rather than
    // refused, because the fit is a heuristic and a clamped shape is still a
    // valid monotone ladder.
    Some((a.clamp(1.035, 1e3), b0.clamp(1e-3, 1e3)))
}

/// Resolve a ladder against measured Ritz values (ASCENDING).
///
/// Returns `mu` unchanged for [`SpectralLadder::Ritz`], and otherwise a ladder
/// spanning `[λmin, λmax]` — the PROBE's safeguarded bounds, which must be the
/// same ones the fit saw.
pub fn resolve(
    ladder: SpectralLadder,
    mu: &[f64],
    lambda_min: f64,
    lambda_max: f64,
) -> Vec<f64> {
    let k = mu.len();
    if k == 0 {
        return Vec::new();
    }
    let (a, b) = match ladder {
        SpectralLadder::Ritz => return mu.to_vec(),
        SpectralLadder::Kumaraswamy { q_a, q_b } => (q_a, q_b),
        SpectralLadder::KumaraswamyFitted => match fit_kuma(mu, lambda_min, lambda_max) {
            Some(ab) => ab,
            None => return mu.to_vec(), // degenerate: no shape to fit
        },
    };
    // Same bounds the fit used, or the two disagree about what `v = 0` means.
    kuma_ladder(lambda_min, lambda_max, k, a, b)
}

#[cfg(test)]
mod tests {
    //! Verification of iterative solver selection and escalation for structured linear systems.

    use super::*;

    /// A preconditioner ladder must be monotone, or the pairing with the
    /// ascending Ritz basis is meaningless.
    #[test]
    fn the_ladder_is_monotone_and_spans_the_bounds() {
        for (a, b) in [(1.05, 0.5), (2.0, 2.0), (1.5, 8.0), (30.0, 0.2)] {
            let l = kuma_ladder(1e-3, 10.0, 16, a, b);
            for w in l.windows(2) {
                assert!(w[1] >= w[0], "not monotone at a={a} b={b}: {l:?}");
            }
            assert!(l[0] >= 1e-3 && *l.last().unwrap() <= 10.0, "outside the bounds: {l:?}");
        }
    }

    /// The midpoint grid, tested where it is unambiguous.
    ///
    /// At `a = b = 1` the Kumaraswamy icdf is the identity, so the ladder's
    /// quantiles ARE `(j+½)/k` and the ends must sit strictly inside. Testing
    /// this through a closeness proxy at arbitrary `(a, b)` does not work — a
    /// steep shape (large `a`, small `b`) legitimately puts the top quantile
    /// within `1e-8` of `λmax`, which is a shape effect and not an endpoint
    /// grid.
    #[test]
    fn the_grid_is_midpoint_not_endpoint() {
        let k = 8;
        for j in 0..k {
            let u = (j as f64 + 0.5) / k as f64;
            assert!((kuma_icdf(u, 1.0, 1.0, 1e-12) - u).abs() < 1e-12, "identity shape moved");
        }
        // An endpoint grid would make these exactly the bounds.
        let l = kuma_ladder(1e-3, 10.0, k, 1.0, 1.0);
        assert!(l[0] > 1e-3 && *l.last().unwrap() < 10.0, "ends are pinned: {l:?}");
        let span = 10f64.ln() - 1e-3f64.ln();
        assert!(
            ((l[0].ln() - 1e-3f64.ln()) / span - 0.5 / k as f64).abs() < 1e-9,
            "first quantile is not the midpoint"
        );
    }

    /// The fit must recover parameters it generated. This is the only test that
    /// says the closed-form `a` and the 1-D search over `b` agree with the
    /// forward map.
    #[test]
    fn the_fit_recovers_the_shape_that_generated_the_ladder() {
        let (lo, hi) = (1e-2, 20.0);
        for (a, b) in [(1.5, 0.8), (3.0, 2.5), (2.0, 6.0)] {
            let truth = kuma_ladder(lo, hi, 24, a, b);
            let (fa, fb) = fit_kuma(&truth, lo, hi).expect("a generated ladder must be fittable");
            let back = kuma_ladder(lo, hi, truth.len(), fa, fb);
            // Compare the LADDERS, not the parameters: (a, b) is only
            // identifiable up to the shape it produces, and the endpoints are
            // re-derived from the data so an exact parameter match is not the
            // claim.
            let rel = truth
                .iter()
                .zip(&back)
                .map(|(t, k)| ((t.ln() - k.ln()) / t.ln().abs().max(1.0)).abs())
                .fold(0.0, f64::max);
            assert!(rel < 0.05, "fit(a={a},b={b}) -> ({fa:.3},{fb:.3}) reproduced the ladder to {rel:.3e}");
        }
    }

    /// A flat or near-flat spectrum has no shape to fit, and fitting noise
    /// would be worse than not fitting. `resolve` must fall back to the Ritz
    /// values rather than inventing a ladder.
    #[test]
    fn a_degenerate_spectrum_falls_back_to_the_ritz_values() {
        let flat = vec![1.0, 1.0, 1.0, 1.0];
        assert!(fit_kuma(&flat, 0.5, 2.0).is_none());
        assert_eq!(resolve(SpectralLadder::KumaraswamyFitted, &flat, 0.5, 2.0), flat);

        let tiny = vec![1.0, 2.0];
        assert!(fit_kuma(&tiny, 0.5, 4.0).is_none(), "two points cannot determine two parameters");
        assert_eq!(resolve(SpectralLadder::KumaraswamyFitted, &tiny, 0.5, 4.0), tiny);
    }

    /// `Ritz` must be exactly a no-op — it is the reference the shaped ladders
    /// are measured against, so any drift here invalidates every comparison.
    #[test]
    fn the_ritz_ladder_is_bitwise_identity() {
        let mu = vec![1e-3, 4e-3, 0.2, 1.0, 1.05, 9.7];
        let out = resolve(SpectralLadder::Ritz, &mu, 1e-4, 20.0);
        for (a, b) in mu.iter().zip(&out) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// The `a ≥ 1.035` floor is the "rogue alpha" guard `model/dirichlet.rs`
    /// records: below ~1 the `1/a` exponent becomes enormous. In the HDP that
    /// produced infinite gradients; here it would produce a ladder spanning
    /// orders of magnitude in one step, i.e. a NaN `E` inside the
    /// preconditioner with no error message.
    #[test]
    fn the_fit_clamps_a_away_from_the_rogue_alpha_regime() {
        // A ladder that hugs λmin then jumps — the shape that drives a → 0.
        let mut mu: Vec<f64> = (0..20).map(|j| 1e-3 * (1.0 + 1e-4 * j as f64)).collect();
        mu.push(50.0);
        if let Some((a, b)) = fit_kuma(&mu, 1e-4, 100.0) {
            assert!(a >= 1.035, "a = {a} is inside the rogue-alpha regime");
            assert!(a.is_finite() && b.is_finite() && b > 0.0);
            let l = kuma_ladder(1e-4, 100.0, mu.len(), a, b);
            assert!(l.iter().all(|v| v.is_finite() && *v > 0.0), "ladder went non-finite: {l:?}");
        }
    }
}
