//! Inferno — Contour quadrature and elliptic-integral tools for matrix functions.

pub mod elliptic;

/// `n`-point Gauss–Hermite nodes/weights in the PHYSICISTS' convention
/// (weight `e^{−x²}`, so `∫ f(x) e^{−x²} dx ≈ Σ wᵢ f(xᵢ)` and `Σ wᵢ = √π`) —
/// the standard `gauher` construction by Newton iteration on the recurrence,
/// in `f64`.
///
/// Nodes come back in DESCENDING order and symmetric about zero
/// (`x[i] = −x[n−1−i]`); an odd `n` has an exact zero node in the middle.
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let m = (n + 1) / 2;
    let mut z = 0.0f64;
    for i in 0..m {
        z = match i {
            0 => {
                let nn = n as f64;
                (2.0 * nn + 1.0).sqrt() - 1.85575 * (2.0 * nn + 1.0).powf(-1.0 / 6.0)
            }
            1 => z - 1.14 * (n as f64).powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0f64;
        for _ in 0..100 {
            let mut p1 = std::f64::consts::PI.powf(-0.25);
            let mut p2 = 0.0f64;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                p1 = z * (2.0 / j as f64).sqrt() * p2 - ((j as f64 - 1.0) / j as f64).sqrt() * p3;
            }
            pp = (2.0 * n as f64).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 3e-14 {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// `(omega [n], amplitude [n])` for the RBF spectral density by Gauss–Hermite
/// QUADRATURE — `FourierTimeEmbedding.make_basis(basis="quadrature")`.
///
/// Bochner gives `k(τ) = ∫ p(ω) cos(2πωτ) dω` with `p = N(0, σ²)` for the RBF.
/// A random Fourier basis estimates that integral by Monte Carlo, which at
/// n=64 is a *noisy* kernel — the very thing that makes "how similar are two
/// times" an accident of the seed. Quadrature evaluates the same integral
/// exactly: with `ω = √2·σ·x` and weights `a = w/√π` (which sum to 1),
///
/// ```text
///   k̂(τ) = Σ aᵢ cos(2π ωᵢ τ)
/// ```
///
/// Measured against the closed-form RBF at σ=1.5, a `2n = 128`-node rule is
/// accurate to 4.4e-12 — the kernel stops being approximate at all.
///
/// Only the POSITIVE nodes are kept, with weights doubled: `cos` is even, so a
/// `±ω` pair contributes identically and the mirror node is pure redundancy. A
/// `2n`-node rule therefore yields exactly `n` frequencies, leaving the feature
/// count unchanged.
///
/// The amplitudes span a very wide range, and that is the rule telling the
/// truth about a 1-D kernel: only ~25 of 64 nodes carry non-negligible mass and
/// the rest are numerically zero because a 1-D RBF does not need them. They are
/// inert rather than harmful. No threshold is applied — an arbitrary cutoff
/// would trade an exact statement for a tunable one.
pub fn rbf_quadrature_basis(n_features: usize, sigma: f64) -> (Vec<f64>, Vec<f64>) {
    // Even `2n` ⇒ no zero node, so exactly `n` are positive.
    let (x, w) = gauss_hermite(2 * n_features);
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let mut om = Vec::with_capacity(n_features);
    let mut amp = Vec::with_capacity(n_features);
    for (xi, wi) in x.iter().zip(w.iter()) {
        if *xi > 0.0 {
            om.push(std::f64::consts::SQRT_2 * sigma * xi);
            amp.push((2.0 * wi * inv_sqrt_pi).sqrt()); // ×2 for the mirror node
        }
    }
    (om, amp)
}

/// Gauss–Legendre nodes/weights on `[-1, 1]` by Newton iteration on `P_n`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0f64; n];
    let mut weights = vec![0f64; n];
    for i in 0..n {
        // Chebyshev initial guess
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            // Evaluate P_n(x) and P'_n(x) by the recurrence.
            let (mut p0, mut p1) = (1f64, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        // recompute P'_n at the root for the weight
        let (mut p0, mut p1) = (1f64, x);
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        nodes[i] = -x; // ascending order (cos guess descends)
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// One term of `K^{-1/2} b ≈ Σ_j weight_j · (K + shift_j·I)^{-1} b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContourTerm {
    pub shift: f64,
    pub weight: f64,
}

/// Which rule produces the contour `(shift, weight)` pairs.
///
/// An enum rather than a trait: there are exactly two implementations, both
/// satisfying the identical contract, selected by a `Copy` option field —
/// no downstream extension point is wanted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContourRule {
    /// Gauss–Legendre on the Möbius-mapped integral. Measured node counts for
    /// 1e-4 relative error: Q≈8 at κ=1e2, ≈12 at κ=1e4, ≈24 at κ=1e6, ≈48 at
    /// κ=1e8 — roughly doubling per 100× in κ (see the node-count test).
    #[default]
    GaussLegendre,
    /// Hale–Higham–Trefethen elliptic-function quadrature:
    /// `O(exp(−π²Q/(log κ + 3)))`, so a FIXED small budget (Q≈15) covers even
    /// extreme condition numbers. Wins over GL only at fixed small Q and
    /// large κ.
    Elliptic,
}

/// `(shift, weight)` pairs discretising the inverse-square-root integral over
/// a spectrum contained in `[lambda_min, lambda_max]`.
///
/// Requires `0 < lambda_min <= lambda_max` and `q >= 1`. All returned shifts
/// and weights are strictly positive under either rule.
pub fn contour_terms(
    rule: ContourRule,
    lambda_min: f64,
    lambda_max: f64,
    q: usize,
) -> Vec<ContourTerm> {
    assert!(
        lambda_min > 0.0 && lambda_max >= lambda_min,
        "contour_terms: need 0 < lambda_min <= lambda_max, got [{lambda_min}, {lambda_max}]"
    );
    assert!(q >= 1, "contour_terms: q must be >= 1");
    match rule {
        ContourRule::GaussLegendre => contour_terms_gauss_legendre(lambda_min, lambda_max, q),
        ContourRule::Elliptic => elliptic::contour_terms_elliptic(lambda_min, lambda_max, q),
    }
}

/// Gauss–Legendre discretisation of `(2/π) ∫₀^∞ (K + t²I)^{-1} dt`.
///
/// Substitution: the Möbius map `t = σ(1+u)/(1−u)`, `u ∈ (−1, 1)`, with
/// `σ = (λmin·λmax)^{1/4}`. The scalar integrand `(λ + t²)^{-1}` has poles at
/// `t = ±i√λ`; after the map, GL's convergence rate is set by how close the
/// mapped poles sit to `[−1, 1]`, which is worst at the spectrum's edges.
/// `σ` at the geometric midpoint of `[√λmin, √λmax]` places the two edge
/// cases equidistant in the mapped plane, so neither edge dominates.
///
///   t_j      = σ(1+u_j)/(1−u_j)
///   shift_j  = t_j²
///   weight_j = (4σ/π) · w_j / (1−u_j)²      (dt = 2σ/(1−u)² du, times 2/π)
pub fn contour_terms_gauss_legendre(lambda_min: f64, lambda_max: f64, q: usize) -> Vec<ContourTerm> {
    let sigma = (lambda_min * lambda_max).powf(0.25);
    let (nodes, weights) = gauss_legendre(q);
    nodes
        .iter()
        .zip(&weights)
        .map(|(&u, &w)| {
            let one_minus = 1.0 - u;
            let t = sigma * (1.0 + u) / one_minus;
            ContourTerm {
                shift: t * t,
                weight: (4.0 * sigma / std::f64::consts::PI) * w / (one_minus * one_minus),
            }
        })
        .collect()
}

/// The scalar contract every rule is tested against:
/// `Σ_j weight_j / (lambda + shift_j)` → `lambda^{-1/2}`.
#[cfg(any(test, doc))]
pub fn scalar_inv_sqrt(terms: &[ContourTerm], lambda: f64) -> f64 {
    terms.iter().map(|t| t.weight / (lambda + t.shift)).sum()
}

#[cfg(test)]
mod tests {
    //! Verification of contour quadrature and elliptic-integral tools for matrix functions.

    use super::*;

    #[test]
    fn gl_quadrature_integrates_polynomials() {
        let (u, w) = gauss_legendre(24);
        // ∫₋₁¹ x² dx = 2/3
        let s: f64 = u.iter().zip(&w).map(|(x, wi)| wi * x * x).sum();
        assert!((s - 2.0 / 3.0).abs() < 1e-10, "GL ∫x² = {s}");
        let total: f64 = w.iter().sum();
        assert!((total - 2.0).abs() < 1e-10);
    }

    #[test]
    fn gl_contour_shifts_and_weights_are_strictly_positive() {
        let terms = contour_terms(ContourRule::GaussLegendre, 1e-4, 1e2, 33);
        assert_eq!(terms.len(), 33);
        for t in &terms {
            assert!(t.shift > 0.0 && t.shift.is_finite(), "shift {}", t.shift);
            assert!(t.weight > 0.0 && t.weight.is_finite(), "weight {}", t.weight);
        }
    }

    #[test]
    fn gl_contour_matches_scalar_inv_sqrt() {
        // λ swept across the spectrum the terms were built for; the quadrature
        // must reproduce λ^{-1/2} at every point INSIDE [λmin, λmax], with the
        // hardest cases at the edges. Q chosen per condition number from the
        // κ^{1/4} law (see the node-count test below for the measured table).
        for &(lmin, lmax, q) in &[(1e-2, 1e2, 32usize), (1e-4, 1e2, 64), (1e-6, 1.0, 128)] {
            let terms = contour_terms(ContourRule::GaussLegendre, lmin, lmax, q);
            // geometric sweep including both edges
            let n_pts = 9;
            for i in 0..n_pts {
                let f = i as f64 / (n_pts - 1) as f64;
                let lambda = lmin * (lmax / lmin).powf(f);
                let approx = scalar_inv_sqrt(&terms, lambda);
                let exact = lambda.powf(-0.5);
                let rel = ((approx - exact) / exact).abs();
                assert!(
                    rel < 1e-6,
                    "κ={:.0e} Q={q} λ={lambda:.3e}: rel err {rel:.3e}",
                    lmax / lmin
                );
            }
        }
    }

    /// Pins the measured Q(κ) table so the κ^{1/4} estimates in the plan are
    /// facts: the SMALLEST Q from the candidate ladder reaching 1e-4 relative
    /// error across the whole spectrum, per condition number.
    #[test]
    fn gl_contour_node_count_tracks_condition_number() {
        let q_ladder: &[usize] = &[8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
        let worst_rel = |terms: &[ContourTerm], lmin: f64, lmax: f64| -> f64 {
            let n_pts = 17;
            (0..n_pts)
                .map(|i| {
                    let f = i as f64 / (n_pts - 1) as f64;
                    let lambda = lmin * (lmax / lmin).powf(f);
                    let exact = lambda.powf(-0.5);
                    ((scalar_inv_sqrt(terms, lambda) - exact) / exact).abs()
                })
                .fold(0.0, f64::max)
        };
        let min_q = |kappa: f64| -> usize {
            let (lmin, lmax) = (1.0 / kappa, 1.0);
            *q_ladder
                .iter()
                .find(|&&q| {
                    worst_rel(&contour_terms(ContourRule::GaussLegendre, lmin, lmax, q), lmin, lmax)
                        < 1e-4
                })
                .unwrap_or_else(|| panic!("no Q on the ladder reaches 1e-4 at κ={kappa:.0e}"))
        };

        let (q2, q4, q6) = (min_q(1e2), min_q(1e4), min_q(1e6));
        // MEASURED table (dense-sweep verified 2026-07): 8 / 12 / 24, roughly
        // doubling per 100× in κ — substantially better than the a-priori
        // κ^{1/4}·log(1/ε) estimate. Bounds leave one ladder rung of head-
        // room; this test documents the scaling, it does not tune it.
        assert!(q2 <= 12, "κ=1e2 needed Q={q2} (measured 8)");
        assert!(q4 <= 16, "κ=1e4 needed Q={q4} (measured 12)");
        assert!(q6 <= 32, "κ=1e6 needed Q={q6} (measured 24)");
        assert!(q2 <= q4 && q4 <= q6, "Q not monotone in κ: {q2}, {q4}, {q6}");
        // The elliptic rule's remaining motivation is a FIXED small budget at
        // extreme condition: at Q=16 and κ=1e8 GL is still ~3e-2 off, where
        // HHT's O(exp(−π²Q/(log κ+3))) bound gives ~1e-3 at Q=15. Pin GL's
        // failure there so Phase 6's reason-to-exist stays a measured fact.
        let terms = contour_terms(ContourRule::GaussLegendre, 1e-8, 1.0, 16);
        assert!(
            worst_rel(&terms, 1e-8, 1.0) > 1e-3,
            "GL at Q=16, κ=1e8 now beats 1e-3 — revisit whether Phase 6 is still needed"
        );
    }

    #[test]
    #[should_panic(expected = "lambda_min")]
    fn contour_terms_refuses_a_nonpositive_lambda_min() {
        let _ = contour_terms(ContourRule::GaussLegendre, 0.0, 1.0, 8);
    }
}
