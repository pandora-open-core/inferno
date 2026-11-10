//! Inferno — Elliptic-integral routines used by contour quadrature.

use super::ContourTerm;

/// Arithmetic–geometric mean of `(a0, b0)`, `a0, b0 > 0`. Quadratic
/// convergence: ~8 iterations to f64 precision.
pub fn agm(a0: f64, b0: f64) -> f64 {
    let (mut a, mut b) = (a0, b0);
    for _ in 0..64 {
        if (a - b).abs() <= f64::EPSILON * a.abs().max(1.0) {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    0.5 * (a + b)
}

/// Complete elliptic integral of the first kind, PARAMETER convention:
/// `K(m) = ∫₀^{π/2} dθ/√(1 − m sin²θ)`, `m = k² ∈ [0, 1)`.
///
/// Via the AGM identity `K(m) = π / (2·agm(1, √(1−m)))`.
pub fn ellipk(m: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&m),
        "ellipk: parameter m must be in [0, 1), got {m}"
    );
    std::f64::consts::FRAC_PI_2 / agm(1.0, (1.0 - m).sqrt())
}

/// Jacobi elliptic functions `(sn, cn, dn)` at REAL argument `u` with
/// PARAMETER `m = k²` — the descending-Landen / AGM algorithm (Numerical
/// Recipes `sncndn`, expressed with the parameter rather than the
/// complementary one).
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    const CA: f64 = 1e-16;
    let mc = 1.0 - m; // complementary parameter, what the recurrence runs on

    if mc.abs() < 1e-300 {
        // m = 1: degenerate hyperbolic case.
        let sn = u.tanh();
        let c = 1.0 / u.cosh();
        return (sn, c, c);
    }

    let mut emc = mc;
    let mut uu = u;
    let mut d = 1.0;
    let bo = emc < 0.0;
    if bo {
        d = 1.0 - emc;
        emc = -emc / d;
        d = d.sqrt();
        uu *= d;
    }

    let mut a = 1.0;
    let mut dn = 1.0;
    let mut em = [0.0f64; 14];
    let mut en = [0.0f64; 14];
    let mut c = 0.0;
    let mut l = 0;
    for i in 0..13 {
        l = i;
        em[i] = a;
        emc = emc.sqrt();
        en[i] = emc;
        c = 0.5 * (a + emc);
        if (a - emc).abs() <= CA * a {
            break;
        }
        emc *= a;
        a = c;
    }

    uu *= c;
    let mut sn = uu.sin();
    let mut cn = uu.cos();
    if sn.abs() > 0.0 {
        a = cn / sn;
        c *= a;
        for i in (0..=l).rev() {
            let b = em[i];
            a *= c;
            c *= dn;
            dn = (en[i] + a) / (b + a);
            a = c / b;
        }
        a = 1.0 / (c * c + 1.0).sqrt();
        sn = if sn < 0.0 { -a } else { a };
        cn = c * sn;
    }
    if bo {
        std::mem::swap(&mut dn, &mut cn);
        sn /= d;
    }
    (sn, cn, dn)
}

/// HHT method-3 `(shift, weight)` pairs for `K^{-1/2}` over a spectrum in
/// `[lambda_min, lambda_max]`.
///
/// Derivation (self-contained, rather than transcribed from gpytorch's
/// intermediate variables — see the sign/Jacobian note below): substitute
/// `t = √λmin · sn(u|m₁)/cn(u|m₁)` with `m₁ = 1 − λmin/λmax` into
/// `K^{-1/2} = (2/π)∫₀^∞ (K + t²I)^{-1} dt`. Then
/// `dt = √λmin·(dn/cn²) du`, the range maps to `u ∈ [0, K′]` with
/// `K′ = ellipk(m₁)`, and the integrand extends to a `2K′`-periodic analytic
/// function symmetric about `K′` — which is why the Q-point midpoint rule
/// converges exponentially (`O(exp(−π²Q/(log κ + 3)))`, the whole point of
/// this rule):
///
/// ```text
/// u_j = (j − ½)·K′/Q,  (sn, cn, dn) at (u_j, m₁)
/// shift_j  = λmin · (sn/cn)²
/// weight_j = (2 K′ √λmin)/(π Q) · dn/cn²
/// ```
///
/// NOTE the Jacobian is `dn/cn²`, not the `dn/cn` that appears as the
/// imaginary-transformed `dn` in gpytorch's code — its remaining `1/cn`
/// rides inside its complex `sn` factor. Getting this wrong fails the
/// scalar oracle at O(1), which is exactly what the tests below pin.
pub fn contour_terms_elliptic(lambda_min: f64, lambda_max: f64, q: usize) -> Vec<ContourTerm> {
    let k2 = (lambda_min / lambda_max).min(1.0);
    let m1 = 1.0 - k2; // the parameter the Jacobi functions run at
    let kp = ellipk(m1);
    let scale = 2.0 * kp * lambda_min.sqrt() / (std::f64::consts::PI * q as f64);

    (0..q)
        .map(|j| {
            let u = (j as f64 + 0.5) * kp / q as f64;
            let (sn, cn, dn) = ellipj(u, m1);
            ContourTerm {
                shift: lambda_min * (sn / cn) * (sn / cn),
                weight: scale * dn / (cn * cn),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Verification of elliptic-integral routines used by contour quadrature.

    use super::*;
    use crate::quadrature::{contour_terms, scalar_inv_sqrt, ContourRule};

    #[test]
    fn ellipk_matches_known_values() {
        assert!((ellipk(0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        // K(m = 1/2) — Abramowitz & Stegun 17.3.34 territory, 20 digits.
        assert!(
            (ellipk(0.5) - 1.85407467730137191843).abs() < 1e-14,
            "K(0.5) = {}",
            ellipk(0.5)
        );
        // Legendre-free self-consistency: K(m) from AGM must agree with the
        // defining integral evaluated by brute midpoint quadrature.
        for &m in &[0.1, 0.3, 0.7, 0.9, 0.99] {
            let n = 200_000;
            let num: f64 = (0..n)
                .map(|i| {
                    let th = (i as f64 + 0.5) / n as f64 * std::f64::consts::FRAC_PI_2;
                    1.0 / (1.0 - m * th.sin().powi(2)).sqrt()
                })
                .sum::<f64>()
                * std::f64::consts::FRAC_PI_2
                / n as f64;
            assert!(
                (ellipk(m) - num).abs() < 1e-9,
                "K({m}): AGM {} vs integral {num}",
                ellipk(m)
            );
        }
    }

    #[test]
    fn jacobi_identities_hold() {
        for &m in &[0.0, 0.05, 0.3, 0.5, 0.8, 0.97, 0.9999] {
            for i in 0..12 {
                let u = -2.4 + 0.45 * i as f64;
                let (sn, cn, dn) = ellipj(u, m);
                assert!(
                    (sn * sn + cn * cn - 1.0).abs() < 1e-12,
                    "sn²+cn²≠1 at (u={u}, m={m}): {}",
                    sn * sn + cn * cn
                );
                assert!(
                    (dn * dn + m * sn * sn - 1.0).abs() < 1e-12,
                    "dn²+m·sn²≠1 at (u={u}, m={m})"
                );
            }
            // limits: m=0 → circular, m→1 → hyperbolic
        }
        let (sn, cn, dn) = ellipj(0.7, 0.0);
        assert!((sn - 0.7f64.sin()).abs() < 1e-14 && (cn - 0.7f64.cos()).abs() < 1e-14);
        assert!((dn - 1.0).abs() < 1e-14);
        let (sn1, _, _) = ellipj(0.7, 1.0);
        assert!((sn1 - 0.7f64.tanh()).abs() < 1e-12, "sn(u,1) ≠ tanh(u)");
    }

    #[test]
    fn elliptic_contour_matches_scalar_inv_sqrt_at_extreme_condition() {
        // The whole point of this rule: Q = 15 (the settings default) at
        // condition numbers where GL at that budget is hopeless.
        for &(kappa, q, tol) in &[(1e4f64, 15usize, 1e-5f64), (1e8, 15, 1e-3)] {
            let (lmin, lmax) = (1.0 / kappa, 1.0);
            let terms = contour_terms(ContourRule::Elliptic, lmin, lmax, q);
            for t in &terms {
                assert!(t.shift > 0.0 && t.weight > 0.0, "non-positive term {t:?}");
            }
            let n_pts = 17;
            for i in 0..n_pts {
                let f = i as f64 / (n_pts - 1) as f64;
                let lambda = lmin * (lmax / lmin).powf(f);
                let rel =
                    (scalar_inv_sqrt(&terms, lambda) - lambda.powf(-0.5)).abs() * lambda.sqrt();
                assert!(
                    rel < tol,
                    "κ={kappa:.0e} Q={q} λ={lambda:.3e}: rel {rel:.3e}"
                );
            }
        }
    }

    /// The measured motivation for this phase, pinned: at the fixed default
    /// budget and extreme κ, elliptic beats GL by orders of magnitude.
    #[test]
    fn elliptic_beats_gauss_legendre_node_for_node() {
        let worst = |rule: ContourRule, kappa: f64, q: usize| -> f64 {
            let (lmin, lmax) = (1.0 / kappa, 1.0);
            let terms = contour_terms(rule, lmin, lmax, q);
            let n_pts = 33;
            (0..n_pts)
                .map(|i| {
                    let f = i as f64 / (n_pts - 1) as f64;
                    let lambda = lmin * (lmax / lmin).powf(f);
                    (scalar_inv_sqrt(&terms, lambda) - lambda.powf(-0.5)).abs() * lambda.sqrt()
                })
                .fold(0.0, f64::max)
        };
        for &kappa in &[1e6f64, 1e8] {
            let gl = worst(ContourRule::GaussLegendre, kappa, 15);
            let el = worst(ContourRule::Elliptic, kappa, 15);
            assert!(
                el < gl / 10.0,
                "κ={kappa:.0e} Q=15: elliptic {el:.3e} not decisively below GL {gl:.3e}"
            );
        }
    }
}
