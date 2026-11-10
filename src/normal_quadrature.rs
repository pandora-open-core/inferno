//! Inferno — Normal-distribution quadrature nodes and weights for likelihood integration.

use crate::{ensure, Result};
/// Three-point Gauss-Hermite rule for E[f(X)], X ~ Normal(mean, variance).
/// Exact for polynomials up to degree five; not an adaptive error-controlled rule.
pub fn normal_expectation(mean: f64, variance: f64, f: impl Fn(f64) -> f64) -> Result<f64> {
    ensure(
        mean.is_finite() && variance.is_finite() && variance >= 0.0,
        "invalid normal moments",
    )?;
    let offset = variance.sqrt() * 3.0_f64.sqrt();
    let value = f(mean - offset) / 6.0 + (2.0 / 3.0) * f(mean) + f(mean + offset) / 6.0;
    ensure(value.is_finite(), "nonfinite quadrature result")?;
    Ok(value)
}
