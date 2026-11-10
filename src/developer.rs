//! Inferno — Finite-difference diagnostics for custom components and hyperparameter objectives.

use crate::{ensure, Result};
/// Central finite differences for a caller-supplied finite scalar objective.
/// This is a diagnostic, not autodiff. For positive parameters, differentiate
/// in log space, transforming with `exp()` inside the objective.
/// Evaluates the objective twice per parameter; propagates errors unchanged.
pub fn finite_difference<F>(
    parameters: &[f64],
    relative_step: f64,
    mut objective: F,
) -> Result<Vec<f64>>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    ensure(
        relative_step.is_finite() && relative_step > 0.0,
        "invalid finite difference step",
    )?;
    ensure(
        parameters.iter().all(|v| v.is_finite()),
        "nonfinite parameters",
    )?;
    let mut gradient = Vec::with_capacity(parameters.len());
    for (i, p) in parameters.iter().enumerate() {
        let step = relative_step * p.abs().max(1.0);
        let mut plus = parameters.to_vec();
        let mut minus = parameters.to_vec();
        plus[i] += step;
        minus[i] -= step;
        ensure(
            plus[i].is_finite() && minus[i].is_finite() && plus[i] != minus[i],
            "finite difference step overflow or underflow",
        )?;
        let a = objective(&plus)?;
        let b = objective(&minus)?;
        ensure(a.is_finite() && b.is_finite(), "nonfinite objective")?;
        let denominator = plus[i] - minus[i];
        ensure(
            denominator.is_finite(),
            "finite difference denominator overflow",
        )?;
        let g = (a - b) / denominator;
        ensure(g.is_finite(), "gradient overflow")?;
        gradient.push(g);
    }
    Ok(gradient)
}
/// KL[N(mean,variance) || N(prior_mean,prior_variance)] for one dimension.
pub fn gaussian_kl(mean: f64, variance: f64, prior_mean: f64, prior_variance: f64) -> Result<f64> {
    ensure(
        mean.is_finite()
            && prior_mean.is_finite()
            && variance.is_finite()
            && variance > 0.0
            && prior_variance.is_finite()
            && prior_variance > 0.0,
        "invalid Gaussian KL arguments",
    )?;
    let ratio = variance / prior_variance;
    let delta = (mean - prior_mean) / prior_variance.sqrt();
    let result = 0.5 * (ratio - 1.0 + prior_variance.ln() - variance.ln() + delta * delta);
    ensure(
        result.is_finite() && result >= -1e-12,
        "Gaussian KL overflow",
    )?;
    Ok(result.max(0.0))
}
