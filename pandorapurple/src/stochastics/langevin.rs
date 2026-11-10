//! Inferno — Langevin stochastic differential equations with caller-supplied drift objectives.

use candle_core::{Result, Tensor};
use crate::flows::{Direction, anderson_backward_drift};

/// Euler–Maruyama with a forward drift and Anderson backward completion.
/// Drift and score closures may call the VAE or legacy multitask GP.
/// Scalar sigma is constant in space and time. Caller supplies standard-normal
/// noise, making seeds, trajectories and differentiability explicit.
pub struct LangevinSDE<F, S> {
    forward: F,
    score: S,
    sigma: f64,
}
impl<F, S> LangevinSDE<F, S>
where F: Fn(&Tensor, f64) -> Result<Tensor>, S: Fn(&Tensor, f64) -> Result<Tensor> {
    pub fn new(forward: F, score: S, sigma: f64) -> Result<Self> {
        if !sigma.is_finite() || sigma < 0.0 { candle_core::bail!("sigma must be finite and nonnegative"); }
        Ok(Self { forward, score, sigma })
    }
    pub fn drift(&self, x: &Tensor, time: f64, direction: Direction) -> Result<Tensor> {
        if !time.is_finite() { candle_core::bail!("time must be finite"); }
        let forward = (self.forward)(x, time)?;
        match direction {
            Direction::Forward => Ok(forward),
            Direction::Backward => anderson_backward_drift(&forward, &(self.score)(x, time)?, self.sigma),
        }
    }
    /// dt is a positive step magnitude; backward motion uses -dt for drift
    /// and sqrt(dt) for diffusion. No detach is applied.
    pub fn step(&self, x: &Tensor, time: f64, dt: f64, direction: Direction, noise: &Tensor) -> Result<Tensor> {
        if !dt.is_finite() || dt <= 0.0 || x.dims() != noise.dims() {
            candle_core::bail!("positive finite dt and matching noise shape required");
        }
        let sign = if direction == Direction::Forward { 1.0 } else { -1.0 };
        x.add(&self.drift(x, time, direction)?.affine(sign * dt, 0.0)?)?
            .add(&noise.affine(self.sigma * dt.sqrt(), 0.0)?)
    }
}
