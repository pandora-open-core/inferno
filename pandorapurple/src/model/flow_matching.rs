//! Inferno — Flow-matching neural models with caller-supplied training data.

use candle_core::{Result, Tensor};

/// Caller-supplied velocity network v(x,t), independent of the VAE/GP option.
pub struct FlowMatchingModel<F> { pub velocity: F }
impl<F: Fn(&Tensor, f64) -> Result<Tensor>> FlowMatchingModel<F> {
    /// Straight-path conditional flow matching, t in [0,1].
    pub fn loss(&self, source: &Tensor, target: &Tensor, t: f64) -> Result<Tensor> {
        if !(0.0..=1.0).contains(&t) || source.dims() != target.dims() {
            candle_core::bail!("matching endpoint shapes and t in [0,1] required");
        }
        let x = source.affine(1.0 - t, 0.0)?.add(&target.affine(t, 0.0)?)?;
        (self.velocity)(&x, t)?.sub(&target.sub(source)?)?.sqr()?.mean_all()
    }
}
