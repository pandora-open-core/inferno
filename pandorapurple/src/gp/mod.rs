//! Inferno — Legacy neural-kernel Gaussian processes with sparse and differentiable prediction paths.

use candle_core::Tensor;
pub mod kernel;
pub use kernel::{KernelHyper, LmcHyper, ProbabilisticMixtureMean, RawHyper, SvgpPredictor, layout, Prediction};
pub mod diff;
pub use diff::{KernelHyperTensor, kernel_base_candle, DifferentiableGp};
pub mod fast;
pub use fast::FastGp;
#[derive(Clone, Debug)]
pub struct ServePrediction {
    pub mean: f32,
    pub variance: f32,
    pub per_latent_mean: [f32; 8],
    pub per_latent_variance: [f32; 8],
    pub latency_us: u64,
}

#[derive(Clone, Debug)]
pub struct DifferentiablePrediction {
    pub mean: Tensor,                // [B]
    pub variance: Tensor,            // [B]
    pub per_latent_mean: Tensor,     // [B, 8]
    pub per_latent_variance: Tensor, // [B, 8]
}


/// Reusable multitask kernel evaluation interface for lazy operators.
pub trait KernelEval: Send + Sync {
    fn kmm(&self, z: &Tensor) -> candle_core::Result<Tensor>;
    fn kxm(&self, x: &Tensor, z: &Tensor) -> candle_core::Result<Tensor>;
    fn kdiag(&self, x: &Tensor, l: usize) -> candle_core::Result<Tensor>;
}
