//! Inferno — Composable neural layers and spectral variational-autoencoder components.

use anyhow::{anyhow, Context, Result};
use candle_core::Tensor;

// ============================================================================
// Execution mode
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirichletMode {
    /// Median-uniform / mean-evidence. Reproducible from inputs alone.
    Deterministic,
    /// Fresh random draws for both sticks and the low-rank MVN.
    Stochastic,
}

impl Default for DirichletMode {
    fn default() -> Self {
        DirichletMode::Deterministic
    }
}

#[derive(Debug, Clone)]
pub struct GpParams {
    pub gates: Tensor,
    pub gates_logits: Tensor,
    pub mat52: Tensor,
    pub rbf: Tensor,
    pub rq: Tensor,
}

pub mod decoder;
pub use decoder::{Decoder, DecoderMode, SpectralDecoder};
pub mod dirichlet;
pub use dirichlet::{AmortisedDirichlet, Dirichlet, DirichletOutput, KernelNetworkModule};
pub mod encoder;
pub use encoder::{ConvolutionalLoopEncoder, Encoder, EncoderOutput};
pub mod nkn;
pub use nkn::KernelNetwork;
pub mod vae;
pub use vae::{
    draw_step_bundle, lmc_project, noise_site_shapes, pack_features, spectral_mixture,
    RefinementCertainty, SpectralVae, StateSpaceOutput,
};
pub mod counterfactual;
pub mod parametrize;
pub use parametrize::{build_or_absent, SpectralNormLinear, WeightNormLinear};
pub mod conv_t;
pub mod ewma;
pub mod mamba;
pub mod moments;
pub mod norm;
pub mod time;
mod utils;
pub use utils::*;
pub type Vae =
    SpectralVae<ConvolutionalLoopEncoder, AmortisedDirichlet<KernelNetwork>, SpectralDecoder>;

mod dirichlet_backward;
pub use dirichlet_backward::{
    coreg_matrix, coreg_vjp, interval_vjp, kumaraswamy_icdf, kumaraswamy_icdf_vjp, softplus_vjp,
    stick_break, stick_break_vjp, DirichletGrads,
};

pub fn vae_forward(
    vae: &Vae,
    certainty: RefinementCertainty,
    x: &Tensor,
    state: Option<&StateSpaceOutput>,
) -> Result<(StateSpaceOutput, Vec<f32>, [f32; 8])> {
    let state_out = vae
        .refinement_loop(x, 0.0, certainty, None, state.cloned(), false, None)
        .context("VAE refinement failed")?;

    let zz = pack_features(&state_out).context("feature packing failed")?;
    if zz.len() != 198 {
        return Err(anyhow!("packed feature vector length {} ≠ 198", zz.len()));
    }
    let lmc_w =
        lmc_project(&state_out.lmc_consensus).map_err(|e| anyhow!("LMC projection failed: {e}"))?;
    Ok((state_out, zz, lmc_w))
}

pub mod flow_matching;
pub use flow_matching::FlowMatchingModel;

pub mod concentration;
pub mod gaussian;
pub mod noise;
