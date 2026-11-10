//! Inferno — Composable neural layers and spectral variational-autoencoder components.

use candle_core::{Tensor};
use anyhow::{anyhow, Context, Result};

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
pub use decoder::{DecoderMode, SpectralDecoder, Decoder};
pub mod dirichlet;
pub use dirichlet::{AmortisedDirichlet, KernelNetworkModule, DirichletOutput, Dirichlet};
pub mod encoder;
pub use encoder::{ConvolutionalLoopEncoder, EncoderOutput, Encoder};
pub mod nkn;
pub use nkn::KernelNetwork;
pub mod vae;
pub use vae::{SpectralVae, StateSpaceOutput, RefinementCertainty, pack_features, lmc_project, spectral_mixture, draw_step_bundle, noise_site_shapes};
pub mod counterfactual;
pub mod parametrize;
pub use parametrize::{build_or_absent, SpectralNormLinear, WeightNormLinear};
pub mod norm;
pub mod ewma;
pub mod mamba;
pub mod time;
pub mod conv_t;
pub mod moments;
mod utils;
pub use utils::*;
pub type Vae = SpectralVae<
    ConvolutionalLoopEncoder,
    AmortisedDirichlet<KernelNetwork>,
    SpectralDecoder,
>;

mod dirichlet_backward;
pub use dirichlet_backward::{DirichletGrads, coreg_vjp, coreg_matrix, stick_break_vjp, stick_break, kumaraswamy_icdf_vjp, kumaraswamy_icdf, softplus_vjp, interval_vjp};

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
    let lmc_w = lmc_project(&state_out.lmc_consensus)
        .map_err(|e| anyhow!("LMC projection failed: {e}"))?;
    Ok((state_out, zz, lmc_w))
}


pub mod flow_matching;
pub use flow_matching::FlowMatchingModel;

pub mod noise;
pub mod concentration;
pub mod gaussian;
