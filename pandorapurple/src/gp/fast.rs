//! Inferno — Cached sparse GP prediction using caller-supplied legacy posterior parameters.

use anyhow::Result;
use candle_core::Tensor;
use std::time::Instant;

use crate::model::{RefinementCertainty, StateSpaceOutput, vae_forward, Vae};
use crate::gp::{Prediction, SvgpPredictor};
use super::ServePrediction;
pub type FastGp = SvgpPredictor;

/// Run the fast GP serving path for one applicant.
///
/// Mutable `&mut gp` because `set_lmc_weights` rewrites the per-request
/// LMC weights from the VAE's `lmc_consensus`. Caller doesn't otherwise
/// hold state; the predictor's `k_star` / `v_buf` / `pt_v_buf` scratch
/// buffers are reused across calls.
pub fn serve(
    vae: &Vae,
    gp: &mut FastGp,
    certainty: RefinementCertainty,
    x: &Tensor,
    state: Option<&StateSpaceOutput>,
) -> Result<(ServePrediction, StateSpaceOutput)> {
    let t0 = Instant::now();
    let (state_out, zz, lmc_w) = vae_forward(vae, certainty, x, state)?;
    gp.set_lmc_weights(lmc_w);
    let pred: Prediction = gp.predict(&zz);
    Ok((
        ServePrediction {
            mean: pred.mean,
            variance: pred.variance,
            per_latent_mean: pred.per_latent_mean,
            per_latent_variance: pred.per_latent_variance,
            latency_us: t0.elapsed().as_micros() as u64,
        },
        state_out,
    ))
}
