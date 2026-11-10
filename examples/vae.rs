//! Inferno — Construct a trainable spectral variational autoencoder from library components.

use inferno::candle::{DType, Device, Tensor};
use inferno::model::{DecoderMode, DirichletMode, RefinementCertainty};
use inferno::models::PandoraVae;
use inferno::nn::{VarBuilder, VarMap};
use inferno::pandorapurple::config::RootConfig;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = RootConfig::default();
    let parameters = VarMap::new();
    let builder = VarBuilder::from_varmap(&parameters, DType::F32, &device);
    let vae = PandoraVae::build(
        builder,
        config.clone(),
        DirichletMode::Deterministic,
        DecoderMode::Deterministic,
        device.clone(),
    )?;
    let input = Tensor::zeros(
        (1, config.encoder.seq_len, config.encoder.input_dim),
        DType::F32,
        &device,
    )?;
    let output = vae.forward(&input, None, None, RefinementCertainty::Quick)?;
    println!(
        "Latent shape: {:?}; trainable tensors: {}",
        output
            .z
            .as_ref()
            .ok_or("VAE returned no latent tensor")?
            .dims(),
        parameters.all_vars().len()
    );
    Ok(())
}
