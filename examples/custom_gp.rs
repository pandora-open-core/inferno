//! Inferno — Construct and evaluate a CPU Gaussian process with custom covariance components.

use inferno::{
    cache::CachedCovariance,
    covariance::{Covariance, FunctionCovariance, Matern, MaternOrder, Rbf, Sum},
    developer::finite_difference,
    gp::ExactGp,
    likelihood::{FunctionNoise, Gaussian},
    mean::Linear,
    DType, Result, Tensor,
};
fn main() -> Result<()> {
    let x = Tensor::from_f64_slice(&[0.0, 1.0, 2.0], &[3, 1], DType::F64)?;
    let y = Tensor::from_f64_slice(&[1.0, 2.2, 2.9], &[3], DType::F64)?;
    // Replace this closure with a domain-specific PSD covariance.
    let custom = FunctionCovariance(|x: &Tensor, y: &Tensor| Rbf::new(0.5, 2.0)?.covariance(x, y));
    let kernel = Sum(
        Matern::new(1.0, vec![1.0], MaternOrder::FiveHalves)?,
        custom,
    );
    let kernel = CachedCovariance::lru(kernel, 4, 0);
    // Revision must change when kernel parameters change.
    let noise = FunctionNoise(|x: &Tensor| Ok(vec![0.05; x.shape()[0]]));
    let posterior = ExactGp::with_components(
        kernel,
        Linear {
            weights: vec![1.0],
            bias: 1.0,
        },
        noise,
        1e-9,
    )?
    .condition(&x, &y)?;
    println!(
        "log marginal likelihood: {}",
        posterior.log_marginal_likelihood()
    );
    println!(
        "latent mean: {:?}",
        posterior.predict(&x)?.mean.to_vec_f64()?
    );
    println!(
        "observed variance: {:?}",
        posterior.predict_observed(&x)?.variance.to_vec_f64()?
    );
    let _joint = posterior.posterior_covariance(&x)?;
    // Check or tune positive hyperparameters in log space.
    let gradient = finite_difference(&[0.0, 0.0], 1e-4, |p| {
        Ok(-ExactGp::new(
            Rbf::new(p[0].exp(), p[1].exp())?,
            Gaussian::new(0.05)?,
            1e-9,
        )?
        .condition(&x, &y)?
        .log_marginal_likelihood())
    })?;
    println!("negative log marginal likelihood gradient: {gradient:?}");
    Ok(())
}
