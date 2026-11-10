//! Inferno — CPU exact Gaussian-process regression with an RBF covariance and Gaussian noise.

use inferno::{covariance::Rbf, gp::ExactGp, likelihood::Gaussian, DType, Result, Tensor};
fn main() -> Result<()> {
    let x = Tensor::from_f64_slice(&[-1.0, 0.0, 1.0], &[3, 1], DType::F64)?;
    let y = Tensor::from_f64_slice(&[-0.8, 0.0, 0.8], &[3], DType::F64)?;
    let posterior =
        ExactGp::new(Rbf::new(1.0, 1.0)?, Gaussian::new(0.05)?, 1e-8)?.condition(&x, &y)?;
    let test = Tensor::from_f64_slice(&[-0.5, 0.5], &[2, 1], DType::F64)?;
    let prediction = posterior.predict(&test)?;
    println!("mean: {:?}", prediction.mean.to_vec_f64()?);
    println!("latent variance: {:?}", prediction.variance.to_vec_f64()?);
    println!(
        "log marginal likelihood: {}",
        posterior.log_marginal_likelihood()
    );
    Ok(())
}
