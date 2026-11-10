//! Inferno — Candle Gaussian distributions with differentiable covariance operations.

use inferno::candle::{DType, Device, Tensor, Var};
use inferno::distributions::MultivariateNormal;
use inferno::gp::prediction::DefaultPredictionStrategy;
use inferno::operators::DenseOperator;

fn main() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    // A learnable variance demonstrates the copied CustomOp backward path.
    let scale = Var::new(1.0f64, &dev)?;
    let base = Tensor::new(&[[1.0f64, 0.5], [0.5, 1.0]], &dev)?;
    let covariance = base.broadcast_mul(scale.as_tensor())?;
    let mean = Tensor::zeros(2, DType::F64, &dev)?;
    let labels = Tensor::new(&[1.0f64, -0.5], &dev)?;
    let prior = MultivariateNormal::from_dense(mean.clone(), covariance.clone());
    let log_prob = prior.log_prob(&labels)?;
    let gradients = log_prob.backward()?;
    println!("log p(y) = {}", log_prob.to_scalar::<f64>()?);
    println!(
        "d log p(y) / d scale = {}",
        gradients
            .get(&scale)
            .expect("scale gradient")
            .to_scalar::<f64>()?
    );

    let strategy =
        DefaultPredictionStrategy::new(&DenseOperator::new(covariance), &mean, &labels, 1e-5)?;
    let cross = Tensor::new(&[[0.4f64, 0.2]], &dev)?;
    let predicted_mean =
        strategy.exact_predictive_mean(&Tensor::zeros(1, DType::F64, &dev)?, &cross)?;
    let predicted_variance =
        strategy.exact_predictive_variance(&Tensor::new(&[1.0f64], &dev)?, &cross)?;
    println!(
        "prediction: mean={:?}, variance={:?}",
        predicted_mean.to_vec1::<f64>()?,
        predicted_variance.to_vec1::<f64>()?
    );
    Ok(())
}
