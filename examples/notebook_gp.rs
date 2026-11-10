//! Inferno — Interactive Candle Gaussian-process construction and prediction.

use inferno::prelude::*;
fn main() -> Result<()> {
    let x = Tensor::new(&[[-1f64], [0.], [1.]], &Device::Cpu)?;
    let y = Tensor::new(&[-0.8f64, 0., 0.8], &Device::Cpu)?;
    let mut model = GaussianProcess::new(1., 1., 0.05)?;
    model.fit(&x, &y)?;
    let query = Tensor::new(&[[-0.5f64], [0.5]], &Device::Cpu)?;
    let prediction = model.predict(&query)?;
    println!("mean: {:?}", prediction.mean.to_vec1::<f64>()?);
    println!("variance: {:?}", prediction.variance.to_vec1::<f64>()?);
    Ok(())
}
