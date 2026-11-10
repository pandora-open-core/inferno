//! Inferno — Schedule a Candle calculation through the generalised autograd engine.

use inferno::{autograd, candle_apply, DType, Tensor, Variable};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Variable::leaf(Tensor::from_f64_slice(&[2., 3.], &[2], DType::F64)?, true);
    let loss = candle_apply(&[&x], |xs| xs[0].sqr()?.sum_all())?;
    autograd::backward(&[loss], None, false)?;
    println!("gradient: {:?}", x.grad().unwrap().to_vec_f64()?);
    Ok(())
}
