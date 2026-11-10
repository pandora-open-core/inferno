//! Inferno — Bidirectional latent stochastic dynamics with caller-supplied mixture parameters.

use inferno::candle::{Device, Result, Tensor};
use inferno::stochastics::{Direction, LangevinSDE};
fn main() -> Result<()> {
    let x = Tensor::new(&[[1f32, 2.]], &Device::Cpu)?;
    let sde = LangevinSDE::new(
        |x: &Tensor, _| x.affine(-0.5, 0.0),
        |x: &Tensor, _| x.neg(),
        0.5,
    )?;
    let noise = x.zeros_like()?;
    let y = sde.step(&x, 0.5, 0.01, Direction::Backward, &noise)?;
    println!("Backward latent step: {:?}", y.to_vec2::<f32>()?);
    Ok(())
}
