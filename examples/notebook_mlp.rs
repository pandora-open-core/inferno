//! Inferno — Train a neural network with registered Candle parameters.

use inferno::prelude::*;

struct Regressor {
    hidden: nn::Linear,
    output: nn::Linear,
}
impl Regressor {
    fn new(vb: nn::VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            hidden: nn::linear(1, 16, vb.pp("hidden"))?,
            output: nn::linear(16, 1, vb.pp("output"))?,
        })
    }
}
impl Module for Regressor {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.output.forward(&self.hidden.forward(x)?.tanh()?)
    }
}
fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device, Regressor::new)?;
    let mut optimizer = model.adam(0.02)?;
    let x = Tensor::new(&[[-1f32], [-0.5], [0.], [0.5], [1.]], &device)?;
    let y = x.affine(2., 1.)?;
    model.train();
    for step in 0..300 {
        let loss = nn::loss::mse(&model.forward(&x)?, &y)?;
        if step % 100 == 0 {
            println!("step={step} loss={:.6}", loss.to_scalar::<f32>()?);
        }
        optimizer.backward_step(&loss)?;
    }
    model.eval();
    println!("prediction: {:?}", model.forward(&x)?.to_vec2::<f32>()?);
    Ok(())
}
