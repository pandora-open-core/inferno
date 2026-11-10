//! Inferno — Parameter registration and training checks for spectral model components.

use inferno::prelude::*;

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn spectral_buffers_are_checkpoint_state_not_optimizer_parameters() -> Result<()> {
    use inferno::model::parametrize::{initial_power_buffers, SpectralNormLinear};
    let device = Device::Cpu;
    let variables = nn::VarMap::new();
    let (u, v) = initial_power_buffers(2, 2, &device, DType::F32)?;
    let u = inferno::candle::Var::from_tensor(&u)?;
    let v = inferno::candle::Var::from_tensor(&v)?;
    variables
        .data()
        .lock()
        .unwrap()
        .insert("weight_u".into(), u.clone());
    variables
        .data()
        .lock()
        .unwrap()
        .insert("weight_v".into(), v.clone());
    let layer = SpectralNormLinear::new(
        nn::VarBuilder::from_varmap(&variables, DType::F32, &device),
        2,
        2,
        nn::Init::Const(0.2),
    )?;
    let before = flat(&u);
    let x = Tensor::new(&[[1f32, 2.]], &device)?;
    let loss = layer.forward(&x)?.sqr()?.sum_all()?;
    assert_ne!(before, flat(&u));
    let grads = loss.backward()?;
    assert!(grads.get(&u).is_none());
    assert!(grads.get(&v).is_none());
    layer.set_power_iteration(false);
    let frozen = flat(&u);
    layer.forward(&x)?;
    assert_eq!(frozen, flat(&u));
    Ok(())
}
