//! Inferno — Interactive model training, persistence and Gaussian-process interface checks.

use inferno::prelude::*;

#[test]
fn registered_parameters_train_and_checkpoint_restores_values() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device, |vb| nn::linear(1, 1, vb.pp("output")))?;
    assert_eq!(model.parameters().len(), 2);
    for parameter in model.parameters() {
        parameter.set(&parameter.zeros_like()?)?;
    }
    let mut optimizer = model.adam(0.05)?;
    let x = Tensor::new(&[[-1f32], [0.], [1.]], &device)?;
    let y = x.affine(2., 1.)?;
    let loss = || nn::loss::mse(&model.forward(&x)?, &y);
    let initial = loss()?.to_scalar::<f32>()?;
    for _ in 0..250 {
        optimizer.backward_step(&loss()?)?;
    }
    assert!(loss()?.to_scalar::<f32>()? < initial * 0.01);
    let path = std::env::temp_dir().join(format!(
        "inferno-notebook-{}.safetensors",
        std::process::id()
    ));
    model.save(&path)?;
    let expected = model.forward(&x)?.to_vec2::<f32>()?;
    let mut restored = Model::new(&device, |vb| nn::linear(1, 1, vb.pp("output")))?;
    restored.load(&path)?;
    std::fs::remove_file(path).unwrap();
    assert_eq!(restored.forward(&x)?.to_vec2::<f32>()?, expected);
    model.eval();
    assert!(!model.is_training());
    assert!(model.adam(f64::NAN).is_err());
    Ok(())
}

struct Mode;
impl ModuleT for Mode {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        xs.affine(1., if train { 1. } else { 0. })
    }
}
#[test]
fn mode_reaches_user_module() -> Result<()> {
    let mut model = Model::new(&Device::Cpu, |_| Ok(Mode))?;
    let x = Tensor::new(&[2f32], &Device::Cpu)?;
    assert_eq!(model.forward(&x)?.to_vec1::<f32>()?, [3.]);
    model.eval();
    assert_eq!(model.forward(&x)?.to_vec1::<f32>()?, [2.]);
    model.train();
    assert_eq!(model.forward(&x)?.to_vec1::<f32>()?, [3.]);
    Ok(())
}

#[test]
fn gp_agrees_with_reference_and_preserves_fit_on_error() -> Result<()> {
    let x = Tensor::new(&[[-1f32], [0.], [1.]], &Device::Cpu)?;
    let y = Tensor::new(&[-0.8f32, 0., 0.8], &Device::Cpu)?;
    let mut gp = GaussianProcess::new(1., 1., 0.05)?;
    assert!(gp.predict(&x).is_err());
    gp.fit(&x, &y)?;
    let prediction = gp.predict(&x)?;
    let rx = inferno::Tensor::from_f64_slice(&[-1., 0., 1.], &[3, 1], inferno::DType::F64).unwrap();
    let ry = inferno::Tensor::from_f64_slice(
        &[-0.8f32 as f64, 0., 0.8f32 as f64],
        &[3],
        inferno::DType::F64,
    )
    .unwrap();
    let expected = inferno::gp::ExactGp::new(
        inferno::covariance::Rbf::new(1., 1.).unwrap(),
        inferno::likelihood::Gaussian::new(0.05).unwrap(),
        1e-8,
    )
    .unwrap()
    .condition(&rx, &ry)
    .unwrap()
    .predict(&rx)
    .unwrap();
    assert_eq!(
        prediction.mean.to_vec1::<f64>()?,
        expected.mean.to_vec_f64().unwrap()
    );
    assert_eq!(
        prediction.variance.to_vec1::<f64>()?,
        expected.variance.to_vec_f64().unwrap()
    );
    let invalid = Tensor::new(&[f32::NAN], &Device::Cpu)?;
    assert!(gp.fit(&x, &invalid).is_err());
    assert_eq!(
        gp.predict(&x)?.mean.to_vec1::<f64>()?,
        prediction.mean.to_vec1::<f64>()?
    );
    assert!(GaussianProcess::new(1., 0., 0.05).is_err());
    assert!(GaussianProcess::new(1., 1., -1.).is_err());
    Ok(())
}
