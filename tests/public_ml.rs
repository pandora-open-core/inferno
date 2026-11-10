//! Inferno — Public tensor-model interfaces and differentiability integration checks.

use inferno::candle::{Device, Tensor, Var};
use inferno::stochastics::{Direction, LangevinSDE};

#[test]
fn reverse_sde_uses_signed_drift_and_positive_noise_scale() -> inferno::candle::Result<()> {
    let x = Tensor::new(&[[1f64]], &Device::Cpu)?;
    let sde = LangevinSDE::new(
        |x: &Tensor, _| x.affine(0., 2.),
        |x: &Tensor, _| x.affine(0., 3.),
        0.5,
    )?;
    let noise = x.affine(0., 1.)?;
    let f = sde
        .step(&x, 0.5, 0.04, Direction::Forward, &noise)?
        .to_vec2::<f64>()?[0][0];
    let b = sde
        .step(&x, 0.5, 0.04, Direction::Backward, &noise)?
        .to_vec2::<f64>()?[0][0];
    assert!((f - 1.18).abs() < 1e-10);
    assert!((b - 1.05).abs() < 1e-10);
    assert!(sde.step(&x, 0., -0.1, Direction::Forward, &noise).is_err());
    Ok(())
}

#[test]
fn sinkhorn_preserves_marginals_and_cost_gradient() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    let a = Tensor::new(&[0.3f64, 0.7], &dev)?;
    let b = Tensor::new(&[0.4f64, 0.6], &dev)?;
    let c = Var::new(&[[0.2f64, 1.0], [0.7, 0.1]], &dev)?;
    let e = Tensor::new(0.5f64, &dev)?;
    let w = inferno::kernels::sinkhorn::sinkhorn_coupling_t(&a, &b, c.as_tensor(), &e, 150, 1e-9)?;
    for (actual, expected) in w.sum(1)?.to_vec1::<f64>()?.iter().zip([0.3, 0.7]) {
        assert!((actual - expected).abs() < 1e-8);
    }
    for (actual, expected) in w.sum(0)?.to_vec1::<f64>()?.iter().zip([0.4, 0.6]) {
        assert!((actual - expected).abs() < 1e-8);
    }
    let loss = w.mul(c.as_tensor())?.sum_all()?;
    let gradients = loss.backward()?;
    let g = gradients.get(c.as_tensor()).unwrap().to_vec2::<f64>()?;
    assert!(g.iter().flatten().all(|x| x.is_finite()));
    assert!(g.iter().flatten().any(|x| x.abs() > 1e-6));
    let evaluate = |delta: f64| -> inferno::candle::Result<f64> {
        let cost = Tensor::new(&[[0.2f64 + delta, 1.0], [0.7, 0.1]], &dev)?;
        let plan = inferno::kernels::sinkhorn::sinkhorn_coupling_t(&a, &b, &cost, &e, 150, 1e-9)?;
        plan.mul(&cost)?.sum_all()?.to_scalar::<f64>()
    };
    let finite_difference = (evaluate(1e-5)? - evaluate(-1e-5)?) / 2e-5;
    assert!((g[0][0] - finite_difference).abs() < 1e-6);

    Ok(())
}

#[test]
fn candle_vjp_connects_to_generalised_autograd() -> Result<(), Box<dyn std::error::Error>> {
    use inferno::{autograd, candle_apply, DType, Variable};
    let x = Variable::leaf(
        inferno::Tensor::from_f64_slice(&[2., 3.], &[2], DType::F64)?,
        true,
    );
    let a = autograd::derivatives::elementwise::affine(&x, 2., 0.)?;
    let y = candle_apply(&[&a], |xs| xs[0].sqr()?.sum_all())?;
    autograd::backward(&[y], None, false)?;
    assert_eq!(x.grad().unwrap().to_vec_f64()?, vec![16., 24.]);
    Ok(())
}

#[test]
fn bridge_weights_only_feed_component_losses() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    let plan = Var::new(&[[0.2f64, 0.3], [0.1, 0.4]], &dev)?;
    let p = Tensor::new(&[[0f64], [2.]], &dev)?;
    let q = Tensor::new(&[[1f64], [3.]], &dev)?;
    let loss = inferno::model::decoder::bridge_weighted_reconstruction(&p, &q, plan.as_tensor())?;
    assert!((loss.to_scalar::<f64>()? - 3.4).abs() < 1e-10);
    assert!(loss.backward()?.get(plan.as_tensor()).is_some());
    let p = Tensor::new(&[[0.2f64, 0.8], [0.7, 0.3]], &dev)?;
    let q = Tensor::new(&[[0.5f64, 0.5], [0.8, 0.2]], &dev)?;
    let loss = inferno::model::dirichlet::bridge_weighted_divergence(&p, &q, plan.as_tensor())?;
    assert!(loss.to_scalar::<f64>()? > 0.);
    assert!(loss.backward()?.get(plan.as_tensor()).is_some());
    Ok(())
}

#[test]
fn convolution_views_match_contiguous_forward_and_backward() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    for two_d in [false, true] {
        let (base, weight) = if two_d {
            (
                Tensor::arange(0f32, 192., &dev)?
                    .reshape((1, 8, 3, 8))?
                    .affine(0.01, 0.)?,
                Tensor::arange(0f32, 108., &dev)?
                    .reshape((3, 4, 3, 3))?
                    .affine(0.01, 0.)?,
            )
        } else {
            (
                Tensor::arange(0f32, 72., &dev)?
                    .reshape((2, 12, 3))?
                    .affine(0.01, 0.)?,
                Tensor::arange(0f32, 36., &dev)?
                    .reshape((3, 4, 3))?
                    .affine(0.01, 0.)?,
            )
        };
        let input = Var::from_tensor(&base)?;
        let weights = Var::from_tensor(&weight)?;
        let x = input.transpose(1, 2)?;
        let w = weights.transpose(0, 1)?;
        let conv = |x: &Tensor, w: &Tensor| {
            if two_d {
                x.conv2d(w, 1, 1, 1, 1)
            } else {
                x.conv1d(w, 1, 1, 1, 1)
            }
        };
        let viewed = conv(&x, &w)?;
        let packed = conv(&x.contiguous()?, &w.contiguous()?)?;
        let error = viewed.sub(&packed)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(error < 1e-4, "convolution view forward error: {error}");
        let gv = viewed.sqr()?.sum_all()?.backward()?;
        let gp = packed.sqr()?.sum_all()?.backward()?;
        for variable in [&input, &weights] {
            let error = gv
                .get(variable.as_tensor())
                .unwrap()
                .sub(gp.get(variable.as_tensor()).unwrap())?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error < 1e-3, "convolution view gradient error: {error}");
        }
    }
    Ok(())
}

#[test]
fn bridge_backward_drift_uses_physical_time_convention() -> inferno::candle::Result<()> {
    use inferno::stochastics::{GaussianMixture, GmmSchrodingerBridge, SdeConfig};
    let dev = Device::Cpu;
    let mixture = |mean: f64| -> inferno::candle::Result<GaussianMixture> {
        Ok(GaussianMixture::new(
            Tensor::new(&[1f64], &dev)?,
            Tensor::new(&[[mean]], &dev)?,
            Tensor::new(&[[1f64]], &dev)?,
        ))
    };
    let bridge = GmmSchrodingerBridge::fit(
        mixture(0.)?,
        mixture(2.)?,
        SdeConfig {
            sigma: 0.,
            ..Default::default()
        },
    )?;
    let x = Tensor::new(&[[1f64]], &dev)?;
    for direction in [Direction::Forward, Direction::Backward] {
        let drift = bridge.drift(&x, 0.5, direction)?.to_vec2::<f64>()?[0][0];
        assert!((drift - 2.).abs() < 1e-10);
    }
    assert!(bridge.drift(&x, -1., Direction::Forward).is_err());
    Ok(())
}

#[test]
fn latent_geoc_reference_scales_displacement_by_remaining_horizon() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    let drift = inferno::flows::LatentGeocDrift::new(
        Tensor::new(&[[1f64]], &dev)?,
        Tensor::new(&[0f64], &dev)?,
        Tensor::new(&[[4f64]], &dev)?,
    )?;
    let x = Tensor::new(&[[1f64]], &dev)?;
    let atoms = Tensor::new(&[[3f64]], &dev)?;
    let plan = Tensor::new(&[[1f64]], &dev)?;
    assert_eq!(
        drift.ground_cost(&x, &atoms)?.to_vec2::<f64>()?,
        vec![vec![16.]]
    );
    assert_eq!(
        drift.drift(&x, &atoms, &plan)?.to_vec2::<f64>()?,
        vec![vec![8.]]
    );
    Ok(())
}

#[test]
fn independent_flow_matching_model_has_velocity_gradients() -> inferno::candle::Result<()> {
    let dev = Device::Cpu;
    let scale = Var::new(3f64, &dev)?;
    let model = inferno::model::FlowMatchingModel {
        velocity: |x: &Tensor, _| x.broadcast_mul(scale.as_tensor()),
    };
    let source = Tensor::new(&[[0f64]], &dev)?;
    let target = Tensor::new(&[[2f64]], &dev)?;
    let loss = model.loss(&source, &target, 0.5)?;
    assert_eq!(loss.to_scalar::<f64>()?, 1.);
    assert_eq!(
        loss.backward()?
            .get(scale.as_tensor())
            .unwrap()
            .to_scalar::<f64>()?,
        2.
    );
    Ok(())
}
