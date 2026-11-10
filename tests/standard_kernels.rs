//! Inferno — Standard-kernel value, validation and first-order gradient checks.
use inferno::candle::{Device, Tensor as CT, Var};
use inferno::covariance::{self as cpu, Covariance, MaternOrder};
use inferno::kernels::standard::Kernel;
use inferno::{DType, Tensor};
fn scalar(v: f64) -> CT {
    CT::new(v, &Device::Cpu).unwrap()
}
fn families(v: CT, l: CT, p: CT) -> Vec<Kernel> {
    vec![
        Kernel::Rbf {
            variance: v.clone(),
            lengthscale: l.clone(),
        },
        Kernel::Matern {
            variance: v.clone(),
            lengthscale: l.clone(),
            order: MaternOrder::Half,
        },
        Kernel::Matern {
            variance: v.clone(),
            lengthscale: l.clone(),
            order: MaternOrder::ThreeHalves,
        },
        Kernel::Matern {
            variance: v.clone(),
            lengthscale: l.clone(),
            order: MaternOrder::FiveHalves,
        },
        Kernel::Linear {
            variance: v.clone(),
        },
        Kernel::Polynomial {
            variance: v.clone(),
            offset: p.clone(),
            degree: 3,
        },
        Kernel::Periodic {
            variance: v.clone(),
            lengthscale: l.clone(),
            period: p.clone(),
        },
        Kernel::RationalQuadratic {
            variance: v.clone(),
            lengthscale: l,
            alpha: p,
        },
        Kernel::Constant {
            variance: v.clone(),
        },
        Kernel::WhiteNoise { variance: v },
    ]
}
fn references() -> Vec<Box<dyn Covariance>> {
    vec![
        Box::new(cpu::Rbf::new(1.7, 0.8).unwrap()),
        Box::new(cpu::Matern::new(1.7, vec![0.8, 0.8], MaternOrder::Half).unwrap()),
        Box::new(cpu::Matern::new(1.7, vec![0.8, 0.8], MaternOrder::ThreeHalves).unwrap()),
        Box::new(cpu::Matern::new(1.7, vec![0.8, 0.8], MaternOrder::FiveHalves).unwrap()),
        Box::new(cpu::Linear::new(1.7).unwrap()),
        Box::new(cpu::Polynomial::new(1.7, 1.3, 3).unwrap()),
        Box::new(cpu::Periodic::new(1.7, 0.8, 1.3).unwrap()),
        Box::new(cpu::RationalQuadratic::new(1.7, 0.8, 1.3).unwrap()),
        Box::new(cpu::Constant::new(1.7).unwrap()),
        Box::new(cpu::WhiteNoise::new(1.7).unwrap()),
    ]
}
#[test]
fn candle_matches_cpu_on_rectangular_and_duplicate_inputs() {
    let x = [0.2, -0.4, 1.1, 0.6, 0.2, -0.4];
    let y = [0.2, -0.4, -0.7, 0.9];
    let cx = CT::from_slice(&x, (3, 2), &Device::Cpu).unwrap();
    let cy = CT::from_slice(&y, (2, 2), &Device::Cpu).unwrap();
    let dx = Tensor::from_f64_slice(&x, &[3, 2], DType::F64).unwrap();
    let dy = Tensor::from_f64_slice(&y, &[2, 2], DType::F64).unwrap();
    for (k, cpu) in families(scalar(1.7), scalar(0.8), scalar(1.3))
        .iter()
        .zip(references())
    {
        let got = k
            .covariance(&cx, &cy)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let want = cpu.covariance(&dx, &dy).unwrap().to_vec_f64().unwrap();
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-11, "{k:?}: {a} != {b}");
        }
    }
}
#[test]
fn gradients_match_finite_differences_for_inputs_and_hyperparameters() {
    let x =
        Var::from_tensor(&CT::new(&[[0.2f64, -0.4], [1.1, 0.6]], &Device::Cpu).unwrap()).unwrap();
    let y = CT::new(&[[-0.3f64, 0.9], [0.5, 1.2]], &Device::Cpu).unwrap();
    let vars = [
        Var::from_tensor(&scalar(1.7)).unwrap(),
        Var::from_tensor(&scalar(0.8)).unwrap(),
        Var::from_tensor(&scalar(1.3)).unwrap(),
    ];
    let ks = families(
        vars[0].as_tensor().clone(),
        vars[1].as_tensor().clone(),
        vars[2].as_tensor().clone(),
    );
    for (i, k) in ks.iter().enumerate() {
        let grads = k
            .covariance(&x, &y)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        for j in 0..4 {
            let h = 1e-5;
            let eval = |sign: f64| {
                let mut pars = [1.7, 0.8, 1.3];
                let mut values = vec![0.2, -0.4, 1.1, 0.6];
                if j < 3 {
                    pars[j] += sign * h;
                } else {
                    values[0] += sign * h;
                }
                let tx = CT::from_slice(&values, (2, 2), &Device::Cpu).unwrap();
                families(scalar(pars[0]), scalar(pars[1]), scalar(pars[2]))[i]
                    .covariance(&tx, &y)
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f64>()
                    .unwrap()
            };
            let numerical = (eval(1.) - eval(-1.)) / (2. * h);
            let analytic = if j < 3 {
                grads
                    .get(&vars[j])
                    .map(|g| g.to_scalar::<f64>().unwrap())
                    .unwrap_or(0.)
            } else {
                grads
                    .get(&x)
                    .map(|g| g.to_vec2::<f64>().unwrap()[0][0])
                    .unwrap_or(0.)
            };
            assert!(
                (analytic - numerical).abs() < 2e-7 * (1. + numerical.abs()),
                "family {i}, parameter {j}: {analytic} != {numerical}"
            );
        }
    }
}
#[test]
fn matern_coincident_gradients_are_finite() {
    let x = Var::from_tensor(&CT::new(&[[0.3f64], [0.3], [0.9]], &Device::Cpu).unwrap()).unwrap();
    let l = Var::from_tensor(&scalar(0.8)).unwrap();
    for order in [
        MaternOrder::Half,
        MaternOrder::ThreeHalves,
        MaternOrder::FiveHalves,
    ] {
        let k = Kernel::Matern {
            variance: scalar(1.7),
            lengthscale: l.as_tensor().clone(),
            order,
        };
        let grads = k
            .covariance(&x, &x)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert!(grads
            .get(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(grads
            .get(&l)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
            .is_finite());
    }
}
#[test]
fn invalid_parameters_and_shapes_are_rejected() {
    assert!(cpu::Linear::new(f64::NAN).is_err());
    assert!(cpu::Polynomial::new(1., -1., 2).is_err());
    assert!(cpu::Polynomial::new(1., 1., 0).is_err());
    assert!(cpu::Periodic::new(1., 1., 0.).is_err());
    assert!(cpu::RationalQuadratic::new(1., 1., -1.).is_err());
    assert!(cpu::Constant::new(0.).is_err());
    assert!(cpu::WhiteNoise::new(-1.).is_err());
    let x = CT::new(&[[0.2f64]], &Device::Cpu).unwrap();
    for k in families(scalar(-1.), scalar(0.8), scalar(1.3)) {
        assert!(k.covariance(&x, &x).is_err());
    }
    for k in families(scalar(1.), scalar(0.8), scalar(1.3)) {
        assert!(k
            .covariance(&x, &CT::new(&[[0.1f64, 0.2]], &Device::Cpu).unwrap())
            .is_err());
    }
}

#[test]
fn kernels_are_symmetric_psd_and_obey_known_values() {
    let data = [0.2, -0.4, 1.1, 0.6, 0.2, -0.4];
    let x = Tensor::from_f64_slice(&data, &[3, 2], DType::F64).unwrap();
    for kernel in references() {
        let k = kernel.covariance(&x, &x).unwrap().to_vec_f64().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((k[i * 3 + j] - k[j * 3 + i]).abs() < 1e-12);
            }
        }
        for weights in [[1., -2., 0.5], [-0.7, 0.3, 1.2], [1., 0., -1.]] {
            let mut q = 0.;
            for i in 0..3 {
                for j in 0..3 {
                    q += weights[i] * k[i * 3 + j] * weights[j];
                }
            }
            assert!(q >= -1e-10, "negative quadratic form: {q}");
        }
    }
    let a = Tensor::from_f64_slice(&[0., 1.3], &[2, 1], DType::F64).unwrap();
    let k = cpu::Periodic::new(1.7, 0.8, 1.3)
        .unwrap()
        .covariance(&a, &a)
        .unwrap()
        .to_vec_f64()
        .unwrap();
    assert!(k.iter().all(|v| (v - 1.7).abs() < 1e-12));
    let w = cpu::WhiteNoise::new(1.7)
        .unwrap()
        .covariance(&x, &x)
        .unwrap()
        .to_vec_f64()
        .unwrap();
    assert_eq!(w, vec![1.7, 0., 1.7, 0., 1.7, 0., 1.7, 0., 1.7]);
    let l = cpu::Linear::new(2.).unwrap();
    let p = cpu::Polynomial::new(2., 1., 2).unwrap();
    assert!((l.covariance(&a, &a).unwrap().to_vec_f64().unwrap()[3] - 3.38).abs() < 1e-12);
    assert!(
        (p.covariance(&a, &a).unwrap().to_vec_f64().unwrap()[3] - 2. * 2.69f64.powi(2)).abs()
            < 1e-12
    );
}

#[test]
fn candle_f32_backward_is_supported() {
    let x = Var::from_tensor(&CT::new(&[[0.2f32], [0.7]], &Device::Cpu).unwrap()).unwrap();
    let v = Var::from_tensor(&CT::new(1.7f32, &Device::Cpu).unwrap()).unwrap();
    for k in families(
        v.as_tensor().clone(),
        CT::new(0.8f32, &Device::Cpu).unwrap(),
        CT::new(1.3f32, &Device::Cpu).unwrap(),
    ) {
        let out = k.covariance(&x, &x).unwrap();
        assert_eq!(out.dtype(), inferno::candle::DType::F32);
        let grads = out.sum_all().unwrap().backward().unwrap();
        assert!(grads
            .get(&v)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_finite());
    }
}
