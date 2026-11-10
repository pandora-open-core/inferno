//! Inferno — Custom Gaussian-process component integration and validation checks.

use inferno::{
    cache::{self, Cache, CovarianceKey, LruCache},
    covariance::{Covariance, FunctionCovariance, Matern, MaternOrder, Product, Rbf, Sum},
    gp::ExactGp,
    likelihood::{Bernoulli, Gaussian, GaussianNoise, Likelihood},
    mean::{Constant, FunctionMean, Linear, Mean, Zero},
    DType, Tensor,
};
fn t(v: &[f64], s: &[usize]) -> Tensor {
    Tensor::from_f64_slice(v, s, DType::F64).unwrap()
}
fn values(t: &Tensor) -> Vec<f64> {
    t.to_vec_f64().unwrap()
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}
#[test]
fn custom_mean_matches_centered_analytic_solution() {
    let x = t(&[0.0], &[1, 1]);
    let p = ExactGp::with_components(
        Rbf::new(2.0, 1.0).unwrap(),
        Constant(10.0),
        Gaussian::new(0.5).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &t(&[13.0], &[1]))
    .unwrap();
    near(values(&p.predict(&x).unwrap().mean)[0], 12.4);
    near(values(&p.predict_observed(&x).unwrap().variance)[0], 0.9);
    near(values(&p.posterior_covariance(&x).unwrap())[0], 0.4);
    near(
        p.log_marginal_likelihood(),
        -0.5 * (9.0 / 2.5 + 2.5_f64.ln() + (2.0 * std::f64::consts::PI).ln()),
    );
}
struct Noise;
impl GaussianNoise for Noise {
    fn variances(&self, x: &Tensor) -> inferno::Result<Vec<f64>> {
        Ok(vec![0.2; x.shape()[0]])
    }
}
#[test]
fn custom_noise_and_mean_are_usable() {
    let x = t(&[1.0, 2.0], &[2, 1]);
    let p = ExactGp::with_components(
        Rbf::new(1.0, 1.0).unwrap(),
        Linear {
            weights: vec![2.0],
            bias: 1.0,
        },
        Noise,
        0.0,
    )
    .unwrap()
    .condition(&x, &t(&[3.0, 5.0], &[2]))
    .unwrap();
    assert_eq!(values(&p.predict(&x).unwrap().mean), vec![3.0, 5.0]);
}
#[test]
fn joint_covariance_matches_closed_form_off_diagonal() {
    let x = t(&[0.0], &[1, 1]);
    let z = t(&[0.0, 1.0], &[2, 1]);
    let p = ExactGp::new(
        Rbf::new(1.0, 1.0).unwrap(),
        Gaussian::new(0.25).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &t(&[1.0], &[1]))
    .unwrap();
    let cov = values(&p.posterior_covariance(&z).unwrap());
    let r = (-0.5_f64).exp();
    near(cov[0], 0.2);
    near(cov[1], 0.2 * r);
    near(cov[2], 0.2 * r);
    near(cov[3], 1.0 - r * r / 1.25);
    assert!(cov[0] * cov[3] - cov[1] * cov[2] >= 0.0);
}
#[test]
fn jitter_is_not_observation_noise() {
    let x = t(&[0.0], &[1, 1]);
    let p = ExactGp::new(
        Rbf::new(1.0, 1.0).unwrap(),
        Gaussian::new(0.2).unwrap(),
        0.3,
    )
    .unwrap()
    .condition(&x, &t(&[0.0], &[1]))
    .unwrap();
    near(values(&p.predict(&x).unwrap().variance)[0], 1.0 - 1.0 / 1.5);
    near(
        values(&p.predict_observed(&x).unwrap().variance)[0],
        1.2 - 1.0 / 1.5,
    );
}
#[test]
fn training_order_is_irrelevant() {
    let model = || {
        ExactGp::new(
            Rbf::new(1.2, 0.7).unwrap(),
            Gaussian::new(0.1).unwrap(),
            1e-9,
        )
        .unwrap()
    };
    let a = model()
        .condition(&t(&[0.0, 1.0, 2.0], &[3, 1]), &t(&[2.0, 0.0, -1.0], &[3]))
        .unwrap();
    let b = model()
        .condition(&t(&[2.0, 0.0, 1.0], &[3, 1]), &t(&[-1.0, 2.0, 0.0], &[3]))
        .unwrap();
    let z = t(&[0.5, 1.5], &[2, 1]);
    for (a, b) in values(&a.predict(&z).unwrap().mean)
        .iter()
        .zip(values(&b.predict(&z).unwrap().mean))
    {
        near(*a, b)
    }
    near(a.log_marginal_likelihood(), b.log_marginal_likelihood());
}
#[test]
fn duplicate_inputs_are_regularized() {
    let x = t(&[0.0, 0.0], &[2, 1]);
    let p = ExactGp::new(
        Rbf::new(1.0, 1.0).unwrap(),
        Gaussian::new(0.01).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &t(&[1.0, 1.0], &[2]))
    .unwrap();
    near(values(&p.predict(&x).unwrap().mean)[0], 2.0 / 2.01);
}
#[test]
fn malformed_means_are_rejected() {
    for bad in [vec![], vec![f64::NAN]] {
        let mean = FunctionMean(move |_: &Tensor| Ok(bad.clone()));
        assert!(
            ExactGp::with_components(Rbf::new(1.0, 1.0).unwrap(), mean, Noise, 0.0)
                .unwrap()
                .condition(&t(&[0.0], &[1, 1]), &t(&[0.0], &[1]))
                .is_err()
        );
    }
    assert!(Constant(f64::INFINITY)
        .evaluate(&t(&[0.0], &[1, 1]))
        .is_err());
    assert!(Linear {
        weights: vec![],
        bias: 0.0
    }
    .evaluate(&t(&[0.0], &[1, 1]))
    .is_err());
}
#[test]
fn malformed_kernels_are_rejected() {
    let bad = FunctionCovariance(|_: &Tensor, _: &Tensor| Ok(t(&[1.0], &[1, 1])));
    assert!(ExactGp::with_components(bad, Zero, Noise, 0.0)
        .unwrap()
        .condition(&t(&[0.0, 1.0], &[2, 1]), &t(&[0.0, 1.0], &[2]))
        .is_err());
    let asymmetric =
        FunctionCovariance(|_: &Tensor, _: &Tensor| Ok(t(&[1.0, 0.0, 0.5, 1.0], &[2, 2])));
    assert!(ExactGp::with_components(asymmetric, Zero, Noise, 0.0)
        .unwrap()
        .condition(&t(&[0.0, 1.0], &[2, 1]), &t(&[0.0, 1.0], &[2]))
        .is_err());
}
#[test]
fn rejects_nonfinite_data_and_feature_mismatch() {
    let model = || {
        ExactGp::new(
            Rbf::new(1.0, 1.0).unwrap(),
            Gaussian::new(0.1).unwrap(),
            0.0,
        )
        .unwrap()
    };
    assert!(model()
        .condition(&t(&[f64::NAN], &[1, 1]), &t(&[0.0], &[1]))
        .is_err());
    assert!(model()
        .condition(&t(&[0.0], &[1, 1]), &t(&[f64::INFINITY], &[1]))
        .is_err());
    let p = model()
        .condition(&t(&[0.0], &[1, 1]), &t(&[0.0], &[1]))
        .unwrap();
    assert!(p.predict(&t(&[0.0, 1.0], &[1, 2])).is_err());
}
#[test]
fn matern_values_and_composition() {
    let x = t(&[0.0], &[1, 1]);
    let y = t(&[1.0], &[1, 1]);
    for (order, z, p) in [
        (MaternOrder::Half, 1.0, 1.0),
        (
            MaternOrder::ThreeHalves,
            3.0_f64.sqrt(),
            1.0 + 3.0_f64.sqrt(),
        ),
        (
            MaternOrder::FiveHalves,
            5.0_f64.sqrt(),
            1.0 + 5.0_f64.sqrt() + 5.0 / 3.0,
        ),
    ] {
        let k = Matern::new(2.0, vec![1.0], order).unwrap();
        near(values(&k.covariance(&x, &x).unwrap())[0], 2.0);
        near(
            values(&k.covariance(&x, &y).unwrap())[0],
            2.0 * p * (-z).exp(),
        );
    }
    let a = Rbf::new(2.0, 1.0).unwrap();
    let b = Rbf::new(3.0, 2.0).unwrap();
    near(values(&Sum(a, b).covariance(&x, &x).unwrap())[0], 5.0);
    near(values(&Product(a, b).covariance(&x, &x).unwrap())[0], 6.0);
    assert!(Matern::new(1.0, vec![0.0], MaternOrder::Half).is_err());
    assert!(Matern::new(1.0, vec![1.0, 1.0], MaternOrder::Half)
        .unwrap()
        .covariance(&x, &x)
        .is_err());
}
#[test]
fn cache_lru_replacement_eviction_and_disable() {
    let mut c = LruCache::new(2);
    c.insert(1, 10);
    c.insert(2, 20);
    assert_eq!(c.get(&1), Some(&10));
    c.insert(3, 30);
    assert_eq!(c.get(&2), None);
    c.insert(1, 11);
    assert_eq!(c.len(), 2);
    assert_eq!(c.get(&1), Some(&11));
    c.clear();
    assert!(c.is_empty());
    let mut c = LruCache::new(0);
    c.insert(1, 1);
    assert!(c.is_empty());
}
#[test]
fn cache_separates_revisions_shapes_and_data() {
    let a = t(&[0.0, 1.0], &[2, 1]);
    let b = t(&[0.0, 2.0], &[2, 1]);
    let c = t(&[0.0, 1.0], &[1, 2]);
    let key = CovarianceKey::new(0, &a, &a).unwrap();
    assert_ne!(key, CovarianceKey::new(1, &a, &a).unwrap());
    assert_ne!(key, CovarianceKey::new(0, &b, &a).unwrap());
    assert_ne!(key, CovarianceKey::new(0, &c, &c).unwrap());
}
#[test]
fn cache_computes_only_on_miss() {
    let calls = std::cell::Cell::new(0);
    let k = FunctionCovariance(|x: &Tensor, y: &Tensor| {
        calls.set(calls.get() + 1);
        Rbf::new(1.0, 1.0)?.covariance(x, y)
    });
    let x = t(&[0.0], &[1, 1]);
    let mut c = LruCache::new(2);
    for _ in 0..2 {
        cache::covariance(&mut c, &k, 0, &x, &x).unwrap();
    }
    assert_eq!(calls.get(), 1);
    cache::covariance(&mut c, &k, 1, &x, &x).unwrap();
    assert_eq!(calls.get(), 2);
}
#[test]
fn bernoulli_extreme_logits_and_normalization() {
    for z in [-1000.0, -10.0, 0.0, 10.0, 1000.0] {
        near(
            Bernoulli.log_prob(0.0, z).unwrap().exp() + Bernoulli.log_prob(1.0, z).unwrap().exp(),
            1.0,
        );
    }
    near(Bernoulli.log_prob(1.0, 1000.0).unwrap(), 0.0);
    near(Bernoulli.log_prob(0.0, 1000.0).unwrap(), -1000.0);
    assert!(Bernoulli.log_prob(0.5, 0.0).is_err());
    assert!(Bernoulli.log_prob(1.0, f64::NAN).is_err());
}

#[test]
fn finite_differences_match_analytic_gradient_and_propagate_errors() {
    use inferno::developer::finite_difference;
    let g = finite_difference(&[2.0, -3.0], 1e-5, |p| Ok(p[0] * p[0] + 2.0 * p[1] * p[1])).unwrap();
    assert!((g[0] - 4.0).abs() < 1e-8);
    assert!((g[1] + 12.0).abs() < 1e-8);
    assert!(finite_difference(&[1.0], 0.0, |_| Ok(0.0)).is_err());
    assert!(finite_difference(&[1.0], 1e-5, |_| Ok(f64::NAN)).is_err());
    assert!(finite_difference(&[1.0], 1e-5, |_| Err(inferno::Error(
        "custom failure".into()
    )))
    .is_err());
}
#[test]
fn gaussian_kl_matches_known_values() {
    use inferno::developer::gaussian_kl;
    near(gaussian_kl(3.0, 2.0, 3.0, 2.0).unwrap(), 0.0);
    near(gaussian_kl(2.0, 1.0, 0.0, 1.0).unwrap(), 2.0);
    assert!(gaussian_kl(0.0, 0.0, 0.0, 1.0).is_err());
}
#[test]
fn boxed_components_support_runtime_selection() {
    let k: Box<dyn Covariance> = Box::new(Rbf::new(1.0, 1.0).unwrap());
    let m: Box<dyn Mean> = Box::new(Constant(2.0));
    let n: Box<dyn GaussianNoise> = Box::new(Gaussian::new(0.1).unwrap());
    let x = t(&[0.0], &[1, 1]);
    let p = ExactGp::with_components(k, m, n, 0.0)
        .unwrap()
        .condition(&x, &t(&[2.0], &[1]))
        .unwrap();
    near(values(&p.predict(&x).unwrap().mean)[0], 2.0);
}

#[test]
fn noise_contract_rejects_negative_nan_and_wrong_lengths() {
    use inferno::likelihood::FunctionNoise;
    let x = t(&[0.0], &[1, 1]);
    let y = t(&[0.0], &[1]);
    for bad in [vec![], vec![-0.1], vec![f64::NAN], vec![f64::INFINITY]] {
        let n = FunctionNoise(move |_: &Tensor| Ok(bad.clone()));
        assert!(
            ExactGp::with_components(Rbf::new(1.0, 1.0).unwrap(), Zero, n, 0.0)
                .unwrap()
                .condition(&x, &y)
                .is_err()
        );
    }
    // Zero-noise deterministic conditioning is valid when K is positive definite.
    let n = FunctionNoise(|_: &Tensor| Ok(vec![0.0]));
    let p = ExactGp::with_components(Rbf::new(1.0, 1.0).unwrap(), Zero, n, 0.0)
        .unwrap()
        .condition(&x, &t(&[1.0], &[1]))
        .unwrap();
    near(values(&p.predict(&x).unwrap().mean)[0], 1.0);
    near(values(&p.predict(&x).unwrap().variance)[0], 0.0);
}
#[test]
fn composed_kernels_have_nonnegative_quadratic_forms() {
    let x = t(&[-2.0, -0.5, 0.0, 0.7, 2.0], &[5, 1]);
    for order in [
        MaternOrder::Half,
        MaternOrder::ThreeHalves,
        MaternOrder::FiveHalves,
    ] {
        let kernel = Product(
            Sum(
                Matern::new(1.5, vec![0.8], order).unwrap(),
                Rbf::new(0.7, 1.1).unwrap(),
            ),
            Rbf::new(0.8, 2.0).unwrap(),
        );
        let a = values(&kernel.covariance(&x, &x).unwrap());
        for phase in 0..20 {
            let v: Vec<_> = (0..5).map(|i| ((i * 7 + phase) as f64).sin()).collect();
            let mut q = 0.0;
            for i in 0..5 {
                for j in 0..5 {
                    near(a[i * 5 + j], a[j * 5 + i]);
                    q += v[i] * a[i * 5 + j] * v[j];
                }
            }
            assert!(q >= -1e-12);
        }
    }
}
#[test]
fn failed_cache_computations_are_not_retained() {
    let calls = std::cell::Cell::new(0);
    let bad = FunctionCovariance(|_: &Tensor, _: &Tensor| {
        calls.set(calls.get() + 1);
        Err(inferno::Error("failure".into()))
    });
    let mut c = LruCache::new(2);
    let x = t(&[0.0], &[1, 1]);
    assert!(cache::covariance(&mut c, &bad, 0, &x, &x).is_err());
    assert!(cache::covariance(&mut c, &bad, 0, &x, &x).is_err());
    assert_eq!(calls.get(), 2);
    assert!(c.is_empty());
}
#[test]
fn custom_likelihood_checks_log_density_contract() {
    use inferno::likelihood::FunctionLikelihood;
    assert!(FunctionLikelihood(|_, _| Ok(f64::NAN))
        .log_prob(0.0, 0.0)
        .is_err());
    assert!(FunctionLikelihood(|_, _| Ok(f64::INFINITY))
        .log_prob(0.0, 0.0)
        .is_err());
    assert_eq!(
        FunctionLikelihood(|_, _| Ok(f64::NEG_INFINITY))
            .log_prob(0.0, 0.0)
            .unwrap(),
        f64::NEG_INFINITY
    );
    assert!(FunctionLikelihood(|_, _| Ok(0.0))
        .log_prob(f64::NAN, 0.0)
        .is_err());
}

#[test]
fn leave_one_out_matches_brute_force_refits() {
    let model = || {
        ExactGp::with_components(
            Rbf::new(1.3, 0.8).unwrap(),
            Constant(0.7),
            Gaussian::new(0.2).unwrap(),
            0.0,
        )
        .unwrap()
    };
    let x = [-1.0, 0.2, 1.5];
    let y = [1.0, 2.0, -1.0];
    let p = model().condition(&t(&x, &[3, 1]), &t(&y, &[3])).unwrap();
    let loo = p.leave_one_out().unwrap();
    let means = values(&loo.mean);
    let variances = values(&loo.variance);
    for i in 0..3 {
        let xr: Vec<_> = (0..3).filter(|j| *j != i).map(|j| x[j]).collect();
        let yr: Vec<_> = (0..3).filter(|j| *j != i).map(|j| y[j]).collect();
        let refit = model().condition(&t(&xr, &[2, 1]), &t(&yr, &[2])).unwrap();
        let q = refit.predict_observed(&t(&[x[i]], &[1, 1])).unwrap();
        near(means[i], values(&q.mean)[0]);
        near(variances[i], values(&q.variance)[0]);
    }
}
#[test]
fn posterior_draws_transform_innovations_and_respect_jitter() {
    let x = t(&[0.0], &[1, 1]);
    let p = ExactGp::new(
        Rbf::new(2.0, 1.0).unwrap(),
        Gaussian::new(0.5).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &t(&[3.0], &[1]))
    .unwrap();
    let samples = values(
        &p.sample_latent(&x, &t(&[-1.0, 0.0, 1.0], &[3, 1]), 0.1)
            .unwrap(),
    );
    near(samples[0], 2.4 - 0.5_f64.sqrt());
    near(samples[1], 2.4);
    near(samples[2], 2.4 + 0.5_f64.sqrt());
    assert!(p.sample_latent(&x, &t(&[1.0, 2.0], &[1, 2]), 0.0).is_err());
    assert!(p.sample_latent(&x, &t(&[0.0], &[1, 1]), -1.0).is_err());
}

#[test]
fn cache_key_includes_dtype() {
    let a = Tensor::from_f64_slice(&[1.0], &[1, 1], DType::F32).unwrap();
    let b = t(&[1.0], &[1, 1]);
    assert_ne!(
        CovarianceKey::new(0, &a, &a).unwrap(),
        CovarianceKey::new(0, &b, &b).unwrap()
    );
}

#[test]
fn kernels_handle_large_coordinates_and_rescued_exponential_tails() {
    let x = t(&[f64::MAX], &[1, 1]);
    let y = t(&[-f64::MAX], &[1, 1]);
    near(
        values(&Rbf::new(1.0, f64::MAX).unwrap().covariance(&x, &y).unwrap())[0],
        (-2.0_f64).exp(),
    );
    near(
        values(
            &Matern::new(1.0, vec![f64::MAX], MaternOrder::Half)
                .unwrap()
                .covariance(&x, &y)
                .unwrap(),
        )[0],
        (-2.0_f64).exp(),
    );
    let k = Matern::new(1e308, vec![1.0], MaternOrder::Half).unwrap();
    let got = values(
        &k.covariance(&t(&[0.0], &[1, 1]), &t(&[710.0], &[1, 1]))
            .unwrap(),
    )[0];
    let expected = 1e308 * (-710.0_f64).exp();
    assert!(got > 0.0);
    assert!((got / expected - 1.0).abs() < 1e-11);
    let k = Matern::new(1e308, vec![1.0], MaternOrder::ThreeHalves).unwrap();
    let got = values(
        &k.covariance(&t(&[0.0], &[1, 1]), &t(&[1.0], &[1, 1]))
            .unwrap(),
    )[0];
    let z = 3.0_f64.sqrt();
    let expected = 1e308 * ((1.0 + z) * (-z).exp());
    assert!((got / expected - 1.0).abs() < 1e-11);
}

#[test]
fn heteroscedastic_noise_uses_training_and_prediction_locations() {
    use inferno::likelihood::FunctionNoise;
    let noise = FunctionNoise(|x: &Tensor| {
        Ok(x.to_vec_f64()?.into_iter().map(|v| 0.1 + v.abs()).collect())
    });
    let x = t(&[0.0], &[1, 1]);
    let z = t(&[2.0], &[1, 1]);
    let posterior = ExactGp::with_components(Rbf::new(1.0, 1.0).unwrap(), Zero, noise, 0.0)
        .unwrap()
        .condition(&x, &t(&[1.0], &[1]))
        .unwrap();
    let latent = posterior.predict(&z).unwrap();
    let observed = posterior.predict_observed(&z).unwrap();
    near(values(&latent.mean)[0], (-2.0_f64).exp() / 1.1);
    near(values(&latent.variance)[0], 1.0 - (-4.0_f64).exp() / 1.1);
    near(
        values(&observed.variance)[0] - values(&latent.variance)[0],
        2.1,
    );
}

#[test]
fn rbf_preserves_tails_rescued_by_large_variance() {
    let x = t(&[0.0], &[1, 1]);
    let y = t(&[1500.0_f64.sqrt()], &[1, 1]);
    let value = values(&Rbf::new(1e308, 1.0).unwrap().covariance(&x, &y).unwrap())[0];
    let expected = (1e154 * (-375.0_f64).exp()).powi(2);
    assert!(value > 0.0);
    assert!((value / expected - 1.0).abs() < 1e-11);
}

#[test]
fn cache_adapter_reuses_conditioning_covariance_during_gp_predictions() {
    use inferno::cache::CachedCovariance;
    let calls = std::cell::Cell::new(0);
    let kernel = FunctionCovariance(|x: &Tensor, y: &Tensor| {
        calls.set(calls.get() + 1);
        Rbf::new(1.0, 1.0)?.covariance(x, y)
    });
    let cached = CachedCovariance::new(kernel, LruCache::new(2), 0);
    let x = t(&[0.0], &[1, 1]);
    let posterior = ExactGp::new(cached, Gaussian::new(0.2).unwrap(), 0.0)
        .unwrap()
        .condition(&x, &t(&[1.0], &[1]))
        .unwrap();
    for _ in 0..3 {
        near(values(&posterior.predict(&x).unwrap().mean)[0], 1.0 / 1.2);
    }
    assert_eq!(calls.get(), 1);
}
#[test]
fn cache_adapter_invalidates_before_kernel_updates() {
    use inferno::cache::CachedCovariance;
    let mut kernel = CachedCovariance::lru(Rbf::new(1.0, 1.0).unwrap(), 2, 0);
    let x = t(&[0.0], &[1, 1]);
    near(values(&kernel.covariance(&x, &x).unwrap())[0], 1.0);
    *kernel.kernel_mut().unwrap() = Rbf::new(2.0, 1.0).unwrap();
    near(values(&kernel.covariance(&x, &x).unwrap())[0], 2.0);
    kernel.set_revision(1).unwrap();
    near(values(&kernel.covariance(&x, &x).unwrap())[0], 2.0);
}
