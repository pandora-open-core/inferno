//! Inferno — Numerical accuracy and failure-mode checks for CPU reference operations.

use inferno::{
    covariance::Rbf,
    gp::ExactGp,
    likelihood::Gaussian,
    operators::{conjugate_gradient, Dense, DiagonalPreconditioner},
    quadrature::normal_expectation,
    DType, Tensor,
};
fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64_slice(data, shape, DType::F64).unwrap()
}
#[test]
fn single_observation_matches_analytic_gaussian_conditioning() {
    // Prior variance 2, noise variance 0.5, y=3 => mean=2.4, variance=0.4.
    let x = tensor(&[0.0], &[1, 1]);
    let posterior = ExactGp::new(
        Rbf::new(2.0, 1.0).unwrap(),
        Gaussian::new(0.5).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &tensor(&[3.0], &[1]))
    .unwrap();
    let p = posterior.predict(&x).unwrap();
    assert!((p.mean.to_vec_f64().unwrap()[0] - 2.4).abs() < 1e-12);
    assert!((p.variance.to_vec_f64().unwrap()[0] - 0.4).abs() < 1e-12);
    let lml = -0.5 * (9.0 / 2.5 + (2.5_f64).ln() + (2.0 * std::f64::consts::PI).ln());
    assert!((posterior.log_marginal_likelihood() - lml).abs() < 1e-12);
}
#[test]
fn correlated_observations_match_closed_form_inverse() {
    let x = tensor(&[0.0, 1.0], &[2, 1]);
    let posterior = ExactGp::new(
        Rbf::new(1.0, 1.0).unwrap(),
        Gaussian::new(0.2).unwrap(),
        0.0,
    )
    .unwrap()
    .condition(&x, &tensor(&[1.0, -1.0], &[2]))
    .unwrap();
    let p = posterior.predict(&x).unwrap();
    let r = (-0.5_f64).exp();
    let expected_mean = (1.0 - r) / (1.2 - r);
    let expected_variance = 1.0 - (1.2 * (1.0 + r * r) - 2.0 * r * r) / (1.44 - r * r);
    for (i, mean) in p.mean.to_vec_f64().unwrap().iter().enumerate() {
        assert!(
            (mean
                - if i == 0 {
                    expected_mean
                } else {
                    -expected_mean
                })
            .abs()
                < 1e-12
        );
    }
    for variance in p.variance.to_vec_f64().unwrap() {
        assert!((variance - expected_variance).abs() < 1e-12);
    }
}
#[test]
fn quadrature_matches_normal_fourth_moment() {
    let value = normal_expectation(2.0, 3.0, |x| x.powi(4)).unwrap();
    assert!((value - (16.0 + 6.0 * 4.0 * 3.0 + 3.0 * 9.0)).abs() < 1e-10);
}
#[test]
fn preconditioned_solve_matches_known_solution() {
    let a = Dense::new(&tensor(&[4.0, 1.0, 1.0, 3.0], &[2, 2])).unwrap();
    let p = DiagonalPreconditioner::new(&[4.0, 3.0]).unwrap();
    let x = conjugate_gradient(&a, &[1.0, 2.0], &p, 1e-12, 10).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    assert!(conjugate_gradient(&a, &[1.0, 2.0], &p, 1e-12, 1).is_err());
}
#[test]
fn rejects_invalid_parameters_and_shapes() {
    assert!(Rbf::new(1.0, 0.0).is_err());
    assert!(Gaussian::new(f64::NAN).is_err());
    assert!(DiagonalPreconditioner::new(&[-1.0]).is_err());
    assert!(normal_expectation(0.0, -1.0, |x| x).is_err());
    let model = ExactGp::new(
        Rbf::new(1.0, 1.0).unwrap(),
        Gaussian::new(0.1).unwrap(),
        0.0,
    )
    .unwrap();
    assert!(model
        .condition(&tensor(&[0.0], &[1, 1]), &tensor(&[1.0, 2.0], &[2]))
        .is_err());
}
