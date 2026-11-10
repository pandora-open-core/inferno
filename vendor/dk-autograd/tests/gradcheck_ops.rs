//! Phase 2 — every curated op against central finite differences.
//!
//! Test data is chosen to stay AWAY from points where the op is not
//! differentiable or where FD straddles a kink: no zeros for `abs`/`relu`,
//! no ties for `maximum`/`minimum`/`max_dim`, strictly positive inputs for
//! `log`/`sqrt`/`rsqrt`/`recip`/`pow`, and lengthscales well above the FD
//! step for `gibbs`. A gradcheck failure at a kink is a property of the
//! test, not the gradient — so the tests do not go there.

use dk_autograd::derivatives as ops;
use dk_autograd::derivatives::{elementwise as ew, kernels as kn, reduction as rd, shape as sp};
use dk_autograd::gradcheck::{gradcheck, GradcheckCfg};
use dk_autograd::{DType, Tensor, Variable};

fn t(data: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64_slice(data, shape, DType::F64).unwrap()
}

/// Deterministic, well-conditioned values in roughly [-1.4, 1.4], never 0.
fn spread(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let v = ((i as f64 + seed) * 0.7).sin() * 1.3;
            if v.abs() < 0.15 {
                v + 0.4
            } else {
                v
            }
        })
        .collect()
}

/// Strictly positive values in roughly [0.3, 1.9].
fn positive(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 1.1 + 0.8 * ((i as f64 + seed) * 0.9).sin())
        .collect()
}

/// Scalar loss so gradcheck has a single output to weight; the harness
/// applies its own cotangent, so any reduction works — `sum_all` keeps the
/// op under test the only interesting part of the graph.
fn total(v: &Variable) -> dk_tensor::Result<Variable> {
    ops::sum_all(v)
}

// ---------------------------------------------------------------------------
// D1 — elementwise
// ---------------------------------------------------------------------------

macro_rules! check_unary {
    ($name:ident, $f:expr, $data:expr) => {
        #[test]
        fn $name() {
            let x = t(&$data, &[$data.len()]);
            gradcheck(|v| total(&$f(&v[0])?), &[x], &GradcheckCfg::default()).unwrap();
        }
    };
}

check_unary!(gradcheck_log, ew::log, positive(5, 0.0));
check_unary!(gradcheck_sqrt, ew::sqrt, positive(5, 1.0));
check_unary!(gradcheck_rsqrt, ew::rsqrt, positive(5, 2.0));
check_unary!(gradcheck_recip, ew::recip, positive(5, 3.0));
check_unary!(gradcheck_abs, ew::abs, spread(6, 0.5));
check_unary!(gradcheck_sin, ew::sin, spread(5, 1.5));
check_unary!(gradcheck_cos, ew::cos, spread(5, 2.5));
check_unary!(gradcheck_tanh, ew::tanh, spread(5, 3.5));
check_unary!(gradcheck_sigmoid, ew::sigmoid, spread(5, 4.5));
check_unary!(gradcheck_silu, ew::silu, spread(5, 5.5));
check_unary!(gradcheck_gelu_tanh, ew::gelu_tanh, spread(5, 6.5));
check_unary!(gradcheck_exp, ops::exp, spread(5, 7.5));
check_unary!(gradcheck_relu, ops::relu, spread(6, 8.5));
check_unary!(gradcheck_neg, ops::neg, spread(4, 9.5));

#[test]
fn gradcheck_powf_and_affine_and_smooth_clamp() {
    let x = t(&positive(5, 4.0), &[5]);
    gradcheck(
        |v| total(&ew::powf(&v[0], 1.7)?),
        &[x.clone()],
        &GradcheckCfg::default(),
    )
    .unwrap();
    gradcheck(
        |v| total(&ew::affine(&v[0], -2.3, 0.9)?),
        &[x.clone()],
        &GradcheckCfg::default(),
    )
    .unwrap();
    // smooth_clamp is the gradient-preserving replacement for clamp: its
    // derivative is nonzero even far outside the bounds, which is exactly
    // what a hard clamp loses.
    gradcheck(
        |v| total(&ew::smooth_clamp(&v[0], -0.5, 2.5)?),
        &[x],
        &GradcheckCfg::default(),
    )
    .unwrap();
}

/// The property smooth_clamp exists for: a hard `clamp` has EXACTLY zero
/// gradient outside its bounds, this does not — 3x past the bound the
/// gradient is small but alive.
#[test]
fn smooth_clamp_keeps_gradient_outside_the_bounds() {
    let x = Variable::leaf(t(&[3.0], &[1]), true);
    let y = ops::sum_all(&ew::smooth_clamp(&x, -1.0, 1.0).unwrap()).unwrap();
    dk_autograd::backward(&[y], None, false).unwrap();
    let g = x.grad().unwrap().to_scalar_f64().unwrap();
    assert!(g > 0.0 && g.is_finite(), "gradient died outside bounds: {g}");
}

/// ...and the limit of that property, pinned so nobody relies on it too far
/// out. `tanh` hits exactly 1.0 in f64 at 20, so past
/// `SMOOTH_CLAMP_SATURATION` the derivative `1 − tanh²` is exactly 0 and
/// smooth_clamp is no better than a hard clamp. Documented on the function;
/// asserted here so a future libm change is caught rather than assumed.
#[test]
fn smooth_clamp_saturates_past_the_documented_threshold() {
    let bound = ew::SMOOTH_CLAMP_SATURATION;
    // half = 1.0 for bounds [-1, 1], so x IS the normalised distance
    let inside = Variable::leaf(t(&[bound * 0.4], &[1]), true);
    let y = ops::sum_all(&ew::smooth_clamp(&inside, -1.0, 1.0).unwrap()).unwrap();
    dk_autograd::backward(&[y], None, false).unwrap();
    assert!(inside.grad().unwrap().to_scalar_f64().unwrap() > 0.0);

    let outside = Variable::leaf(t(&[bound * 1.5], &[1]), true);
    let y = ops::sum_all(&ew::smooth_clamp(&outside, -1.0, 1.0).unwrap()).unwrap();
    dk_autograd::backward(&[y], None, false).unwrap();
    let g = outside.grad().unwrap().to_scalar_f64().unwrap();
    assert_eq!(
        g, 0.0,
        "expected exact f64 saturation past {bound}, got {g} — if libm changed, \
         update SMOOTH_CLAMP_SATURATION and the doc comment"
    );
}

#[test]
fn gradcheck_binary_ops() {
    let a = t(&positive(6, 0.0), &[2, 3]);
    let b = t(&positive(3, 5.0), &[3]); // broadcast against a
    let cfg = GradcheckCfg::default();
    gradcheck(|v| total(&ew::div(&v[0], &v[1])?), &[a.clone(), b.clone()], &cfg).unwrap();
    gradcheck(|v| total(&ew::pow(&v[0], &v[1])?), &[a.clone(), b.clone()], &cfg).unwrap();
    // deliberately no ties: maximum/minimum are not differentiable there
    gradcheck(|v| total(&ew::maximum(&v[0], &v[1])?), &[a.clone(), b.clone()], &cfg).unwrap();
    gradcheck(|v| total(&ew::minimum(&v[0], &v[1])?), &[a, b], &cfg).unwrap();
}

// ---------------------------------------------------------------------------
// D2 — reductions
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_reductions_over_each_dim() {
    let x = t(&spread(24, 0.3), &[2, 3, 4]);
    let cfg = GradcheckCfg::default();
    for dim in 0..3 {
        gradcheck(|v| total(&rd::sum_dim(&v[0], dim)?), &[x.clone()], &cfg).unwrap();
        gradcheck(|v| total(&rd::mean_dim(&v[0], dim)?), &[x.clone()], &cfg).unwrap();
        gradcheck(|v| total(&rd::logsumexp_dim(&v[0], dim)?), &[x.clone()], &cfg).unwrap();
        gradcheck(|v| total(&rd::softmax(&v[0], dim)?), &[x.clone()], &cfg).unwrap();
        gradcheck(|v| total(&rd::log_softmax(&v[0], dim)?), &[x.clone()], &cfg).unwrap();
        gradcheck(
            |v| total(&rd::max_dim(&v[0], dim)?.0),
            &[x.clone()],
            &cfg,
        )
        .unwrap();
    }
}

#[test]
fn softmax_rows_sum_to_one_and_log_softmax_agrees() {
    let x = Variable::new(t(&spread(12, 2.0), &[3, 4]));
    let s = rd::softmax(&x, 1).unwrap();
    for row in s.tensor().to_vec_f64().unwrap().chunks(4) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row sums to {sum}");
    }
    let ls = rd::log_softmax(&x, 1).unwrap();
    for (a, b) in ls
        .tensor()
        .to_vec_f64()
        .unwrap()
        .iter()
        .zip(s.tensor().to_vec_f64().unwrap())
    {
        assert!((a.exp() - b).abs() < 1e-12);
    }
}

#[test]
fn logsumexp_is_stable_for_large_magnitudes() {
    // Naive exp() overflows here; the max-shift makes it exact.
    let x = Variable::new(t(&[800.0, 801.0, -800.0], &[3]));
    let y = rd::logsumexp_dim(&x, 0).unwrap().tensor().to_scalar_f64().unwrap();
    assert!(y.is_finite(), "logsumexp overflowed: {y}");
    let want = 801.0 + (1.0 + (-1.0f64).exp()).ln();
    assert!((y - want).abs() < 1e-9, "{y} vs {want}");
}

// ---------------------------------------------------------------------------
// D3 — shape / indexing
// ---------------------------------------------------------------------------

#[test]
fn gradcheck_shape_ops() {
    let x = t(&spread(24, 1.1), &[2, 3, 4]);
    let cfg = GradcheckCfg::default();
    gradcheck(|v| total(&sp::reshape(&v[0], &[6, 4])?), &[x.clone()], &cfg).unwrap();
    gradcheck(|v| total(&sp::transpose(&v[0], 0, 2)?), &[x.clone()], &cfg).unwrap();
    gradcheck(|v| total(&sp::permute(&v[0], &[2, 0, 1])?), &[x.clone()], &cfg).unwrap();
    gradcheck(|v| total(&sp::narrow(&v[0], 1, 1, 2)?), &[x.clone()], &cfg).unwrap();
    gradcheck(|v| total(&sp::unsqueeze(&v[0], 1)?), &[x.clone()], &cfg).unwrap();
    // repeated indices — the accumulate-vs-overwrite case
    gradcheck(
        |v| total(&sp::index_select(&v[0], 1, &[0, 2, 2, 1])?),
        &[x.clone()],
        &cfg,
    )
    .unwrap();
    gradcheck(|v| total(&sp::expand(&sp::narrow(&v[0], 1, 0, 1)?, &[2, 3, 4])?), &[x], &cfg)
        .unwrap();
}

/// The bug this pins: using the FORWARD permutation in the backward instead
/// of its inverse. It is invisible for involutions like `[1,0]`, so the test
/// uses a 3-cycle where the two differ.
#[test]
fn permute_backward_uses_the_inverse_permutation() {
    let x = Variable::leaf(t(&spread(24, 4.0), &[2, 3, 4]), true);
    // weight by a non-symmetric cotangent so a wrong axis order shows up
    let w = Variable::new(t(&spread(24, 9.0), &[4, 2, 3]));
    let p = sp::permute(&x, &[2, 0, 1]).unwrap();
    let y = ops::sum_all(&ops::mul(&p, &w).unwrap()).unwrap();
    dk_autograd::backward(&[y], None, false).unwrap();
    let got = x.grad().unwrap().to_vec_f64().unwrap();
    // dL/dx must be w permuted BACK, i.e. w.permute(inverse) == w.permute([1,2,0])
    let want = Variable::new(w.tensor().permute(&[1, 2, 0]).unwrap())
        .tensor()
        .to_vec_f64()
        .unwrap();
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn gradcheck_cat_over_multiple_inputs() {
    let a = t(&spread(6, 0.0), &[2, 3]);
    let b = t(&spread(4, 3.0), &[2, 2]);
    let c = t(&spread(2, 6.0), &[2, 1]);
    gradcheck(
        |v| total(&sp::cat(&[&v[0], &v[1], &v[2]], 1)?),
        &[a, b, c],
        &GradcheckCfg::default(),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Fused pairwise kernels
// ---------------------------------------------------------------------------

/// Kernel ops carry `exp` over a contraction, so FD noise is larger than for
/// a plain elementwise op; the step is widened and the tolerance relaxed by
/// the same order. Still ~5 significant figures of agreement.
fn kernel_cfg() -> GradcheckCfg {
    GradcheckCfg {
        eps: 1e-5,
        rtol: 1e-5,
        atol: 1e-7,
    }
}

#[test]
fn gradcheck_matern52_all_four_inputs() {
    let x1 = t(&spread(8, 0.0), &[4, 2]);
    let x2 = t(&spread(6, 3.0), &[3, 2]);
    let w = t(&positive(2, 1.0), &[2]);
    let s = t(&[1.15], &[]);
    gradcheck(
        |v| total(&kn::matern52(&v[0], &v[1], &v[2], &v[3])?),
        &[x1, x2, w, s],
        &kernel_cfg(),
    )
    .unwrap();
}

#[test]
fn gradcheck_rbf_all_three_inputs() {
    let x1 = t(&spread(8, 1.0), &[4, 2]);
    let x2 = t(&spread(6, 4.0), &[3, 2]);
    let w = t(&positive(2, 2.0), &[2]);
    gradcheck(
        |v| total(&kn::rbf(&v[0], &v[1], &v[2])?),
        &[x1, x2, w],
        &kernel_cfg(),
    )
    .unwrap();
}

/// Covers the alpha gradient specifically — it is the one that carries BOTH
/// the exponent and the alpha inside the base, and dropping the second term
/// is the classic rational-quadratic gradient bug.
#[test]
fn gradcheck_rational_quadratic_including_alpha() {
    let x1 = t(&spread(8, 2.0), &[4, 2]);
    let x2 = t(&spread(6, 5.0), &[3, 2]);
    let w = t(&positive(2, 3.0), &[2]);
    let alpha = t(&[1.8], &[]);
    gradcheck(
        |v| total(&kn::rational_quadratic(&v[0], &v[1], &v[2], &v[3])?),
        &[x1, x2, w, alpha],
        &kernel_cfg(),
    )
    .unwrap();
}

#[test]
fn gradcheck_gibbs_including_both_lengthscales() {
    let x1 = t(&spread(8, 3.0), &[4, 2]);
    let x2 = t(&spread(6, 6.0), &[3, 2]);
    let l1 = t(&positive(8, 4.0), &[4, 2]);
    let l2 = t(&positive(6, 7.0), &[3, 2]);
    gradcheck(
        |v| total(&kn::gibbs(&v[0], &v[1], &v[2], &v[3])?),
        &[x1, x2, l1, l2],
        &kernel_cfg(),
    )
    .unwrap();
}

/// All four softmin inputs, INCLUDING eps. The eps gradient is the one the
/// existing kernel loses by taking `eps: Option<f64>` — a scalar cannot
/// carry a gradient, so `raw_global_langevin_temp` never trains.
#[test]
fn gradcheck_softmin_including_eps() {
    let c = t(&positive(12, 0.0), &[3, 4]);
    let log_b = t(&spread(4, 2.0), &[4]);
    let g = t(&spread(4, 5.0), &[4]);
    let eps = t(&[0.45], &[]);
    gradcheck(
        |v| total(&kn::softmin(&v[0], &v[1], &v[2], &v[3])?),
        &[c, log_b, g, eps],
        &kernel_cfg(),
    )
    .unwrap();
}

/// eps must actually receive a nonzero gradient — a backward that returned
/// zeros would still pass a loose gradcheck if the numeric value were also
/// near zero, so assert it directly.
#[test]
fn softmin_eps_receives_a_real_gradient() {
    let c = Variable::new(t(&positive(12, 1.0), &[3, 4]));
    let log_b = Variable::new(t(&spread(4, 3.0), &[4]));
    let g = Variable::new(t(&spread(4, 6.0), &[4]));
    let eps = Variable::leaf(t(&[0.35], &[]), true);
    let out = kn::softmin(&c, &log_b, &g, &eps).unwrap();
    dk_autograd::backward(&[ops::sum_all(&out).unwrap()], None, false).unwrap();
    let ge = eps.grad().unwrap().to_scalar_f64().unwrap();
    assert!(ge.is_finite() && ge.abs() > 1e-6, "eps gradient is dead: {ge}");
}

/// Kernels composed into an NKN-style product must still gradcheck — the
/// case where one input feeds several kernels and the gradients accumulate.
#[test]
fn gradcheck_composed_kernel_product() {
    let x1 = t(&spread(8, 5.0), &[4, 2]);
    let x2 = t(&spread(6, 8.0), &[3, 2]);
    let w = t(&positive(2, 6.0), &[2]);
    let s = t(&[0.9], &[]);
    gradcheck(
        |v| {
            let m = kn::matern52(&v[0], &v[1], &v[2], &v[3])?;
            let r = kn::rbf(&v[0], &v[1], &v[2])?;
            total(&ops::mul(&m, &r)?)
        },
        &[x1, x2, w, s],
        &kernel_cfg(),
    )
    .unwrap();
}

/// End-to-end: a cost matrix built by a kernel, fed to softmin, with the
/// gradient flowing all the way back to the kernel inputs. This is the
/// actual Sinkhorn-with-a-learned-cost shape.
#[test]
fn gradcheck_kernel_into_softmin() {
    let x1 = t(&spread(6, 7.0), &[3, 2]);
    let x2 = t(&spread(8, 9.0), &[4, 2]);
    let w = t(&positive(2, 8.0), &[2]);
    let log_b = t(&spread(4, 11.0), &[4]);
    let g = t(&spread(4, 13.0), &[4]);
    let eps = t(&[0.5], &[]);
    gradcheck(
        |v| {
            // cost = 1 - rbf, a valid non-negative cost
            let k = kn::rbf(&v[0], &v[1], &v[2])?;
            let cost = ew::affine(&k, -1.0, 1.0)?;
            total(&kn::softmin(&cost, &v[3], &v[4], &v[5])?)
        },
        &[x1, x2, w, log_b, g, eps],
        &kernel_cfg(),
    )
    .unwrap();
}
