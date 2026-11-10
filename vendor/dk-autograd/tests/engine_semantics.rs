//! PyTorch-semantics tests for the ported engine, hand-picked from the
//! scenarios test_autograd.py exercises: diamond graphs, grad accumulation,
//! `.grad()` pruning + non-pollution, retain_graph, inplace-version errors,
//! saved-output (exp) unpacking, broadcast backward, and gradcheck itself.

use dk_autograd::derivatives as ops;
use dk_autograd::gradcheck::{gradcheck, GradcheckCfg};
use dk_autograd::{backward, grad, grad_mode, DType, Tensor, Variable};

fn leaf(data: &[f64], shape: &[usize]) -> Variable {
    Variable::leaf(
        Tensor::from_f64_slice(data, shape, DType::F64).unwrap(),
        true,
    )
}

fn assert_close(t: &Tensor, want: &[f64]) {
    let got = t.to_vec_f64().unwrap();
    assert_eq!(got.len(), want.len(), "length mismatch: {got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-9, "got {got:?}, want {want:?}");
    }
}

#[test]
fn simple_chain_backward() {
    // y = sum(a * b); dy/da = b, dy/db = a
    let a = leaf(&[1., 2., 3.], &[3]);
    let b = leaf(&[4., 5., 6.], &[3]);
    let y = ops::sum_all(&ops::mul(&a, &b).unwrap()).unwrap();
    backward(&[y], None, false).unwrap();
    assert_close(&a.grad().unwrap(), &[4., 5., 6.]);
    assert_close(&b.grad().unwrap(), &[1., 2., 3.]);
}

#[test]
fn diamond_graph_accumulates_both_paths() {
    // y = sum(x*x + x*x) — x reached via two mul nodes AND each mul uses x
    // twice; total dy/dx = 4x.
    let x = leaf(&[1., 2.], &[2]);
    let p1 = ops::mul(&x, &x).unwrap();
    let p2 = ops::mul(&x, &x).unwrap();
    let y = ops::sum_all(&ops::add(&p1, &p2).unwrap()).unwrap();
    backward(&[y], None, false).unwrap();
    assert_close(&x.grad().unwrap(), &[4., 8.]);
}

#[test]
fn grad_accumulates_across_backward_calls() {
    let x = leaf(&[3.], &[1]);
    let y1 = ops::sum_all(&ops::mul(&x, &x).unwrap()).unwrap();
    backward(&[y1], None, false).unwrap();
    let y2 = ops::sum_all(&ops::mul(&x, &x).unwrap()).unwrap();
    backward(&[y2], None, false).unwrap();
    // 2x + 2x = 12
    assert_close(&x.grad().unwrap(), &[12.]);
}

#[test]
fn non_scalar_backward_requires_explicit_grad() {
    let x = leaf(&[1., 2.], &[2]);
    let y = ops::mul(&x, &x).unwrap();
    let err = backward(&[y.clone()], None, false).unwrap_err();
    assert!(err.to_string().contains("scalar"), "{err}");
    backward(
        &[y],
        Some(vec![Tensor::from_f64_slice(&[1., 1.], &[2], DType::F64).unwrap()]),
        false,
    )
    .unwrap();
    assert_close(&x.grad().unwrap(), &[2., 4.]);
}

#[test]
fn grad_does_not_touch_leaf_grads() {
    // The capture-only path: grad() must return dy/dx without writing .grad.
    let x = leaf(&[2.], &[1]);
    let y = ops::sum_all(&ops::mul(&x, &x).unwrap()).unwrap();
    let gs = grad(&[y], &[x.clone()], None, false).unwrap();
    assert_close(gs[0].as_ref().unwrap(), &[4.]);
    assert!(x.grad().is_none(), ".grad polluted by grad()");
}

#[test]
fn grad_prunes_unneeded_subgraph() {
    // y = sum(a*b + c*c); grad wrt a only — the c*c branch must not run.
    // Observable effect: c's grad is untouched AND the result is correct.
    let a = leaf(&[2.], &[1]);
    let b = leaf(&[5.], &[1]);
    let c = leaf(&[7.], &[1]);
    let y = ops::sum_all(
        &ops::add(&ops::mul(&a, &b).unwrap(), &ops::mul(&c, &c).unwrap()).unwrap(),
    )
    .unwrap();
    let gs = grad(&[y], &[a.clone()], None, false).unwrap();
    assert_close(gs[0].as_ref().unwrap(), &[5.]);
    assert!(a.grad().is_none());
    assert!(c.grad().is_none());
}

#[test]
fn second_backward_without_retain_fails_cleanly() {
    let x = leaf(&[1.5], &[1]);
    let y = ops::sum_all(&ops::exp(&x).unwrap()).unwrap();
    backward(&[y.clone()], None, false).unwrap();
    let err = backward(&[y], None, false).unwrap_err();
    assert!(
        err.to_string().contains("second time") || err.to_string().contains("retain"),
        "unexpected error: {err}"
    );
}

#[test]
fn retain_graph_allows_second_backward() {
    let x = leaf(&[1.5], &[1]);
    let y = ops::sum_all(&ops::exp(&x).unwrap()).unwrap();
    backward(&[y.clone()], None, true).unwrap();
    backward(&[y], None, false).unwrap();
    let e = (1.5f64).exp();
    assert_close(&x.grad().unwrap(), &[2.0 * e]);
}

#[test]
fn inplace_mutation_of_saved_input_errors() {
    let x = leaf(&[1., 2.], &[2]);
    let y = ops::sum_all(&ops::mul(&x, &x).unwrap()).unwrap();
    // simulate an in-place op on x's data (any view bump counts)
    x.tensor().bump_version();
    let err = backward(&[y], None, false).unwrap_err();
    assert!(
        err.to_string().contains("modified by an inplace operation"),
        "unexpected error: {err}"
    );
}

#[test]
fn no_grad_mode_records_nothing() {
    let x = leaf(&[2.], &[1]);
    let y = {
        let _g = grad_mode::no_grad();
        ops::mul(&x, &x).unwrap()
    };
    assert!(y.grad_fn().is_none());
    assert!(!y.requires_grad());
}

#[test]
fn broadcast_backward_reduces_correctly() {
    // a: [2,3], b: [3] -> dy/db must sum over the broadcast dim.
    let a = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let b = leaf(&[10., 20., 30.], &[3]);
    let y = ops::sum_all(&ops::add(&a, &b).unwrap()).unwrap();
    backward(&[y], None, false).unwrap();
    assert_close(&b.grad().unwrap(), &[2., 2., 2.]);
    assert_close(&a.grad().unwrap(), &[1., 1., 1., 1., 1., 1.]);
}

#[test]
fn mm_backward_matches_manual() {
    let a = leaf(&[1., 2., 3., 4.], &[2, 2]);
    let b = leaf(&[5., 6., 7., 8.], &[2, 2]);
    let y = ops::sum_all(&ops::mm(&a, &b).unwrap()).unwrap();
    backward(&[y], None, false).unwrap();
    // dY = ones; dA = 1 @ B^T ; dB = A^T @ 1
    assert_close(&a.grad().unwrap(), &[11., 15., 11., 15.]);
    assert_close(&b.grad().unwrap(), &[4., 4., 6., 6.]);
}

#[test]
fn relu_masks_gradient() {
    let x = leaf(&[-1., 2., -3., 4.], &[4]);
    let y = ops::sum_all(&ops::relu(&x).unwrap()).unwrap();
    backward(&[y], None, false).unwrap();
    assert_close(&x.grad().unwrap(), &[0., 1., 0., 1.]);
}

#[test]
fn gradcheck_passes_on_composite() {
    // f(a, b) = sum(exp(a) * b + a*a) — exercises saved-output, saved-input
    // and accumulation together, against central differences.
    let a = Tensor::from_f64_slice(&[0.3, -0.7, 1.1], &[3], DType::F64).unwrap();
    let b = Tensor::from_f64_slice(&[2.0, 0.5, -1.5], &[3], DType::F64).unwrap();
    gradcheck(
        |vs| {
            let e = ops::exp(&vs[0])?;
            let t1 = ops::mul(&e, &vs[1])?;
            let t2 = ops::mul(&vs[0], &vs[0])?;
            ops::sum_all(&ops::add(&t1, &t2)?)
        },
        &[a, b],
        &GradcheckCfg::default(),
    )
    .unwrap();
}

#[test]
fn gradcheck_catches_a_wrong_gradient() {
    // relu's kink at 0 makes FD disagree if evaluated ACROSS the kink; use
    // a deliberately wrong composite instead: claim d(mul)/da = b via mul,
    // but feed mismatched forward (a*b vs backward of a+b) by abusing sub.
    // Simplest honest check: f = sum(mul(a,a)) but perturb the analytic
    // path by comparing against sum(mul(a, a.detach())) which drops one
    // product-rule term.
    let a = Tensor::from_f64_slice(&[0.4, -1.2], &[2], DType::F64).unwrap();
    let err = gradcheck(
        |vs| {
            let detached = vs[0].detach();
            ops::sum_all(&ops::mul(&vs[0], &detached)?)
        },
        &[a],
        &GradcheckCfg::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("gradcheck failed"), "{err}");
}
