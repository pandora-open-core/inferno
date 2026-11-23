use inferno::pudding::{Check, Engine, TensorSnapshot as T, Tolerance};

#[test]
fn probability_and_shape_obligations() {
    let p = T::new(vec![2], vec![0.25, 0.75]);
    let mut engine = Engine::new();
    engine
        .check(
            "posterior shape",
            Check::Shape {
                tensor: &p,
                expected: &[2],
            },
        )
        .check(
            "posterior mass",
            Check::Probability {
                tensor: &p,
                tolerance: Tolerance::new(1e-12, 0.0),
            },
        );
    assert!(engine.finish().passed());
}
#[test]
fn invalid_observations_fail_closed() {
    for t in [
        T::new(vec![2], vec![1.0]),
        T::new(vec![1], vec![f64::NAN]),
        T::new(vec![usize::MAX, 2], vec![]),
    ] {
        let mut e = Engine::new();
        e.check("invalid", Check::Finite(&t));
        assert!(!e.finish().passed());
    }
    assert!(!Engine::new().finish().passed());
}
#[test]
fn probabilities_reject_negative_empty_and_unnormalized_values() {
    for p in [vec![], vec![-0.1, 1.1], vec![0.2, 0.2], vec![f64::INFINITY]] {
        let t = T::new(vec![p.len()], p);
        let mut e = Engine::new();
        e.check(
            "mass",
            Check::Probability {
                tensor: &t,
                tolerance: Tolerance::new(1e-10, 0.0),
            },
        );
        assert!(!e.finish().passed());
    }
}
fn close(a: T, b: T, tolerance: Tolerance) -> bool {
    let mut e = Engine::new();
    e.check(
        "compare",
        Check::Close {
            actual: &a,
            expected: &b,
            tolerance,
        },
    );
    e.finish().passed()
}
#[test]
fn tolerance_handles_overflow_and_invalid_configuration() {
    let t = |v| T::new(vec![1], vec![v]);
    assert!(!close(t(f64::MAX), t(-f64::MAX), Tolerance::new(0.0, 1.0)));
    assert!(close(t(f64::MAX), t(f64::MAX), Tolerance::new(0.0, 0.0)));
    assert!(close(t(1.0), t(1.00001), Tolerance::new(0.0001, 0.0)));
    for tolerance in [
        Tolerance::new(-1.0, 0.0),
        Tolerance::new(f64::NAN, 0.0),
        Tolerance::new(0.0, f64::INFINITY),
    ] {
        assert!(!close(t(1.0), t(1.0), tolerance));
    }
    assert!(!close(
        T::new(vec![0], vec![]),
        T::new(vec![0], vec![]),
        Tolerance::new(0.0, 0.0)
    ));
    assert!(!close(
        T::new(vec![1], vec![1.0]),
        T::new(vec![1, 1], vec![1.0]),
        Tolerance::new(0.0, 0.0)
    ));
}
