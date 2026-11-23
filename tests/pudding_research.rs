use inferno::pudding::{equilibrium::AtomReference, research::Study};
#[test]
fn endpoint_studies_expose_observed_mass_and_equilibrium_scope() {
    for equilibrium in [false, true] {
        let s = Study::demo(equilibrium).unwrap();
        assert_eq!(s.is_equilibrium(), equilibrium);
        assert!(s.bridge.marginal_residual <= s.tolerance);
        assert!(s.baseline_residual > 0.001);
        for (a, b) in s.observed_target().iter().zip(&s.target) {
            assert!((a - b).abs() < 1e-12);
        }
        if equilibrium {
            assert!(s.symmetry_residual() < 1e-11);
        } else {
            assert!(s.symmetry_residual() > 0.01);
        }
        assert_eq!(s.svg().matches("class=\"edge\"").count(), 36);
        assert!(s.json().contains("\"formal_certificate\":false"));
        assert!(!s.html().contains("__STUDY_"));
    }
}
#[test]
fn failed_solver_never_becomes_visual_evidence() {
    let r = AtomReference::langevin(&[0., 0.], &[vec![1., 1.], vec![1., 1.]], 1., 0.1).unwrap();
    assert!(Study::solve(&r, &[0.8, 0.2], &[0.1, 0.9], 1e-14, 1).is_err());
    assert!(Study::solve(&r, &[0.8, 0.3], &[0.1, 0.9], 1e-12, 100).is_err());
}
