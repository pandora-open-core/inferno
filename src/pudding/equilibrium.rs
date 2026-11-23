//! Finite atom-support Langevin reference and balanced Schrödinger endpoint law.
//! Floating-point diagnostics, not proof certificates. See proofs/spec/schrodinger-equilibrium.md.

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidInput,
    NumericalRange,
    NotConverged,
}

/// Frozen reversible reference. Rows of `transition` are conditional probabilities.
#[derive(Debug, Clone)]
pub struct AtomReference {
    invariant: Vec<f64>,
    transition: Vec<Vec<f64>>,
}

/// A balanced endpoint coupling and its source-conditional stochastic return.
#[derive(Debug, Clone)]
pub struct Bridge {
    pub coupling: Vec<Vec<f64>>,
    pub transition: Vec<Vec<f64>>,
    pub marginal_residual: f64,
    pub iterations: usize,
}

fn probability(p: &[f64]) -> bool {
    !p.is_empty()
        && p.iter().all(|x| x.is_finite() && *x > 0.0)
        && (p.iter().sum::<f64>() - 1.0).abs() <= 1e-12
}

impl AtomReference {
    pub fn invariant(&self) -> &[f64] {
        &self.invariant
    }
    pub fn transition(&self) -> &[Vec<f64>] {
        &self.transition
    }

    /// For log Gibbs weights U_i = log pi0_i + beta R_i / 2, use graph rates
    /// Q_ij = D w_ij exp((U_j-U_i)/2). Symmetric positive w gives detailed balance.
    /// K=I+dt Q is a stochastic, lazy uniformisation step, not an Euler SDE step.
    /// `dt` must leave every diagonal strictly positive. Complete support is required.
    pub fn langevin(
        log_weights: &[f64],
        conductance: &[Vec<f64>],
        diffusion: f64,
        dt: f64,
    ) -> Result<Self, Error> {
        let n = log_weights.len();
        if n == 0
            || !log_weights.iter().all(|x| x.is_finite())
            || !diffusion.is_finite()
            || diffusion <= 0.0
            || !dt.is_finite()
            || dt <= 0.0
            || conductance.len() != n
            || conductance.iter().any(|r| r.len() != n)
        {
            return Err(Error::InvalidInput);
        }
        let max = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut invariant: Vec<_> = log_weights.iter().map(|u| (u - max).exp()).collect();
        let z: f64 = invariant.iter().sum();
        invariant.iter_mut().for_each(|x| *x /= z);
        if !probability(&invariant) {
            return Err(Error::NumericalRange);
        }
        let mut transition = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut exit = 0.0;
            for j in 0..n {
                let w = conductance[i][j];
                if !w.is_finite() || w < 0.0 || w != conductance[j][i] || (i != j && w == 0.0) {
                    return Err(Error::InvalidInput);
                }
                if i != j {
                    let rate = dt * diffusion * w * ((log_weights[j] - log_weights[i]) / 2.0).exp();
                    if !rate.is_finite() || rate <= 0.0 {
                        return Err(Error::NumericalRange);
                    }
                    transition[i][j] = rate;
                    exit += rate;
                }
            }
            if !exit.is_finite() || exit >= 1.0 {
                return Err(Error::InvalidInput);
            }
            transition[i][i] = 1.0 - exit;
        }
        Ok(Self {
            invariant,
            transition,
        })
    }

    /// Minimise KL(P | diag(mu) K) under endpoint marginals a,b by Sinkhorn scaling.
    /// Equal endpoints give stationarity. Reversibility of this reference additionally
    /// gives symmetry at convergence and detailed balance of P_ij/a_i.
    pub fn bridge(
        &self,
        a: &[f64],
        b: &[f64],
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Bridge, Error> {
        let n = self.invariant.len();
        if !probability(&self.invariant)
            || !probability(a)
            || !probability(b)
            || a.len() != n
            || b.len() != n
            || !tolerance.is_finite()
            || tolerance <= 0.0
            || tolerance > 1e-6
            || max_iterations == 0
            || self.transition.len() != n
            || self.transition.iter().any(|r| {
                r.len() != n
                    || r.iter().any(|x| !x.is_finite() || *x <= 0.0)
                    || (r.iter().sum::<f64>() - 1.0).abs() > 1e-12
            })
        {
            return Err(Error::InvalidInput);
        }
        let reference: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                self.transition[i]
                    .iter()
                    .map(|k| self.invariant[i] * k)
                    .collect()
            })
            .collect();
        let mut v = vec![1.0; n];
        let mut u = vec![1.0; n];
        for iteration in 1..=max_iterations {
            for i in 0..n {
                u[i] = a[i] / (0..n).map(|j| reference[i][j] * v[j]).sum::<f64>();
            }
            for j in 0..n {
                v[j] = b[j] / (0..n).map(|i| reference[i][j] * u[i]).sum::<f64>();
            }
            if u.iter().chain(&v).any(|x| !x.is_finite() || *x <= 0.0) {
                return Err(Error::NumericalRange);
            }
            let coupling: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| u[i] * reference[i][j] * v[j]).collect())
                .collect();
            if coupling
                .iter()
                .flatten()
                .any(|x| !x.is_finite() || *x <= 0.0)
            {
                return Err(Error::NumericalRange);
            }
            let mut residual = 0.0_f64;
            for i in 0..n {
                residual = residual.max((coupling[i].iter().sum::<f64>() - a[i]).abs());
                residual = residual.max(((0..n).map(|j| coupling[j][i]).sum::<f64>() - b[i]).abs());
            }
            if residual <= tolerance {
                let transition = (0..n)
                    .map(|i| coupling[i].iter().map(|p| p / a[i]).collect())
                    .collect();
                return Ok(Bridge {
                    coupling,
                    transition,
                    marginal_residual: residual,
                    iterations: iteration,
                });
            }
        }
        Err(Error::NotConverged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference() -> AtomReference {
        AtomReference::langevin(&[0.0, -0.7, 0.4], &vec![vec![1.0; 3]; 3], 1.0, 0.1).unwrap()
    }
    #[test]
    fn langevin_detailed_balance_and_reference_fixed_point() {
        let r = reference();
        let b = r.bridge(&r.invariant, &r.invariant, 1e-13, 100).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r.invariant[i] * r.transition[i][j] - r.invariant[j] * r.transition[j][i])
                        .abs()
                        < 1e-14
                );
                assert!((b.transition[i][j] - r.transition[i][j]).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn schrodinger_equal_endpoints_are_reversible_and_stationary() {
        let r = reference();
        let a = [0.2, 0.5, 0.3];
        let b = r.bridge(&a, &a, 1e-14, 10000).unwrap();
        for i in 0..3 {
            let next: f64 = (0..3).map(|j| a[j] * b.transition[j][i]).sum();
            assert!((next - a[i]).abs() < 1e-13);
            assert!((b.transition[i].iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for j in 0..3 {
                assert!((b.coupling[i][j] - b.coupling[j][i]).abs() < 1e-13);
            }
        }
    }
    #[test]
    fn unequal_endpoints_transport_instead_of_equilibrium() {
        let r = reference();
        let a = [0.2, 0.5, 0.3];
        let target = [0.6, 0.1, 0.3];
        let b = r.bridge(&a, &target, 1e-14, 10000).unwrap();
        for (j, expected) in target.iter().enumerate() {
            let next: f64 = (0..3).map(|i| a[i] * b.transition[i][j]).sum();
            assert!((next - expected).abs() < 1e-13);
        }
        assert!((b.coupling[0][1] - b.coupling[1][0]).abs() > 1e-3);
    }
    #[test]
    fn rejects_bad_domains_and_nonconvergence() {
        let r = reference();
        for a in [
            &[0.0, 0.5, 0.5][..],
            &[f64::NAN, 0.5, 0.5],
            &[0.1, 0.2, 0.3],
        ] {
            assert!(r.bridge(a, &r.invariant, 1e-12, 100).is_err());
        }
        assert_eq!(
            r.bridge(&[0.2, 0.5, 0.3], &[0.6, 0.1, 0.3], 1e-14, 1)
                .unwrap_err(),
            Error::NotConverged
        );
        assert!(AtomReference::langevin(&[0.0, 1.0], &vec![vec![1.0; 2]; 2], 1.0, 10.0).is_err());
        assert!(
            AtomReference::langevin(&[0.0, 10000.0], &vec![vec![1.0; 2]; 2], 1.0, 0.1).is_err()
        );
        assert!(
            AtomReference::langevin(&[0.0, 1.0], &[vec![0.0, 1.0], vec![2.0, 0.0]], 1.0, 0.1)
                .is_err()
        );
    }
}
