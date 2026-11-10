//! Inferno — CPU reference matrix operators and preconditioned conjugate-gradient solves.

use crate::{ensure, matrix, Result, Tensor};
pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn matvec(&self, x: &[f64]) -> Result<Vec<f64>>;
}
pub struct Dense {
    n: usize,
    values: Vec<f64>,
}
impl Dense {
    pub fn new(t: &Tensor) -> Result<Self> {
        let (n, m, values) = matrix(t)?;
        ensure(n == m && n > 0, "operator must be nonempty and square")?;
        Ok(Self { n, values })
    }
}
impl LinearOperator for Dense {
    fn dimension(&self) -> usize {
        self.n
    }
    fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        ensure(
            x.len() == self.n && x.iter().all(|v| v.is_finite()),
            "invalid operator input",
        )?;
        Ok(self
            .values
            .chunks(self.n)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect())
    }
}
/// Positive diagonal inverse: usable for Jacobi solves or gradient preconditioning.
pub struct DiagonalPreconditioner {
    inverse: Vec<f64>,
}
impl DiagonalPreconditioner {
    pub fn new(diagonal: &[f64]) -> Result<Self> {
        ensure(
            !diagonal.is_empty()
                && diagonal
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0 && v.recip().is_finite()),
            "diagonal must have finite positive reciprocals",
        )?;
        Ok(Self {
            inverse: diagonal.iter().map(|v| v.recip()).collect(),
        })
    }
    pub fn apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        ensure(
            x.len() == self.inverse.len() && x.iter().all(|v| v.is_finite()),
            "invalid preconditioner input",
        )?;
        Ok(x.iter().zip(&self.inverse).map(|(x, d)| x * d).collect())
    }
}
/// Preconditioned conjugate gradients for a symmetric positive definite operator.
/// Returns an error on breakdown or failure to meet the relative residual tolerance.
pub fn conjugate_gradient(
    a: &impl LinearOperator,
    b: &[f64],
    preconditioner: &DiagonalPreconditioner,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>> {
    ensure(
        b.len() == a.dimension()
            && b.iter().all(|v| v.is_finite())
            && tolerance.is_finite()
            && tolerance > 0.0
            && tolerance < 1.0
            && max_iterations > 0,
        "invalid CG arguments",
    )?;
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut z = preconditioner.apply(&r)?;
    let mut p = z.clone();
    let norm = dot(b, b).sqrt();
    ensure(norm.is_finite(), "CG norm overflow")?;
    if norm == 0.0 {
        return Ok(x);
    }
    let mut rz = dot(&r, &z);
    for _ in 0..max_iterations {
        let ap = a.matvec(&p)?;
        ensure(ap.len() == b.len(), "operator returned wrong dimension")?;
        let curvature = dot(&p, &ap);
        ensure(
            curvature.is_finite() && curvature > 0.0 && rz.is_finite() && rz > 0.0,
            "CG breakdown: operator/preconditioner must be positive definite",
        )?;
        let alpha = rz / curvature;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let residual = dot(&r, &r).sqrt();
        ensure(
            residual.is_finite() && x.iter().all(|v| v.is_finite()),
            "CG arithmetic overflow",
        )?;
        if residual <= tolerance * norm {
            return Ok(x);
        }
        z = preconditioner.apply(&r)?;
        let next = dot(&r, &z);
        for i in 0..p.len() {
            p[i] = z[i] + (next / rz) * p[i];
        }
        rz = next;
    }
    Err(crate::Error("CG did not converge".into()))
}
