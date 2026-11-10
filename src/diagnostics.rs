//! Inferno — Pudding-backed observations of tensor values, operator products and solve residuals.

use crate::candle::{DType, Device, Result, Tensor};
use crate::operators::LinearOperator;
use pudding::{Check, Engine, Report, TensorSnapshot, Tolerance};

/// Take an owned host observation of the current tensor values.
/// This takes `crate::candle::Tensor`, not the distinct root `crate::Tensor`.
pub fn snapshot(tensor: &Tensor) -> Result<TensorSnapshot> {
    Ok(TensorSnapshot::new(
        tensor.dims().to_vec(),
        tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?,
    ))
}

/// Compare the operator's matmul with its dense representation on this rhs.
/// Materializes the operator: only use for deliberately small diagnostics.
/// Agreement does not prove that either implementation is mathematically correct.
pub fn check_operator<O: LinearOperator + ?Sized>(
    operator: &O,
    rhs: &Tensor,
    tolerance: Tolerance,
) -> Result<Report> {
    let dense = operator.to_dense()?;
    let dense_snapshot = snapshot(&dense)?;
    let rhs_snapshot = snapshot(rhs)?;
    let actual = snapshot(&operator.matmul(rhs)?)?;
    let expected = snapshot(&dense.broadcast_matmul(rhs)?)?;
    let shape = operator.shape();
    let mut engine = Engine::new();
    engine
        .check(
            "operator declared shape",
            Check::Shape {
                tensor: &dense_snapshot,
                expected: &shape,
            },
        )
        .check("rhs finite", Check::Finite(&rhs_snapshot))
        .check(
            "operator matmul agreement",
            Check::Close {
                actual: &actual,
                expected: &expected,
                tolerance,
            },
        );
    Ok(engine.finish())
}

/// Check A*x against b for a supplied candidate solution, without solving again.
/// This is a residual observation, not a bound on solution or gradient error.
pub fn check_solution<O: LinearOperator + ?Sized>(
    operator: &O,
    solution: &Tensor,
    rhs: &Tensor,
    tolerance: Tolerance,
) -> Result<Report> {
    let solution_snapshot = snapshot(solution)?;
    let actual = snapshot(&operator.matmul(solution)?)?;
    let expected = snapshot(rhs)?;
    let mut engine = Engine::new();
    engine
        .check("solution finite", Check::Finite(&solution_snapshot))
        .check(
            "linear solve residual",
            Check::Close {
                actual: &actual,
                expected: &expected,
                tolerance,
            },
        );
    Ok(engine.finish())
}

#[cfg(test)]
mod tests {
    //! Verification of pudding-backed observations of tensor values, operator products and solve residuals.

    use super::*;
    use crate::operators::DenseOperator;
    #[test]
    fn checks_candle_operator_and_solution() -> Result<()> {
        let a = DenseOperator::new(Tensor::from_vec(
            vec![2f64, 0., 0., 4.],
            (2, 2),
            &Device::Cpu,
        )?);
        let b = Tensor::from_vec(vec![2f64, 8.], (2, 1), &Device::Cpu)?;
        let x = Tensor::from_vec(vec![1f64, 2.], (2, 1), &Device::Cpu)?;
        let tolerance = Tolerance::new(1e-12, 1e-12);
        assert!(check_operator(&a, &b, tolerance)?.passed());
        assert!(check_solution(&a, &x, &b, tolerance)?.passed());
        assert!(!check_solution(&a, &b, &b, tolerance)?.passed());
        Ok(())
    }
}
