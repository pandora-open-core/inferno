//! Inferno — Composable prior mean functions for CPU Gaussian-process regression.

use crate::{ensure, matrix, Result, Tensor};
pub trait Mean {
    /// Return one finite mean per input row.
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>>;
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Zero;
impl Mean for Zero {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        Ok(vec![0.0; matrix(inputs)?.0])
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Constant(pub f64);
impl Mean for Constant {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        ensure(self.0.is_finite(), "mean must be finite")?;
        Ok(vec![self.0; matrix(inputs)?.0])
    }
}
#[derive(Clone, Debug)]
pub struct Linear {
    pub weights: Vec<f64>,
    pub bias: f64,
}
impl Mean for Linear {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        let (n, d, x) = matrix(inputs)?;
        ensure(d > 0 && self.weights.len() == d, "mean dimension mismatch")?;
        ensure(
            self.bias.is_finite() && self.weights.iter().all(|v| v.is_finite()),
            "mean parameters must be finite",
        )?;
        let values: Vec<_> = (0..n)
            .map(|i| self.bias + (0..d).map(|j| x[i * d + j] * self.weights[j]).sum::<f64>())
            .collect();
        ensure(values.iter().all(|v| v.is_finite()), "mean overflow")?;
        Ok(values)
    }
}
/// Adapter for a fallible user-defined mean without a named struct.
pub struct FunctionMean<F>(pub F);
impl<F: Fn(&Tensor) -> Result<Vec<f64>>> Mean for FunctionMean<F> {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        (self.0)(inputs)
    }
}

impl<T: Mean + ?Sized> Mean for Box<T> {
    fn evaluate(&self, inputs: &Tensor) -> Result<Vec<f64>> {
        (**self).evaluate(inputs)
    }
}
