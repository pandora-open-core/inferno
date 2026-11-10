//! Inferno — Stable softplus transforms and inverses for constrained parameters.

use candle_core::{Result, Tensor};

pub fn get_softplus(x: &Tensor) -> Result<Tensor> {
    // The shift is numerical stabilization, not part of the derivative.
    // Differentiating max(x, 0) + log1p(exp(-abs(x))) directly can lose the
    // derivative at zero under Candle's max/abs subgradient conventions.
    let shift = x.maximum(&x.zeros_like()?)?.detach();
    let log_part = x.sub(&shift)?.exp()?.add(&shift.neg()?.exp()?)?.log()?;
    shift.add(&log_part)
}

#[cfg(test)]
mod tests {
    //! Verification of stable softplus transforms and inverses for constrained parameters.

    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn values_and_gradients_match_softplus_including_zero_and_extreme_tails() -> Result<()> {
        let values = [-1000.0f64, -1.0, 0.0, 1.0, 1000.0];
        let x = Var::new(&values, &Device::Cpu)?;
        let y = get_softplus(&x)?;
        let gradients = y.sum_all()?.backward()?;
        let actual = y.to_vec1::<f64>()?;
        let grad = gradients
            .get(&x)
            .expect("softplus gradient")
            .to_vec1::<f64>()?;
        for (i, &v) in values.iter().enumerate() {
            let expected = v.max(0.0) + (-v.abs()).exp().ln_1p();
            let sigmoid = if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                v.exp() / (1.0 + v.exp())
            };
            assert!((actual[i] - expected).abs() < 1e-10, "value at {v}");
            assert!((grad[i] - sigmoid).abs() < 1e-10, "gradient at {v}");
        }
        Ok(())
    }
}
