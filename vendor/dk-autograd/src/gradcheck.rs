//! Finite-difference gradient checking, modeled on torch.autograd.gradcheck.
//!
//! Central differences in f64 (f32 central FD is hopeless — the same
//! conclusion the pandorapurple symbolic-layer work reached). Verifies
//! J^T v against the analytic backward for a fixed non-degenerate cotangent
//! v, per input, per element.

use dk_tensor::{DType, Tensor};

use crate::engine;
use crate::node::{Error, Result};
use crate::variable::Variable;

pub struct GradcheckCfg {
    pub eps: f64,
    pub rtol: f64,
    pub atol: f64,
}

impl Default for GradcheckCfg {
    fn default() -> Self {
        GradcheckCfg {
            eps: 1e-4,
            rtol: 1e-6,
            atol: 1e-8,
        }
    }
}

/// Check `f`'s analytic gradients against central finite differences.
/// `inputs` are treated as requires-grad leaves; every input must be F64.
pub fn gradcheck<F>(f: F, inputs: &[Tensor], cfg: &GradcheckCfg) -> Result<()>
where
    F: Fn(&[Variable]) -> Result<Variable>,
{
    for (i, t) in inputs.iter().enumerate() {
        if t.dtype() != DType::F64 {
            return Err(Error::Msg(format!(
                "gradcheck: input {i} must be f64 (got {})",
                t.dtype()
            )));
        }
    }

    let vars: Vec<Variable> = inputs
        .iter()
        .map(|t| Variable::leaf(t.clone(), true))
        .collect();
    let y = f(&vars)?;
    let y_shape = y.tensor().shape().to_vec();
    let y_n = y.tensor().elem_count();

    // fixed, non-degenerate cotangent
    let v: Vec<f64> = (0..y_n).map(|i| ((i + 1) as f64 * 0.7).sin() + 0.1).collect();
    let v_t = Tensor::from_f64_slice(&v, &y_shape, DType::F64)?;

    let analytic = engine::grad(&[y], &vars, Some(vec![v_t]), false)?;

    for (k, input) in inputs.iter().enumerate() {
        let base = input.to_vec_f64()?;
        let a_grad = match &analytic[k] {
            Some(g) => g.to_vec_f64()?,
            None => vec![0.0; base.len()],
        };
        for j in 0..base.len() {
            let mut plus = base.clone();
            plus[j] += cfg.eps;
            let mut minus = base.clone();
            minus[j] -= cfg.eps;

            let yp = eval_no_grad(&f, inputs, k, &plus)?;
            let ym = eval_no_grad(&f, inputs, k, &minus)?;

            let mut num = 0.0f64;
            for i in 0..y_n {
                num += v[i] * (yp[i] - ym[i]) / (2.0 * cfg.eps);
            }

            let ana = a_grad[j];
            let tol = cfg.atol + cfg.rtol * num.abs().max(ana.abs());
            if (num - ana).abs() > tol {
                return Err(Error::Msg(format!(
                    "gradcheck failed: input {k} elem {j}: numeric {num:.9e} vs analytic {ana:.9e} (tol {tol:.3e})"
                )));
            }
        }
    }
    Ok(())
}

fn eval_no_grad<F>(f: &F, inputs: &[Tensor], k: usize, replaced: &[f64]) -> Result<Vec<f64>>
where
    F: Fn(&[Variable]) -> Result<Variable>,
{
    let _guard = crate::grad_mode::no_grad();
    let vars: Result<Vec<Variable>> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let t = if i == k {
                Tensor::from_f64_slice(replaced, t.shape(), DType::F64)?
            } else {
                t.clone()
            };
            Ok(Variable::new(t))
        })
        .collect();
    f(&vars?)?.tensor().to_vec_f64()
}
