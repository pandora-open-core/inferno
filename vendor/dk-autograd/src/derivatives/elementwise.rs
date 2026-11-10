//! D1 — elementwise ops, from derivatives.yaml (pytorch @ 540f3a53).
//!
//! Nearly all of these differ only in a scalar local derivative, so they go
//! through two generic nodes rather than one struct apiece. The derivative
//! is a plain `fn` pointer, not a boxed closure: no allocation, and
//! `Send + Sync` comes free (a `Box<dyn Fn>` would need explicit bounds and
//! is the kind of thing that breaks the `Node: Send + Sync` bound at a
//! distance).
//!
//! Convention: `UnaryBackward` saves the INPUT and expresses the derivative
//! as `f'(x)`, even where `f'` is cheaper in terms of the output (tanh,
//! sigmoid). Recomputing one `exp` in backward is worth having a single
//! uniform mechanism. `exp` itself keeps its hand-written save-the-output
//! node in the parent module, so the weak-grad_fn path stays exercised.

use std::sync::Arc;

use dk_tensor::{cpu_ops, Tensor};

use super::{meta_for, should_record};
use crate::node::{Node, NodeMeta, Result};
use crate::saved_variable::SavedVariable;
use crate::variable::Variable;

// ---------------------------------------------------------------------------
// generic machinery
// ---------------------------------------------------------------------------

struct UnaryBackward {
    meta: NodeMeta,
    input: SavedVariable,
    /// f'(x)
    d: fn(f64) -> f64,
}

impl Node for UnaryBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("unary: missing grad");
        let x = self.input.unpack_data()?;
        let local = cpu_ops::unary_map(&x, self.d)?;
        Ok(vec![Some(cpu_ops::mul(&g, &local)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

fn unary(
    name: &'static str,
    a: &Variable,
    f: fn(f64) -> f64,
    d: fn(f64) -> f64,
) -> Result<Variable> {
    let out = cpu_ops::unary_map(a.tensor(), f)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(UnaryBackward {
        meta: meta_for(name, &[a], &out),
        input: SavedVariable::save(a, false),
        d,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

struct BinaryBackward {
    meta: NodeMeta,
    a: SavedVariable,
    b: SavedVariable,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    /// (∂f/∂a, ∂f/∂b) at (a, b)
    d: fn(f64, f64) -> (f64, f64),
}

impl Node for BinaryBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("binary: missing grad");
        let a = self.a.unpack_data()?;
        let b = self.b.unpack_data()?;
        let da = cpu_ops::binary_map("d/da", &a, &b, |x, y| (self.d)(x, y).0)?;
        let db = cpu_ops::binary_map("d/db", &a, &b, |x, y| (self.d)(x, y).1)?;
        Ok(vec![
            Some(cpu_ops::sum_to_shape(&cpu_ops::mul(&g, &da)?, &self.a_shape)?),
            Some(cpu_ops::sum_to_shape(&cpu_ops::mul(&g, &db)?, &self.b_shape)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

fn binary(
    name: &'static str,
    a: &Variable,
    b: &Variable,
    f: fn(f64, f64) -> f64,
    d: fn(f64, f64) -> (f64, f64),
) -> Result<Variable> {
    let out = cpu_ops::binary_map(name, a.tensor(), b.tensor(), f)?;
    if !should_record(&[a, b]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(BinaryBackward {
        meta: meta_for(name, &[a, b], &out),
        a: SavedVariable::save(a, false),
        b: SavedVariable::save(b, false),
        a_shape: a.tensor().shape().to_vec(),
        b_shape: b.tensor().shape().to_vec(),
        d,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// unary ops
// ---------------------------------------------------------------------------

pub fn log(a: &Variable) -> Result<Variable> {
    unary("LogBackward", a, f64::ln, |x| 1.0 / x)
}

pub fn sqrt(a: &Variable) -> Result<Variable> {
    // yaml: "grad / (2 * result)" — equivalent, expressed in x here.
    unary("SqrtBackward", a, f64::sqrt, |x| 0.5 / x.sqrt())
}

pub fn rsqrt(a: &Variable) -> Result<Variable> {
    unary(
        "RsqrtBackward",
        a,
        |x| 1.0 / x.sqrt(),
        |x| -0.5 * x.powf(-1.5),
    )
}

pub fn recip(a: &Variable) -> Result<Variable> {
    unary("RecipBackward", a, |x| 1.0 / x, |x| -1.0 / (x * x))
}

/// `abs`. The subgradient at 0 is 0 — PyTorch's `sgn(0) == 0` convention.
/// Finite differences straddle the kink there, so gradcheck must avoid 0.
pub fn abs(a: &Variable) -> Result<Variable> {
    unary("AbsBackward", a, f64::abs, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

pub fn sin(a: &Variable) -> Result<Variable> {
    unary("SinBackward", a, f64::sin, f64::cos)
}

pub fn cos(a: &Variable) -> Result<Variable> {
    unary("CosBackward", a, f64::cos, |x| -x.sin())
}

pub fn tanh(a: &Variable) -> Result<Variable> {
    unary("TanhBackward", a, f64::tanh, |x| {
        let t = x.tanh();
        1.0 - t * t
    })
}

fn sigmoid_scalar(x: f64) -> f64 {
    // Branch on sign so exp never sees a large positive argument.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn sigmoid(a: &Variable) -> Result<Variable> {
    unary("SigmoidBackward", a, sigmoid_scalar, |x| {
        let s = sigmoid_scalar(x);
        s * (1.0 - s)
    })
}

/// SiLU / swish: `x·σ(x)`, derivative `σ(x)(1 + x(1−σ(x)))`.
pub fn silu(a: &Variable) -> Result<Variable> {
    unary(
        "SiluBackward",
        a,
        |x| x * sigmoid_scalar(x),
        |x| {
            let s = sigmoid_scalar(x);
            s * (1.0 + x * (1.0 - s))
        },
    )
}

/// GELU, **tanh approximation** (PyTorch's `approximate="tanh"`), NOT the
/// exact erf form: Rust's std has no `erf`, and pulling in a special-function
/// dependency for one op is not worth it. Named and documented so nobody
/// assumes exact-GELU parity with a torch default.
pub fn gelu_tanh(a: &Variable) -> Result<Variable> {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    const K: f64 = 0.044715;
    unary(
        "GeluTanhBackward",
        a,
        |x| 0.5 * x * (1.0 + (C * (x + K * x * x * x)).tanh()),
        |x| {
            let inner = C * (x + K * x * x * x);
            let t = inner.tanh();
            let dinner = C * (1.0 + 3.0 * K * x * x);
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
        },
    )
}

/// `x^p` for a CONSTANT exponent. The tensor-exponent case is
/// [`super::kernels`]-adjacent and lives with the ops that need it.
pub fn powf(a: &Variable, p: f64) -> Result<Variable> {
    struct PowfBackward {
        meta: NodeMeta,
        input: SavedVariable,
        p: f64,
    }
    impl Node for PowfBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("powf: missing grad");
            let x = self.input.unpack_data()?;
            let p = self.p;
            let local = cpu_ops::unary_map(&x, move |v| p * v.powf(p - 1.0))?;
            Ok(vec![Some(cpu_ops::mul(&g, &local)?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = cpu_ops::unary_map(a.tensor(), |v| v.powf(p))?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(PowfBackward {
        meta: meta_for("PowfBackward", &[a], &out),
        input: SavedVariable::save(a, false),
        p,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// Smooth saturating bound (the `smooth_clamp` the coupling kernel uses):
/// `mid + half·tanh((x−mid)/half)`. Analytically its gradient is nonzero
/// everywhere, which is why the GEOC path prefers it over `clamp` — but
/// that is an ANALYTIC statement, not a numerical one.
///
/// LIMIT: `tanh` reaches exactly 1.0 in f64 at argument 20, so the
/// derivative `1 − tanh²` underflows to exactly 0 once
/// `|x − mid| / half ≥ 20` (and by ≈9 it is already below 1e-7). Past that
/// point this is as dead as a hard clamp. Keep the bounds within roughly an
/// order of magnitude of the data, or the "gradient-preserving" property is
/// nominal only. `SMOOTH_CLAMP_SATURATION` records the threshold and
/// `smooth_clamp_saturates_past_the_documented_threshold` pins it.
/// Normalised distance past which [`smooth_clamp`]'s gradient is exactly
/// zero in f64. Measured, not derived: `f64::tanh(20.0) == 1.0`.
pub const SMOOTH_CLAMP_SATURATION: f64 = 20.0;

pub fn smooth_clamp(a: &Variable, low: f64, high: f64) -> Result<Variable> {
    let mid = 0.5 * (low + high);
    let half = 0.5 * (high - low);
    if half <= 0.0 {
        return Err(dk_tensor::Error::Msg(format!(
            "smooth_clamp: need low < high, got [{low}, {high}]"
        )));
    }
    let scaled = affine(a, 1.0 / half, -mid / half)?;
    let t = tanh(&scaled)?;
    affine(&t, half, mid)
}

/// `mul·x + add` with constants — fused so the graph stays shallow, matching
/// the `LazyTensor::affine` node in the symbolic layer.
pub fn affine(a: &Variable, mul: f64, add: f64) -> Result<Variable> {
    struct AffineBackward {
        meta: NodeMeta,
        mul: f64,
    }
    impl Node for AffineBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("affine: missing grad");
            Ok(vec![Some(cpu_ops::scale(&g, self.mul)?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = cpu_ops::unary_map(a.tensor(), |v| mul * v + add)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(AffineBackward {
        meta: meta_for("AffineBackward", &[a], &out),
        mul,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// binary ops
// ---------------------------------------------------------------------------

pub fn div(a: &Variable, b: &Variable) -> Result<Variable> {
    binary(
        "DivBackward",
        a,
        b,
        |x, y| x / y,
        |x, y| (1.0 / y, -x / (y * y)),
    )
}

/// `a^b` with a TENSOR exponent — the case `powf` cannot express, and the
/// one the rational-quadratic kernel needs. `∂/∂b = a^b·ln a` requires
/// `a > 0`; at `a == 0` the b-gradient is defined as 0 (PyTorch's choice).
pub fn pow(a: &Variable, b: &Variable) -> Result<Variable> {
    binary(
        "PowBackward",
        a,
        b,
        |x, y| x.powf(y),
        |x, y| {
            let da = y * x.powf(y - 1.0);
            let db = if x > 0.0 { x.powf(y) * x.ln() } else { 0.0 };
            (da, db)
        },
    )
}

/// Ties go to `a`, matching PyTorch's `maximum` (and making the backward
/// deterministic rather than depending on comparison order).
pub fn maximum(a: &Variable, b: &Variable) -> Result<Variable> {
    binary(
        "MaximumBackward",
        a,
        b,
        f64::max,
        |x, y| if x >= y { (1.0, 0.0) } else { (0.0, 1.0) },
    )
}

pub fn minimum(a: &Variable, b: &Variable) -> Result<Variable> {
    binary(
        "MinimumBackward",
        a,
        b,
        f64::min,
        |x, y| if x <= y { (1.0, 0.0) } else { (0.0, 1.0) },
    )
}
