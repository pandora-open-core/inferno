//! Fused pairwise kernel ops: Matérn 5/2, RBF, rational quadratic, Gibbs,
//! and softmin.
//!
//! Thin `Node` wrappers over `dk_tensor::kernels`, which holds the maths and
//! the closed-form VJPs (and documents why these are fused rather than
//! composed — briefly: composition materialises an `[M,N,D]` intermediate in
//! both directions, and softmin's backward needs only the transport plan).
//!
//! Every op here is MULTI-INPUT, so each returns one gradient per
//! `next_edge` in declaration order — the engine's `validate_outputs`
//! enforces the count, and gradcheck enforces the values.
//!
//! Shapes: `x1 [M,D]`, `x2 [N,D]` -> `[M,N]`. Inputs must be f64 to go
//! through gradcheck; the ops themselves accept any float dtype but
//! accumulate in f64 regardless.

use std::sync::Arc;

use dk_tensor::kernels::{self, PairShape};
use dk_tensor::{DType, Tensor};

use super::{meta_for, should_record};
use crate::input_metadata::InputMetadata;
use crate::node::{collect_next_edges, Node, NodeMeta, Result};
use crate::saved_variable::SavedVariable;
use crate::variable::Variable;

/// Validate the `[M,D] x [N,D]` contract once, up front, so a downstream
/// panic can never be the first sign of a shape error.
fn pair_shape(x1: &Tensor, x2: &Tensor, op: &'static str) -> Result<PairShape> {
    if x1.rank() != 2 || x2.rank() != 2 || x1.shape()[1] != x2.shape()[1] {
        return Err(dk_tensor::Error::ShapeMismatch {
            op,
            lhs: x1.shape().to_vec(),
            rhs: x2.shape().to_vec(),
        });
    }
    Ok(PairShape {
        m: x1.shape()[0],
        n: x2.shape()[0],
        d: x1.shape()[1],
    })
}

fn flat(v: &Variable) -> Result<Vec<f64>> {
    v.tensor().to_vec_f64()
}

fn out_tensor(vals: Vec<f64>, sh: PairShape, dtype: DType) -> Result<Tensor> {
    Tensor::from_f64_slice(&vals, &[sh.m, sh.n], dtype)
}

fn meta_multi(name: &'static str, inputs: &[&Variable], out: &Tensor) -> NodeMeta {
    NodeMeta::new(name, collect_next_edges(inputs))
        .with_input_metadata(vec![InputMetadata::of(out)])
}

// ---------------------------------------------------------------------------
// Matern 5/2
// ---------------------------------------------------------------------------

struct Matern52Backward {
    meta: NodeMeta,
    x1: SavedVariable,
    x2: SavedVariable,
    w: SavedVariable,
    scale: SavedVariable,
    sh: PairShape,
    dtype: DType,
}

impl Node for Matern52Backward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("matern52: missing grad");
        let (x1, x2) = (self.x1.unpack_data()?, self.x2.unpack_data()?);
        let (w, s) = (self.w.unpack_data()?, self.scale.unpack_data()?);
        let (gx1, gx2, gw, gs) = kernels::matern52_backward(
            &x1.to_vec_f64()?,
            &x2.to_vec_f64()?,
            &w.to_vec_f64()?,
            s.to_scalar_f64()?,
            &g.to_vec_f64()?,
            self.sh,
        )?;
        Ok(vec![
            Some(Tensor::from_f64_slice(&gx1, &[self.sh.m, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gx2, &[self.sh.n, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gw, &[self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&[gs], s.shape(), self.dtype)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// `k = (1 + √5·r·s + (5/3)·d²·s²)·exp(−√5·r·s)`, `r = √(d²+1e-12)`.
/// `inv_bw [D]` and `scale` are INVERSE lengthscales.
pub fn matern52(
    x1: &Variable,
    x2: &Variable,
    inv_bw: &Variable,
    scale: &Variable,
) -> Result<Variable> {
    let sh = pair_shape(x1.tensor(), x2.tensor(), "matern52")?;
    let dtype = x1.tensor().dtype();
    let vals = kernels::matern52(
        &flat(x1)?,
        &flat(x2)?,
        &flat(inv_bw)?,
        scale.tensor().to_scalar_f64()?,
        sh,
    )?;
    let out = out_tensor(vals, sh, dtype)?;
    if !should_record(&[x1, x2, inv_bw, scale]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(Matern52Backward {
        meta: meta_multi("Matern52Backward", &[x1, x2, inv_bw, scale], &out),
        x1: SavedVariable::save(x1, false),
        x2: SavedVariable::save(x2, false),
        w: SavedVariable::save(inv_bw, false),
        scale: SavedVariable::save(scale, false),
        sh,
        dtype,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// RBF
// ---------------------------------------------------------------------------

struct RbfBackward {
    meta: NodeMeta,
    x1: SavedVariable,
    x2: SavedVariable,
    w: SavedVariable,
    sh: PairShape,
    dtype: DType,
}

impl Node for RbfBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("rbf: missing grad");
        let (x1, x2, w) = (
            self.x1.unpack_data()?,
            self.x2.unpack_data()?,
            self.w.unpack_data()?,
        );
        let (gx1, gx2, gw) = kernels::rbf_backward(
            &x1.to_vec_f64()?,
            &x2.to_vec_f64()?,
            &w.to_vec_f64()?,
            &g.to_vec_f64()?,
            self.sh,
        )?;
        Ok(vec![
            Some(Tensor::from_f64_slice(&gx1, &[self.sh.m, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gx2, &[self.sh.n, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gw, &[self.sh.d], self.dtype)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// `k = exp(−d²)` — NO factor of 1/2, matching `pair_covar_lazy`.
pub fn rbf(x1: &Variable, x2: &Variable, inv_bw: &Variable) -> Result<Variable> {
    let sh = pair_shape(x1.tensor(), x2.tensor(), "rbf")?;
    let dtype = x1.tensor().dtype();
    let vals = kernels::rbf(&flat(x1)?, &flat(x2)?, &flat(inv_bw)?, sh)?;
    let out = out_tensor(vals, sh, dtype)?;
    if !should_record(&[x1, x2, inv_bw]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(RbfBackward {
        meta: meta_multi("RbfBackward", &[x1, x2, inv_bw], &out),
        x1: SavedVariable::save(x1, false),
        x2: SavedVariable::save(x2, false),
        w: SavedVariable::save(inv_bw, false),
        sh,
        dtype,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// Rational quadratic
// ---------------------------------------------------------------------------

struct RationalBackward {
    meta: NodeMeta,
    x1: SavedVariable,
    x2: SavedVariable,
    w: SavedVariable,
    alpha: SavedVariable,
    sh: PairShape,
    dtype: DType,
}

impl Node for RationalBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("rq: missing grad");
        let (x1, x2) = (self.x1.unpack_data()?, self.x2.unpack_data()?);
        let (w, al) = (self.w.unpack_data()?, self.alpha.unpack_data()?);
        let (gx1, gx2, gw, ga) = kernels::rational_quadratic_backward(
            &x1.to_vec_f64()?,
            &x2.to_vec_f64()?,
            &w.to_vec_f64()?,
            al.to_scalar_f64()?,
            &g.to_vec_f64()?,
            self.sh,
        )?;
        Ok(vec![
            Some(Tensor::from_f64_slice(&gx1, &[self.sh.m, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gx2, &[self.sh.n, self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&gw, &[self.sh.d], self.dtype)?),
            Some(Tensor::from_f64_slice(&[ga], al.shape(), self.dtype)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// `k = (1 + d²/(2α))^(−α)`. The α gradient carries BOTH the exponent and
/// the α inside the base — this is the case a constant-exponent `powf`
/// cannot express, and the reason the symbolic layer has a `Pow` node.
pub fn rational_quadratic(
    x1: &Variable,
    x2: &Variable,
    inv_bw: &Variable,
    alpha: &Variable,
) -> Result<Variable> {
    let sh = pair_shape(x1.tensor(), x2.tensor(), "rational_quadratic")?;
    let dtype = x1.tensor().dtype();
    let vals = kernels::rational_quadratic(
        &flat(x1)?,
        &flat(x2)?,
        &flat(inv_bw)?,
        alpha.tensor().to_scalar_f64()?,
        sh,
    )?;
    let out = out_tensor(vals, sh, dtype)?;
    if !should_record(&[x1, x2, inv_bw, alpha]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(RationalBackward {
        meta: meta_multi("RationalQuadraticBackward", &[x1, x2, inv_bw, alpha], &out),
        x1: SavedVariable::save(x1, false),
        x2: SavedVariable::save(x2, false),
        w: SavedVariable::save(inv_bw, false),
        alpha: SavedVariable::save(alpha, false),
        sh,
        dtype,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// Gibbs (non-stationary)
// ---------------------------------------------------------------------------

struct GibbsBackward {
    meta: NodeMeta,
    x1: SavedVariable,
    x2: SavedVariable,
    l1: SavedVariable,
    l2: SavedVariable,
    sh: PairShape,
    dtype: DType,
}

impl Node for GibbsBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("gibbs: missing grad");
        let (x1, x2) = (self.x1.unpack_data()?, self.x2.unpack_data()?);
        let (l1, l2) = (self.l1.unpack_data()?, self.l2.unpack_data()?);
        let (gx1, gx2, gl1, gl2) = kernels::gibbs_backward(
            &x1.to_vec_f64()?,
            &x2.to_vec_f64()?,
            &l1.to_vec_f64()?,
            &l2.to_vec_f64()?,
            &g.to_vec_f64()?,
            self.sh,
        )?;
        let m = [self.sh.m, self.sh.d];
        let n = [self.sh.n, self.sh.d];
        Ok(vec![
            Some(Tensor::from_f64_slice(&gx1, &m, self.dtype)?),
            Some(Tensor::from_f64_slice(&gx2, &n, self.dtype)?),
            Some(Tensor::from_f64_slice(&gl1, &m, self.dtype)?),
            Some(Tensor::from_f64_slice(&gl2, &n, self.dtype)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// Non-stationary Gibbs covariance with per-point lengthscales
/// `l1 [M,D]`, `l2 [N,D]` (strictly positive). Lengthscales are INPUTS, not
/// parameters, so a network emitting them is differentiable through the
/// kernel — that is the whole point of this kernel. Reduces to `rbf` when
/// the lengthscale is constant.
pub fn gibbs(
    x1: &Variable,
    x2: &Variable,
    l1: &Variable,
    l2: &Variable,
) -> Result<Variable> {
    let sh = pair_shape(x1.tensor(), x2.tensor(), "gibbs")?;
    let dtype = x1.tensor().dtype();
    let vals = kernels::gibbs(&flat(x1)?, &flat(x2)?, &flat(l1)?, &flat(l2)?, sh)?;
    let out = out_tensor(vals, sh, dtype)?;
    if !should_record(&[x1, x2, l1, l2]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(GibbsBackward {
        meta: meta_multi("GibbsBackward", &[x1, x2, l1, l2], &out),
        x1: SavedVariable::save(x1, false),
        x2: SavedVariable::save(x2, false),
        l1: SavedVariable::save(l1, false),
        l2: SavedVariable::save(l2, false),
        sh,
        dtype,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// softmin  —  the Sinkhorn potential update
// ---------------------------------------------------------------------------

struct SoftminBackward {
    meta: NodeMeta,
    c: SavedVariable,
    log_b: SavedVariable,
    g: SavedVariable,
    eps: SavedVariable,
    m: usize,
    n: usize,
    dtype: DType,
}

impl Node for SoftminBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let v = grads.pop().flatten().expect("softmin: missing grad");
        let c = self.c.unpack_data()?;
        let log_b = self.log_b.unpack_data()?;
        let g = self.g.unpack_data()?;
        let eps = self.eps.unpack_data()?;
        let (dc, dlog_b, dg, deps) = kernels::softmin_backward(
            &c.to_vec_f64()?,
            &log_b.to_vec_f64()?,
            &g.to_vec_f64()?,
            eps.to_scalar_f64()?,
            &v.to_vec_f64()?,
            self.m,
            self.n,
        )?;
        Ok(vec![
            Some(Tensor::from_f64_slice(&dc, &[self.m, self.n], self.dtype)?),
            Some(Tensor::from_f64_slice(&dlog_b, &[self.n], self.dtype)?),
            Some(Tensor::from_f64_slice(&dg, &[self.n], self.dtype)?),
            Some(Tensor::from_f64_slice(&[deps], eps.shape(), self.dtype)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// `out_i = −ε · logsumexp_j[(g_j − C_ij)/ε + log b_j]`.
///
/// Matches `coupling_kernel::softmin_lazy` and the pykeops `_keops_softmin`
/// it was ported from. `eps` is a TENSOR, not a scalar, precisely so the
/// gradient reaches a learnable temperature — the severed-`eps` bug that
/// left `raw_global_langevin_temp` dead in the existing kernel.
///
/// `C [M,N]`, `log_b [N]`, `g [N]`, `eps` scalar -> `[M]`.
pub fn softmin(
    c: &Variable,
    log_b: &Variable,
    g: &Variable,
    eps: &Variable,
) -> Result<Variable> {
    if c.tensor().rank() != 2 {
        return Err(dk_tensor::Error::Msg(format!(
            "softmin: C must be 2-D, got {:?}",
            c.tensor().shape()
        )));
    }
    let (m, n) = (c.tensor().shape()[0], c.tensor().shape()[1]);
    let dtype = c.tensor().dtype();
    let vals = kernels::softmin(
        &flat(c)?,
        &flat(log_b)?,
        &flat(g)?,
        eps.tensor().to_scalar_f64()?,
        m,
        n,
    )?;
    let out = Tensor::from_f64_slice(&vals, &[m], dtype)?;
    if !should_record(&[c, log_b, g, eps]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SoftminBackward {
        meta: meta_for("SoftminBackward", &[c, log_b, g, eps], &out),
        c: SavedVariable::save(c, false),
        log_b: SavedVariable::save(log_b, false),
        g: SavedVariable::save(g, false),
        eps: SavedVariable::save(eps, false),
        m,
        n,
        dtype,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}
