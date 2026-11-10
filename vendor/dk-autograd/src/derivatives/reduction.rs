//! D2 — dim-wise reductions, from derivatives.yaml.
//!
//! `logsumexp`, `softmax` and `log_softmax` get hand-written backwards
//! rather than being composed, for the same reason `softmin` does: the
//! closed forms need only the OUTPUT, so the backward is O(1) graph depth
//! and numerically exact, where a composition would re-derive the shift and
//! carry a deep graph.
//!
//!   logsumexp:   dx = g ⊙ softmax(x)          (broadcast g along dim)
//!   softmax:     dx = y ⊙ (g − Σ_dim(g ⊙ y))
//!   log_softmax: dx = g − exp(y)·Σ_dim(g)

use std::sync::Arc;

use dk_tensor::{cpu_ops, Tensor};

use super::{meta_for, should_record};
use crate::node::{Node, NodeMeta, Result};
use crate::variable::Variable;

/// Save an intermediate that is neither an input nor a graph output (e.g.
/// `softmax`'s own result) — plain data, no version tracking needed because
/// nothing else can alias it.
type Cached = std::sync::Mutex<Option<Tensor>>;

// ---------------------------------------------------------------------------
// sum / mean along a dim
// ---------------------------------------------------------------------------

struct SumDimBackward {
    meta: NodeMeta,
    in_shape: Vec<usize>,
    dim: usize,
    /// 1 for sum, 1/len for mean — the only difference between the two.
    scale: f64,
}

impl Node for SumDimBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("sum_dim: missing grad");
        let e = cpu_ops::expand_along_dim(&g, &self.in_shape, self.dim)?;
        Ok(vec![Some(if self.scale == 1.0 {
            e
        } else {
            cpu_ops::scale(&e, self.scale)?
        })])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn sum_dim(a: &Variable, dim: usize) -> Result<Variable> {
    let out = cpu_ops::sum_dim(a.tensor(), dim)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SumDimBackward {
        meta: meta_for("SumDimBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
        dim,
        scale: 1.0,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

pub fn mean_dim(a: &Variable, dim: usize) -> Result<Variable> {
    let len = *a
        .tensor()
        .shape()
        .get(dim)
        .ok_or_else(|| dk_tensor::Error::Msg(format!("mean_dim: bad dim {dim}")))?;
    let out = cpu_ops::scale(&cpu_ops::sum_dim(a.tensor(), dim)?, 1.0 / len as f64)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SumDimBackward {
        meta: meta_for("MeanDimBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
        dim,
        scale: 1.0 / len as f64,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// max along a dim
// ---------------------------------------------------------------------------

struct MaxDimBackward {
    meta: NodeMeta,
    in_shape: Vec<usize>,
    dim: usize,
    idx: Vec<usize>,
}

impl Node for MaxDimBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("max_dim: missing grad");
        Ok(vec![Some(cpu_ops::scatter_along_dim(
            &g,
            &self.idx,
            &self.in_shape,
            self.dim,
        )?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

/// `(values, argmax)`. Gradient goes only to the argmax element; ties go to
/// the lowest index (see `cpu_ops::max_dim`), so it is deterministic.
pub fn max_dim(a: &Variable, dim: usize) -> Result<(Variable, Vec<usize>)> {
    let (out, idx) = cpu_ops::max_dim(a.tensor(), dim)?;
    if !should_record(&[a]) {
        return Ok((Variable::new(out), idx));
    }
    let node = Arc::new(MaxDimBackward {
        meta: meta_for("MaxDimBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
        dim,
        idx: idx.clone(),
    });
    Ok((Variable::from_grad_fn(out, node, 0), idx))
}

// ---------------------------------------------------------------------------
// logsumexp / softmax / log_softmax
// ---------------------------------------------------------------------------

struct LogSumExpBackward {
    meta: NodeMeta,
    /// softmax(x) along `dim` — the whole Jacobian, cached at forward time.
    soft: Cached,
    in_shape: Vec<usize>,
    dim: usize,
}

impl Node for LogSumExpBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("logsumexp: missing grad");
        let guard = self.soft.lock().unwrap();
        let soft = guard.as_ref().ok_or_else(|| {
            dk_tensor::Error::Msg("logsumexp: saved softmax already freed (retain_graph?)".into())
        })?;
        let ge = cpu_ops::expand_along_dim(&g, &self.in_shape, self.dim)?;
        Ok(vec![Some(cpu_ops::mul(&ge, soft)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn release_saved(&self) {
        *self.soft.lock().unwrap() = None;
    }
}

pub fn logsumexp_dim(a: &Variable, dim: usize) -> Result<Variable> {
    let out = cpu_ops::logsumexp_dim(a.tensor(), dim)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(LogSumExpBackward {
        meta: meta_for("LogSumExpBackward", &[a], &out),
        soft: std::sync::Mutex::new(Some(cpu_ops::softmax_dim(a.tensor(), dim)?)),
        in_shape: a.tensor().shape().to_vec(),
        dim,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

struct SoftmaxBackward {
    meta: NodeMeta,
    result: Cached,
    dim: usize,
    /// true for log_softmax
    logged: bool,
}

impl Node for SoftmaxBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("softmax: missing grad");
        let guard = self.result.lock().unwrap();
        let y = guard.as_ref().ok_or_else(|| {
            dk_tensor::Error::Msg("softmax: saved result already freed (retain_graph?)".into())
        })?;
        let shape = y.shape().to_vec();
        let out = if self.logged {
            // dx = g − exp(y)·Σ_dim g
            let gsum = cpu_ops::expand_along_dim(&cpu_ops::sum_dim(&g, self.dim)?, &shape, self.dim)?;
            let p = cpu_ops::unary_map(y, f64::exp)?;
            cpu_ops::sub(&g, &cpu_ops::mul(&p, &gsum)?)?
        } else {
            // dx = y ⊙ (g − Σ_dim(g ⊙ y))
            let gy = cpu_ops::mul(&g, y)?;
            let s = cpu_ops::expand_along_dim(&cpu_ops::sum_dim(&gy, self.dim)?, &shape, self.dim)?;
            cpu_ops::mul(y, &cpu_ops::sub(&g, &s)?)?
        };
        Ok(vec![Some(out)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn release_saved(&self) {
        *self.result.lock().unwrap() = None;
    }
}

pub fn softmax(a: &Variable, dim: usize) -> Result<Variable> {
    let out = cpu_ops::softmax_dim(a.tensor(), dim)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SoftmaxBackward {
        meta: meta_for("SoftmaxBackward", &[a], &out),
        result: std::sync::Mutex::new(Some(out.clone())),
        dim,
        logged: false,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

pub fn log_softmax(a: &Variable, dim: usize) -> Result<Variable> {
    let out = cpu_ops::log_softmax_dim(a.tensor(), dim)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SoftmaxBackward {
        meta: meta_for("LogSoftmaxBackward", &[a], &out),
        result: std::sync::Mutex::new(Some(out.clone())),
        dim,
        logged: true,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}
