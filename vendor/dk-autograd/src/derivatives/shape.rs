//! D3 — shape / indexing ops.
//!
//! The view ops are metadata-only in the forward and a pure re-layout in the
//! backward, so each backward is the inverse view rather than a scatter. The
//! exception is `expand`, whose backward is a genuine reduction
//! (`sum_to_shape`), and `cat` / `index_select`, which move data.
//!
//! NOTE: these produce views that SHARE storage with their input, so the
//! version counter is shared too. That is deliberate — it is what lets
//! `SavedVariable` catch an in-place write through a view, exactly as
//! upstream does.

use std::sync::Arc;

use dk_tensor::{cpu_ops, Tensor};

use super::{meta_for, should_record};
use crate::node::{Node, NodeMeta, Result};
use crate::variable::Variable;

/// Backward of any op whose gradient just needs re-shaping to the input's
/// layout: reshape, squeeze, unsqueeze (and `permute`'s inverse handled by
/// the caller supplying the right closure would be over-engineering, so
/// permute/transpose get their own tiny nodes below).
struct ReshapeBackward {
    meta: NodeMeta,
    in_shape: Vec<usize>,
}

impl Node for ReshapeBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("reshape: missing grad");
        Ok(vec![Some(g.contiguous()?.reshape(&self.in_shape)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

fn reshape_like(name: &'static str, a: &Variable, out: Tensor) -> Result<Variable> {
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(ReshapeBackward {
        meta: meta_for(name, &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

pub fn reshape(a: &Variable, shape: &[usize]) -> Result<Variable> {
    reshape_like("ReshapeBackward", a, a.tensor().reshape(shape)?)
}

pub fn squeeze(a: &Variable, dim: usize) -> Result<Variable> {
    reshape_like("SqueezeBackward", a, a.tensor().squeeze(dim)?)
}

pub fn unsqueeze(a: &Variable, dim: usize) -> Result<Variable> {
    reshape_like("UnsqueezeBackward", a, a.tensor().unsqueeze(dim)?)
}

/// `transpose`'s backward is itself — an involution.
pub fn transpose(a: &Variable, d0: usize, d1: usize) -> Result<Variable> {
    struct TransposeBackward {
        meta: NodeMeta,
        d0: usize,
        d1: usize,
    }
    impl Node for TransposeBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("transpose: missing grad");
            Ok(vec![Some(g.transpose(self.d0, self.d1)?.contiguous()?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = a.tensor().transpose(d0, d1)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(TransposeBackward {
        meta: meta_for("TransposeBackward", &[a], &out),
        d0,
        d1,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// `permute`'s backward is the INVERSE permutation, not the same one — the
/// classic off-by-a-permutation bug. `inv[dims[i]] = i`.
pub fn permute(a: &Variable, dims: &[usize]) -> Result<Variable> {
    struct PermuteBackward {
        meta: NodeMeta,
        inv: Vec<usize>,
    }
    impl Node for PermuteBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("permute: missing grad");
            Ok(vec![Some(g.permute(&self.inv)?.contiguous()?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = a.tensor().permute(dims)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let mut inv = vec![0usize; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inv[d] = i;
    }
    let node = Arc::new(PermuteBackward {
        meta: meta_for("PermuteBackward", &[a], &out),
        inv,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// `narrow`: the backward pads with zeros outside the window.
pub fn narrow(a: &Variable, dim: usize, start: usize, len: usize) -> Result<Variable> {
    struct NarrowBackward {
        meta: NodeMeta,
        in_shape: Vec<usize>,
        dim: usize,
        start: usize,
    }
    impl Node for NarrowBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("narrow: missing grad");
            let (outer, full, inner) = cpu_ops::split_at_dim(&self.in_shape, self.dim)?;
            let (_, glen, _) = cpu_ops::split_at_dim(g.shape(), self.dim)?;
            let gv = g.to_vec_f64()?;
            let mut out = vec![0.0; outer * full * inner];
            for o in 0..outer {
                for l in 0..glen {
                    for i in 0..inner {
                        out[(o * full + self.start + l) * inner + i] = gv[(o * glen + l) * inner + i];
                    }
                }
            }
            Ok(vec![Some(Tensor::from_f64_slice(
                &out,
                &self.in_shape,
                g.dtype(),
            )?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = a.tensor().narrow(dim, start, len)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(NarrowBackward {
        meta: meta_for("NarrowBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
        dim,
        start,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// `expand`: forward is stride-0 broadcasting, backward is the genuine
/// reduction `sum_to_shape` — the only D3 op whose backward loses
/// information rather than moving it.
pub fn expand(a: &Variable, shape: &[usize]) -> Result<Variable> {
    struct ExpandBackward {
        meta: NodeMeta,
        in_shape: Vec<usize>,
    }
    impl Node for ExpandBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("expand: missing grad");
            Ok(vec![Some(cpu_ops::sum_to_shape(&g, &self.in_shape)?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let out = a.tensor().expand(shape)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(ExpandBackward {
        meta: meta_for("ExpandBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// `index_select` along `dim`. The backward ACCUMULATES (`+=`), because an
/// index may repeat — overwriting instead would silently drop gradient for
/// every repeat but the last.
pub fn index_select(a: &Variable, dim: usize, idx: &[usize]) -> Result<Variable> {
    struct IndexSelectBackward {
        meta: NodeMeta,
        in_shape: Vec<usize>,
        dim: usize,
        idx: Vec<usize>,
    }
    impl Node for IndexSelectBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("index_select: missing grad");
            let (outer, full, inner) = cpu_ops::split_at_dim(&self.in_shape, self.dim)?;
            let k = self.idx.len();
            let gv = g.to_vec_f64()?;
            let mut out = vec![0.0; outer * full * inner];
            for o in 0..outer {
                for (l, &src) in self.idx.iter().enumerate() {
                    for i in 0..inner {
                        out[(o * full + src) * inner + i] += gv[(o * k + l) * inner + i];
                    }
                }
            }
            Ok(vec![Some(Tensor::from_f64_slice(
                &out,
                &self.in_shape,
                g.dtype(),
            )?)])
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    let shape = a.tensor().shape().to_vec();
    let (outer, full, inner) = cpu_ops::split_at_dim(&shape, dim)?;
    if let Some(&bad) = idx.iter().find(|&&i| i >= full) {
        return Err(dk_tensor::Error::Msg(format!(
            "index_select: index {bad} out of range for dim {dim} of size {full}"
        )));
    }
    let v = a.tensor().to_vec_f64()?;
    let mut sel = vec![0.0; outer * idx.len() * inner];
    for o in 0..outer {
        for (l, &src) in idx.iter().enumerate() {
            for i in 0..inner {
                sel[(o * idx.len() + l) * inner + i] = v[(o * full + src) * inner + i];
            }
        }
    }
    let mut out_shape = shape.clone();
    out_shape[dim] = idx.len();
    let out = Tensor::from_f64_slice(&sel, &out_shape, a.tensor().dtype())?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(IndexSelectBackward {
        meta: meta_for("IndexSelectBackward", &[a], &out),
        in_shape: shape,
        dim,
        idx: idx.to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

/// `cat` along `dim`. The backward is a narrow per input — one output, N
/// next_edges, which is the multi-input node shape the engine's
/// `validate_outputs` checks.
pub fn cat(parts: &[&Variable], dim: usize) -> Result<Variable> {
    struct CatBackward {
        meta: NodeMeta,
        shapes: Vec<Vec<usize>>,
        dim: usize,
    }
    impl Node for CatBackward {
        fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
            let g = grads.pop().flatten().expect("cat: missing grad");
            let mut out = Vec::with_capacity(self.shapes.len());
            let mut off = 0usize;
            for s in &self.shapes {
                let len = s[self.dim];
                out.push(Some(g.narrow(self.dim, off, len)?.contiguous()?));
                off += len;
            }
            Ok(out)
        }
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
    }
    if parts.is_empty() {
        return Err(dk_tensor::Error::Msg("cat: no inputs".into()));
    }
    let base = parts[0].tensor().shape().to_vec();
    let mut total = 0usize;
    for p in parts {
        let s = p.tensor().shape();
        if s.len() != base.len()
            || s.iter()
                .enumerate()
                .any(|(i, &v)| i != dim && v != base[i])
        {
            return Err(dk_tensor::Error::ShapeMismatch {
                op: "cat",
                lhs: base.clone(),
                rhs: s.to_vec(),
            });
        }
        total += s[dim];
    }
    let mut out_shape = base.clone();
    out_shape[dim] = total;
    let (outer, _, inner) = cpu_ops::split_at_dim(&out_shape, dim)?;

    let mut buf = vec![0.0; outer * total * inner];
    let mut off = 0usize;
    for p in parts {
        let s = p.tensor().shape().to_vec();
        let len = s[dim];
        let v = p.tensor().to_vec_f64()?;
        for o in 0..outer {
            for l in 0..len {
                for i in 0..inner {
                    buf[(o * total + off + l) * inner + i] = v[(o * len + l) * inner + i];
                }
            }
        }
        off += len;
    }
    let out = Tensor::from_f64_slice(&buf, &out_shape, parts[0].tensor().dtype())?;
    if !should_record(parts) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(CatBackward {
        meta: meta_for("CatBackward", parts, &out),
        shapes: parts.iter().map(|p| p.tensor().shape().to_vec()).collect(),
        dim,
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}
