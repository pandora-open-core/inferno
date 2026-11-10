//! The hand-written "VariableType" layer: forward op + backward Node,
//! curated from tools/autograd/derivatives.yaml (pytorch @ 540f3a53).
//!
//! Pattern per op (what gen_variable_type.py generates upstream):
//!   1. run the kernel on the tensors;
//!   2. if grad mode is on and any input requires grad, build the
//!      `FooBackward` node with `collect_next_edges(inputs)` and
//!      `SavedVariable`s, and wire the output to it.
//!
//! Phase 1 carries the ops the engine tests need (add/mul/sum/exp/mm/relu);
//! Phase 2 (D1–D3) splits this into elementwise/reduction/shape modules and
//! completes the curated list.
//!
//! yaml references (derivatives.yaml):
//!   add:  "grad" / "grad"            (+ sum_to_size for broadcast)
//!   mul:  "grad * other" / "grad * self"
//!   sum:  "grad.expand(self.sym_sizes())"
//!   exp:  "grad * result"            (result saved as output)
//!   mm:   mm_mat1_backward / mm_mat2_backward (FunctionsManual.cpp) —
//!         grad @ mat2^T and mat1^T @ grad in the contiguous case
//!   relu: "threshold_backward(grad, result, 0)"

use std::sync::Arc;

use dk_tensor::{cpu_ops, Tensor};

use crate::grad_mode;
use crate::input_metadata::InputMetadata;
use crate::node::{collect_next_edges, Node, NodeMeta, Result};
use crate::saved_variable::SavedVariable;
use crate::variable::Variable;

pub mod elementwise;
pub mod kernels;
pub mod reduction;
pub mod shape;

pub(crate) fn should_record(inputs: &[&Variable]) -> bool {
    grad_mode::is_enabled() && inputs.iter().any(|v| v.requires_grad())
}

pub(crate) fn meta_for(name: &'static str, inputs: &[&Variable], out: &Tensor) -> NodeMeta {
    NodeMeta::new(name, collect_next_edges(inputs))
        .with_input_metadata(vec![InputMetadata::of(out)])
}

// ---------------------------------------------------------------------------
// add / sub / mul / neg
// ---------------------------------------------------------------------------

struct AddBackward {
    meta: NodeMeta,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl Node for AddBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("AddBackward: missing grad");
        Ok(vec![
            Some(cpu_ops::sum_to_shape(&g, &self.a_shape)?),
            Some(cpu_ops::sum_to_shape(&g, &self.b_shape)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn add(a: &Variable, b: &Variable) -> Result<Variable> {
    let out = cpu_ops::add(a.tensor(), b.tensor())?;
    if !should_record(&[a, b]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(AddBackward {
        meta: meta_for("AddBackward", &[a, b], &out),
        a_shape: a.tensor().shape().to_vec(),
        b_shape: b.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

struct SubBackward {
    meta: NodeMeta,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl Node for SubBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("SubBackward: missing grad");
        Ok(vec![
            Some(cpu_ops::sum_to_shape(&g, &self.a_shape)?),
            Some(cpu_ops::sum_to_shape(&cpu_ops::neg(&g)?, &self.b_shape)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn sub(a: &Variable, b: &Variable) -> Result<Variable> {
    let out = cpu_ops::sub(a.tensor(), b.tensor())?;
    if !should_record(&[a, b]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SubBackward {
        meta: meta_for("SubBackward", &[a, b], &out),
        a_shape: a.tensor().shape().to_vec(),
        b_shape: b.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

struct MulBackward {
    meta: NodeMeta,
    a: SavedVariable,
    b: SavedVariable,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl Node for MulBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("MulBackward: missing grad");
        let a = self.a.unpack_data()?;
        let b = self.b.unpack_data()?;
        // yaml: self: "grad * other", other: "grad * self"
        Ok(vec![
            Some(cpu_ops::sum_to_shape(&cpu_ops::mul(&g, &b)?, &self.a_shape)?),
            Some(cpu_ops::sum_to_shape(&cpu_ops::mul(&g, &a)?, &self.b_shape)?),
        ])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn mul(a: &Variable, b: &Variable) -> Result<Variable> {
    let out = cpu_ops::mul(a.tensor(), b.tensor())?;
    if !should_record(&[a, b]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(MulBackward {
        meta: meta_for("MulBackward", &[a, b], &out),
        a: SavedVariable::save(a, false),
        b: SavedVariable::save(b, false),
        a_shape: a.tensor().shape().to_vec(),
        b_shape: b.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

struct NegBackward {
    meta: NodeMeta,
}

impl Node for NegBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("NegBackward: missing grad");
        Ok(vec![Some(cpu_ops::neg(&g)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn neg(a: &Variable) -> Result<Variable> {
    let out = cpu_ops::neg(a.tensor())?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(NegBackward {
        meta: meta_for("NegBackward", &[a], &out),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// sum (all elements) — yaml: "grad.expand(self.sym_sizes())"
// ---------------------------------------------------------------------------

struct SumBackward {
    meta: NodeMeta,
    in_shape: Vec<usize>,
}

impl Node for SumBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("SumBackward: missing grad");
        Ok(vec![Some(g.expand(&self.in_shape)?.contiguous()?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn sum_all(a: &Variable) -> Result<Variable> {
    let out = cpu_ops::sum_all(a.tensor())?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(SumBackward {
        meta: meta_for("SumBackward", &[a], &out),
        in_shape: a.tensor().shape().to_vec(),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// exp — result saved AS OUTPUT (the SavedVariable weak-grad_fn path)
// ---------------------------------------------------------------------------

struct ExpBackward {
    meta: NodeMeta,
    result: std::sync::Mutex<Option<SavedVariable>>,
}

impl Node for ExpBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("ExpBackward: missing grad");
        let guard = self.result.lock().unwrap();
        let saved = guard.as_ref().ok_or_else(|| {
            dk_tensor::Error::Msg(
                "trying to backward through the graph a second time (saved tensors \
                 already freed); pass retain_graph=true"
                    .into(),
            )
        })?;
        let r = saved.unpack_data()?;
        Ok(vec![Some(cpu_ops::mul(&g, &r)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn release_saved(&self) {
        *self.result.lock().unwrap() = None;
    }
}

pub fn exp(a: &Variable) -> Result<Variable> {
    let out_t = cpu_ops::unary_map(a.tensor(), f64::exp)?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out_t));
    }
    let node = Arc::new(ExpBackward {
        meta: meta_for("ExpBackward", &[a], &out_t),
        result: std::sync::Mutex::new(None),
    });
    let out = Variable::from_grad_fn(out_t, node.clone(), 0);
    // save the OUTPUT into its own grad_fn — weak reference, no cycle
    *node.result.lock().unwrap() = Some(SavedVariable::save(&out, true));
    Ok(out)
}

// ---------------------------------------------------------------------------
// relu — yaml: threshold_backward(grad, result, 0)
// ---------------------------------------------------------------------------

struct ReluBackward {
    meta: NodeMeta,
    input: SavedVariable,
}

impl Node for ReluBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("ReluBackward: missing grad");
        let x = self.input.unpack_data()?;
        let mask = cpu_ops::unary_map(&x, |v| if v > 0.0 { 1.0 } else { 0.0 })?;
        Ok(vec![Some(cpu_ops::mul(&g, &mask)?)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn relu(a: &Variable) -> Result<Variable> {
    let out = cpu_ops::unary_map(a.tensor(), |v| v.max(0.0))?;
    if !should_record(&[a]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(ReluBackward {
        meta: meta_for("ReluBackward", &[a], &out),
        input: SavedVariable::save(a, false),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}

// ---------------------------------------------------------------------------
// mm — FunctionsManual: mm_mat1_backward / mm_mat2_backward
// ---------------------------------------------------------------------------

struct MmBackward {
    meta: NodeMeta,
    a: SavedVariable,
    b: SavedVariable,
}

impl Node for MmBackward {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let g = grads.pop().flatten().expect("MmBackward: missing grad");
        let a = self.a.unpack_data()?;
        let b = self.b.unpack_data()?;
        // contiguous case of mm_mat1_backward / mm_mat2_backward:
        //   dA = g @ B^T ; dB = A^T @ g
        let da = cpu_ops::mm(&g, &b.transpose(0, 1)?.contiguous()?)?;
        let db = cpu_ops::mm(&a.transpose(0, 1)?.contiguous()?, &g)?;
        Ok(vec![Some(da), Some(db)])
    }
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}

pub fn mm(a: &Variable, b: &Variable) -> Result<Variable> {
    let out = cpu_ops::mm(a.tensor(), b.tensor())?;
    if !should_record(&[a, b]) {
        return Ok(Variable::new(out));
    }
    let node = Arc::new(MmBackward {
        meta: meta_for("MmBackward", &[a, b], &out),
        a: SavedVariable::save(a, false),
        b: SavedVariable::save(b, false),
    });
    Ok(Variable::from_grad_fn(out, node, 0))
}
