//! Port of torch/csrc/autograd/functions/accumulate_grad.{h,cpp} — the leaf
//! node that does `.grad += incoming`.
//!
//! Upstream's grad layout contract (accumulate into a tensor matching the
//! leaf's layout, stealing the incoming grad when uniquely owned) reduces
//! here to: first grad is stored contiguous; later grads are added. The
//! stream/layout subtleties return with the GPU phases.

use std::sync::Arc;

use dk_tensor::{cpu_ops, Tensor};

use crate::input_metadata::InputMetadata;
use crate::node::{Node, NodeMeta, Result};
use crate::variable::AutogradMeta;

pub struct AccumulateGrad {
    meta: NodeMeta,
    /// Strong ref, matching upstream (AccumulateGrad holds its Variable
    /// strongly; the variable's meta holds the accumulator weakly — no
    /// cycle, and `.grad` survives while the graph does).
    target: Arc<AutogradMeta>,
}

impl AccumulateGrad {
    pub fn new(target: Arc<AutogradMeta>, leaf: &Tensor) -> AccumulateGrad {
        AccumulateGrad {
            meta: NodeMeta::new("AccumulateGrad", Vec::new())
                .with_input_metadata(vec![InputMetadata::of(leaf)]),
            target,
        }
    }
}

impl Node for AccumulateGrad {
    fn apply(&self, mut grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let incoming = match grads.pop().flatten() {
            Some(g) => g,
            None => return Ok(vec![]),
        };
        let mut slot = self.target.grad.lock().unwrap();
        let next = match slot.take() {
            None => incoming.contiguous()?,
            Some(existing) => cpu_ops::add(&existing, &incoming)?,
        };
        *slot = Some(next);
        Ok(vec![])
    }

    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}
