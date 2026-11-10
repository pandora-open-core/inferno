//! Port of torch/csrc/autograd/functions/basic_ops.h `GraphRoot` — the
//! synthetic node `Engine::execute` seeds the backward with: its outputs
//! are the caller-supplied root gradients, its next_edges are the roots'
//! gradient edges.

use dk_tensor::Tensor;

use crate::edge::Edge;
use crate::node::{Node, NodeMeta, Result};

pub struct GraphRoot {
    meta: NodeMeta,
    outputs: Vec<Option<Tensor>>,
}

impl GraphRoot {
    pub fn new(next_edges: Vec<Edge>, outputs: Vec<Option<Tensor>>) -> GraphRoot {
        GraphRoot {
            meta: NodeMeta::new("GraphRoot", next_edges),
            outputs,
        }
    }
}

impl Node for GraphRoot {
    fn apply(&self, _grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        Ok(self.outputs.clone())
    }

    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
}
