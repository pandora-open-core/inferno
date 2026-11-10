//! Port of torch/csrc/autograd/edge.h.
//!
//! `Edge{ function, input_nr }` — "the particular input of a function".
//! Upstream's undefined edge (null function) is `Edge::undefined()` here;
//! it means "gradient stops" (input doesn't require grad).

use std::sync::Arc;

use crate::node::{node_key, Node};

#[derive(Clone)]
pub struct Edge {
    pub node: Option<Arc<dyn Node>>,
    /// Which input of `node` this edge feeds (backward-flow input index).
    pub input_nr: u32,
}

impl Edge {
    pub fn new(node: Arc<dyn Node>, input_nr: u32) -> Edge {
        Edge {
            node: Some(node),
            input_nr,
        }
    }

    pub fn undefined() -> Edge {
        Edge {
            node: None,
            input_nr: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.node.is_some()
    }

    pub fn key(&self) -> Option<usize> {
        self.node.as_ref().map(node_key)
    }
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.node {
            Some(n) => write!(f, "Edge({} #{}, in={})", n.meta().name, n.meta().sequence_nr, self.input_nr),
            None => write!(f, "Edge(undefined)"),
        }
    }
}
