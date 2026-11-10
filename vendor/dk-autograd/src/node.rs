//! Port of torch/csrc/autograd/node.h (the `Node` class) — as a trait.
//!
//! Trait object, not enum (per the plan): engine.cpp dispatches virtually,
//! and extensibility is the point — pandorapurple' `GenredBackwardNode` must
//! be able to implement this without dk-autograd knowing it exists.
//! `Arc<dyn Node>` replaces `c10::intrusive_ptr<Node>`.
//!
//! `sequence_nr` comes from one global counter (upstream uses thread-local
//! counters only so profiler correlation works; we don't carry a profiler).
//! Ready-queue ordering by descending sequence_nr — the property that gives
//! reverse-creation-order execution — is preserved.
//!
//! `topological_nr` (upstream's pruning accelerator) is deliberately
//! omitted in Milestone A; `compute_min_topological_nr` is an optimization,
//! not a semantic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dk_tensor::Tensor;

use crate::edge::Edge;
use crate::input_metadata::InputMetadata;

pub type Result<T> = dk_tensor::Result<T>;
pub use dk_tensor::Error;

static SEQUENCE_NR: AtomicU64 = AtomicU64::new(0);

pub fn next_sequence_nr() -> u64 {
    SEQUENCE_NR.fetch_add(1, Ordering::Relaxed)
}

/// A pre-hook: may observe/replace the incoming gradients before `apply`.
pub type PreHook = Box<dyn Fn(&mut Vec<Option<Tensor>>) + Send + Sync>;

pub struct NodeMeta {
    pub name: &'static str,
    pub sequence_nr: u64,
    pub next_edges: Vec<Edge>,
    /// Metadata of this node's *inputs* — i.e. of the gradients that flow
    /// into it. Indexed by input_nr; used by validate_outputs.
    pub input_metadata: Vec<InputMetadata>,
    pub pre_hooks: Mutex<Vec<PreHook>>,
    /// Creation backtrace when anomaly mode was on at construction.
    pub anomaly_trace: Option<String>,
}

impl NodeMeta {
    pub fn new(name: &'static str, next_edges: Vec<Edge>) -> NodeMeta {
        NodeMeta {
            name,
            sequence_nr: next_sequence_nr(),
            next_edges,
            input_metadata: Vec::new(),
            pre_hooks: Mutex::new(Vec::new()),
            anomaly_trace: crate::anomaly::capture_trace(),
        }
    }

    pub fn with_input_metadata(mut self, meta: Vec<InputMetadata>) -> NodeMeta {
        self.input_metadata = meta;
        self
    }

    pub fn num_inputs(&self) -> usize {
        // At least one slot even when metadata wasn't recorded: a node with
        // no declared inputs still receives its single incoming gradient.
        self.input_metadata.len().max(1)
    }
}

/// The autograd graph node — one backward function.
///
/// `apply` consumes the gradients w.r.t. this node's outputs-in-the-forward
/// sense (its "inputs" in backward flow) and produces gradients for each of
/// `next_edges` (None = nothing to propagate along that edge).
pub trait Node: Send + Sync + 'static {
    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>>;
    fn meta(&self) -> &NodeMeta;

    /// Drop saved tensors after a non-retaining backward
    /// (upstream `release_variables`). Default: nothing saved.
    fn release_saved(&self) {}
}

/// Identity key for graph bookkeeping (upstream keys on `Node*`).
pub fn node_key(node: &Arc<dyn Node>) -> usize {
    Arc::as_ptr(node) as *const () as usize
}

/// Upstream `collect_next_edges` (function.h): the edges gradients flow to
/// after this op — each differentiable input's gradient edge, undefined
/// (empty) for non-differentiable inputs.
pub fn collect_next_edges(inputs: &[&crate::variable::Variable]) -> Vec<Edge> {
    inputs.iter().map(|v| v.grad_edge()).collect()
}
