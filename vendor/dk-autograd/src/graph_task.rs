//! Port of torch/csrc/autograd/graph_task.h — the per-backward bookkeeping.
//!
//! `dependencies` counts unresolved in-edges per node (engine.cpp
//! `compute_dependencies`); `not_ready` holds InputBuffers of nodes still
//! waiting; `exec_info`, when present, restricts execution to nodes on a
//! path to a capture target (`.grad(inputs=...)` pruning, upstream
//! `init_to_execute`) and records where captured gradients land.

use std::collections::HashMap;

use dk_tensor::Tensor;

use crate::input_buffer::InputBuffer;

#[derive(Default)]
pub struct ExecInfo {
    /// Execute this node (it lies on a path to some capture target).
    pub needed: bool,
    /// (input_nr, slot in captured_grads): gradients arriving at this
    /// node's input_nr are the requested output at `slot`.
    pub captures: Vec<(u32, usize)>,
}

pub struct GraphTask {
    /// node_key -> number of unresolved incoming edges.
    pub dependencies: HashMap<usize, usize>,
    /// node_key -> partially accumulated inputs.
    pub not_ready: HashMap<usize, InputBuffer>,
    /// None => execute everything reachable (plain `.backward()`).
    pub exec_info: Option<HashMap<usize, ExecInfo>>,
    /// Filled via ExecInfo.captures during execution (`.grad()` results).
    pub captured_grads: Vec<Option<Tensor>>,
    /// Keep saved tensors alive after this backward (retain_graph).
    pub keep_graph: bool,
}

impl GraphTask {
    pub fn new(keep_graph: bool, n_captures: usize) -> GraphTask {
        GraphTask {
            dependencies: HashMap::new(),
            not_ready: HashMap::new(),
            exec_info: None,
            captured_grads: (0..n_captures).map(|_| None).collect(),
            keep_graph,
        }
    }

    /// Should this node's `apply` run? Upstream `exec_info_.needed_`.
    /// Capture-only nodes (e.g. the AccumulateGrad of a `.grad()` input)
    /// receive gradients but never execute — that is what keeps `.grad()`
    /// from polluting leaf `.grad`s.
    pub fn should_execute(&self, key: usize) -> bool {
        match &self.exec_info {
            None => true,
            Some(map) => map.get(&key).map(|e| e.needed).unwrap_or(false),
        }
    }

    /// Should gradients be routed to this node at all (executed OR captured)?
    pub fn should_schedule(&self, key: usize) -> bool {
        match &self.exec_info {
            None => true,
            Some(map) => map
                .get(&key)
                .map(|e| e.needed || !e.captures.is_empty())
                .unwrap_or(false),
        }
    }
}
