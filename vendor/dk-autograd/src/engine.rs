//! Port of torch/csrc/autograd/engine.cpp — Milestone A: single-threaded.
//!
//! The ready-queue loop runs on the calling thread; scheduling semantics
//! (dependency counting, reverse-sequence_nr ordering, not-ready buffers,
//! exec_info pruning, capture) are the full upstream algorithm from
//! `Engine::execute` (engine.cpp:1339) -> `compute_dependencies` (:1301) ->
//! `evaluate_function` (:1109) -> `call_function` (:1061). Per-device
//! worker threads, the reentrant thread pool (MAX_DEPTH=60) and stream
//! sync are Milestone B and slot in behind this same API.
//!
//! Entry points `backward`/`grad` port torch/csrc/autograd/autograd.cpp
//! (`run_backward`, `_make_grads`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dk_tensor::Tensor;

use crate::edge::Edge;
use crate::graph_root::GraphRoot;
use crate::graph_task::{ExecInfo, GraphTask};
use crate::input_buffer::InputBuffer;
use crate::node::{node_key, Error, Node, Result};
use crate::ready_queue::{NodeTask, ReadyQueue};
use crate::variable::Variable;

pub struct Engine;

impl Engine {
    /// Upstream `Engine::execute`. `roots` are the gradient edges of the
    /// outputs; `inputs` (when Some) are capture edges — the `.grad()` mode
    /// that prunes execution to paths reaching them and returns their
    /// gradients instead of accumulating leaves.
    pub fn execute(
        roots: Vec<Edge>,
        root_grads: Vec<Option<Tensor>>,
        keep_graph: bool,
        inputs: Option<Vec<Edge>>,
    ) -> Result<Vec<Option<Tensor>>> {
        let n_captures = inputs.as_ref().map(|i| i.len()).unwrap_or(0);
        let mut task = GraphTask::new(keep_graph, n_captures);

        let root = Arc::new(GraphRoot::new(roots, root_grads)) as Arc<dyn Node>;

        if let Some(capture_edges) = &inputs {
            task.exec_info = Some(Self::init_to_execute(&root, capture_edges));
        }

        Self::compute_dependencies(&root, &mut task);

        let queue = ReadyQueue::new();
        queue.push(NodeTask {
            node: root,
            inputs: InputBuffer::new(0),
        });

        // thread_main, single-threaded
        let mut executed: Vec<Arc<dyn Node>> = Vec::new();
        while let Some(NodeTask { node, inputs }) = queue.pop() {
            Self::evaluate_function(node, inputs, &mut task, &queue, &mut executed)?;
        }

        // exec_post_processing: release saved tensors unless retaining.
        if !task.keep_graph {
            for node in &executed {
                node.release_saved();
            }
        }

        Ok(task.captured_grads)
    }

    /// Upstream `GraphTask::init_to_execute`.
    ///
    /// Two distinct properties per node (upstream's exact split):
    /// - `captures`: gradients arriving at (node, input_nr) are a requested
    ///   output. Capture-only nodes RECEIVE gradients but never execute —
    ///   that is what keeps `.grad()` from touching leaf `.grad`s.
    /// - `needed`: executing this node produces gradients that flow toward
    ///   some capture — i.e. some child edge leads to a capture target or
    ///   to another needed node.
    fn init_to_execute(
        root: &Arc<dyn Node>,
        capture_edges: &[Edge],
    ) -> HashMap<usize, ExecInfo> {
        let mut exec: HashMap<usize, ExecInfo> = HashMap::new();
        for (slot, edge) in capture_edges.iter().enumerate() {
            if let Some(key) = edge.key() {
                exec.entry(key).or_default().captures.push((edge.input_nr, slot));
            }
        }

        // Post-order DFS with memo over the DAG:
        // needed(n) = any child c with (c has captures) or needed(c).
        fn mark(
            node: &Arc<dyn Node>,
            exec: &mut HashMap<usize, ExecInfo>,
            memo: &mut HashMap<usize, bool>,
        ) -> bool {
            let key = node_key(node);
            if let Some(&v) = memo.get(&key) {
                return v;
            }
            let mut needed = false;
            for edge in &node.meta().next_edges {
                if let Some(child) = &edge.node {
                    let child_key = node_key(child);
                    let child_captures = exec
                        .get(&child_key)
                        .map(|e| !e.captures.is_empty())
                        .unwrap_or(false);
                    // note: mark() must run even when child_captures is true,
                    // so deeper captures still mark their own paths
                    let child_needed = mark(child, exec, memo);
                    needed |= child_captures || child_needed;
                }
            }
            if needed {
                exec.entry(key).or_default().needed = true;
            }
            memo.insert(key, needed);
            needed
        }

        let mut memo = HashMap::new();
        mark(root, &mut exec, &mut memo);
        exec
    }

    /// Upstream `Engine::compute_dependencies`: BFS from the root counting
    /// in-edges. Only executing nodes propagate, so traversal never expands
    /// past a capture-only (non-executing) node — its children would wait
    /// on gradients that can never arrive.
    fn compute_dependencies(root: &Arc<dyn Node>, task: &mut GraphTask) {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut stack: Vec<Arc<dyn Node>> = vec![root.clone()];
        seen.insert(node_key(root));
        while let Some(node) = stack.pop() {
            if !task.should_execute(node_key(&node)) {
                continue;
            }
            for edge in &node.meta().next_edges {
                let Some(child) = &edge.node else { continue };
                let key = node_key(child);
                if !task.should_schedule(key) {
                    continue;
                }
                *task.dependencies.entry(key).or_insert(0) += 1;
                if seen.insert(key) {
                    stack.push(child.clone());
                }
            }
        }
    }

    /// Upstream `Engine::evaluate_function` + `call_function`.
    fn evaluate_function(
        node: Arc<dyn Node>,
        inputs: InputBuffer,
        task: &mut GraphTask,
        queue: &ReadyQueue,
        executed: &mut Vec<Arc<dyn Node>>,
    ) -> Result<Vec<()>> {
        let key = node_key(&node);
        let mut input_vec = inputs.into_vec();

        // pre-hooks
        for hook in node.meta().pre_hooks.lock().unwrap().iter() {
            hook(&mut input_vec);
        }

        // captures (the `.grad()` outputs)
        if let Some(exec) = &task.exec_info {
            if let Some(info) = exec.get(&key) {
                for &(input_nr, slot) in &info.captures {
                    let g = input_vec.get(input_nr as usize).cloned().flatten();
                    task.captured_grads[slot] = match (task.captured_grads[slot].take(), g) {
                        (None, g) => g,
                        (Some(acc), None) => Some(acc),
                        (Some(acc), Some(g)) => Some(dk_tensor::cpu_ops::add(&acc, &g)?),
                    };
                }
            }
        }

        // capture-only nodes stop here: gradients captured, apply skipped
        if !task.should_execute(key) {
            return Ok(vec![]);
        }

        // call_function
        let outputs = node.apply(input_vec).map_err(|e| {
            match &node.meta().anomaly_trace {
                Some(trace) => Error::Msg(format!(
                    "error in {} during backward: {e}\nforward trace (anomaly mode):\n{trace}",
                    node.meta().name
                )),
                None => Error::Msg(format!(
                    "error in {} during backward: {e}",
                    node.meta().name
                )),
            }
        })?;
        executed.push(node.clone());

        // validate_outputs
        let edges = &node.meta().next_edges;
        if outputs.len() != edges.len() {
            return Err(Error::Msg(format!(
                "{} returned {} gradients but has {} next edges",
                node.meta().name,
                outputs.len(),
                edges.len()
            )));
        }

        for (edge, grad) in edges.iter().zip(outputs) {
            let (Some(child), Some(grad)) = (&edge.node, grad) else {
                continue;
            };
            let child_key = node_key(child);
            if !task.should_schedule(child_key) {
                continue;
            }
            // shape/dtype check against the consumer's declared input
            if let Some(im) = child.meta().input_metadata.get(edge.input_nr as usize) {
                if !im.matches(&grad) {
                    return Err(Error::Msg(format!(
                        "invalid gradient for {} input {}: expected shape {:?} {} got {:?} {}",
                        child.meta().name,
                        edge.input_nr,
                        im.shape,
                        im.dtype,
                        grad.shape(),
                        grad.dtype()
                    )));
                }
            }

            let deps = task.dependencies.get_mut(&child_key).ok_or_else(|| {
                Error::Msg(format!(
                    "dependency not found for {} (graph mutated during backward?)",
                    child.meta().name
                ))
            })?;
            *deps -= 1;
            let ready = *deps == 0;

            let mut buffer = task
                .not_ready
                .remove(&child_key)
                .unwrap_or_else(|| InputBuffer::new(child.meta().num_inputs()));
            buffer.add(edge.input_nr as usize, grad)?;

            if ready {
                queue.push(NodeTask {
                    node: child.clone(),
                    inputs: buffer,
                });
            } else {
                task.not_ready.insert(child_key, buffer);
            }
        }
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// autograd.cpp entry points
// ---------------------------------------------------------------------------

/// Upstream `_make_grads`: default root gradient is ones for scalar
/// outputs; non-scalar outputs require an explicit gradient.
fn make_grads(outputs: &[Variable], grads: Option<Vec<Tensor>>) -> Result<Vec<Option<Tensor>>> {
    match grads {
        Some(gs) => {
            if gs.len() != outputs.len() {
                return Err(Error::Msg(format!(
                    "got {} gradients for {} outputs",
                    gs.len(),
                    outputs.len()
                )));
            }
            for (v, g) in outputs.iter().zip(&gs) {
                if v.tensor().shape() != g.shape() {
                    return Err(Error::Msg(format!(
                        "gradient shape {:?} does not match output shape {:?}",
                        g.shape(),
                        v.tensor().shape()
                    )));
                }
            }
            Ok(gs.into_iter().map(Some).collect())
        }
        None => outputs
            .iter()
            .map(|v| {
                if v.tensor().elem_count() == 1 {
                    Ok(Some(Tensor::ones(v.tensor().shape(), v.tensor().dtype())))
                } else {
                    Err(Error::Msg(
                        "grad can be implicitly created only for scalar outputs".into(),
                    ))
                }
            })
            .collect(),
    }
}

fn root_edges(outputs: &[Variable]) -> Result<Vec<Edge>> {
    let edges: Vec<Edge> = outputs.iter().map(|v| v.grad_edge()).collect();
    if edges.iter().all(|e| !e.is_valid()) {
        return Err(Error::Msg(
            "element 0 of tensors does not require grad and does not have a grad_fn".into(),
        ));
    }
    Ok(edges)
}

/// `torch.autograd.backward`: run the graph, accumulating into leaf
/// `.grad`s via their AccumulateGrad nodes.
pub fn backward(
    outputs: &[Variable],
    grad_outputs: Option<Vec<Tensor>>,
    retain_graph: bool,
) -> Result<()> {
    let roots = root_edges(outputs)?;
    let grads = make_grads(outputs, grad_outputs)?;
    Engine::execute(roots, grads, retain_graph, None)?;
    Ok(())
}

/// `torch.autograd.grad`: return gradients of `outputs` w.r.t. `inputs`
/// without touching `.grad`, executing only the needed subgraph.
pub fn grad(
    outputs: &[Variable],
    inputs: &[Variable],
    grad_outputs: Option<Vec<Tensor>>,
    retain_graph: bool,
) -> Result<Vec<Option<Tensor>>> {
    let roots = root_edges(outputs)?;
    let grads = make_grads(outputs, grad_outputs)?;
    let capture_edges: Vec<Edge> = inputs.iter().map(|v| v.grad_edge()).collect();
    for (i, e) in capture_edges.iter().enumerate() {
        if !e.is_valid() {
            return Err(Error::Msg(format!(
                "input {i} to grad() does not require grad"
            )));
        }
    }
    Engine::execute(roots, grads, retain_graph, Some(capture_edges))
}
