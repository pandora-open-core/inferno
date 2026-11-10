//! Port of engine.cpp's `ReadyQueue`/`NodeTask`/`CompareNodeTaskTime`.
//!
//! Ordering: higher `sequence_nr` first — reverse creation order, which is
//! what makes gradient accumulation deterministic and keeps memory bounded
//! (upstream's comparator, minus the shutdown-task and reentrant-depth
//! terms that arrive with Milestone B's worker threads).
//!
//! Milestone A runs single-threaded, so this is a plain BinaryHeap behind a
//! Mutex with a non-blocking pop; the blocking Condvar pop comes with the
//! per-device threads.

use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use crate::input_buffer::InputBuffer;
use crate::node::Node;

pub struct NodeTask {
    pub node: Arc<dyn Node>,
    pub inputs: InputBuffer,
}

struct HeapEntry(NodeTask);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.node.meta().sequence_nr == other.0.node.meta().sequence_nr
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // max-heap on sequence_nr => newest nodes first
        self.0
            .node
            .meta()
            .sequence_nr
            .cmp(&other.0.node.meta().sequence_nr)
    }
}

pub struct ReadyQueue {
    heap: Mutex<BinaryHeap<HeapEntry>>,
}

impl ReadyQueue {
    pub fn new() -> ReadyQueue {
        ReadyQueue {
            heap: Mutex::new(BinaryHeap::new()),
        }
    }

    pub fn push(&self, task: NodeTask) {
        self.heap.lock().unwrap().push(HeapEntry(task));
    }

    pub fn pop(&self) -> Option<NodeTask> {
        self.heap.lock().unwrap().pop().map(|e| e.0)
    }

    pub fn is_empty(&self) -> bool {
        self.heap.lock().unwrap().is_empty()
    }
}

impl Default for ReadyQueue {
    fn default() -> Self {
        Self::new()
    }
}
