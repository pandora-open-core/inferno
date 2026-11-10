//! dk-autograd — a from-first-principles port of PyTorch's autograd engine.
//!
//! Provenance: `torch/csrc/autograd/` at pytorch 540f3a53 (v2.14.0a0).
//! The *scheduling* core (engine.cpp, node.h, edge.h, graph_task.h,
//! input_buffer.cpp) ports cleanly — it is dispatcher-free. The *variable*
//! layer is a redesign, not a port: PyTorch's `AutogradMeta` lives inside
//! `c10::TensorImpl` and `Variable` IS `at::Tensor`; here `Variable`
//! composes over an autograd-free `dk_tensor::Tensor` instead.
//!
//! Map from upstream files to modules:
//!   node.h + function.h        -> node.rs   (Node trait, NodeMeta, sequence_nr)
//!   edge.h                     -> edge.rs
//!   graph_task.h + engine.cpp  -> graph_task.rs, ready_queue.rs, engine.rs
//!   input_buffer.{h,cpp}       -> input_buffer.rs
//!   input_metadata.h           -> input_metadata.rs
//!   saved_variable.{h,cpp}     -> saved_variable.rs
//!   variable.{h,cpp}           -> variable.rs (redesigned)
//!   functions/basic_ops        -> graph_root.rs
//!   functions/accumulate_grad  -> accumulate_grad.rs
//!   autograd.{h,cpp}           -> lib.rs entry points (backward / grad)
//!   anomaly_mode               -> anomaly.rs
//!   tools/autograd/derivatives.yaml (curated) -> derivatives/*
//!
//! Threading: Milestone A runs the ready-queue loop single-threaded on the
//! calling thread — deterministic and semantically complete. Milestone B
//! (per-device worker threads + reentrant thread pool, engine.cpp's
//! thread_main/reentrant_thread_init) lands with the GPU phases.

pub mod anomaly;
pub mod edge;
pub mod engine;
pub mod graph_task;
pub mod input_buffer;
pub mod input_metadata;
pub mod node;
pub mod ready_queue;
pub mod saved_variable;
pub mod variable;

pub mod accumulate_grad;
pub mod graph_root;

pub mod derivatives;
pub mod gradcheck;

pub use dk_tensor::{DType, Device, Tensor};
pub use edge::Edge;
pub use engine::{backward, grad, Engine};
pub use node::{Node, NodeMeta};
pub use variable::{AutogradMeta, Variable};

/// Grad-mode flag (PyTorch's `GradMode`). Thread-local, like upstream.
pub mod grad_mode {
    use std::cell::Cell;
    thread_local! {
        static ENABLED: Cell<bool> = const { Cell::new(true) };
    }
    pub fn is_enabled() -> bool {
        ENABLED.with(|e| e.get())
    }
    pub fn set_enabled(v: bool) {
        ENABLED.with(|e| e.set(v));
    }
    /// RAII guard: `let _g = no_grad();`
    pub fn no_grad() -> NoGradGuard {
        let prev = is_enabled();
        set_enabled(false);
        NoGradGuard { prev }
    }
    pub struct NoGradGuard {
        prev: bool,
    }
    impl Drop for NoGradGuard {
        fn drop(&mut self) {
            set_enabled(self.prev);
        }
    }
}
