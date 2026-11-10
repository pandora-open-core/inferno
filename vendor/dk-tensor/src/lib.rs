//! dk-tensor — a lightweight, autograd-free tensor.
//!
//! Part of the PyTorch/KeOps port (see the plan in the repo root's plans):
//! `dk_autograd::Variable` composes over this type, the way PyTorch could
//! not (its `AutogradMeta` is welded inside `c10::TensorImpl`). CPU storage
//! always; CUDA storage via cudarc behind the `cuda` feature, pinned to the
//! same cudarc version candle resolves so both stacks share one primary
//! context per device.
//!
//! Deliberately NOT here: ops with derivatives (dk-autograd), symbolic
//! formulas (pandorapurple::operators), candle interop (feature-gated
//! converters land with the GPU phases).

mod dtype;
mod error;
mod storage;
mod tensor;

pub mod cache_model;
pub mod panel;

pub mod cpu_ops;
pub mod kernels;
pub mod pairwise;

pub use dtype::DType;
pub use error::{Error, Result};
pub use storage::{Device, Storage};
pub use tensor::{contiguous_strides, StridedIndex, Tensor};
