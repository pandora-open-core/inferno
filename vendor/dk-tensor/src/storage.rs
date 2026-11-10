//! Storage: where the bytes live.
//!
//! CPU storage is an `Arc<Vec<u8>>` — untyped bytes, typed access goes
//! through the dtype on the owning `Tensor`. Sharing is by `Arc` clone;
//! views alias the same allocation with different (shape, strides, offset).
//!
//! CUDA storage (behind the `cuda` feature) wraps a cudarc `CudaSlice<u8>`
//! on the same primary context candle retains, so device pointers handed to
//! the cxx-bridged kernels and to candle interop are on one context.
//!
//! In-place mutation of shared storage is the caller's contract to manage —
//! `Tensor::bump_version` is how autograd's SavedVariable detects it, exactly
//! like PyTorch's VariableVersion.

use std::sync::Arc;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// CUDA ordinal.
    Cuda(usize),
}

impl Device {
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }
}

#[derive(Clone)]
pub enum Storage {
    Cpu(Arc<Vec<u8>>),
    #[cfg(feature = "cuda")]
    Cuda {
        buf: Arc<cudarc::driver::CudaSlice<u8>>,
        ordinal: usize,
    },
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda { ordinal, .. } => Device::Cuda(*ordinal),
        }
    }

    pub fn len_bytes(&self) -> usize {
        match self {
            Storage::Cpu(b) => b.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda { buf, .. } => buf.len(),
        }
    }

    /// CPU bytes, or None on a device tensor.
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match self {
            Storage::Cpu(b) => Some(b.as_slice()),
            #[cfg(feature = "cuda")]
            _ => None,
        }
    }

    /// Same allocation? Views compare equal; copies do not.
    pub fn ptr_eq(&self, other: &Storage) -> bool {
        match (self, other) {
            (Storage::Cpu(a), Storage::Cpu(b)) => Arc::ptr_eq(a, b),
            #[cfg(feature = "cuda")]
            (Storage::Cuda { buf: a, .. }, Storage::Cuda { buf: b, .. }) => Arc::ptr_eq(a, b),
            #[cfg(feature = "cuda")]
            _ => false,
        }
    }
}
