//! Error type. One enum, no error-crate dependency — this crate sits under
//! everything and must stay dependency-light.

use crate::DType;

#[derive(Debug)]
pub enum Error {
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    DTypeMismatch {
        op: &'static str,
        lhs: DType,
        rhs: DType,
    },
    DeviceMismatch {
        op: &'static str,
    },
    /// Storage does not cover the strided view (internal invariant breach).
    OutOfBounds {
        op: &'static str,
    },
    /// Op requires a contiguous tensor and the caller didn't provide one.
    NotContiguous {
        op: &'static str,
    },
    /// Built without the `cuda` feature, or no device available.
    CudaUnavailable,
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: shape mismatch {lhs:?} vs {rhs:?}")
            }
            Error::DTypeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: dtype mismatch {lhs} vs {rhs}")
            }
            Error::DeviceMismatch { op } => write!(f, "{op}: tensors on different devices"),
            Error::OutOfBounds { op } => write!(f, "{op}: strided view escapes storage"),
            Error::NotContiguous { op } => write!(f, "{op}: tensor must be contiguous"),
            Error::CudaUnavailable => write!(f, "cuda support not compiled in or unavailable"),
            Error::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(m: String) -> Self {
        Error::Msg(m)
    }
}
