//! Port of torch/csrc/autograd/input_metadata.h — heavily reduced.
//! Upstream carries SymInt shapes, nested/batched tensor cases and stream
//! info; here it is the plain (shape, dtype, device) triple the engine's
//! validate_outputs needs.

use dk_tensor::{DType, Device, Tensor};

#[derive(Clone, Debug)]
pub struct InputMetadata {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

impl InputMetadata {
    pub fn of(t: &Tensor) -> InputMetadata {
        InputMetadata {
            shape: t.shape().to_vec(),
            dtype: t.dtype(),
            device: t.device(),
        }
    }

    pub fn matches(&self, grad: &Tensor) -> bool {
        grad.shape() == self.shape.as_slice()
            && grad.dtype() == self.dtype
            && grad.device() == self.device
    }
}
