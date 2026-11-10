//! Port of torch/csrc/autograd/input_buffer.{h,cpp} — accumulates the
//! gradients arriving at each input of a not-yet-ready node.
//!
//! Upstream's per-device stream/event choreography is Milestone-B work;
//! with one stream per device the accumulate is just `out = out + grad`
//! (upstream's own CPU path).

use dk_tensor::{cpu_ops, Tensor};

use crate::node::Result;

pub struct InputBuffer {
    slots: Vec<Option<Tensor>>,
}

impl InputBuffer {
    pub fn new(size: usize) -> InputBuffer {
        InputBuffer {
            slots: (0..size).map(|_| None).collect(),
        }
    }

    pub fn add(&mut self, pos: usize, grad: Tensor) -> Result<()> {
        if pos >= self.slots.len() {
            self.slots.resize_with(pos + 1, || None);
        }
        self.slots[pos] = match self.slots[pos].take() {
            None => Some(grad),
            Some(existing) => Some(cpu_ops::add(&existing, &grad)?),
        };
        Ok(())
    }

    pub fn get(&self, pos: usize) -> Option<&Tensor> {
        self.slots.get(pos).and_then(|s| s.as_ref())
    }

    pub fn into_vec(self) -> Vec<Option<Tensor>> {
        self.slots
    }
}
