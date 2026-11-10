//! Port of torch/csrc/autograd/saved_variable.{h,cpp}.
//!
//! Two save modes, as upstream:
//! - saving an *input*: hold the meta strongly (no cycle possible);
//! - saving an *output of the node doing the saving*: holding the grad_fn
//!   strongly would create a Node -> SavedVariable -> Node cycle, so hold
//!   it weakly and rebuild on unpack.
//!
//! `unpack` re-checks the tensor's version counter against the version at
//! save time and fails with upstream's message when an in-place op
//! intervened.

use std::sync::{Arc, Weak};

use dk_tensor::Tensor;

use crate::node::{Error, Node, Result};
use crate::variable::Variable;

pub struct SavedVariable {
    tensor: Tensor,
    saved_version: u64,
    /// Strong meta for non-output saves.
    meta: Option<Arc<crate::variable::AutogradMeta>>,
    /// Weak grad_fn for output saves (upstream `weak_grad_fn_`).
    weak_grad_fn: Option<Weak<dyn Node>>,
    output_nr: u32,
}

impl SavedVariable {
    /// `is_output`: the saved variable is an output of the node that is
    /// saving it (e.g. `exp` saving its result).
    pub fn save(v: &Variable, is_output: bool) -> SavedVariable {
        let (meta, weak_grad_fn) = if is_output {
            (
                None,
                v.grad_fn().as_ref().map(Arc::downgrade),
            )
        } else {
            (v.meta.clone(), None)
        };
        SavedVariable {
            tensor: v.tensor().clone(),
            saved_version: v.tensor().version(),
            meta,
            weak_grad_fn,
            output_nr: v.output_nr(),
        }
    }

    pub fn unpack(&self) -> Result<Variable> {
        if self.tensor.version() != self.saved_version {
            return Err(Error::Msg(
                "one of the variables needed for gradient computation has been \
                 modified by an inplace operation"
                    .into(),
            ));
        }
        if let Some(weak) = &self.weak_grad_fn {
            let grad_fn = weak.upgrade().ok_or_else(|| {
                Error::Msg(
                    "trying to unpack a SavedVariable whose graph has been freed \
                     (backward through the graph a second time without retain_graph=True?)"
                        .into(),
                )
            })?;
            return Ok(Variable::from_grad_fn(
                self.tensor.clone(),
                grad_fn,
                self.output_nr,
            ));
        }
        Ok(Variable {
            tensor: self.tensor.clone(),
            meta: self.meta.clone(),
        })
    }

    /// Data only, no graph — for backwards that never differentiate again.
    pub fn unpack_data(&self) -> Result<Tensor> {
        Ok(self.unpack()?.into_tensor())
    }
}
