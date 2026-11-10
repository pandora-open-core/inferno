//! The variable layer — a redesign of torch/csrc/autograd/variable.{h,cpp},
//! not a port. Upstream `Variable` IS `at::Tensor`, with `AutogradMeta`
//! stored inside `c10::TensorImpl`; that intrusion is why it can't be
//! lifted. Here the relationship is inverted: composition over intrusion.
//! `dk_tensor::Tensor` stays autograd-free and `Variable` wraps it with an
//! optional `Arc<AutogradMeta>` shared by clones.
//!
//! Semantics preserved from upstream:
//! - leaf + requires_grad  -> grad_edge is the (lazily created, cached)
//!   `AccumulateGrad` node (variable.cpp `grad_accumulator()`);
//! - non-leaf              -> grad_edge is (grad_fn, output_nr);
//! - `.grad` lives on the meta, shared by every clone;
//! - view tracking (DifferentiableViewMeta) is NOT carried — in-place ops
//!   on views are out of scope for the curated op set; the version counter
//!   still catches mutations for SavedVariable.

use std::sync::{Arc, Mutex, Weak};

use dk_tensor::Tensor;

use crate::accumulate_grad::AccumulateGrad;
use crate::edge::Edge;
use crate::node::{Node, Result};

pub struct AutogradMeta {
    /// Accumulated gradient for leaves (upstream `AutogradMeta::grad_`).
    pub grad: Mutex<Option<Tensor>>,
    /// The producing backward node (non-leaf) — fixed at construction.
    pub grad_fn: Option<Arc<dyn Node>>,
    /// Which output of `grad_fn` this variable is.
    pub output_nr: u32,
    /// Leaf flag: participate in autograd even without a grad_fn.
    pub requires_grad: bool,
    /// Cached AccumulateGrad (upstream keeps a weak ref and re-creates on
    /// expiry — same here).
    pub grad_accumulator: Mutex<Weak<AccumulateGrad>>,
}

impl AutogradMeta {
    fn new_leaf() -> AutogradMeta {
        AutogradMeta {
            grad: Mutex::new(None),
            grad_fn: None,
            output_nr: 0,
            requires_grad: true,
            grad_accumulator: Mutex::new(Weak::new()),
        }
    }

    fn new_output(grad_fn: Arc<dyn Node>, output_nr: u32) -> AutogradMeta {
        AutogradMeta {
            grad: Mutex::new(None),
            grad_fn: Some(grad_fn),
            output_nr,
            requires_grad: false,
            grad_accumulator: Mutex::new(Weak::new()),
        }
    }
}

#[derive(Clone)]
pub struct Variable {
    pub(crate) tensor: Tensor,
    pub(crate) meta: Option<Arc<AutogradMeta>>,
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Variable[{:?}, requires_grad={}, has_grad_fn={}]",
            self.tensor,
            self.requires_grad(),
            self.grad_fn().is_some()
        )
    }
}

impl Variable {
    /// Plain data, no autograd participation.
    pub fn new(tensor: Tensor) -> Variable {
        Variable { tensor, meta: None }
    }

    pub fn leaf(tensor: Tensor, requires_grad: bool) -> Variable {
        let meta = requires_grad.then(|| Arc::new(AutogradMeta::new_leaf()));
        Variable { tensor, meta }
    }

    /// An op output wired to its backward node (the generated-VariableType
    /// pattern: `make_variable(result, grad_fn, output_nr)`).
    pub fn from_grad_fn(tensor: Tensor, grad_fn: Arc<dyn Node>, output_nr: u32) -> Variable {
        Variable {
            meta: Some(Arc::new(AutogradMeta::new_output(grad_fn, output_nr))),
            tensor,
        }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Detach: same data, no graph (fresh Variable, shared storage).
    pub fn detach(&self) -> Variable {
        Variable::new(self.tensor.clone())
    }

    pub fn requires_grad(&self) -> bool {
        self.meta
            .as_ref()
            .map(|m| m.requires_grad || m.grad_fn.is_some())
            .unwrap_or(false)
    }

    /// Turn a plain/leaf variable into a requires-grad leaf. Errors on
    /// non-leaves, like upstream.
    pub fn requires_grad_(&mut self) -> Result<()> {
        match &self.meta {
            Some(m) if m.grad_fn.is_some() => Err(dk_tensor::Error::Msg(
                "you can only change requires_grad flags of leaf variables".into(),
            )),
            Some(_) => Ok(()),
            None => {
                self.meta = Some(Arc::new(AutogradMeta::new_leaf()));
                Ok(())
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.meta.as_ref().map(|m| m.grad_fn.is_none()).unwrap_or(true)
    }

    pub fn grad_fn(&self) -> Option<Arc<dyn Node>> {
        self.meta.as_ref().and_then(|m| m.grad_fn.clone())
    }

    pub fn output_nr(&self) -> u32 {
        self.meta.as_ref().map(|m| m.output_nr).unwrap_or(0)
    }

    /// Accumulated `.grad` (leaves, after backward).
    pub fn grad(&self) -> Option<Tensor> {
        self.meta
            .as_ref()
            .and_then(|m| m.grad.lock().unwrap().clone())
    }

    pub fn zero_grad(&self) {
        if let Some(m) = &self.meta {
            *m.grad.lock().unwrap() = None;
        }
    }

    /// The edge gradients flow along to reach this variable — upstream
    /// `impl::gradient_edge` (variable.cpp).
    pub fn grad_edge(&self) -> Edge {
        match &self.meta {
            None => Edge::undefined(),
            Some(m) => match &m.grad_fn {
                Some(f) => Edge::new(f.clone(), m.output_nr),
                None if m.requires_grad => Edge::new(self.grad_accumulator(), 0),
                None => Edge::undefined(),
            },
        }
    }

    /// Lazily created, weakly cached AccumulateGrad — upstream
    /// `impl::grad_accumulator`.
    pub fn grad_accumulator(&self) -> Arc<AccumulateGrad> {
        let meta = self
            .meta
            .as_ref()
            .expect("grad_accumulator on a variable without autograd meta");
        let mut slot = meta.grad_accumulator.lock().unwrap();
        if let Some(acc) = slot.upgrade() {
            return acc;
        }
        let acc = Arc::new(AccumulateGrad::new(meta.clone(), &self.tensor));
        *slot = Arc::downgrade(&acc);
        acc
    }
}
