//! Inferno — CPU adapter for first-order F32/F64 gradients between dk-autograd and Candle.

use candle_core::{Tensor as CandleTensor, Var};
use dk_autograd::node::{collect_next_edges, Result};
use dk_autograd::{Node, NodeMeta, Variable};
use dk_tensor::{DType, Device, Error, Tensor};
use std::sync::{Arc, Mutex};

fn error(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}
fn to_candle(x: &Tensor) -> Result<CandleTensor> {
    if x.device() != Device::Cpu {
        return Err(error("Candle VJP adapter currently requires CPU tensors"));
    }
    let dtype = match x.dtype() {
        DType::F32 => candle_core::DType::F32,
        DType::F64 => candle_core::DType::F64,
        _ => return Err(error("Candle VJP adapter requires F32 or F64")),
    };
    CandleTensor::from_vec(x.to_vec_f64()?, x.shape(), &candle_core::Device::Cpu)
        .and_then(|x| x.to_dtype(dtype))
        .map_err(error)
}
fn from_candle(x: &CandleTensor) -> Result<Tensor> {
    let dtype = match x.dtype() {
        candle_core::DType::F32 => DType::F32,
        candle_core::DType::F64 => DType::F64,
        _ => return Err(error("Candle VJP output requires F32 or F64")),
    };
    if !matches!(x.device(), candle_core::Device::Cpu) {
        return Err(error("Candle VJP output must remain on CPU"));
    }
    let values = x
        .to_dtype(candle_core::DType::F64)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f64>())
        .map_err(error)?;
    Tensor::from_f64_slice(&values, x.dims(), dtype)
}
struct CandleVjp {
    meta: NodeMeta,
    saved: Mutex<Option<(Vec<Var>, CandleTensor)>>,
}
impl Node for CandleVjp {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let guard = self.saved.lock().map_err(error)?;
        let (inputs, output) = guard.as_ref().ok_or_else(|| {
            error("Candle graph released; use retain_graph for repeated backward")
        })?;
        let Some(seed) = grads.first().and_then(Option::as_ref) else {
            return Ok(vec![None; inputs.len()]);
        };
        let seed = to_candle(seed)?;
        let grad = output
            .mul(&seed)
            .and_then(|x| x.sum_all())
            .and_then(|x| x.backward())
            .map_err(error)?;
        inputs
            .iter()
            .map(|input| grad.get(input.as_tensor()).map(from_candle).transpose())
            .collect()
    }
    fn release_saved(&self) {
        if let Ok(mut guard) = self.saved.lock() {
            *guard = None;
        }
    }
}

/// Wrap a differentiable Candle function in dk-autograd. Single tensor output;
/// arbitrary input shapes. First-order gradients only. Captured tensors are
/// constants with respect to dk-autograd; pass parameters through `inputs`.
pub fn candle_apply<F>(inputs: &[&Variable], forward: F) -> Result<Variable>
where
    F: FnOnce(&[CandleTensor]) -> candle_core::Result<CandleTensor>,
{
    let variables: Vec<Var> = inputs
        .iter()
        .map(|x| Var::from_tensor(&to_candle(x.tensor())?).map_err(error))
        .collect::<Result<_>>()?;
    let tensors: Vec<_> = variables.iter().map(|x| x.as_tensor().clone()).collect();
    let output = forward(&tensors).map_err(error)?;
    let tensor = from_candle(&output)?;
    if !dk_autograd::grad_mode::is_enabled() || !inputs.iter().any(|x| x.requires_grad()) {
        return Ok(Variable::new(tensor));
    }
    let meta = NodeMeta::new("CandleVjp", collect_next_edges(inputs)).with_input_metadata(vec![
        dk_autograd::input_metadata::InputMetadata::of(&tensor),
    ]);
    Ok(Variable::from_grad_fn(
        tensor,
        Arc::new(CandleVjp {
            meta,
            saved: Mutex::new(Some((variables, output))),
        }),
        0,
    ))
}
