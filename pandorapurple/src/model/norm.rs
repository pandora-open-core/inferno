//! Inferno — Normalisation layers and tensor statistics for neural networks.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Init, Module, VarBuilder};

pub use candle_nn::LayerNormConfig;

/// Layer normalisation that keeps its input in the graph.
///
/// Semantically identical to [`candle_nn::LayerNorm`]; see the module docs for
/// why it is not that type.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias: Some(bias), remove_mean: true, eps }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self { weight, bias: None, remove_mean: true, eps }
    }

    /// RMS norm — no mean removal.
    pub fn rms_norm(weight: Tensor, eps: f64) -> Self {
        Self { weight, bias: None, remove_mean: false, eps }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Deliberately NO fast-path branch. `candle_nn` selects the fused
        // kernel on `x.is_contiguous()`, which makes differentiability depend
        // on the memory layout of the input — a property no caller thinks
        // about and none of them controls.
        let x_dtype = x.dtype();
        let internal = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal)?;
        let x = if self.remove_mean {
            let mean = (x.sum_keepdim(D::Minus1)? / hidden as f64)?;
            x.broadcast_sub(&mean)?
        } else {
            x
        };
        let norm = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let normed = x.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        let out = normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight.to_dtype(x_dtype)?)?;
        match &self.bias {
            None => Ok(out),
            Some(b) => out.broadcast_add(&b.to_dtype(x_dtype)?),
        }
    }
}

/// Drop-in replacement for [`candle_nn::layer_norm`].
///
/// Same registered names (`weight`, `bias`), same inits, same defaults.
pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let config: LayerNormConfig = config.into();
    let weight = vb.get_with_hints(size, "weight", Init::Const(1.))?;
    let bias = if config.affine {
        Some(vb.get_with_hints(size, "bias", Init::Const(0.))?)
    } else {
        None
    };
    Ok(LayerNorm { weight, bias, remove_mean: config.remove_mean, eps: config.eps })
}

#[cfg(test)]
mod tests {
    //! Verification of normalisation layers and tensor statistics for neural networks.

    use super::*;
    use candle_core::{Device, Var};
    use candle_nn::VarMap;

    fn dx_and_dparams(size: usize, cfg: LayerNormConfig) -> (Option<f32>, usize) {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let v: Vec<f32> = (0..4 * size).map(|i| ((i as f32) * 0.7).sin()).collect();
        let x = Var::from_tensor(&Tensor::from_vec(v, (4, size), &dev).unwrap()).unwrap();
        let ln = layer_norm(size, cfg, vb.pp("ln")).unwrap();
        let y = ln.forward(x.as_tensor()).unwrap();
        let g = y.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let dx = g
            .get(x.as_tensor())
            .map(|t| t.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap());
        let data = vm.data().lock().unwrap();
        let hit = data.values().filter(|v| g.get(v.as_tensor()).is_some()).count();
        (dx, hit)
    }

    /// The whole point. `candle_nn`'s own `LayerNorm` returns `None` here —
    /// verified directly in the sibling test below, so this is a claim about
    /// a measured difference and not about a reading of the source.
    #[test]
    fn the_input_and_both_parameters_stay_in_the_graph() {
        let (dx, params) = dx_and_dparams(8, LayerNormConfig::from(1e-5));
        let dx = dx.expect("gradient must reach the input");
        assert!(dx > 0.0 && dx.is_finite(), "dx = {dx}");
        assert_eq!(params, 2, "weight AND bias must both receive gradient");
    }

    /// The behaviour being replaced, asserted rather than described.
    ///
    /// It fails LOUDLY the day candle fixes this upstream, which is exactly
    /// when someone should come back and delete this module. A comment saying
    /// "candle is broken" would rot silently instead.
    #[test]
    fn candle_layer_norm_silently_drops_the_gradient_on_its_default_config() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let v: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.7).sin()).collect();
        let x = Var::from_tensor(&Tensor::from_vec(v, (4, 8), &dev).unwrap()).unwrap();
        let ln = candle_nn::layer_norm(8, 1e-5, vb.pp("ln")).unwrap();
        let y = ln.forward(x.as_tensor()).unwrap();

        // Note it does not ERROR — `apply_op3_no_bwd` records no op, so the
        // backward walk never reaches the node and reports success.
        let g = y.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        assert!(
            g.get(x.as_tensor()).is_none(),
            "candle_nn::LayerNorm now propagates to its input — delete \
             model::norm and go back to the upstream type"
        );
    }

    /// Values agree with candle's, so the swap is a gradient change and not a
    /// numerical one. Any checkpoint's forward pass is unaffected.
    #[test]
    fn the_forward_value_matches_candles_to_the_last_bit_of_f32() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let v: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).cos() * 3.0).collect();
        let x = Tensor::from_vec(v, (8, 8), &dev).unwrap();

        // ONE parameter set, read by both norms — a fresh VarBuilder per norm
        // would compare two different weights and pass for the wrong reason.
        let ours = layer_norm(8, LayerNormConfig::from(1e-5), vb.pp("ln")).unwrap();
        let theirs = candle_nn::layer_norm(8, 1e-5, vb.pp("ln")).unwrap();

        let a = ours.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = theirs.forward(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert!((p - q).abs() <= 1e-6, "element {i}: {p} vs {q}");
        }
    }
}
