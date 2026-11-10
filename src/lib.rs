//! Inferno — Tensor operations, automatic differentiation and probabilistic ML.
//!
//! Operators, distributions, likelihoods and differentiable linear
//! algebra use [`candle`] tensors. Root [`Tensor`], [`DType`] and [`Device`]
//! retain the dk-tensor backend used by the small [`gp::ExactGp`] reference.
//! These backends have distinct tensor types and gradient behavior.
pub mod covariance;
pub mod gp;
pub mod likelihood;
pub mod operators;
pub mod quadrature;
#[cfg(feature = "torch-kernels")]
pub use dk_cuda_ffi as torch_kernels;
pub use dk_tensor::{DType, Device, Tensor};
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<dk_tensor::Error> for Error {
    fn from(e: dk_tensor::Error) -> Self {
        Self(e.to_string())
    }
}
pub(crate) fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
pub(crate) fn matrix(t: &Tensor) -> Result<(usize, usize, Vec<f64>)> {
    ensure(t.shape().len() == 2, "expected a rank-two CPU tensor")?;
    ensure(
        matches!(t.device(), Device::Cpu),
        "CPU reference path requires CPU tensors",
    )?;
    let data = t.to_vec_f64()?;
    ensure(
        data.iter().all(|v| v.is_finite()),
        "tensor contains nonfinite values",
    )?;
    Ok((t.shape()[0], t.shape()[1], data))
}

/// Source-native differentiable tensor backend for the extracted ML stack.
pub use candle_core as candle;
pub mod distributions;
pub mod frechet;
pub mod kernels;
pub mod likelihoods;
pub mod linalg;
pub mod linalg_diff;
mod normal_quadrature;
pub mod reference_operators;
pub mod settings;
pub mod utils;

/// Candle neural-network layers, optimisers and parameter stores.
pub use candle_nn as nn;
/// Generalised autograd engine over the same dk-tensor type as the root API.
pub use dk_autograd as autograd;
pub use dk_autograd::Variable;
/// Curated source-native VAE and legacy multitask GP components.
pub use pandorapurple;
pub use pandorapurple::{flows, model, stochastics};

pub mod cache;
pub mod developer;
pub mod mean;

pub mod autograd_bridge;
pub use autograd_bridge::candle_apply;

/// Probabilistic model components and public aliases.
pub mod models;
/// Object-oriented, Candle-native notebook development API.
pub mod notebook;
/// A single tensor family for interactive model development.
pub mod prelude {
    //! Candle tensors, neural layers and model types for interactive development.
    pub use crate::candle::{DType, Device, Module, ModuleT, Result, Tensor};
    pub use crate::models::{PandoraGP, PandoraVae};
    pub use crate::nn::{self, Optimizer};
    pub use crate::notebook::{GaussianProcess, Model};
}

/// Runtime consistency checks and finite-support transport tools.
pub use pudding;
/// Candle tensor and operator diagnostics backed by Pudding.
pub mod diagnostics;
