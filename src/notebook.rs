//! Inferno — Candle model construction, registered parameters, training and exact GP prediction.

use crate::candle::{DType, Device, ModuleT, Result, Tensor, Var};
use crate::nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

/// Owns a module, its named parameters, device and training mode.
/// Build every trainable child from the supplied builder (use `vb.pp("child")`).
/// External variables are not automatically discovered or registered.
pub struct Model<M> {
    module: M,
    variables: VarMap,
    device: Device,
    training: bool,
}
impl<M: ModuleT> Model<M> {
    pub fn new(device: &Device, build: impl FnOnce(VarBuilder<'_>) -> Result<M>) -> Result<Self> {
        let variables = VarMap::new();
        let module = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
        Ok(Self {
            module,
            variables,
            device: device.clone(),
            training: true,
        })
    }
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.module.forward_t(input, self.training)
    }
    pub fn train(&mut self) {
        self.training = true;
    }
    pub fn eval(&mut self) {
        self.training = false;
    }
    pub fn is_training(&self) -> bool {
        self.training
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    /// Shared parameter handles; create optimizers after building all children.
    pub fn parameters(&self) -> Vec<Var> {
        self.variables.all_vars()
    }
    pub fn module(&self) -> &M {
        &self.module
    }
    pub fn adam(&self, learning_rate: f64) -> Result<AdamW> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            crate::candle::bail!("learning rate must be finite and positive")
        }
        AdamW::new(
            self.parameters(),
            ParamsAdamW {
                lr: learning_rate,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.variables.save(path)
    }
    /// Construct the same architecture before loading. Updates shared handles;
    /// a failed load may have updated some variables (Candle VarMap semantics).
    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.variables.load(path)
    }
}

/// CPU-only, small-data RBF Gaussian process with fixed hyperparameters.
/// `fit` performs exact conditioning, not gradient-based hyperparameter learning.
/// O(n^3) fit and O(n^2) storage; prediction materializes a query covariance.
pub struct GaussianProcess {
    kernel: crate::covariance::Rbf,
    noise: f64,
    posterior: Option<crate::gp::Posterior<crate::covariance::Rbf>>,
}
pub struct Prediction {
    pub mean: Tensor,
    /// Latent marginal variance, excluding observation noise and jitter.
    pub variance: Tensor,
}
fn numerical_error(error: crate::Error) -> crate::candle::Error {
    crate::candle::Error::Msg(error.to_string())
}
fn reference(input: &Tensor) -> Result<crate::Tensor> {
    if !input.device().is_cpu() {
        crate::candle::bail!("GaussianProcess requires CPU tensors")
    }
    let values = input
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    crate::Tensor::from_f64_slice(&values, input.dims(), crate::DType::F64)
        .map_err(|e| crate::candle::Error::Msg(e.to_string()))
}
impl GaussianProcess {
    /// Kernel variance, kernel lengthscale, observation noise variance.
    pub fn new(variance: f64, lengthscale: f64, noise: f64) -> Result<Self> {
        let kernel = crate::covariance::Rbf::new(variance, lengthscale).map_err(numerical_error)?;
        crate::likelihood::Gaussian::new(noise).map_err(numerical_error)?;
        Ok(Self {
            kernel,
            noise,
            posterior: None,
        })
    }
    /// Inputs `[n, features]`, targets `[n]`. A failed refit preserves the old fit.
    pub fn fit(&mut self, inputs: &Tensor, targets: &Tensor) -> Result<()> {
        let posterior = crate::gp::ExactGp::new(
            self.kernel,
            crate::likelihood::Gaussian::new(self.noise).map_err(numerical_error)?,
            1e-8,
        )
        .map_err(numerical_error)?
        .condition(&reference(inputs)?, &reference(targets)?)
        .map_err(numerical_error)?;
        self.posterior = Some(posterior);
        Ok(())
    }
    pub fn predict(&self, inputs: &Tensor) -> Result<Prediction> {
        let posterior = self
            .posterior
            .as_ref()
            .ok_or_else(|| crate::candle::Error::Msg("call fit before predict".into()))?;
        let prediction = posterior
            .predict(&reference(inputs)?)
            .map_err(numerical_error)?;
        let convert = |t: crate::Tensor| -> Result<Tensor> {
            let values = t
                .to_vec_f64()
                .map_err(|e| crate::candle::Error::Msg(e.to_string()))?;
            Tensor::from_vec(values, t.shape(), &Device::Cpu)
        };
        Ok(Prediction {
            mean: convert(prediction.mean)?,
            variance: convert(prediction.variance)?,
        })
    }
}
