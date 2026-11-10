//! Inferno — Public aliases for the spectral VAE and legacy differentiable multitask GP.

pub use crate::model::SpectralVae as PandoraVae;
pub use crate::notebook::{GaussianProcess as ExactGP, Model};
pub use crate::pandorapurple::gp::{DifferentiableGp as PandoraGP, KernelHyperTensor};
