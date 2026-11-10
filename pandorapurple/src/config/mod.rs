//! Inferno — Typed configuration for model components and numerical methods.

pub mod config;
pub mod sections;
pub use config::{SpectralVaeConfig, EncoderConfig, DirichletConfig, DecoderConfig, RootConfig, KernelNetworkConfig};
pub use sections::EWMAConfig;
