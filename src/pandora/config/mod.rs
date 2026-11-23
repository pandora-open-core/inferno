//! Inferno — Typed configuration for model components and numerical methods.

pub mod config;
pub mod sections;
pub use config::{
    DecoderConfig, DirichletConfig, EncoderConfig, KernelNetworkConfig, RootConfig,
    SpectralVaeConfig,
};
pub use sections::EWMAConfig;
