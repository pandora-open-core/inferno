//! Inferno — Gaussian-mixture bridges and Langevin dynamics with caller-owned objectives.

mod sde;
mod langevin;
pub use sde::{GaussianMixture, GmmSchrodingerBridge, SdeConfig};
pub use langevin::LangevinSDE;
pub use crate::flows::Direction;
