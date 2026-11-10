//! Inferno — Shared tensor transforms and checkpoint-loading helpers.

pub mod load_weights;
pub use load_weights::load_varbuilder;
pub mod softplus;
pub use softplus::get_softplus;