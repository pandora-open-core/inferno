//! Pudding: an alpha proof engine for mathematical consistency in ML workflows.
//!
//! Runtime checks produce observations, not formal certificates. The inherited
//! Lean development is in `proofs/lean`; its hypotheses and source
//! freezes do not automatically apply to a running Inferno workflow.
pub mod bytecode;
pub mod consistency;
pub use consistency::{Check, CheckResult, Engine, Report, TensorSnapshot, Tolerance};

/// Finite-support numerical reference; outputs are not formal certificates.
pub mod equilibrium;

/// Reproducible experiments and portable graph artifacts.
pub mod research;
