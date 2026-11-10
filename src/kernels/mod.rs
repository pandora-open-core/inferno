//! Inferno — Pairwise distribution comparisons and covariance kernels on Candle tensors.

pub mod mmd;
pub mod sinkhorn;

pub use mmd::{mmd_imq, mmd_rbf, mmd_rbf_multi, scalar_f64, silverman_bandwidth, KernelChoice};
pub use sinkhorn::{
    sinkhorn_divergence, sinkhorn_potentials, SinkhornCost, SinkhornOptions, SinkhornOutput,
};

use candle_core::{Result as CResult, Tensor, D};

/// Squared pairwise distance `||x_i - y_j||^2`, shape `[n, m]`. Numerical
/// floor at 0 (clamps tiny negatives caused by cancellation).
pub(crate) fn sq_pdist(x: &Tensor, y: &Tensor) -> CResult<Tensor> {
    // ||x||^2: [n,1], ||y||^2: [1,m], xy: [n,m]
    let xx = x.sqr()?.sum_keepdim(D::Minus1)?; // [n, 1]
    let yy_row = y.sqr()?.sum_keepdim(D::Minus1)?; // [m, 1]
    let yy = yy_row.transpose(0, 1)?; // [1, m]
    let xy = x.matmul(&y.transpose(0, 1)?)?; // [n, m]
                                             // d = xx + yy - 2 xy
    let two_xy = (xy * 2.0_f64)?;
    let d = xx.broadcast_add(&yy)?.broadcast_sub(&two_xy)?;
    // Floor tiny negatives caused by catastrophic cancellation.
    // (No `.relu()` in candle-core proper — clamp via `maximum(0)`.)
    let zero = Tensor::zeros_like(&d)?;
    d.maximum(&zero)
}

pub mod standard;
