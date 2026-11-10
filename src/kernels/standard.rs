//! Inferno — Standard covariance kernels on Candle tensors, preserving first-order autograd.
//!
//! Inputs are floating-point matrices `[n, features]`. Parameters are scalar
//! tensors on the same device and dtype; wrap them in `Var` to train them.
//! White noise uses equality of feature vectors, so it has no input derivative.
pub use crate::covariance::MaternOrder;
use candle_core::{DType, Result, Tensor, D};

/// Covariance families with caller-owned tensor hyperparameters.
#[derive(Clone, Debug)]
pub enum Kernel {
    Rbf {
        variance: Tensor,
        lengthscale: Tensor,
    },
    Matern {
        variance: Tensor,
        lengthscale: Tensor,
        order: MaternOrder,
    },
    Linear {
        variance: Tensor,
    },
    Polynomial {
        variance: Tensor,
        offset: Tensor,
        degree: u32,
    },
    Periodic {
        variance: Tensor,
        lengthscale: Tensor,
        period: Tensor,
    },
    RationalQuadratic {
        variance: Tensor,
        lengthscale: Tensor,
        alpha: Tensor,
    },
    Constant {
        variance: Tensor,
    },
    WhiteNoise {
        variance: Tensor,
    },
}
fn parameter(p: &Tensor, x: &Tensor, allow_zero: bool) -> Result<()> {
    if p.rank() != 0 || p.dtype() != x.dtype() || !p.device().same_device(x.device()) {
        candle_core::bail!("kernel parameters must be scalars matching the input dtype and device")
    }
    let value = p.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    if !value.is_finite() || value < 0. || (!allow_zero && value == 0.) {
        candle_core::bail!("invalid kernel parameter")
    }
    Ok(())
}
impl Kernel {
    /// Evaluate `[n, m]` covariance without detaching inputs or parameters.
    /// Matérn uses the zero subgradient at coincident points; the 1/2 kernel
    /// is not classically differentiable there. Higher derivatives through
    /// the guarded square root are not supported at coincidence.
    pub fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let (n, d) = x.dims2()?;
        let (m, e) = y.dims2()?;
        if d == 0
            || d != e
            || x.dtype() != y.dtype()
            || !matches!(x.dtype(), DType::F32 | DType::F64)
            || !x.device().same_device(y.device())
        {
            candle_core::bail!(
                "kernel inputs must have matching nonempty features, floating dtype and device"
            )
        }
        let variance = match self {
            Self::Rbf { variance, .. }
            | Self::Matern { variance, .. }
            | Self::Linear { variance }
            | Self::Polynomial { variance, .. }
            | Self::Periodic { variance, .. }
            | Self::RationalQuadratic { variance, .. }
            | Self::Constant { variance }
            | Self::WhiteNoise { variance } => variance,
        };
        parameter(variance, x, false)?;
        let delta = || x.unsqueeze(1)?.broadcast_sub(&y.unsqueeze(0)?);
        let dist = |lengthscale: &Tensor| -> Result<Tensor> {
            parameter(lengthscale, x, false)?;
            delta()?.broadcast_div(lengthscale)?.sqr()?.sum(D::Minus1)
        };
        let base = match self {
            Self::Rbf { lengthscale, .. } => dist(lengthscale)?.affine(-0.5, 0.)?.exp()?,
            Self::Matern {
                lengthscale, order, ..
            } => {
                let r2 = dist(lengthscale)?;
                // Never take sqrt(0) on the tape: its backward divides by zero.
                let mask = r2.detach().gt(0.)?;
                let safe = mask.where_cond(&r2, &r2.ones_like()?)?.sqrt()?;
                let r = mask.where_cond(&safe, &r2.zeros_like()?)?;
                match order {
                    MaternOrder::Half => r.neg()?.exp()?,
                    MaternOrder::ThreeHalves => {
                        let z = r.affine(3f64.sqrt(), 0.)?;
                        z.affine(1., 1.)?.mul(&z.neg()?.exp()?)?
                    }
                    MaternOrder::FiveHalves => {
                        let z = r.affine(5f64.sqrt(), 0.)?;
                        z.affine(1., 1.)?
                            .add(&z.sqr()?.affine(1. / 3., 0.)?)?
                            .mul(&z.neg()?.exp()?)?
                    }
                }
            }
            Self::Linear { .. } => x.matmul(&y.t()?)?,
            Self::Polynomial { offset, degree, .. } => {
                parameter(offset, x, true)?;
                if *degree == 0 || *degree > i32::MAX as u32 {
                    candle_core::bail!("invalid polynomial degree")
                }
                // Integer powers support negative inner products.
                let b = x.matmul(&y.t()?)?.broadcast_add(offset)?;
                let mut out = b.ones_like()?;
                let mut power = b;
                let mut k = *degree;
                while k > 0 {
                    if k & 1 == 1 {
                        out = out.mul(&power)?;
                    }
                    k >>= 1;
                    if k > 0 {
                        power = power.sqr()?;
                    }
                }
                out
            }
            Self::Periodic {
                lengthscale,
                period,
                ..
            } => {
                parameter(lengthscale, x, false)?;
                parameter(period, x, false)?;
                delta()?
                    .broadcast_div(period)?
                    .affine(std::f64::consts::PI, 0.)?
                    .sin()?
                    .broadcast_div(lengthscale)?
                    .sqr()?
                    .sum(D::Minus1)?
                    .affine(-2., 0.)?
                    .exp()?
            }
            Self::RationalQuadratic {
                lengthscale, alpha, ..
            } => {
                parameter(alpha, x, false)?;
                dist(lengthscale)?
                    .broadcast_div(&alpha.affine(2., 0.)?)?
                    .affine(1., 1.)?
                    .log()?
                    .broadcast_mul(alpha)?
                    .neg()?
                    .exp()?
            }
            Self::Constant { .. } => Tensor::ones((n, m), x.dtype(), x.device())?,
            Self::WhiteNoise { .. } => delta()?
                .detach()
                .eq(0.)?
                .to_dtype(x.dtype())?
                .sum(D::Minus1)?
                .eq(d as f64)?
                .to_dtype(x.dtype())?,
        };
        base.broadcast_mul(variance)
    }
}
