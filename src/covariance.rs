//! Inferno — Composable CPU covariance kernels with validated scalar hyperparameters.
//! This dk-tensor reference API returns F64 values and does not preserve autograd.
//! Use `kernels::standard` for differentiable Candle covariance matrices.

use crate::{ensure, matrix, DType, Result, Tensor};
pub trait Covariance {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor>;
}
#[derive(Debug, Clone, Copy)]
pub struct Rbf {
    variance: f64,
    lengthscale: f64,
}
impl Rbf {
    pub fn new(variance: f64, lengthscale: f64) -> Result<Self> {
        ensure(
            variance.is_finite() && variance > 0.0 && lengthscale.is_finite() && lengthscale > 0.0,
            "variance and lengthscale must be finite and positive",
        )?;
        Ok(Self {
            variance,
            lengthscale,
        })
    }
}
impl Covariance for Rbf {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let (n, d, x) = matrix(x)?;
        let (m, e, y) = matrix(y)?;
        ensure(
            d == e && d > 0,
            "covariance input feature dimensions must match and be nonempty",
        )?;
        let mut out = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let distance: f64 = (0..d)
                    .map(|k| {
                        scaled_difference(x[i * d + k], y[j * d + k], self.lengthscale).powi(2)
                    })
                    .sum();
                out[i * m + j] = if distance == 0.0 {
                    self.variance
                } else {
                    (self.variance.ln() - 0.5 * distance)
                        .exp()
                        .min(self.variance)
                };
            }
        }
        ensure(
            out.iter().all(|v| v.is_finite()),
            "covariance computation overflowed",
        )?;
        Ok(Tensor::from_f64_slice(&out, &[n, m], DType::F64)?)
    }
}

/// Fallible closure adapter for custom covariance implementations.
pub struct FunctionCovariance<F>(pub F);
impl<F: Fn(&Tensor, &Tensor) -> Result<Tensor>> Covariance for FunctionCovariance<F> {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        (self.0)(x, y)
    }
}
/// Sum and product preserve positive semidefiniteness for valid input kernels.
pub struct Sum<A, B>(pub A, pub B);
pub struct Product<A, B>(pub A, pub B);
fn combine<A: Covariance, B: Covariance>(
    a: &A,
    b: &B,
    x: &Tensor,
    y: &Tensor,
    product: bool,
) -> Result<Tensor> {
    let (n, d, _) = matrix(x)?;
    let (m, e, _) = matrix(y)?;
    ensure(d == e && d > 0, "kernel feature mismatch")?;
    let (r, c, av) = matrix(&a.covariance(x, y)?)?;
    let (s, t, bv) = matrix(&b.covariance(x, y)?)?;
    ensure(
        (r, c) == (n, m) && (s, t) == (n, m),
        "component kernel shape mismatch",
    )?;
    let out: Vec<_> = av
        .iter()
        .zip(bv)
        .map(|(a, b)| if product { a * b } else { a + b })
        .collect();
    ensure(
        out.iter().all(|v| v.is_finite()),
        "kernel composition overflow",
    )?;
    Ok(Tensor::from_f64_slice(&out, &[n, m], DType::F64)?)
}
impl<A: Covariance, B: Covariance> Covariance for Sum<A, B> {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        combine(&self.0, &self.1, x, y, false)
    }
}
impl<A: Covariance, B: Covariance> Covariance for Product<A, B> {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        combine(&self.0, &self.1, x, y, true)
    }
}
#[derive(Clone, Copy, Debug)]
pub enum MaternOrder {
    Half,
    ThreeHalves,
    FiveHalves,
}
#[derive(Clone, Debug)]
pub struct Matern {
    variance: f64,
    lengthscales: Vec<f64>,
    order: MaternOrder,
}
impl Matern {
    /// ARD lengthscales: one strictly positive scale per feature.
    pub fn new(variance: f64, lengthscales: Vec<f64>, order: MaternOrder) -> Result<Self> {
        ensure(
            variance.is_finite()
                && variance > 0.0
                && !lengthscales.is_empty()
                && lengthscales.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid Matern parameters",
        )?;
        Ok(Self {
            variance,
            lengthscales,
            order,
        })
    }
}
impl Covariance for Matern {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let (n, d, x) = matrix(x)?;
        let (m, e, y) = matrix(y)?;
        ensure(
            d == e && d == self.lengthscales.len(),
            "Matern dimension mismatch",
        )?;
        let mut out = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                // hypot avoids squaring large finite distances.
                let r = (0..d).fold(0.0_f64, |r, k| {
                    r.hypot(scaled_difference(
                        x[i * d + k],
                        y[j * d + k],
                        self.lengthscales[k],
                    ))
                });
                out[i * m + j] = if !r.is_finite() {
                    0.0
                } else {
                    let (z, log_p) = match self.order {
                        MaternOrder::Half => (r, 0.0),
                        MaternOrder::ThreeHalves => {
                            let z = 3.0_f64.sqrt() * r;
                            (z, z.ln_1p())
                        }
                        MaternOrder::FiveHalves => {
                            let z = 5.0_f64.sqrt() * r;
                            let log_p = if z <= 1.0 {
                                (z + z * z / 3.0).ln_1p()
                            } else {
                                2.0 * z.ln() + (1.0 / 3.0 + 1.0 / z + (1.0 / z).powi(2)).ln()
                            };
                            (z, log_p)
                        }
                    };
                    if !z.is_finite() {
                        0.0
                    } else if z == 0.0 {
                        self.variance
                    } else {
                        // Log space avoids overflow before the exponential decay
                        // and preserves tiny tails rescued by a large variance.
                        (self.variance.ln() + (log_p - z).min(0.0))
                            .exp()
                            .min(self.variance)
                    }
                };
            }
        }
        ensure(out.iter().all(|v| v.is_finite()), "Matern overflow")?;
        Ok(Tensor::from_f64_slice(&out, &[n, m], DType::F64)?)
    }
}

impl<T: Covariance + ?Sized> Covariance for Box<T> {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        (**self).covariance(x, y)
    }
}

fn scaled_difference(x: f64, y: f64, scale: f64) -> f64 {
    let difference = x - y;
    if difference.is_finite() {
        difference / scale
    } else {
        x / scale - y / scale
    }
}

/// Linear covariance `variance * <x, y>`.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    variance: f64,
}
/// Polynomial covariance `variance * (<x, y> + offset)^degree`.
#[derive(Clone, Copy, Debug)]
pub struct Polynomial {
    variance: f64,
    offset: f64,
    degree: u32,
}
/// Product periodic covariance with period and lengthscale shared across features.
#[derive(Clone, Copy, Debug)]
pub struct Periodic {
    variance: f64,
    lengthscale: f64,
    period: f64,
}
/// Rational-quadratic covariance, a scale mixture of RBF kernels.
#[derive(Clone, Copy, Debug)]
pub struct RationalQuadratic {
    variance: f64,
    lengthscale: f64,
    alpha: f64,
}
/// Constant covariance `variance` for every pair.
#[derive(Clone, Copy, Debug)]
pub struct Constant {
    variance: f64,
}
/// White covariance on equal feature vectors (including duplicate rows).
/// Observation-index noise belongs in the likelihood instead.
#[derive(Clone, Copy, Debug)]
pub struct WhiteNoise {
    variance: f64,
}
fn positive(values: &[f64]) -> Result<()> {
    ensure(
        values.iter().all(|v| v.is_finite() && *v > 0.),
        "kernel parameters must be finite and positive",
    )
}
impl Linear {
    pub fn new(variance: f64) -> Result<Self> {
        positive(&[variance])?;
        Ok(Self { variance })
    }
}
impl Constant {
    pub fn new(variance: f64) -> Result<Self> {
        positive(&[variance])?;
        Ok(Self { variance })
    }
}
impl WhiteNoise {
    pub fn new(variance: f64) -> Result<Self> {
        positive(&[variance])?;
        Ok(Self { variance })
    }
}
impl Polynomial {
    pub fn new(variance: f64, offset: f64, degree: u32) -> Result<Self> {
        positive(&[variance])?;
        ensure(
            offset.is_finite() && offset >= 0. && degree > 0 && degree <= i32::MAX as u32,
            "polynomial requires a finite nonnegative offset and positive i32 degree",
        )?;
        Ok(Self {
            variance,
            offset,
            degree,
        })
    }
}
impl Periodic {
    pub fn new(variance: f64, lengthscale: f64, period: f64) -> Result<Self> {
        positive(&[variance, lengthscale, period])?;
        Ok(Self {
            variance,
            lengthscale,
            period,
        })
    }
}
impl RationalQuadratic {
    pub fn new(variance: f64, lengthscale: f64, alpha: f64) -> Result<Self> {
        positive(&[variance, lengthscale, alpha])?;
        Ok(Self {
            variance,
            lengthscale,
            alpha,
        })
    }
}
fn pairwise(x: &Tensor, y: &Tensor, f: impl Fn(&[f64], &[f64]) -> f64) -> Result<Tensor> {
    let (n, d, x) = matrix(x)?;
    let (m, e, y) = matrix(y)?;
    ensure(
        d == e && d > 0,
        "kernel feature dimensions must match and be nonempty",
    )?;
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        for j in 0..m {
            out.push(f(&x[i * d..(i + 1) * d], &y[j * d..(j + 1) * d]));
        }
    }
    ensure(
        out.iter().all(|v| v.is_finite()),
        "kernel computation overflowed",
    )?;
    Ok(Tensor::from_f64_slice(&out, &[n, m], DType::F64)?)
}
impl Covariance for Linear {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |a, b| {
            self.variance * a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>()
        })
    }
}
impl Covariance for Polynomial {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |a, b| {
            self.variance
                * (self.offset + a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>())
                    .powi(self.degree as i32)
        })
    }
}
impl Covariance for Periodic {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |a, b| {
            let r2 = a
                .iter()
                .zip(b)
                .map(|(a, b)| {
                    ((std::f64::consts::PI * scaled_difference(*a, *b, self.period)).sin()
                        / self.lengthscale)
                        .powi(2)
                })
                .sum::<f64>();
            (self.variance.ln() - 2. * r2).exp().min(self.variance)
        })
    }
}
impl Covariance for RationalQuadratic {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |a, b| {
            let r = a.iter().zip(b).fold(0f64, |r, (a, b)| {
                r.hypot(scaled_difference(*a, *b, self.lengthscale))
            });
            // Compute log(1 + r²/(2 alpha)) without overflowing r².
            let z = 2. * r.ln() - std::f64::consts::LN_2 - self.alpha.ln();
            let log_base = if z > 0. {
                z + (-z).exp().ln_1p()
            } else {
                z.exp().ln_1p()
            };
            (self.variance.ln() - self.alpha * log_base)
                .exp()
                .min(self.variance)
        })
    }
}
impl Covariance for Constant {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |_, _| self.variance)
    }
}
impl Covariance for WhiteNoise {
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        pairwise(x, y, |a, b| if a == b { self.variance } else { 0. })
    }
}
