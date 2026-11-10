//! Reference CPU kernels.
//!
//! These are correctness-first implementations (f64 accumulation, naive
//! loops): the oracle for gradcheck and the GPU paths, and the executor for
//! GPU-less tests. Broadcast follows PyTorch: shapes right-aligned, size-1
//! dims stretch.

use crate::error::{Error, Result};
use crate::Tensor;

/// Right-aligned broadcast of two shapes.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return Err(Error::ShapeMismatch {
                op: "broadcast",
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            });
        };
    }
    Ok(out)
}

fn check_same_dtype(op: &'static str, a: &Tensor, b: &Tensor) -> Result<()> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(())
}

/// Elementwise binary op with broadcasting; math in f64, output in the
/// (shared) input dtype.
pub fn binary_map(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Tensor> {
    check_same_dtype(op, a, b)?;
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let ae = a.expand(&shape)?;
    let be = b.expand(&shape)?;
    let av = ae.to_vec_f64()?;
    let bv = be.to_vec_f64()?;
    let out: Vec<f64> = av.iter().zip(&bv).map(|(&x, &y)| f(x, y)).collect();
    Tensor::from_f64_slice(&out, &shape, a.dtype())
}

pub fn unary_map(a: &Tensor, f: impl Fn(f64) -> f64) -> Result<Tensor> {
    let v: Vec<f64> = a.to_vec_f64()?.into_iter().map(f).collect();
    Tensor::from_f64_slice(&v, a.shape(), a.dtype())
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_map("add", a, b, |x, y| x + y)
}
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_map("sub", a, b, |x, y| x - y)
}
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_map("mul", a, b, |x, y| x * y)
}
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_map("div", a, b, |x, y| x / y)
}
pub fn neg(a: &Tensor) -> Result<Tensor> {
    unary_map(a, |x| -x)
}

pub fn scale(a: &Tensor, s: f64) -> Result<Tensor> {
    unary_map(a, |x| x * s)
}

/// Sum of every element, as a scalar tensor of the same dtype.
pub fn sum_all(a: &Tensor) -> Result<Tensor> {
    let s: f64 = a.to_vec_f64()?.iter().sum();
    Ok(Tensor::scalar(s, a.dtype()))
}

pub fn ones_like(a: &Tensor) -> Tensor {
    Tensor::ones(a.shape(), a.dtype())
}
pub fn zeros_like(a: &Tensor) -> Tensor {
    Tensor::zeros(a.shape(), a.dtype())
}

/// Reduce a broadcasted gradient back to `target_shape` by summing the
/// stretched dims — the standard broadcast-backward companion. This is what
/// PyTorch's `sum_to_size` does inside generated backwards.
pub fn sum_to_shape(grad: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    if grad.shape() == target_shape {
        return grad.contiguous();
    }
    let gshape = grad.shape().to_vec();
    let rank = gshape.len();
    let lead = rank - target_shape.len();
    let gv = grad.to_vec_f64()?;
    let tn: usize = target_shape.iter().product();
    let mut out = vec![0.0f64; tn.max(1)];
    // walk grad in logical order, computing the target flat index
    let gstrides = crate::contiguous_strides(&gshape);
    let tstrides = crate::contiguous_strides(target_shape);
    for (flat, v) in gv.iter().enumerate() {
        let mut tflat = 0usize;
        for d in lead..rank {
            let gi = (flat / gstrides[d]) % gshape[d];
            let td = d - lead;
            let ti = if target_shape[td] == 1 { 0 } else { gi };
            tflat += ti * tstrides[td];
        }
        out[tflat] += v;
    }
    Tensor::from_f64_slice(&out, target_shape, grad.dtype())
}

// ---------------------------------------------------------------------------
// dim-wise reductions
// ---------------------------------------------------------------------------

/// Split a shape at `dim` into (outer, len, inner) so a dim-wise reduction
/// is one flat triple loop regardless of rank.
pub fn split_at_dim(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    if dim >= shape.len() {
        return Err(Error::Msg(format!(
            "dim {dim} out of range for shape {shape:?}"
        )));
    }
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    Ok((outer, shape[dim], inner))
}

pub fn shape_without_dim(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    s.remove(dim);
    s
}

pub fn sum_dim(a: &Tensor, dim: usize) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(a.shape(), dim)?;
    let v = a.to_vec_f64()?;
    let mut out = vec![0.0; outer * inner];
    for o in 0..outer {
        for l in 0..len {
            for i in 0..inner {
                out[o * inner + i] += v[(o * len + l) * inner + i];
            }
        }
    }
    Tensor::from_f64_slice(&out, &shape_without_dim(a.shape(), dim), a.dtype())
}

/// `logsumexp` along `dim`, max-shifted. The shift cancels exactly, so this
/// is an identity rather than an approximation.
pub fn logsumexp_dim(a: &Tensor, dim: usize) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(a.shape(), dim)?;
    let v = a.to_vec_f64()?;
    let mut out = vec![0.0; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut max = f64::NEG_INFINITY;
            for l in 0..len {
                let x = v[(o * len + l) * inner + i];
                if x > max {
                    max = x;
                }
            }
            if max == f64::NEG_INFINITY {
                out[o * inner + i] = f64::NEG_INFINITY;
                continue;
            }
            let mut s = 0.0;
            for l in 0..len {
                s += (v[(o * len + l) * inner + i] - max).exp();
            }
            out[o * inner + i] = max + s.ln();
        }
    }
    Tensor::from_f64_slice(&out, &shape_without_dim(a.shape(), dim), a.dtype())
}

/// `softmax` along `dim` (shape preserved).
pub fn softmax_dim(a: &Tensor, dim: usize) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(a.shape(), dim)?;
    let v = a.to_vec_f64()?;
    let mut out = vec![0.0; v.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut max = f64::NEG_INFINITY;
            for l in 0..len {
                let x = v[(o * len + l) * inner + i];
                if x > max {
                    max = x;
                }
            }
            let mut s = 0.0;
            for l in 0..len {
                let e = (v[(o * len + l) * inner + i] - max).exp();
                out[(o * len + l) * inner + i] = e;
                s += e;
            }
            for l in 0..len {
                out[(o * len + l) * inner + i] /= s;
            }
        }
    }
    Tensor::from_f64_slice(&out, a.shape(), a.dtype())
}

pub fn log_softmax_dim(a: &Tensor, dim: usize) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(a.shape(), dim)?;
    let v = a.to_vec_f64()?;
    let mut out = vec![0.0; v.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut max = f64::NEG_INFINITY;
            for l in 0..len {
                let x = v[(o * len + l) * inner + i];
                if x > max {
                    max = x;
                }
            }
            let mut s = 0.0;
            for l in 0..len {
                s += (v[(o * len + l) * inner + i] - max).exp();
            }
            let lse = max + s.ln();
            for l in 0..len {
                out[(o * len + l) * inner + i] = v[(o * len + l) * inner + i] - lse;
            }
        }
    }
    Tensor::from_f64_slice(&out, a.shape(), a.dtype())
}

/// `(values, argmax)` along `dim`. Ties resolve to the LOWEST index, which
/// is what makes the backward deterministic.
pub fn max_dim(a: &Tensor, dim: usize) -> Result<(Tensor, Vec<usize>)> {
    let (outer, len, inner) = split_at_dim(a.shape(), dim)?;
    let v = a.to_vec_f64()?;
    let mut vals = vec![0.0; outer * inner];
    let mut idx = vec![0usize; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best = f64::NEG_INFINITY;
            let mut bl = 0usize;
            for l in 0..len {
                let x = v[(o * len + l) * inner + i];
                if x > best {
                    best = x;
                    bl = l;
                }
            }
            vals[o * inner + i] = best;
            idx[o * inner + i] = bl;
        }
    }
    Ok((
        Tensor::from_f64_slice(&vals, &shape_without_dim(a.shape(), dim), a.dtype())?,
        idx,
    ))
}

/// Scatter a reduced-shape gradient back along `dim` at `idx` (max/min
/// backward): zero everywhere except the winning element.
pub fn scatter_along_dim(
    grad: &Tensor,
    idx: &[usize],
    full_shape: &[usize],
    dim: usize,
) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(full_shape, dim)?;
    let g = grad.to_vec_f64()?;
    let mut out = vec![0.0; outer * len * inner];
    for o in 0..outer {
        for i in 0..inner {
            let l = idx[o * inner + i];
            out[(o * len + l) * inner + i] = g[o * inner + i];
        }
    }
    Tensor::from_f64_slice(&out, full_shape, grad.dtype())
}

/// Broadcast a reduced-shape tensor back along `dim` (sum/mean backward).
pub fn expand_along_dim(grad: &Tensor, full_shape: &[usize], dim: usize) -> Result<Tensor> {
    let (outer, len, inner) = split_at_dim(full_shape, dim)?;
    let g = grad.to_vec_f64()?;
    let mut out = vec![0.0; outer * len * inner];
    for o in 0..outer {
        for l in 0..len {
            for i in 0..inner {
                out[(o * len + l) * inner + i] = g[o * inner + i];
            }
        }
    }
    Tensor::from_f64_slice(&out, full_shape, grad.dtype())
}

/// 2-D matmul reference: [m,k] x [k,n] -> [m,n], f64 accumulation.
pub fn mm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_dtype("mm", a, b)?;
    if a.rank() != 2 || b.rank() != 2 || a.shape()[1] != b.shape()[0] {
        return Err(Error::ShapeMismatch {
            op: "mm",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let av = a.to_vec_f64()?;
    let bv = b.to_vec_f64()?;
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for l in 0..k {
            let ail = av[i * k + l];
            if ail == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += ail * bv[l * n + j];
            }
        }
    }
    Tensor::from_f64_slice(&out, &[m, n], a.dtype())
}

/// Strided in-logical-order equality helper for tests.
pub fn allclose(a: &Tensor, b: &Tensor, atol: f64, rtol: f64) -> Result<bool> {
    if a.shape() != b.shape() {
        return Ok(false);
    }
    let av = a.to_vec_f64()?;
    let bv = b.to_vec_f64()?;
    Ok(av
        .iter()
        .zip(&bv)
        .all(|(&x, &y)| (x - y).abs() <= atol + rtol * y.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::StridedIndex;
    use crate::DType;

    #[test]
    fn broadcast_add_matches_manual() {
        let a = Tensor::from_f64_slice(&[1., 2., 3.], &[3], DType::F32).unwrap();
        let b = Tensor::from_f64_slice(&[10., 20.], &[2, 1], DType::F32).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![11., 12., 13., 21., 22., 23.]
        );
    }

    #[test]
    fn sum_to_shape_inverts_broadcast() {
        let g = Tensor::ones(&[2, 3], DType::F64);
        let r = sum_to_shape(&g, &[3]).unwrap();
        assert_eq!(r.to_vec_f64().unwrap(), vec![2., 2., 2.]);
        let r2 = sum_to_shape(&g, &[2, 1]).unwrap();
        assert_eq!(r2.to_vec_f64().unwrap(), vec![3., 3.]);
    }

    #[test]
    fn mm_reference() {
        let a = Tensor::from_f64_slice(&[1., 2., 3., 4.], &[2, 2], DType::F64).unwrap();
        let b = Tensor::from_f64_slice(&[5., 6., 7., 8.], &[2, 2], DType::F64).unwrap();
        let c = mm(&a, &b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![19., 22., 43., 50.]);
    }

    #[test]
    fn strided_index_covers_all_elements_once() {
        let shape = [2usize, 3, 2];
        let strides = crate::contiguous_strides(&shape);
        let offs: Vec<usize> = StridedIndex::new(&shape, &strides, 0).collect();
        assert_eq!(offs.len(), 12);
        let mut sorted = offs.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
    }
}
