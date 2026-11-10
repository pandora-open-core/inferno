//! Inferno — Tensor linear-algebra primitives, factorisations and matrix solves.

use candle_core::{DType, Result, Tensor};

// ---------------------------------------------------------------------------
// dense f64 kernels (row-major, D×D)
// ---------------------------------------------------------------------------

fn eye(d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

fn matmul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0; d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..d {
                c[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    c
}

/// Lower-triangular Cholesky factor `L` (`A = L Lᵀ`), with diagonal jitter for
/// numerical robustness.
/// Why a Cholesky could not be completed even with jitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotPsd {
    /// Diagonal position where the pivot went non-positive.
    pub pivot: usize,
    /// The offending pivot value.
    pub value: f64,
    /// Largest jitter tried.
    pub jitter_tried: f64,
    pub attempts: usize,
}

impl core::fmt::Display for NotPsd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "matrix is not positive definite: pivot {} was {:.3e} after {} attempt(s), \
             up to {:.3e} jitter",
            self.pivot, self.value, self.attempts, self.jitter_tried
        )
    }
}

impl std::error::Error for NotPsd {}

/// Cholesky that REPORTS failure instead of hiding it, retrying with
/// escalating jitter. Port of gpytorch's `psd_safe_cholesky`.
///
/// The plain [`chol`] below clamps a non-positive pivot to `1e-12` and
/// carries on, so an indefinite matrix yields a confident, wrong factor
/// with no error signal anywhere. That is the failure this exists to
/// remove: every caller that actually cares whether its matrix was PSD
/// should be here instead.
///
/// Returns the factor and the jitter it needed — the second value is the
/// diagnostic. A matrix that suddenly starts needing `1e-4` where it used
/// to take `0` is a conditioning problem upstream, and the only way to see
/// that is to be told.
///
/// `dtype` keys the base jitter and should be the width the DATA came in
/// at, not the width the arithmetic runs at. A Gram matrix accumulated in
/// f64 from f32 inputs is conditioned like f32 data, because that is the
/// precision its entries actually carry.
pub fn psd_safe_cholesky(
    a: &[f64],
    d: usize,
    dtype: crate::settings::Dtype,
) -> core::result::Result<(Vec<f64>, f64), NotPsd> {
    use crate::settings::{cholesky_jitter, cholesky_max_tries, verbose_linalg};

    // First attempt is unjittered: a well-conditioned matrix should never
    // pay for the pathological case, and "needed no jitter" is itself the
    // most useful thing the return value can say.
    let mut last = match chol_checked(a, d, 0.0) {
        Ok(l) => return Ok((l, 0.0)),
        Err(e) => e,
    };

    let base = cholesky_jitter::value(dtype);
    let tries = cholesky_max_tries::value();
    let mut jitter = base;
    for attempt in 1..=tries {
        match chol_checked(a, d, jitter) {
            Ok(l) => {
                if verbose_linalg::value() {
                    tracing::debug!(
                        d,
                        dtype = dtype.as_str(),
                        jitter,
                        attempt,
                        "psd_safe_cholesky succeeded with jitter"
                    );
                }
                return Ok((l, jitter));
            }
            Err(e) => {
                last = e;
                jitter *= 10.0;
            }
        }
    }
    // `jitter` was multiplied once past the last attempt; report what was
    // actually tried.
    let tried = if tries == 0 { 0.0 } else { jitter / 10.0 };
    if verbose_linalg::value() {
        tracing::warn!(
            d,
            dtype = dtype.as_str(),
            tries,
            "psd_safe_cholesky exhausted its jitter budget"
        );
    }
    Err(NotPsd {
        pivot: last.0,
        value: last.1,
        jitter_tried: tried,
        attempts: tries + 1,
    })
}

/// One Cholesky attempt. `Err((i, pivot))` on the first non-positive pivot
/// — no clamping, no silent recovery.
fn chol_checked(a: &[f64], d: usize, jitter: f64) -> core::result::Result<Vec<f64>, (usize, f64)> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let piv = s + jitter;
                // Strictly positive AND finite: a NaN input propagates to a
                // NaN pivot, which compares false against `> 0.0` and would
                // otherwise fall through as if it were merely small.
                if !(piv > 0.0) || !piv.is_finite() {
                    return Err((i, piv));
                }
                l[i * d + i] = piv.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// Cholesky with a fixed jitter and a clamped pivot.
///
/// WARNING: this cannot fail, which means it cannot tell you it was given
/// something that is not positive definite — the `.max(1e-12)` below turns
/// an indefinite matrix into a plausible-looking factor. Prefer
/// [`psd_safe_cholesky`] anywhere the answer matters; this is kept for the
/// existing callers that depend on its infallibility.
pub(crate) fn chol(a: &[f64], d: usize, jitter: f64) -> Vec<f64> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                l[i * d + i] = (s + jitter).max(1e-12).sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    l
}

/// Solve `A X = B` given `L` (`A = L Lᵀ`). `b` is `[D, M]` row-major.
fn chol_solve(l: &[f64], b: &[f64], d: usize, m: usize) -> Vec<f64> {
    let mut x = vec![0.0; d * m];
    for col in 0..m {
        let mut y = vec![0.0; d];
        // forward: L y = b
        for i in 0..d {
            let mut s = b[i * m + col];
            for k in 0..i {
                s -= l[i * d + k] * y[k];
            }
            y[i] = s / l[i * d + i];
        }
        // backward: Lᵀ x = y
        for i in (0..d).rev() {
            let mut s = y[i];
            for k in (i + 1)..d {
                s -= l[k * d + i] * x[k * m + col];
            }
            x[i * m + col] = s / l[i * d + i];
        }
    }
    x
}

/// `exp(A)` via scaling-and-squaring with a truncated Taylor series. After
/// scaling `B = A/2^s` so `‖B‖ < ½`, an 18-term series is at machine precision;
/// then square `s` times. No linear solve required.
/// `pub(crate)`: [`crate::linalg_diff::matrix_exp_diff`] reuses this exact
/// kernel for both its forward AND its backward (the adjoint is another
/// `expm`, at `2D`), so the two paths can never disagree numerically.
pub(crate) fn mat_exp(a: &[f64], d: usize) -> Vec<f64> {
    let norm = (0..d)
        .map(|i| (0..d).map(|j| a[i * d + j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let s: u32 = if norm > 0.5 {
        (norm.log2().ceil() as i64 + 1).max(0) as u32
    } else {
        0
    };
    let scale = 2f64.powi(s as i32);
    let b: Vec<f64> = a.iter().map(|v| v / scale).collect();

    let mut term = eye(d); // B^0 / 0!
    let mut e = eye(d);
    for k in 1..=18 {
        let prod = matmul(&term, &b, d);
        term = prod.iter().map(|v| v / k as f64).collect(); // term *= B/k
        for idx in 0..d * d {
            e[idx] += term[idx];
        }
    }
    for _ in 0..s {
        e = matmul(&e, &e, d);
    }
    e
}

/// Symmetric eigendecomposition via cyclic Jacobi rotations. Returns
/// `(eigvals [D], eigvecs [D*D] row-major — column `i` is eigenvector `i`)`.
/// Symmetric eigendecomposition by cyclic Jacobi. Returns
/// `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j`.
///
/// Two things to know before using this outside its original callers:
/// eigenvalues come back UNSORTED, and the convergence test is ABSOLUTE
/// (`off.sqrt() < 1e-14`), so a badly-scaled matrix either burns all 100
/// sweeps or exits having done nothing. Pre-scale to unit Frobenius norm
/// and rescale the eigenvalues afterwards.
pub fn jacobi_eigh(a_in: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = eye(d);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..d {
            for q in (p + 1)..d {
                off += a[p * d + q] * a[p * d + q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A ← Jᵀ A J (rotate rows then columns of the p,q plane)
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig = (0..d).map(|i| a[i * d + i]).collect();
    (eig, v)
}

/// Apply a scalar function to the eigenvalues of a symmetric matrix:
/// `f(M) = Q diag(f(λ)) Qᵀ`.
fn sym_func_dense(a: &[f64], d: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let (eig, v) = jacobi_eigh(a, d);
    let fe: Vec<f64> = eig.iter().map(|&l| f(l)).collect();
    let mut r = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for m in 0..d {
                s += v[i * d + m] * fe[m] * v[j * d + m];
            }
            r[i * d + j] = s;
        }
    }
    r
}

// ---------------------------------------------------------------------------
// candle wrappers (batched over leading dims)
// ---------------------------------------------------------------------------

fn dims_of(a: &Tensor) -> Result<(Vec<usize>, usize, usize)> {
    let dims = a.dims().to_vec();
    if dims.len() < 2 {
        return Err(candle_core::Error::Msg(
            "linalg: expected a [.., D, D] tensor".into(),
        ));
    }
    let d = dims[dims.len() - 1];
    if dims[dims.len() - 2] != d {
        return Err(candle_core::Error::Msg(
            "linalg: trailing dims must be square".into(),
        ));
    }
    let bsz: usize = dims[..dims.len() - 2].iter().product::<usize>().max(1);
    Ok((dims, d, bsz))
}

/// Apply a per-`D×D`-matrix host kernel across the batch.
fn map_square<F: Fn(&[f64], usize) -> Vec<f64>>(a: &Tensor, f: F) -> Result<Tensor> {
    let (dims, d, bsz) = dims_of(a)?;
    let flat = a
        .to_dtype(DType::F64)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let mut out = Vec::with_capacity(flat.len());
    for b in 0..bsz {
        out.extend(f(&flat[b * d * d..(b + 1) * d * d], d));
    }
    Tensor::from_vec(out, dims, a.device())?.to_dtype(a.dtype())
}

/// Matrix exponential of a batch of square matrices `[.., D, D]`.
pub fn matrix_exp(a: &Tensor) -> Result<Tensor> {
    map_square(a, mat_exp)
}

/// Symmetric eigendecomposition of a single `[D, D]` matrix →
/// `(lam [D] ASCENDING, v [D, D])` with eigenvector `j` in COLUMN `j`, so that
/// `A == V diag(lam) Vᵀ`.
///
/// The Tensor-level wrapper over [`jacobi_eigh`] that applies its two
/// documented preconditions ONCE, here, rather than at every call site:
///
/// * **Prescale.** `jacobi_eigh`'s convergence test is ABSOLUTE
///   (`off.sqrt() < 1e-14`), so a badly-scaled matrix either burns all 100
///   sweeps or exits having done nothing. Scale to unit Frobenius norm,
///   decompose, scale the eigenvalues back. Same rule as
///   [`crate::operators::linsolve`] and the `cg.rs` spectrum test — stated in
///   one place now instead of three.
/// * **Sort.** `jacobi_eigh` returns eigenvalues in whatever order the sweeps
///   left them. Pairing `lam[i]` with column `i` of an unsorted result is a
///   silent mismatch: "the top eigenvalue" and its eigenvector come from
///   different indices, and nothing about the shapes says so.
///
/// An exactly-zero matrix returns `(0, I)` rather than dividing by its norm.
pub fn symmetric_eigh(a: &Tensor) -> Result<(Tensor, Tensor)> {
    let (d, d2) = a.dims2()?;
    if d != d2 {
        return Err(candle_core::Error::Msg(format!(
            "symmetric_eigh: expected a square [D, D], got [{d}, {d2}]"
        )));
    }
    let (dev, dt) = (a.device().clone(), a.dtype());
    let host: Vec<f64> = a.to_dtype(DType::F64)?.flatten_all()?.to_vec1()?;

    let fro = host.iter().map(|v| v * v).sum::<f64>().sqrt();
    if fro == 0.0 {
        let lam = Tensor::zeros(d, dt, &dev)?;
        let v = Tensor::from_vec(eye(d), (d, d), &dev)?.to_dtype(dt)?;
        return Ok((lam, v));
    }
    let scaled: Vec<f64> = host.iter().map(|v| v / fro).collect();
    let (lam_s, vecs) = jacobi_eigh(&scaled, d);

    // Sort ascending, carrying the eigenvector COLUMNS with the values.
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| lam_s[i].partial_cmp(&lam_s[j]).unwrap());
    let lam: Vec<f64> = order.iter().map(|&k| lam_s[k] * fro).collect();
    let mut v = vec![0f64; d * d];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..d {
            v[row * d + new_col] = vecs[row * d + old_col];
        }
    }

    Ok((
        Tensor::from_vec(lam, d, &dev)?.to_dtype(dt)?,
        Tensor::from_vec(v, (d, d), &dev)?.to_dtype(dt)?,
    ))
}

/// Symmetric matrix square root `A^½` (SPD batch `[.., D, D]`) via Jacobi eig.
pub fn sym_sqrt(a: &Tensor) -> Result<Tensor> {
    map_square(a, |m, d| sym_func_dense(m, d, |l| l.max(0.0).sqrt()))
}

/// Symmetric inverse square root `A^{-½}` (SPD batch) via Jacobi eig
/// (eigenvalues floored for stability).
pub fn sym_inv_sqrt(a: &Tensor) -> Result<Tensor> {
    map_square(a, |m, d| {
        sym_func_dense(m, d, |l| 1.0 / l.max(1e-12).sqrt())
    })
}

/// A cached ("lazy") Cholesky factorisation of an SPD matrix (batch). Factor
/// once with [`CholeskyFactor::new`], then `solve` / `inverse` / `logdet`
/// reuse the stored lower factor `L`.
pub struct CholeskyFactor {
    l: Tensor, // lower-triangular [.., D, D] (host-computed, detached)
    d: usize,
    bsz: usize,
    dims: Vec<usize>,
}

impl CholeskyFactor {
    /// Factor SPD `a` `[.., D, D]` (diagonal `jitter` added for stability).
    pub fn new(a: &Tensor, jitter: f64) -> Result<Self> {
        let (dims, d, bsz) = dims_of(a)?;
        let l = map_square(a, |m, dd| chol(m, dd, jitter))?;
        Ok(Self { l, d, bsz, dims })
    }

    pub fn factor(&self) -> &Tensor {
        &self.l
    }

    /// Solve `A X = B` for `b: [.., D, M]` → `[.., D, M]`. Host-side (no grad);
    /// for a differentiable solve use [`inverse`](Self::inverse) + `matmul`.
    pub fn solve(&self, b: &Tensor) -> Result<Tensor> {
        let bdims = b.dims().to_vec();
        let m = *bdims.last().unwrap();
        if bdims[bdims.len() - 2] != self.d {
            return Err(candle_core::Error::Msg(
                "CholeskyFactor::solve: B row dim ≠ D".into(),
            ));
        }
        let l_flat = self
            .l
            .to_dtype(DType::F64)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let b_flat = b
            .to_dtype(DType::F64)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let (d, dd) = (self.d, self.d * self.d);
        let mut out = Vec::with_capacity(self.bsz * d * m);
        for bi in 0..self.bsz {
            let lb = &l_flat[bi * dd..(bi + 1) * dd];
            let bb = &b_flat[bi * d * m..(bi + 1) * d * m];
            out.extend(chol_solve(lb, bb, d, m));
        }
        Tensor::from_vec(out, bdims, b.device())?.to_dtype(b.dtype())
    }

    /// `A⁻¹` `[.., D, D]` — detached; matmul it against a grad-tracked vector
    /// for a differentiable `A⁻¹ v`.
    pub fn inverse(&self) -> Result<Tensor> {
        // Identity broadcast to A's (batched) shape, then solve A·A⁻¹ = I.
        let eye =
            Tensor::eye(self.d, self.l.dtype(), self.l.device())?.broadcast_as(self.l.shape())?;
        self.solve(&eye)
    }

    /// `log|A| = 2 Σ_i log L_ii` → `[..]`.
    pub fn logdet(&self) -> Result<Tensor> {
        // diagonal of L: gather L[.., i, i].
        let d = self.d;
        let l_flat = self
            .l
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let mut out = Vec::with_capacity(self.bsz);
        for bi in 0..self.bsz {
            let base = bi * d * d;
            let mut s = 0.0;
            for i in 0..d {
                s += l_flat[base + i * d + i].ln();
            }
            out.push((2.0 * s) as f32);
        }
        let lead = if self.dims.len() > 2 {
            self.dims[..self.dims.len() - 2].to_vec()
        } else {
            vec![]
        };
        let t = Tensor::from_vec(out, lead, self.l.device())?;
        t.to_dtype(self.l.dtype())
    }
}

/// Convenience: solve `A X = B` for SPD `A` in one shot.
pub fn cholesky_solve(a: &Tensor, b: &Tensor, jitter: f64) -> Result<Tensor> {
    CholeskyFactor::new(a, jitter)?.solve(b)
}

#[cfg(test)]
mod tests {
    //! Verification of tensor linear-algebra primitives, factorisations and matrix solves.

    use super::*;
    use candle_core::{Device, Tensor};

    fn approx(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn matrix_exp_known_cases() {
        let dev = Device::Cpu;
        // exp(0) = I
        let z = Tensor::zeros((3, 3), DType::F32, &dev).unwrap();
        let e = matrix_exp(&z).unwrap();
        let i = Tensor::eye(3, DType::F32, &dev).unwrap();
        assert!(approx(&e, &i) < 1e-5);

        // exp(diag(1,2)) = diag(e, e²)
        let diag = Tensor::new(&[[1.0f32, 0.0], [0.0, 2.0]], &dev).unwrap();
        let ed = matrix_exp(&diag).unwrap().to_vec2::<f32>().unwrap();
        assert!((ed[0][0] - 1f32.exp()).abs() < 1e-4);
        assert!((ed[1][1] - 2f32.exp()).abs() < 1e-3);
        assert!(ed[0][1].abs() < 1e-5);

        // nilpotent [[0,1],[0,0]] → [[1,1],[0,1]]
        let nil = Tensor::new(&[[0.0f32, 1.0], [0.0, 0.0]], &dev).unwrap();
        let en = matrix_exp(&nil).unwrap().to_vec2::<f32>().unwrap();
        assert!((en[0][0] - 1.0).abs() < 1e-5 && (en[0][1] - 1.0).abs() < 1e-5);
        assert!(en[1][0].abs() < 1e-5 && (en[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cholesky_solves_spd_system() {
        let dev = Device::Cpu;
        let d = 5;
        // SPD A = M Mᵀ + I
        let m = Tensor::randn(0f32, 1f32, (d, d), &dev).unwrap();
        let a = (m.matmul(&m.t().unwrap()).unwrap() + Tensor::eye(d, DType::F32, &dev).unwrap())
            .unwrap();
        let b = Tensor::randn(0f32, 1f32, (d, 2), &dev).unwrap();

        let chol = CholeskyFactor::new(&a, 1e-6).unwrap();
        let x = chol.solve(&b).unwrap();
        let recon = a.matmul(&x).unwrap(); // A X ?= B
        assert!(approx(&recon, &b) < 1e-3, "A X ≠ B: {}", approx(&recon, &b));

        // inverse: A A⁻¹ ≈ I
        let inv = chol.inverse().unwrap();
        let ii = a.matmul(&inv).unwrap();
        assert!(approx(&ii, &Tensor::eye(d, DType::F32, &dev).unwrap()) < 1e-3);

        // logdet of diag(4,9) = ln 36
        let diag = Tensor::new(&[[4.0f32, 0.0], [0.0, 9.0]], &dev).unwrap();
        let ld = CholeskyFactor::new(&diag, 0.0)
            .unwrap()
            .logdet()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (ld - 36f32.ln()).abs() < 1e-4,
            "logdet {ld} vs {}",
            36f32.ln()
        );
    }

    #[test]
    fn sym_sqrt_roundtrips() {
        let dev = Device::Cpu;
        let d = 4;
        // SPD A = M Mᵀ + I
        let m = Tensor::randn(0f32, 1f32, (d, d), &dev).unwrap();
        let a = (m.matmul(&m.t().unwrap()).unwrap() + Tensor::eye(d, DType::F32, &dev).unwrap())
            .unwrap();
        let half = sym_sqrt(&a).unwrap();
        // (A^½)² ≈ A
        assert!(approx(&half.matmul(&half).unwrap(), &a) < 1e-3);
        // A^{-½} A^½ ≈ I
        let inv_half = sym_inv_sqrt(&a).unwrap();
        let id = inv_half.matmul(&half).unwrap();
        assert!(approx(&id, &Tensor::eye(d, DType::F32, &dev).unwrap()) < 1e-3);
    }

    #[test]
    fn batched_cholesky_solve() {
        let dev = Device::Cpu;
        let (bsz, d) = (4usize, 3usize);
        let m = Tensor::randn(0f32, 1f32, (bsz, d, d), &dev).unwrap();
        let a = (m.matmul(&m.transpose(1, 2).unwrap()).unwrap()
            + Tensor::eye(d, DType::F32, &dev)
                .unwrap()
                .broadcast_as((bsz, d, d))
                .unwrap())
        .unwrap();
        let b = Tensor::randn(0f32, 1f32, (bsz, d, 1), &dev).unwrap();
        let x = cholesky_solve(&a, &b, 1e-6).unwrap();
        let recon = a.matmul(&x).unwrap();
        assert!(approx(&recon, &b) < 1e-3);
    }
}

#[cfg(test)]
mod psd_tests {
    //! Verification of tensor linear-algebra primitives, factorisations and matrix solves.

    use super::*;
    use crate::settings::{cholesky_jitter, cholesky_max_tries, Dtype};

    /// Reconstruct `L Lᵀ` and compare against the input.
    fn recon_err(l: &[f64], a: &[f64], d: usize) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..d {
                    s += l[i * d + k] * l[j * d + k];
                }
                worst = worst.max((s - a[i * d + j]).abs());
            }
        }
        worst
    }

    /// A comfortably positive-definite matrix must cost nothing. "Needed no
    /// jitter" is the most useful thing the return value can say.
    #[test]
    fn a_well_conditioned_matrix_needs_no_jitter() {
        let d = 3;
        #[rustfmt::skip]
        let a = vec![
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.2,
            0.5, 0.2, 2.0,
        ];
        let (l, jitter) = psd_safe_cholesky(&a, d, Dtype::F64).unwrap();
        assert_eq!(jitter, 0.0);
        assert!(recon_err(&l, &a, d) < 1e-12);
    }

    /// THE defect this replaces. A genuinely indefinite matrix — one
    /// negative eigenvalue — is silently absorbed by `chol`'s
    /// `.max(1e-12)` clamp, which returns a confident factor that
    /// reconstructs to something else entirely. `psd_safe_cholesky` says no.
    #[test]
    fn an_indefinite_matrix_is_refused_where_the_clamping_version_lies() {
        let d = 2;
        // eigenvalues 2 and -1
        let a = vec![1.0, 0.0, 0.0, -1.0];

        let e = psd_safe_cholesky(&a, d, Dtype::F64).unwrap_err();
        assert_eq!(e.pivot, 1);
        assert!(e.value < 0.0, "expected a negative pivot, got {}", e.value);
        assert!(e.to_string().contains("not positive definite"), "{e}");

        // ... whereas the legacy path returns a factor with no complaint,
        // and it reconstructs to the WRONG matrix.
        let l = chol(&a, d, 0.0);
        assert!(l.iter().all(|x| x.is_finite()));
        assert!(
            recon_err(&l, &a, d) > 0.5,
            "the clamping path is supposed to be wrong here — if it is not, \
             this test no longer documents anything"
        );
    }

    /// The near-singular case jitter exists for: a Gram matrix of nearly
    /// collinear vectors, which is exactly what `SᵀS` becomes when an
    /// optimiser's steps stop turning. Indefinite by a hair in floating
    /// point, PSD in exact arithmetic.
    #[test]
    fn a_hair_indefinite_matrix_is_rescued_and_reports_what_it_cost() {
        let d = 2;
        // PSD in exact arithmetic (rank 1) but nudged just below zero.
        let a = vec![1.0, 1.0, 1.0, 1.0 - 1e-12];
        let (l, jitter) = psd_safe_cholesky(&a, d, Dtype::F64).unwrap();
        assert!(jitter > 0.0, "should have needed jitter");
        assert_eq!(
            jitter,
            cholesky_jitter::BUILTIN_F64,
            "one escalation should suffice"
        );
        assert!(recon_err(&l, &a, d) < 1e-6);
    }

    /// Escalation is ×10 per attempt, so the budget spans `max_tries`
    /// orders of magnitude and the setting actually governs it.
    #[test]
    fn jitter_escalates_and_the_budget_is_governed_by_the_setting() {
        let d = 2;
        let a = vec![1.0, 1.0, 1.0, 1.0 - 1e-3]; // needs much more jitter

        {
            // One try from a tiny base cannot reach it.
            let _t = cholesky_max_tries::with(1);
            let _j = cholesky_jitter::with(Dtype::F64, 1e-12);
            let e = psd_safe_cholesky(&a, d, Dtype::F64).unwrap_err();
            assert_eq!(e.attempts, 2, "one unjittered attempt plus one jittered");
            assert_eq!(e.jitter_tried, 1e-12);
        }
        {
            // More tries from the same base reaches it by escalation.
            let _t = cholesky_max_tries::with(12);
            let _j = cholesky_jitter::with(Dtype::F64, 1e-12);
            let (_, jitter) = psd_safe_cholesky(&a, d, Dtype::F64).unwrap();
            assert!(jitter > 1e-12, "should have escalated, used {jitter:.1e}");
        }
    }

    /// The dtype argument selects the base jitter, and the f32 default is
    /// two orders above f64 — so a matrix can be rescued at f32's base and
    /// refused at f64's within a single try.
    #[test]
    fn the_dtype_argument_selects_the_base_jitter() {
        let d = 2;
        let a = vec![1.0, 1.0, 1.0, 1.0 - 1e-7];
        let _t = cholesky_max_tries::with(1);

        let (_, j32) = psd_safe_cholesky(&a, d, Dtype::F32).unwrap();
        assert_eq!(j32, cholesky_jitter::BUILTIN_F32);
        assert!(psd_safe_cholesky(&a, d, Dtype::F64).is_err());
    }

    /// A NaN pivot compares false against every bound, so an unguarded
    /// `piv > 0.0` check would let it through as merely small and produce a
    /// factor of NaNs with no error.
    #[test]
    fn a_nan_entry_is_reported_rather_than_propagated() {
        let d = 2;
        let a = vec![1.0, 0.0, 0.0, f64::NAN];
        let e = psd_safe_cholesky(&a, d, Dtype::F64).unwrap_err();
        assert_eq!(e.pivot, 1);
        assert!(e.value.is_nan());
    }

    /// A scoped override must actually reach the routine.
    #[test]
    fn a_scoped_jitter_override_reaches_the_factorization() {
        let d = 2;
        let a = vec![1.0, 1.0, 1.0, 1.0 - 1e-9];
        let _t = cholesky_max_tries::with(1);
        let _g = cholesky_jitter::with(Dtype::F64, 1e-5);
        let (_, jitter) = psd_safe_cholesky(&a, d, Dtype::F64).unwrap();
        assert_eq!(jitter, 1e-5);
    }
}

#[cfg(test)]
mod symmetric_eigh_tests {
    //! Verification of tensor linear-algebra primitives, factorisations and matrix solves.

    use super::*;
    use candle_core::Device;

    fn recon(lam: &Tensor, v: &Tensor) -> Tensor {
        // V diag(lam) Vᵀ
        v.broadcast_mul(&lam.unsqueeze(0).unwrap())
            .unwrap()
            .matmul(&v.t().unwrap().contiguous().unwrap())
            .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    }

    fn symmetrise(raw: Vec<f64>, d: usize, dev: &Device) -> Tensor {
        let m = Tensor::from_vec(raw, (d, d), dev)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        m.add(&m.t().unwrap().contiguous().unwrap())
            .unwrap()
            .affine(0.5, 0.0)
            .unwrap()
    }

    /// `A == V diag(lam) Vᵀ` on the three cases that break naive Jacobi use:
    /// a generic symmetric matrix, one that is exactly singular, and one with
    /// repeated eigenvalues (where the eigenvector choice is not unique but the
    /// reconstruction still must be).
    #[test]
    fn reconstructs_the_matrix_on_generic_singular_and_repeated_spectra() {
        let dev = Device::Cpu;
        let d = 6usize;

        let generic = symmetrise(
            (0..d * d)
                .map(|i| ((i * 37 % 23) as f64 - 11.0) * 0.31)
                .collect(),
            d,
            &dev,
        );
        // Exactly singular: a rank-4 Gram, so two eigenvalues are exactly 0.
        let b = Tensor::from_vec(
            (0..d * 4)
                .map(|i| ((i * 13 % 17) as f64 - 8.0) * 0.2)
                .collect::<Vec<f64>>(),
            (d, 4),
            &dev,
        )
        .unwrap();
        let singular = b.matmul(&b.t().unwrap().contiguous().unwrap()).unwrap();
        // Repeated: diag(2,2,2,5,5,9) in a rotated basis.
        let repeated = {
            let vals = [2.0f64, 2.0, 2.0, 5.0, 5.0, 9.0];
            let mut m = vec![0f64; d * d];
            for (i, v) in vals.iter().enumerate() {
                m[i * d + i] = *v;
            }
            Tensor::from_vec(m, (d, d), &dev).unwrap()
        };

        for (name, a) in [
            ("generic", generic),
            ("singular", singular),
            ("repeated", repeated),
        ] {
            let (lam, v) = symmetric_eigh(&a).unwrap();
            let err = max_abs_diff(&recon(&lam, &v), &a);
            assert!(err < 1e-12, "{name}: reconstruction off by {err}");

            // Eigenvalues ASCENDING. jacobi_eigh returns them unsorted, and an
            // unsorted result pairs lam[i] with a different column's vector.
            let l: Vec<f64> = lam.to_vec1().unwrap();
            assert!(
                l.windows(2).all(|w| w[0] <= w[1] + 1e-15),
                "{name}: unsorted {l:?}"
            );

            // Columns orthonormal.
            let vtv = v.t().unwrap().contiguous().unwrap().matmul(&v).unwrap();
            let eye_t = Tensor::from_vec(eye(d), (d, d), &dev).unwrap();
            let o = max_abs_diff(&vtv, &eye_t);
            assert!(o < 1e-12, "{name}: VᵀV off I by {o}");
        }
    }

    /// The prescale rule, exercised. `jacobi_eigh`'s convergence test is
    /// ABSOLUTE (`off.sqrt() < 1e-14`), so without scaling to unit Frobenius
    /// norm a large matrix burns all 100 sweeps and a tiny one exits having
    /// done nothing. Both must still reconstruct.
    #[test]
    fn a_badly_scaled_matrix_still_converges() {
        let dev = Device::Cpu;
        let d = 5usize;
        let base = symmetrise(
            (0..d * d).map(|i| ((i % 7) as f64) - 3.0).collect(),
            d,
            &dev,
        );
        for scale in [1e-10f64, 1.0, 1e10] {
            let a = base.affine(scale, 0.0).unwrap();
            let (lam, v) = symmetric_eigh(&a).unwrap();
            let err = max_abs_diff(&recon(&lam, &v), &a);
            // Relative: at scale 1e10 an absolute 1e-12 is unreachable in f64.
            let rel = err / scale;
            assert!(
                rel < 1e-11,
                "scale {scale}: relative reconstruction error {rel}"
            );
        }
    }

    /// A zero matrix has no Frobenius norm to divide by. `(0, I)`, not NaN.
    #[test]
    fn the_zero_matrix_gives_zero_eigenvalues_and_the_identity() {
        let dev = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::F64, &dev).unwrap();
        let (lam, v) = symmetric_eigh(&a).unwrap();
        assert!(lam.to_vec1::<f64>().unwrap().iter().all(|x| *x == 0.0));
        let eye_t = Tensor::from_vec(eye(4), (4, 4), &dev).unwrap();
        assert!(max_abs_diff(&v, &eye_t) == 0.0);
    }

    #[test]
    fn a_non_square_input_is_refused() {
        let dev = Device::Cpu;
        let a = Tensor::zeros((3, 4), DType::F64, &dev).unwrap();
        assert!(symmetric_eigh(&a)
            .unwrap_err()
            .to_string()
            .contains("square"));
    }
}
