//! Inferno — Differentiable matrix factorisations and solves on Candle tensors.

use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Tensor, D};

use crate::linalg::{chol, mat_exp};

// ---------------------------------------------------------------------------
// Host helpers (f64, row-major)
// ---------------------------------------------------------------------------

/// Forward substitution: solve `L X = B` for lower-triangular `L [d,d]`,
/// `B [d,m]` row-major.
fn fwd_sub(l: &[f64], b: &[f64], d: usize, m: usize) -> Vec<f64> {
    let mut x = vec![0.0; d * m];
    for col in 0..m {
        for i in 0..d {
            let mut s = b[i * m + col];
            for k in 0..i {
                s -= l[i * d + k] * x[k * m + col];
            }
            x[i * m + col] = s / l[i * d + i];
        }
    }
    x
}

/// Back substitution: solve `Lᵀ X = B`.
fn back_sub(l: &[f64], b: &[f64], d: usize, m: usize) -> Vec<f64> {
    let mut x = vec![0.0; d * m];
    for col in 0..m {
        for i in (0..d).rev() {
            let mut s = b[i * m + col];
            for k in (i + 1)..d {
                s -= l[k * d + i] * x[k * m + col];
            }
            x[i * m + col] = s / l[i * d + i];
        }
    }
    x
}

fn matmul_host(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * m];
    for i in 0..n {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i * m + j] += aip * b[p * m + j];
            }
        }
    }
    out
}

fn transpose_host(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Storage plumbing
// ---------------------------------------------------------------------------

/// Contiguous data as f64 + a flag for restoring the original dtype.
/// `pub(crate)`: `operators::ciq`'s CustomOp reuses the same storage plumbing.
pub(crate) fn storage_f64(s: &CpuStorage, l: &Layout) -> Result<(Vec<f64>, bool)> {
    let (start, end) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("linalg_diff: non-contiguous input".into()))?;
    match s {
        CpuStorage::F32(v) => Ok((v[start..end].iter().map(|x| *x as f64).collect(), true)),
        CpuStorage::F64(v) => Ok((v[start..end].to_vec(), false)),
        _ => Err(candle_core::Error::Msg("linalg_diff: expected f32/f64".into())),
    }
}

pub(crate) fn storage_from_f64(data: Vec<f64>, as_f32: bool) -> CpuStorage {
    if as_f32 {
        CpuStorage::F32(data.into_iter().map(|x| x as f32).collect())
    } else {
        CpuStorage::F64(data)
    }
}

pub(crate) fn square_batch_dims(shape: &Shape) -> Result<(usize, usize)> {
    let dims = shape.dims();
    if dims.len() < 2 || dims[dims.len() - 1] != dims[dims.len() - 2] {
        return Err(candle_core::Error::Msg(format!(
            "linalg_diff: expected [.., D, D], got {dims:?}"
        )));
    }
    let d = dims[dims.len() - 1];
    Ok((dims[..dims.len() - 2].iter().product::<usize>().max(1), d))
}

/// Pull a whole (batched) tensor to host f64.
pub(crate) fn tensor_f64(t: &Tensor) -> Result<Vec<f64>> {
    t.contiguous()?
        .to_dtype(candle_core::DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()
}

// ---------------------------------------------------------------------------
// CholeskyOp
// ---------------------------------------------------------------------------

struct CholeskyOp {
    jitter: f64,
}

impl CustomOp1 for CholeskyOp {
    fn name(&self) -> &'static str {
        "cholesky_diff"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (data, as_f32) = storage_f64(storage, layout)?;
        let (bsz, d) = square_batch_dims(layout.shape())?;
        let mut out = Vec::with_capacity(data.len());
        for b in 0..bsz {
            out.extend(chol(&data[b * d * d..(b + 1) * d * d], d, self.jitter));
        }
        Ok((storage_from_f64(out, as_f32), layout.shape().clone()))
    }

    /// Murray (2016): with `P = Φ ∘ (Lᵀ L̄)` (`Φ` = lower-triangular mask, ½ on
    /// the diagonal), `Ā = ½ (S + Sᵀ)` where `S = L⁻ᵀ P L⁻¹` — two triangular
    /// solves against the saved factor. Symmetrised because `A` is SPD by
    /// construction upstream.
    fn bwd(&self, arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let (bsz, d) = square_batch_dims(arg.shape())?;
        let l_all = tensor_f64(res)?;
        let g_all = tensor_f64(grad_res)?;

        let dd = d * d;
        let mut out = Vec::with_capacity(bsz * dd);
        for bi in 0..bsz {
            let l = &l_all[bi * dd..(bi + 1) * dd];
            let lbar = &g_all[bi * dd..(bi + 1) * dd];

            // P = Φ ∘ (Lᵀ L̄)
            let lt = transpose_host(l, d, d);
            let mut p = matmul_host(&lt, lbar, d, d, d);
            for i in 0..d {
                for j in 0..d {
                    if j > i {
                        p[i * d + j] = 0.0; // strict upper zeroed
                    } else if j == i {
                        p[i * d + j] *= 0.5; // half diagonal
                    }
                }
            }

            // S = L⁻ᵀ P L⁻¹:  Y = L⁻ᵀ P  (back-substitution), then
            // S = Y L⁻¹ ⇔ Sᵀ = L⁻ᵀ Yᵀ (back-substitution again).
            let y = back_sub(l, &p, d, d);
            let yt = transpose_host(&y, d, d);
            let st = back_sub(l, &yt, d, d);
            let s = transpose_host(&st, d, d);

            // Ā = ½(S + Sᵀ)
            let mut abar = vec![0.0; dd];
            for i in 0..d {
                for j in 0..d {
                    abar[i * d + j] = 0.5 * (s[i * d + j] + s[j * d + i]);
                }
            }
            out.extend(abar);
        }

        let t = Tensor::from_vec(out, arg.shape().clone(), arg.device())?
            .to_dtype(arg.dtype())?;
        Ok(Some(t))
    }
}

/// Differentiable batched Cholesky `A [.., D, D] → L` (lower-triangular),
/// diagonal `jitter` added in the forward for stability.
pub fn cholesky_diff(a: &Tensor, jitter: f64) -> Result<Tensor> {
    a.contiguous()?.apply_op1(CholeskyOp { jitter })
}

// ---------------------------------------------------------------------------
// MatrixExpOp
// ---------------------------------------------------------------------------

struct MatrixExpOp;

impl CustomOp1 for MatrixExpOp {
    fn name(&self) -> &'static str {
        "matrix_exp_diff"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (data, as_f32) = storage_f64(storage, layout)?;
        let (bsz, d) = square_batch_dims(layout.shape())?;
        let mut out = Vec::with_capacity(data.len());
        for b in 0..bsz {
            out.extend(mat_exp(&data[b * d * d..(b + 1) * d * d], d));
        }
        Ok((storage_from_f64(out, as_f32), layout.shape().clone()))
    }

    /// The adjoint of `expm` is another `expm`, via Van Loan's block identity:
    ///
    /// ```text
    /// expm([[X, E], [0, X]]) = [[e^X, L(X, E)], [0, e^X]]
    /// ```
    ///
    /// where `L(X, E) = ∫₀¹ e^{Xs} E e^{X(1−s)} ds` is the Fréchet derivative.
    /// The reverse-mode rule for `Y = e^A` with output cotangent `Ḡ` is exactly
    /// `Ā = ∫₀¹ e^{Aᵀs} Ḡ e^{Aᵀ(1−s)} ds = L(Aᵀ, Ḡ)`, so building the `2D × 2D`
    /// block with `X = Aᵀ`, `E = Ḡ` and reading its upper-right block gives the
    /// gradient in ONE extra host `expm`. No quadrature, no unrolled
    /// scaling-and-squaring graph, and the same kernel as the forward.
    ///
    /// Note this is the same block trick the Van Loan controllability Gramian
    /// uses, so a `van_loan_step` built on top costs a `4D × 4D` expm on
    /// backward — 64×64 at the D=16 latent, which is nothing.
    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let (bsz, d) = square_batch_dims(arg.shape())?;
        let a_all = tensor_f64(arg)?;
        let g_all = tensor_f64(grad_res)?;

        let dd = d * d;
        let two = 2 * d;
        let mut out = Vec::with_capacity(bsz * dd);
        for bi in 0..bsz {
            let a = &a_all[bi * dd..(bi + 1) * dd];
            let g = &g_all[bi * dd..(bi + 1) * dd];

            // M = [[Aᵀ, Ḡ], [0, Aᵀ]]  — note the TRANSPOSE on the diagonal
            // blocks; using `A` there computes the pushforward, not the adjoint.
            let mut m = vec![0.0; two * two];
            for i in 0..d {
                for j in 0..d {
                    let at = a[j * d + i];
                    m[i * two + j] = at;
                    m[(i + d) * two + (j + d)] = at;
                    m[i * two + (j + d)] = g[i * d + j];
                }
            }

            let e = mat_exp(&m, two);
            for i in 0..d {
                for j in 0..d {
                    out.push(e[i * two + (j + d)]);
                }
            }
        }

        Tensor::from_vec(out, arg.shape().clone(), arg.device())?
            .to_dtype(arg.dtype())
            .map(Some)
    }
}

/// Differentiable batched matrix exponential `A [.., D, D] → e^A`.
///
/// [`crate::linalg::matrix_exp`] routes through `map_square`, which pulls the
/// tensor to a host `Vec` and rebuilds it with `Tensor::from_vec` — that severs
/// the autograd graph, so anything downstream of it is a constant. Use THIS one
/// on any path whose gradient must reach `A`; the detached one remains correct
/// (and marginally cheaper) for frozen teacher geometry.
// ---------------------------------------------------------------------------
// SlogdetOp — general (NON-symmetric) log|det|
// ---------------------------------------------------------------------------

/// LU with partial pivoting. Returns `(log|det|, inverse-transpose)`, or `None`
/// when the matrix is numerically singular.
///
/// Both come out of ONE factorisation because the backward needs `A^{-T}` and
/// the forward needs `Σ log|U_ii|`; refactorising would double the cost of the
/// only expensive part.
fn lu_slogdet_and_inv_t(a: &[f64], d: usize) -> Option<(f64, Vec<f64>)> {
    let mut m = a.to_vec();
    let mut perm: Vec<usize> = (0..d).collect();
    let mut logdet = 0.0f64;

    for k in 0..d {
        // Partial pivot: without it a zero leading minor fails on a matrix that
        // is perfectly well conditioned, which is not rare for `cI − Jac`.
        let (mut piv, mut best) = (k, m[k * d + k].abs());
        for r in (k + 1)..d {
            let v = m[r * d + k].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= 1e-300 {
            return None;
        }
        if piv != k {
            for c in 0..d {
                m.swap(k * d + c, piv * d + c);
            }
            perm.swap(k, piv);
        }
        let pivot = m[k * d + k];
        logdet += pivot.abs().ln();
        for r in (k + 1)..d {
            let f = m[r * d + k] / pivot;
            m[r * d + k] = f;
            for c in (k + 1)..d {
                m[r * d + c] -= f * m[k * d + c];
            }
        }
    }

    // A⁻¹ column by column: solve `A x = e_j` through the stored LU, then write
    // the result into ROW j of the output, which transposes it in place.
    let mut inv_t = vec![0.0f64; d * d];
    let mut y = vec![0.0f64; d];
    for j in 0..d {
        for i in 0..d {
            let mut acc = if perm[i] == j { 1.0 } else { 0.0 };
            for c in 0..i {
                acc -= m[i * d + c] * y[c];
            }
            y[i] = acc;
        }
        for i in (0..d).rev() {
            let mut acc = y[i];
            for c in (i + 1)..d {
                acc -= m[i * d + c] * inv_t[j * d + c];
            }
            inv_t[j * d + i] = acc / m[i * d + i];
        }
    }
    Some((logdet, inv_t))
}

struct SlogdetOp;

impl CustomOp1 for SlogdetOp {
    fn name(&self) -> &'static str {
        "slogdet_abs_diff"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (data, as_f32) = storage_f64(storage, layout)?;
        let (bsz, d) = square_batch_dims(layout.shape())?;
        let dims = layout.shape().dims();
        let out_shape = Shape::from(&dims[..dims.len() - 2]);
        let mut out = Vec::with_capacity(bsz);
        for b in 0..bsz {
            let blk = &data[b * d * d..(b + 1) * d * d];
            match lu_slogdet_and_inv_t(blk, d) {
                Some((ld, _)) => out.push(ld),
                // A singular block is `−∞`, reported rather than jittered into
                // a finite lie: a log-density that silently absorbs a singular
                // Jacobian is a likelihood nobody can trust.
                None => out.push(f64::NEG_INFINITY),
            }
        }
        Ok((storage_from_f64(out, as_f32), out_shape))
    }

    /// `∂ log|det A| / ∂A = A⁻ᵀ`, scaled by the incoming cotangent.
    ///
    /// Exact and cheap — one extra triangular solve per block off the SAME LU
    /// the forward already paid for. A singular block contributes zero rather
    /// than `NaN`: its forward value is `−∞`, so any finite gradient there is a
    /// fiction, and `NaN` would poison every parameter it touches.
    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let (bsz, d) = square_batch_dims(arg.shape())?;
        let a_all = tensor_f64(arg)?;
        let g_all = tensor_f64(grad_res)?;
        let dd = d * d;
        let mut out = Vec::with_capacity(bsz * dd);
        for b in 0..bsz {
            let g = g_all[b];
            match lu_slogdet_and_inv_t(&a_all[b * dd..(b + 1) * dd], d) {
                Some((_, inv_t)) => out.extend(inv_t.into_iter().map(|v| g * v)),
                None => out.extend(std::iter::repeat(0.0).take(dd)),
            }
        }
        Tensor::from_vec(out, arg.shape().clone(), arg.device())?
            .to_dtype(arg.dtype())
            .map(Some)
    }
}

/// Differentiable `log|det A|` for a **general** square matrix
/// `[.., D, D] → [..]`.
///
/// # Why not the Cholesky route
///
/// [`crate::linalg::CholeskyFactor::log_abs_det`] and everything built on it
/// require SPD. The Jacobian of a coupling-flow backbone step,
/// `J = cI − (M√v)·∂s/∂x`, is not symmetric — `∂s/∂x` for a Gaussian mixture
/// carries a `Σ_k r_k v_k (v_k − s)ᵀ` term that is only symmetric by accident —
/// so a Cholesky would either fail or, worse, succeed on the jitter and return
/// a number for a matrix it did not factorise.
///
/// LU with partial pivoting handles it in `O(D³)` at `D = 16`, and the reverse
/// rule `∂ log|det A|/∂A = A⁻ᵀ` reuses the same factorisation.
///
/// # Sign
///
/// This is `log|det|`, not `logdet` — the sign is dropped. For a
/// change-of-variables that is the correct object (the density carries
/// `|det J|`), and for anything that needs the orientation the sign has to be
/// tracked separately.
pub fn slogdet_abs_diff(a: &Tensor) -> Result<Tensor> {
    a.contiguous()?.apply_op1(SlogdetOp)
}

pub fn matrix_exp_diff(a: &Tensor) -> Result<Tensor> {
    a.contiguous()?.apply_op1(MatrixExpOp)
}

// ---------------------------------------------------------------------------
// TriangularSolveOp
// ---------------------------------------------------------------------------

struct TriSolveOp;

impl CustomOp2 for TriSolveOp {
    fn name(&self) -> &'static str {
        "tri_solve_diff"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (l_data, as_f32) = storage_f64(s1, l1)?;
        let (b_data, _) = storage_f64(s2, l2)?;
        let (bsz, d) = square_batch_dims(l1.shape())?;
        let bdims = l2.shape().dims();
        let m = bdims[bdims.len() - 1];
        if bdims[bdims.len() - 2] != d {
            return Err(candle_core::Error::Msg(format!(
                "tri_solve_diff: B row dim {} ≠ D {d}",
                bdims[bdims.len() - 2]
            )));
        }
        let mut out = Vec::with_capacity(b_data.len());
        for bi in 0..bsz {
            let l = &l_data[bi * d * d..(bi + 1) * d * d];
            let b = &b_data[bi * d * m..(bi + 1) * d * m];
            out.extend(fwd_sub(l, b, d, m));
        }
        Ok((storage_from_f64(out, as_f32), l2.shape().clone()))
    }

    /// Adjoint of `X = L⁻¹ B`: `B̄ = L⁻ᵀ X̄`, `L̄ = −tril(B̄ Xᵀ)`.
    fn bwd(
        &self,
        arg1: &Tensor, // L
        arg2: &Tensor, // B
        res: &Tensor,  // X
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (bsz, d) = square_batch_dims(arg1.shape())?;
        let bdims = arg2.shape().dims();
        let m = bdims[bdims.len() - 1];

        let l_all = tensor_f64(arg1)?;
        let x_all = tensor_f64(res)?;
        let g_all = tensor_f64(grad_res)?;

        let mut lbar_out = Vec::with_capacity(bsz * d * d);
        let mut bbar_out = Vec::with_capacity(bsz * d * m);
        for bi in 0..bsz {
            let l = &l_all[bi * d * d..(bi + 1) * d * d];
            let x = &x_all[bi * d * m..(bi + 1) * d * m];
            let g = &g_all[bi * d * m..(bi + 1) * d * m];

            let bbar = back_sub(l, g, d, m); // L⁻ᵀ X̄  [d, m]
            let xt = transpose_host(x, d, m); // [m, d]
            let mut lbar = matmul_host(&bbar, &xt, d, m, d); // B̄ Xᵀ [d, d]
            for i in 0..d {
                for j in 0..d {
                    lbar[i * d + j] = if j <= i { -lbar[i * d + j] } else { 0.0 };
                }
            }
            lbar_out.extend(lbar);
            bbar_out.extend(bbar);
        }

        let lbar = Tensor::from_vec(lbar_out, arg1.shape().clone(), arg1.device())?
            .to_dtype(arg1.dtype())?;
        let bbar = Tensor::from_vec(bbar_out, arg2.shape().clone(), arg2.device())?
            .to_dtype(arg2.dtype())?;
        Ok((Some(lbar), Some(bbar)))
    }
}

/// Differentiable batched lower-triangular solve `X = L⁻¹ B`
/// (`L [.., D, D]`, `B [.., D, M]`).
pub fn tri_solve_diff(l: &Tensor, b: &Tensor) -> Result<Tensor> {
    l.contiguous()?.apply_op2(&b.contiguous()?, TriSolveOp)
}

// ---------------------------------------------------------------------------
// TriangularSolveTOp — the TRANSPOSED partner of TriSolveOp
// ---------------------------------------------------------------------------

struct TriSolveTOp;

impl CustomOp2 for TriSolveTOp {
    fn name(&self) -> &'static str {
        "tri_solve_t_diff"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (l_data, as_f32) = storage_f64(s1, l1)?;
        let (b_data, _) = storage_f64(s2, l2)?;
        let (bsz, d) = square_batch_dims(l1.shape())?;
        let bdims = l2.shape().dims();
        let m = bdims[bdims.len() - 1];
        if bdims[bdims.len() - 2] != d {
            return Err(candle_core::Error::Msg(format!(
                "tri_solve_t_diff: B row dim {} ≠ D {d}",
                bdims[bdims.len() - 2]
            )));
        }
        let mut out = Vec::with_capacity(b_data.len());
        for bi in 0..bsz {
            let l = &l_data[bi * d * d..(bi + 1) * d * d];
            let b = &b_data[bi * d * m..(bi + 1) * d * m];
            out.extend(back_sub(l, b, d, m)); // Lᵀ X = B
        }
        Ok((storage_from_f64(out, as_f32), l2.shape().clone()))
    }

    /// Adjoint of `X = L⁻ᵀ B`.
    ///
    /// `dX = −L⁻ᵀ dLᵀ X + L⁻ᵀ dB`, so with `B̄ = L⁻¹ X̄`:
    ///   ⟨X̄, L⁻ᵀdB⟩ = ⟨L⁻¹X̄, dB⟩            ⇒  B̄ = L⁻¹ X̄
    ///   −⟨X̄, L⁻ᵀdLᵀX⟩ = −tr(X B̄ᵀ dLᵀ)      ⇒  L̄ = −tril(X B̄ᵀ)
    /// (the mirror of the lower solve's `L̄ = −tril(B̄ Xᵀ)`).
    fn bwd(
        &self,
        arg1: &Tensor, // L
        arg2: &Tensor, // B
        res: &Tensor,  // X
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (bsz, d) = square_batch_dims(arg1.shape())?;
        let bdims = arg2.shape().dims();
        let m = bdims[bdims.len() - 1];

        let l_all = tensor_f64(arg1)?;
        let x_all = tensor_f64(res)?;
        let g_all = tensor_f64(grad_res)?;

        let mut lbar_out = Vec::with_capacity(bsz * d * d);
        let mut bbar_out = Vec::with_capacity(bsz * d * m);
        for bi in 0..bsz {
            let l = &l_all[bi * d * d..(bi + 1) * d * d];
            let x = &x_all[bi * d * m..(bi + 1) * d * m];
            let g = &g_all[bi * d * m..(bi + 1) * d * m];

            let bbar = fwd_sub(l, g, d, m); // L⁻¹ X̄  [d, m]
            let bbart = transpose_host(&bbar, d, m); // [m, d]
            let mut lbar = matmul_host(x, &bbart, d, m, d); // X B̄ᵀ [d, d]
            for i in 0..d {
                for j in 0..d {
                    lbar[i * d + j] = if j <= i { -lbar[i * d + j] } else { 0.0 };
                }
            }
            lbar_out.extend(lbar);
            bbar_out.extend(bbar);
        }

        let lbar = Tensor::from_vec(lbar_out, arg1.shape().clone(), arg1.device())?
            .to_dtype(arg1.dtype())?;
        let bbar = Tensor::from_vec(bbar_out, arg2.shape().clone(), arg2.device())?
            .to_dtype(arg2.dtype())?;
        Ok((Some(lbar), Some(bbar)))
    }
}

/// Differentiable batched UPPER (transposed-lower) triangular solve
/// `X = L⁻ᵀ B` — the back-substitution partner of [`tri_solve_diff`].
/// Together they give the full Cholesky solve
/// `A⁻¹B = L⁻ᵀ(L⁻¹B)` with gradients through both factors.
pub fn tri_solve_t_diff(l: &Tensor, b: &Tensor) -> Result<Tensor> {
    l.contiguous()?.apply_op2(&b.contiguous()?, TriSolveTOp)
}

/// Differentiable batched Cholesky solve `X = A⁻¹ B` for SPD `A`, via
/// `L = chol(A)` then `L⁻ᵀ(L⁻¹B)`.
pub fn cholesky_solve_diff(a: &Tensor, b: &Tensor, jitter: f64) -> Result<Tensor> {
    let l = cholesky_diff(a, jitter)?;
    tri_solve_t_diff(&l, &tri_solve_diff(&l, b)?)
}

/// Lower-triangular mask (including the diagonal) of the last two dims —
/// keeps free Cholesky-factor parameters on the triangle.
pub fn tril(t: &Tensor) -> Result<Tensor> {
    let d = t.dim(D::Minus1)?;
    let mut m = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            m[i * d + j] = 1.0;
        }
    }
    let mask = Tensor::from_vec(m, (d, d), t.device())?.to_dtype(t.dtype())?;
    t.broadcast_mul(&mask)
}

#[cfg(test)]
mod tests {
    //! Verification of differentiable matrix factorisations and solves on candle tensors.

    use super::*;
    use candle_core::{DType, Device, Var};

    /// Central finite difference of `loss(param)` at every element of `param`.
    fn finite_diff(
        param: &Var,
        loss: &dyn Fn(&Tensor) -> Result<f64>,
        h: f64,
    ) -> Result<Vec<f64>> {
        let base = param.as_tensor().flatten_all()?.to_vec1::<f64>()?;
        let dims = param.shape().dims().to_vec();
        let dev = param.device().clone();
        let mut grads = Vec::with_capacity(base.len());
        for i in 0..base.len() {
            let mut up = base.clone();
            up[i] += h;
            let mut dn = base.clone();
            dn[i] -= h;
            let lu = loss(&Tensor::from_vec(up, dims.clone(), &dev)?)?;
            let ld = loss(&Tensor::from_vec(dn, dims.clone(), &dev)?)?;
            grads.push((lu - ld) / (2.0 * h));
        }
        Ok(grads)
    }

    #[test]
    fn cholesky_forward_roundtrip_and_gradcheck() {
        let dev = Device::Cpu;
        let d = 4;
        // A = B Bᵀ + I: SPD, differentiable in B.
        let b0 = Tensor::randn(0f64, 1f64, (d, d), &dev).unwrap();
        let b = Var::from_tensor(&b0).unwrap();

        let make_a = |bt: &Tensor| -> Result<Tensor> {
            bt.matmul(&bt.t()?)?
                .add(&Tensor::eye(d, DType::F64, bt.device())?)
        };

        // round-trip: L Lᵀ = A
        let a = make_a(b.as_tensor()).unwrap();
        let l = cholesky_diff(&a, 0.0).unwrap();
        let rt = l.matmul(&l.t().unwrap()).unwrap();
        let err = rt.sub(&a).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(err < 1e-10, "L Lᵀ − A = {err}");

        // gradcheck: loss = Σ (L ⊙ W)
        let w = Tensor::randn(0f64, 1f64, (d, d), &dev).unwrap();
        let loss_of = |bt: &Tensor| -> Result<f64> {
            let a = make_a(bt)?;
            let l = cholesky_diff(&a, 0.0)?;
            l.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let l = cholesky_diff(&make_a(b.as_tensor()).unwrap(), 0.0).unwrap();
        let loss = l.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_auto = grads.get(b.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_fd = finite_diff(&b, &loss_of, 1e-6).unwrap();

        for (i, (ga, gf)) in g_auto.iter().zip(&g_fd).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-5 * (1.0 + gf.abs()),
                "cholesky grad[{i}]: auto {ga} vs FD {gf}"
            );
        }
    }

    #[test]
    fn matrix_exp_forward_matches_the_host_kernel_and_gradchecks() {
        let dev = Device::Cpu;
        let d = 4;
        // Scaled down so ‖A‖ stays in the regime where FD is trustworthy;
        // the scaling-and-squaring path is exercised separately below.
        let a0 = Tensor::randn(0f64, 0.4f64, (d, d), &dev).unwrap();
        let a = Var::from_tensor(&a0).unwrap();

        // Forward must agree with the detached path bit-for-bit: same kernel.
        let diff = matrix_exp_diff(a.as_tensor()).unwrap();
        let host = crate::linalg::matrix_exp(a.as_tensor()).unwrap();
        let err = diff.sub(&host).unwrap().abs().unwrap().max_all().unwrap()
            .to_scalar::<f64>().unwrap();
        assert!(err < 1e-12, "forward disagrees with linalg::matrix_exp: {err}");

        // gradcheck: loss = Σ (e^A ⊙ W)  ⇒  Ā = L(Aᵀ, W)
        let w = Tensor::randn(0f64, 1f64, (d, d), &dev).unwrap();
        let loss_of = |at: &Tensor| -> Result<f64> {
            matrix_exp_diff(at)?.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let loss = matrix_exp_diff(a.as_tensor()).unwrap().mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_auto = grads.get(a.as_tensor()).unwrap()
            .flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_fd = finite_diff(&a, &loss_of, 1e-6).unwrap();

        for (i, (ga, gf)) in g_auto.iter().zip(&g_fd).enumerate() {
            assert!(
                (ga - gf).abs() < 1e-5 * (1.0 + gf.abs()),
                "matrix_exp grad[{i}]: auto {ga} vs FD {gf}",
            );
        }
    }

    /// The reason `matrix_exp_diff` exists. `linalg::matrix_exp` goes through
    /// `map_square`, which pulls to a host `Vec` and rebuilds the tensor — so
    /// the result is a graph LEAF and `backward()` cannot reach `A` at all.
    /// Asserting the old behaviour alongside the new one documents why both
    /// functions are kept.
    #[test]
    fn the_detached_matrix_exp_yields_no_gradient_but_the_diff_one_does() {
        let dev = Device::Cpu;
        let d = 3;
        let a = Var::from_tensor(&Tensor::randn(0f64, 0.3f64, (d, d), &dev).unwrap()).unwrap();
        let w = Tensor::randn(0f64, 1f64, (d, d), &dev).unwrap();

        let detached = crate::linalg::matrix_exp(a.as_tensor()).unwrap()
            .mul(&w).unwrap().sum_all().unwrap();
        assert!(
            detached.backward().unwrap().get(a.as_tensor()).is_none(),
            "linalg::matrix_exp must NOT carry a gradient — if this starts \
             passing a gradient, the host round-trip was removed and this \
             module's reason for existing changed",
        );

        let tracked = matrix_exp_diff(a.as_tensor()).unwrap()
            .mul(&w).unwrap().sum_all().unwrap();
        let g = tracked.backward().unwrap();
        let norm = g.get(a.as_tensor()).unwrap().sqr().unwrap().sum_all().unwrap()
            .to_scalar::<f64>().unwrap().sqrt();
        assert!(norm > 1e-8, "matrix_exp_diff gradient is {norm}, expected non-zero");
    }

    /// Batched, and in the stiff regime the piecewise reference actually runs
    /// in: `‖A·dt‖ ≈ 227` there, which forces many squarings. The gradient must
    /// still be finite — a NaN here would surface much later as a dead bridge.
    #[test]
    fn matrix_exp_grad_is_finite_for_a_strongly_contractive_batch() {
        let dev = Device::Cpu;
        let (b, d) = (3, 4);
        // Eigenvalues ~ −50 after the scale: contractive, many squarings.
        let raw = Tensor::randn(0f64, 1f64, (b, d, d), &dev).unwrap();
        let a = Var::from_tensor(&raw.affine(-50.0, 0.0).unwrap()).unwrap();

        let e = matrix_exp_diff(a.as_tensor()).unwrap();
        assert_eq!(e.dims(), &[b, d, d]);

        let w = Tensor::randn(0f64, 1f64, (b, d, d), &dev).unwrap();
        let loss = e.mul(&w).unwrap().sum_all().unwrap();
        let g = loss.backward().unwrap();
        let gt = g.get(a.as_tensor()).unwrap();
        let s = gt.abs().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(s.is_finite(), "gradient is not finite under strong contraction: {s}");
    }

    #[test]
    fn tri_solve_gradcheck_both_args() {
        let dev = Device::Cpu;
        let (d, m) = (4, 3);
        // L: well-conditioned lower-triangular from a raw var.
        let raw0 = Tensor::randn(0f64, 0.3f64, (d, d), &dev).unwrap();
        let raw = Var::from_tensor(&raw0).unwrap();
        let rhs0 = Tensor::randn(0f64, 1f64, (d, m), &dev).unwrap();
        let rhs = Var::from_tensor(&rhs0).unwrap();
        let w = Tensor::randn(0f64, 1f64, (d, m), &dev).unwrap();

        let make_l = |r: &Tensor| -> Result<Tensor> {
            tril(r)?.add(&Tensor::eye(d, DType::F64, r.device())?.affine(2.0, 0.0)?)
        };
        let loss_of = |r: &Tensor, b: &Tensor| -> Result<f64> {
            let x = tri_solve_diff(&make_l(r)?, b)?;
            x.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let x = tri_solve_diff(&make_l(raw.as_tensor()).unwrap(), rhs.as_tensor()).unwrap();
        let loss = x.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let g_raw = grads.get(raw.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let g_rhs = grads.get(rhs.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let fd_raw = finite_diff(&raw, &|r| loss_of(r, rhs.as_tensor()), 1e-6).unwrap();
        let fd_rhs = finite_diff(&rhs, &|b| loss_of(raw.as_tensor(), b), 1e-6).unwrap();

        for (i, (ga, gf)) in g_raw.iter().zip(&fd_raw).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "L grad[{i}]: {ga} vs {gf}");
        }
        for (i, (ga, gf)) in g_rhs.iter().zip(&fd_rhs).enumerate() {
            assert!((ga - gf).abs() < 1e-5 * (1.0 + gf.abs()), "B grad[{i}]: {ga} vs {gf}");
        }
    }

    #[test]
    fn batched_shapes_pass_through() {
        let dev = Device::Cpu;
        let (l_batch, d, m) = (3, 5, 2);
        let b0 = Tensor::randn(0f32, 1f32, (l_batch, d, d), &dev).unwrap();
        let a = b0
            .matmul(&b0.transpose(1, 2).unwrap())
            .unwrap()
            .broadcast_add(&Tensor::eye(d, DType::F32, &dev).unwrap())
            .unwrap();
        let l = cholesky_diff(&a, 1e-8).unwrap();
        assert_eq!(l.dims(), &[l_batch, d, d]);
        let b = Tensor::randn(0f32, 1f32, (l_batch, d, m), &dev).unwrap();
        let x = tri_solve_diff(&l, &b).unwrap();
        assert_eq!(x.dims(), &[l_batch, d, m]);
        // L X should reproduce B
        let lx = l.matmul(&x).unwrap();
        let err = lx.sub(&b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(err < 1e-4, "L·(L⁻¹B) − B = {err}");
    }
    /// Forward value against an independent cofactor expansion, on a matrix
    /// that is deliberately NON-symmetric and needs a row swap to factorise —
    /// the two things a Cholesky route would get wrong.
    #[test]
    fn slogdet_matches_an_independent_determinant_on_a_nonsymmetric_matrix() {
        let dev = Device::Cpu;
        // Zero leading minor ⇒ partial pivoting is REQUIRED, not an optimisation.
        let a = Tensor::from_vec(
            vec![0.0f64, 2.0, 1.0, 3.0, -1.0, 4.0, 5.0, 6.0, -2.0],
            (3, 3),
            &dev,
        )
        .unwrap();
        // det = 0·((−1)(−2) − 4·6) − 2·(3(−2) − 4·5) + 1·(3·6 − (−1)·5) = 75
        let got = slogdet_abs_diff(&a).unwrap().to_scalar::<f64>().unwrap();
        assert!((got - 75.0f64.ln()).abs() < 1e-12, "log|det| {got} vs {}", 75.0f64.ln());
        assert_eq!(slogdet_abs_diff(&a).unwrap().dims().len(), 0, "scalar out for a [D,D] in");
    }

    /// `∂ log|det A|/∂A = A⁻ᵀ`, checked against central finite differences on a
    /// non-symmetric matrix. This is the gate: the value could be right with a
    /// wrong adjoint and nothing downstream would say so.
    #[test]
    fn slogdet_gradient_matches_finite_differences() {
        let dev = Device::Cpu;
        let d = 5;
        let a0 = Tensor::randn(0f64, 1f64, (d, d), &dev)
            .unwrap()
            .add(&Tensor::eye(d, DType::F64, &dev).unwrap().affine(3.0, 0.0).unwrap())
            .unwrap();
        let a = Var::from_tensor(&a0).unwrap();

        let loss_of =
            |at: &Tensor| -> Result<f64> { slogdet_abs_diff(at)?.to_scalar::<f64>() };
        let loss = slogdet_abs_diff(a.as_tensor()).unwrap();
        let g_auto = loss
            .backward()
            .unwrap()
            .get(a.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let g_fd = finite_diff(&a, &loss_of, 1e-6).unwrap();
        let err = g_auto
            .iter()
            .zip(&g_fd)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "slogdet grad off FD by {err:.3e}\n auto {g_auto:?}\n fd {g_fd:?}");
    }

    /// Batched: `[B, D, D] → [B]`, each block independent.
    #[test]
    fn slogdet_is_batched_and_blockwise() {
        let dev = Device::Cpu;
        let (b, d) = (4usize, 3usize);
        let blocks = Tensor::randn(0f64, 1f64, (b, d, d), &dev)
            .unwrap()
            .broadcast_add(&Tensor::eye(d, DType::F64, &dev).unwrap().affine(4.0, 0.0).unwrap())
            .unwrap();
        let all = slogdet_abs_diff(&blocks).unwrap();
        assert_eq!(all.dims(), &[b]);
        let v = all.to_vec1::<f64>().unwrap();
        for i in 0..b {
            let one = slogdet_abs_diff(&blocks.narrow(0, i, 1).unwrap().squeeze(0).unwrap())
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            assert!((v[i] - one).abs() < 1e-12, "block {i}: {} vs {one}", v[i]);
        }
    }

    /// A singular block reports `−∞` and contributes ZERO gradient — reported,
    /// not jittered into a finite lie. A log-density that silently absorbs a
    /// singular Jacobian is a likelihood nobody can trust, and a `NaN` there
    /// would poison every parameter it touches.
    #[test]
    fn a_singular_block_is_minus_infinity_with_no_gradient() {
        let dev = Device::Cpu;
        // Row 1 = 2 × row 0, so the matrix is exactly rank 1.
        let sing = Tensor::from_vec(vec![1.0f64, 2.0, 2.0, 4.0], (2, 2), &dev).unwrap();
        let v = Var::from_tensor(&sing).unwrap();
        let out = slogdet_abs_diff(v.as_tensor()).unwrap();
        assert_eq!(out.to_scalar::<f64>().unwrap(), f64::NEG_INFINITY);

        let g = out.backward().unwrap();
        if let Some(gt) = g.get(v.as_tensor()) {
            for x in gt.flatten_all().unwrap().to_vec1::<f64>().unwrap() {
                assert_eq!(x, 0.0, "a singular block must contribute exactly zero, got {x}");
            }
        }
    }

}
