//! Inferno — Low-rank Gaussian sampling, scores, log determinants and Woodbury solves.

use candle_core::{Result, Tensor, D};



fn clamp_min(t: &Tensor, v: f64) -> Result<Tensor> {
    t.maximum(&t.affine(0.0, v)?)
}

/// Numerically-stable `softplus`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = x.abs()?.neg()?;
    relu.add(&neg_abs.exp()?.affine(1.0, 1.0)?.log()?)
}

// ---------------------------------------------------------------------------
// Low-rank + diagonal Gaussian primitives:  Σ = diag(d) + F Fᵀ
// ---------------------------------------------------------------------------

/// `Σ⁻¹ v` for `Σ = diag(d) + F Fᵀ` via Woodbury.
/// `d: [B, n]` (> 0), `factor: [B, n, r]`, `v: [B, n]` → `[B, n]`.
pub fn woodbury_solve(diag: &Tensor, factor: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = clamp_min(diag, 1e-12)?;
    let v_ = v.unsqueeze(D::Minus1)?; // [B, n, 1]
    let d_col = d.unsqueeze(D::Minus1)?; // [B, n, 1]
    let d_inv_v = v_.broadcast_div(&d_col)?; // D⁻¹ v
    let f_d_inv = factor.broadcast_div(&d_col)?; // D⁻¹ F
    let r = factor.dim(D::Minus1)?;
    let eye = Tensor::eye(r, factor.dtype(), factor.device())?;
    // capacitance I_r + Fᵀ D⁻¹ F : [B, r, r]
    let cap = factor.transpose(D::Minus2, D::Minus1)?.matmul(&f_d_inv)?
        .broadcast_add(&eye)?;
    let rhs = factor.transpose(D::Minus2, D::Minus1)?.matmul(&d_inv_v)?; // [B, r, 1]
    let y = batched_solve(&cap, &rhs)?; // [B, r, 1]
    d_inv_v.sub(&f_d_inv.matmul(&y)?)?.squeeze(D::Minus1)
}

/// `log det Σ` for `Σ = diag(d) + F Fᵀ` (matrix-determinant lemma).
/// `d: [B, n]`, `factor: [B, n, r]` → `[B]`.
pub fn lowrank_logdet(diag: &Tensor, factor: &Tensor) -> Result<Tensor> {
    let d = clamp_min(diag, 1e-12)?;
    let f_d_inv = factor.broadcast_div(&d.unsqueeze(D::Minus1)?)?;
    let r = factor.dim(D::Minus1)?;
    let eye = Tensor::eye(r, factor.dtype(), factor.device())?;
    let cap = factor.transpose(D::Minus2, D::Minus1)?.matmul(&f_d_inv)?
        .broadcast_add(&eye)?; // [B, r, r]
    let logdet_cap = batched_logdet(&cap)?; // [B]
    d.log()?.sum(D::Minus1)?.add(&logdet_cap)
}

/// `x = μ + √d ⊙ η_d + F η_f` → `Cov(x) = diag(d) + F Fᵀ`.
pub fn lowrank_sample(
    mu: &Tensor,
    diag: &Tensor,
    factor: &Tensor,
    eta_d: Option<&Tensor>,
    eta_f: Option<&Tensor>,
) -> Result<Tensor> {
    let eta_d = match eta_d {
        Some(e) => e.clone(),
        None => mu.randn_like(0.0, 1.0)?,
    };
    let r = factor.dim(D::Minus1)?;
    let eta_f = match eta_f {
        Some(e) => e.clone(),
        None => {
            let mut dims = mu.dims().to_vec();
            let last = dims.len() - 1;
            dims[last] = r;
            Tensor::randn(0f32, 1f32, dims, mu.device())?.to_dtype(mu.dtype())?
        }
    };
    let sd = clamp_min(diag, 1e-12)?.sqrt()?;
    let fz = factor.matmul(&eta_f.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)?;
    mu.add(&sd.mul(&eta_d)?)?.add(&fz)
}

/// `∇_x log N(x; μ, diag(d)+FFᵀ) = −Σ⁻¹(x − μ)` (Woodbury).
pub fn lowrank_score(x: &Tensor, mu: &Tensor, diag: &Tensor, factor: &Tensor) -> Result<Tensor> {
    woodbury_solve(diag, factor, &x.sub(mu)?)?.affine(-1.0, 0.0)
}

/// Batched dense solve `A y = b` for small `A: [B, r, r]`, `b: [B, r, 1]` via
/// per-sample Cholesky on the CPU-side linalg layer (r is the factor rank, ≤ 8).
fn batched_solve(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let bsz = a.dim(0)?;
    let mut out = Vec::with_capacity(bsz);
    for i in 0..bsz {
        let ai = a.get(i)?;
        let bi = b.get(i)?;
        let yi = crate::linalg::cholesky_solve(&ai, &bi, 1e-10)
            .map_err(|e| candle_core::Error::Msg(format!("woodbury capacitance solve: {e}")))?;
        out.push(yi);
    }
    Tensor::stack(&out, 0)
}

/// Batched `log det A` for SPD `A: [B, r, r]`.
fn batched_logdet(a: &Tensor) -> Result<Tensor> {
    let bsz = a.dim(0)?;
    let mut out = Vec::with_capacity(bsz);
    for i in 0..bsz {
        let ai = a.get(i)?;
        let f = crate::linalg::CholeskyFactor::new(&ai, 1e-10)
            .map_err(|e| candle_core::Error::Msg(format!("woodbury capacitance logdet: {e}")))?;
        out.push(f.logdet().map_err(|e| candle_core::Error::Msg(e.to_string()))?);
    }
    Tensor::stack(&out, 0)
}

