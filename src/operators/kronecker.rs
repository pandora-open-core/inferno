//! Inferno — Kronecker-structured matrix operators and tensor products.

use candle_core::{Result, Tensor, D};

use super::{diag_embed, DiagOperator, LinearOperator};

/// `K₁ ⊗ K₂ ⊗ … ⊗ K_P`, factors held lazily.
pub struct KroneckerOperator {
    pub factors: Vec<Box<dyn LinearOperator>>,
}

impl KroneckerOperator {
    pub fn new(factors: Vec<Box<dyn LinearOperator>>) -> Self {
        assert!(!factors.is_empty(), "KroneckerOperator needs ≥ 1 factor");
        Self { factors }
    }

    /// Total rows `Π mᵢ`.
    fn total_rows(&self) -> usize {
        self.factors.iter().map(|f| f.nrows()).product()
    }

    /// Total cols `Π nᵢ`.
    fn total_cols(&self) -> usize {
        self.factors.iter().map(|f| f.ncols()).product()
    }
}

/// The dense Kronecker product of two matrices `[.., m, n] ⊗ [.., p, q]`
/// → `[.., m·p, n·q]`.
///
/// Built by outer-product-then-reshape rather than an index loop, so it
/// stays a differentiable Candle expression:
/// `A[.., m, 1, n, 1] * B[.., 1, p, 1, q]` → `[.., m, p, n, q]` → merge.
pub fn kron_dense(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (ad, bd) = (a.dims(), b.dims());
    let (m, n) = (ad[ad.len() - 2], ad[ad.len() - 1]);
    let (p, q) = (bd[bd.len() - 2], bd[bd.len() - 1]);
    let batch: Vec<usize> = ad[..ad.len() - 2].to_vec();

    // a → [.., m, 1, n, 1]
    let mut a_shape = batch.clone();
    a_shape.extend_from_slice(&[m, 1, n, 1]);
    // b → [.., 1, p, 1, q]
    let mut b_shape = batch.clone();
    b_shape.extend_from_slice(&[1, p, 1, q]);
    // product → [.., m, p, n, q]
    let mut full = batch.clone();
    full.extend_from_slice(&[m, p, n, q]);
    // final → [.., m·p, n·q]
    let mut out = batch;
    out.extend_from_slice(&[m * p, n * q]);

    let av = a.reshape(a_shape)?;
    let bv = b.reshape(b_shape)?;
    av.broadcast_mul(&bv)?
        .broadcast_as(full)?
        .contiguous()?
        .reshape(out)
}

/// `diag(A ⊗ B) = vec(diag(A) diag(B)ᵀ)` — recursive over the factors.
fn kron_diag(diags: &[Tensor]) -> Result<Tensor> {
    let lead = diags[0].clone();
    if diags.len() == 1 {
        return Ok(lead);
    }
    let trail = kron_diag(&diags[1..])?;
    // lead[.., n, 1] * trail[.., 1, m] → [.., n, m] → flatten
    let outer = lead
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&trail.unsqueeze(D::Minus2)?)?;
    let d = outer.dims();
    let mut out: Vec<usize> = d[..d.len() - 2].to_vec();
    out.push(d[d.len() - 2] * d[d.len() - 1]);
    outer.contiguous()?.reshape(out)
}

impl LinearOperator for KroneckerOperator {
    fn shape(&self) -> Vec<usize> {
        let mut s = self.factors[0].batch_shape();
        s.push(self.total_rows());
        s.push(self.total_cols());
        s
    }

    fn to_dense(&self) -> Result<Tensor> {
        let mut acc = self.factors[0].to_dense()?;
        for f in &self.factors[1..] {
            acc = kron_dense(&acc, &f.to_dense()?)?;
        }
        Ok(acc)
    }

    /// The reshape/multiply/transpose cycle: apply each factor to its own
    /// axis in turn, so the `Πnᵢ`-square matrix is never formed.
    ///
    /// Port of `linear_operator`'s `_matmul`: for each factor, view the rhs
    /// as `[.., nᵢ, −1]`, apply the factor, then rotate that axis to the
    /// back so the next factor sees its own axis in front.
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let num_cols = rhs.dim(D::Minus1)?;
        let batch = self.factors[0].batch_shape();
        let mut res = rhs.clone();

        for f in &self.factors {
            let (fr, fc) = (f.nrows(), f.ncols());
            // [.., N, C] → [.., fc, -1]
            let total: usize = res.elem_count() / (batch.iter().product::<usize>().max(1));
            let mut v = batch.clone();
            v.push(fc);
            v.push(total / fc);
            res = f.matmul(&res.contiguous()?.reshape(v)?)?; // [.., fr, rest]

            // [.., fr, rest] → [.., fr, rest/C, C] → transpose → [.., rest/C, fr, C]
            // (candle's `D` has no `Minus3`, so the two swapped axes are
            // addressed absolutely, just past the batch dims.)
            let rest = res.dim(D::Minus1)?;
            let mut v2 = batch.clone();
            v2.push(fr);
            v2.push(rest / num_cols);
            v2.push(num_cols);
            let ax = batch.len();
            let t = res.reshape(v2)?.transpose(ax, ax + 1)?;

            let mut v3 = batch.clone();
            v3.push(fr * (rest / num_cols));
            v3.push(num_cols);
            res = t.contiguous()?.reshape(v3)?;
        }
        Ok(res)
    }

    fn diagonal(&self) -> Result<Tensor> {
        let diags: Vec<Tensor> = self
            .factors
            .iter()
            .map(|f| f.diagonal())
            .collect::<Result<_>>()?;
        kron_diag(&diags)
    }

    /// `chol(A ⊗ B) = chol(A) ⊗ chol(B)` — factor-wise, then Kronecker.
    fn cholesky(&self, jitter: f64) -> Result<Tensor> {
        let mut acc = self.factors[0].cholesky(jitter)?;
        for f in &self.factors[1..] {
            acc = kron_dense(&acc, &f.cholesky(jitter)?)?;
        }
        Ok(acc)
    }

    /// `logdet(A ⊗ B) = n_B·logdet(A) + n_A·logdet(B)`, generalised:
    /// each factor's logdet is scaled by the product of the OTHER factors'
    /// sizes.
    fn log_abs_det(&self, jitter: f64) -> Result<Tensor> {
        let n_total = self.total_cols();
        let mut acc: Option<Tensor> = None;
        for f in &self.factors {
            let scale = (n_total / f.ncols()) as f64;
            let term = f.log_abs_det(jitter)?.affine(scale, 0.0)?;
            acc = Some(match acc {
                None => term,
                Some(a) => a.broadcast_add(&term)?,
            });
        }
        Ok(acc.unwrap())
    }

    /// `root(A ⊗ B) rhs = (root(A) ⊗ root(B)) rhs` — each factor's Cholesky
    /// applied through the same reshape/transpose cycle as
    /// [`matmul`](LinearOperator::matmul), so the `Πnᵢ`-square root is never
    /// formed. No routing gate: the factor-wise closed form beats both trait
    /// arms at every size.
    fn sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let chol_factors: Vec<Box<dyn LinearOperator>> = self
            .factors
            .iter()
            .map(|f| -> Result<Box<dyn LinearOperator>> {
                Ok(Box::new(super::DenseOperator::new(f.cholesky(jitter)?)))
            })
            .collect::<Result<_>>()?;
        KroneckerOperator::new(chol_factors).matmul(rhs)
    }

    /// `root(A ⊗ B)⁻¹ rhs` — per-factor triangular solves through the same
    /// cycle. Mirrors [`cholesky_solve`](LinearOperator::cholesky_solve)'s
    /// loop with `L_f⁻¹ ·` in place of `(L_f L_fᵀ)⁻¹ ·`.
    fn inv_sqrt_matmul(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let num_cols = rhs.dim(D::Minus1)?;
        let batch = self.factors[0].batch_shape();
        let mut res = rhs.clone();

        for f in &self.factors {
            let n = f.ncols();
            let l_f = f.cholesky(jitter)?;
            let total: usize = res.elem_count() / (batch.iter().product::<usize>().max(1));
            let mut v = batch.clone();
            v.push(n);
            v.push(total / n);
            res = crate::linalg_diff::tri_solve_diff(&l_f, &res.contiguous()?.reshape(v)?)?;

            let rest = res.dim(D::Minus1)?;
            let mut v2 = batch.clone();
            v2.push(n);
            v2.push(rest / num_cols);
            v2.push(num_cols);
            let ax = batch.len();
            let t = res.reshape(v2)?.transpose(ax, ax + 1)?;

            let mut v3 = batch.clone();
            v3.push(n * (rest / num_cols));
            v3.push(num_cols);
            res = t.contiguous()?.reshape(v3)?;
        }
        Ok(res)
    }

    /// `(A ⊗ B)⁻¹ rhs = (A⁻¹ ⊗ B⁻¹) rhs` — solve factor-by-factor with the
    /// same cycling as [`matmul`](LinearOperator::matmul), so the big
    /// system never materialises.
    fn cholesky_solve(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        let num_cols = rhs.dim(D::Minus1)?;
        let batch = self.factors[0].batch_shape();
        let mut res = rhs.clone();

        for f in &self.factors {
            let n = f.ncols();
            let total: usize = res.elem_count() / (batch.iter().product::<usize>().max(1));
            let mut v = batch.clone();
            v.push(n);
            v.push(total / n);
            res = f.cholesky_solve(&res.contiguous()?.reshape(v)?, jitter)?;

            let rest = res.dim(D::Minus1)?;
            let mut v2 = batch.clone();
            v2.push(n);
            v2.push(rest / num_cols);
            v2.push(num_cols);
            let ax = batch.len();
            let t = res.reshape(v2)?.transpose(ax, ax + 1)?;

            let mut v3 = batch.clone();
            v3.push(n * (rest / num_cols));
            v3.push(num_cols);
            res = t.contiguous()?.reshape(v3)?;
        }
        Ok(res)
    }
}

// ---------------------------------------------------------------------------
// KroneckerDiagOperator
// ---------------------------------------------------------------------------

/// `D₁ ⊗ D₂ ⊗ …` where every factor is diagonal — itself diagonal, so the
/// whole thing collapses to one `O(Πnᵢ)` diagonal vector.
///
/// This is the exact shape of the multitask likelihood's rank-0 noise
/// (`I_n ⊗ D_T`): representing it this way keeps the marginal covariance's
/// noise term O(N·T) instead of O(N²T²).
#[derive(Clone)]
pub struct KroneckerDiagOperator {
    pub factors: Vec<DiagOperator>,
}

impl KroneckerDiagOperator {
    pub fn new(factors: Vec<DiagOperator>) -> Self {
        assert!(
            !factors.is_empty(),
            "KroneckerDiagOperator needs ≥ 1 factor"
        );
        Self { factors }
    }

    /// Collapse to the single equivalent [`DiagOperator`].
    pub fn to_diag(&self) -> Result<DiagOperator> {
        let diags: Vec<Tensor> = self.factors.iter().map(|f| f.diag.clone()).collect();
        Ok(DiagOperator::new(kron_diag(&diags)?))
    }
}

impl LinearOperator for KroneckerDiagOperator {
    fn shape(&self) -> Vec<usize> {
        let n: usize = self.factors.iter().map(|f| f.ncols()).product();
        let mut s = self.factors[0].batch_shape();
        s.push(n);
        s.push(n);
        s
    }
    fn to_dense(&self) -> Result<Tensor> {
        diag_embed(&self.diagonal()?)
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.to_diag()?.matmul(rhs)
    }
    fn t_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul(rhs)
    }
    fn diagonal(&self) -> Result<Tensor> {
        self.to_diag().and_then(|d| d.diagonal())
    }
    fn cholesky(&self, jitter: f64) -> Result<Tensor> {
        self.to_diag()?.cholesky(jitter)
    }
    fn cholesky_solve(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        self.to_diag()?.cholesky_solve(rhs, jitter)
    }
    fn log_abs_det(&self, jitter: f64) -> Result<Tensor> {
        self.to_diag()?.log_abs_det(jitter)
    }
    fn inv_quad(&self, rhs: &Tensor, jitter: f64) -> Result<Tensor> {
        self.to_diag()?.inv_quad(rhs, jitter)
    }
}

#[cfg(test)]
mod tests {
    //! Verification of kronecker-structured matrix operators and tensor products.

    use super::*;
    use crate::operators::{DenseOperator, DiagOperator};
    use candle_core::{DType, Device};

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn spd(n: usize, dev: &Device) -> Tensor {
        let a = Tensor::randn(0f32, 1f32, (n, n), dev).unwrap();
        a.matmul(&a.t().unwrap())
            .unwrap()
            .add(
                &Tensor::eye(n, DType::F32, dev)
                    .unwrap()
                    .affine(n as f64, 0.0)
                    .unwrap(),
            )
            .unwrap()
    }

    /// `kron_dense` must match the textbook definition
    /// `(A ⊗ B)[i·p+k, j·q+l] = A[i,j]·B[k,l]`.
    #[test]
    fn kron_dense_matches_definition() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &dev).unwrap();
        let b = Tensor::new(&[[0.0f32, 5.0], [6.0, 7.0]], &dev).unwrap();
        let k = kron_dense(&a, &b).unwrap();
        assert_eq!(k.dims(), &[4, 4]);
        let got: Vec<Vec<f32>> = k.to_vec2().unwrap();
        let av: Vec<Vec<f32>> = a.to_vec2().unwrap();
        let bv: Vec<Vec<f32>> = b.to_vec2().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                for kk in 0..2 {
                    for l in 0..2 {
                        let want = av[i][j] * bv[kk][l];
                        assert!(
                            (got[i * 2 + kk][j * 2 + l] - want).abs() < 1e-6,
                            "({i},{j},{kk},{l})"
                        );
                    }
                }
            }
        }
    }

    /// The cycling matmul must equal the dense Kronecker product's matmul —
    /// this is the identity the whole operator rests on.
    #[test]
    fn kronecker_matmul_matches_dense() {
        let dev = Device::Cpu;
        let a = spd(3, &dev);
        let b = spd(4, &dev);
        let op = KroneckerOperator::new(vec![
            Box::new(DenseOperator::new(a.clone())),
            Box::new(DenseOperator::new(b.clone())),
        ]);
        assert_eq!(op.shape(), vec![12, 12]);

        let dense = op.to_dense().unwrap();
        assert!(max_abs(&dense, &kron_dense(&a, &b).unwrap()) < 1e-5);

        let x = Tensor::randn(0f32, 1f32, (12, 2), &dev).unwrap();
        let fast = op.matmul(&x).unwrap();
        let slow = dense.matmul(&x).unwrap();
        assert!(
            max_abs(&fast, &slow) < 1e-3,
            "kron matmul: {}",
            max_abs(&fast, &slow)
        );
    }

    #[test]
    fn kronecker_diag_chol_logdet_and_solve_match_dense() {
        let dev = Device::Cpu;
        let a = spd(3, &dev);
        let b = spd(2, &dev);
        let op = KroneckerOperator::new(vec![
            Box::new(DenseOperator::new(a.clone())),
            Box::new(DenseOperator::new(b.clone())),
        ]);
        let dense = DenseOperator::new(op.to_dense().unwrap());

        // diag(A ⊗ B)
        assert!(max_abs(&op.diagonal().unwrap(), &dense.diagonal().unwrap()) < 1e-4);

        // chol(A ⊗ B) == chol(A) ⊗ chol(B)  →  L Lᵀ == A ⊗ B
        let l = op.cholesky(1e-9).unwrap();
        let llt = l.matmul(&l.t().unwrap()).unwrap();
        assert!(
            max_abs(&llt, &dense.to_dense().unwrap()) < 1e-2,
            "chol(A⊗B)"
        );

        // logdet identity
        let fast = op.log_abs_det(1e-9).unwrap().to_scalar::<f32>().unwrap();
        let slow = dense.log_abs_det(1e-9).unwrap().to_scalar::<f32>().unwrap();
        assert!(
            (fast - slow).abs() / slow.abs().max(1.0) < 1e-3,
            "{fast} vs {slow}"
        );

        // factor-wise solve == dense solve, and it actually inverts
        let x = Tensor::randn(0f32, 1f32, (6, 2), &dev).unwrap();
        let sol = op.cholesky_solve(&x, 1e-9).unwrap();
        let back = dense.to_dense().unwrap().matmul(&sol).unwrap();
        assert!(max_abs(&back, &x) < 1e-2, "kron solve must invert");
    }

    /// The likelihood's actual shape: `I_n ⊗ D_T` stays diagonal and O(N·T).
    #[test]
    fn kronecker_of_diagonals_collapses_to_one_diagonal() {
        let dev = Device::Cpu;
        let eye_n = DiagOperator::new(Tensor::ones(4, DType::F32, &dev).unwrap());
        let d_t = DiagOperator::new(Tensor::new(&[0.5f32, 2.0, 3.0], &dev).unwrap());
        let op = KroneckerDiagOperator::new(vec![eye_n, d_t.clone()]);

        assert_eq!(op.shape(), vec![12, 12]);
        let collapsed = op.to_diag().unwrap();
        assert_eq!(collapsed.diag.dims(), &[12]);
        // I ⊗ D repeats D's diagonal n times
        let v: Vec<f32> = collapsed.diag.to_vec1().unwrap();
        for blk in 0..4 {
            assert!((v[blk * 3] - 0.5).abs() < 1e-6);
            assert!((v[blk * 3 + 1] - 2.0).abs() < 1e-6);
            assert!((v[blk * 3 + 2] - 3.0).abs() < 1e-6);
        }
        // and it agrees with the general (dense) Kronecker path
        let general = KroneckerOperator::new(vec![
            Box::new(DiagOperator::new(
                Tensor::ones(4, DType::F32, &dev).unwrap(),
            )),
            Box::new(d_t),
        ]);
        assert!(max_abs(&op.diagonal().unwrap(), &general.diagonal().unwrap()) < 1e-6);
    }

    /// Gradients must reach each Kronecker factor separately.
    #[test]
    fn gradients_reach_each_kronecker_factor() {
        use candle_core::Var;
        let dev = Device::Cpu;
        let va = Var::from_tensor(&spd(3, &dev)).unwrap();
        let vb = Var::from_tensor(&spd(2, &dev)).unwrap();
        let op = KroneckerOperator::new(vec![
            Box::new(DenseOperator::new(va.as_tensor().clone())),
            Box::new(DenseOperator::new(vb.as_tensor().clone())),
        ]);
        let x = Tensor::randn(0f32, 1f32, (6, 1), &dev).unwrap();
        let loss = op.matmul(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for (name, v) in [("A", &va), ("B", &vb)] {
            let g = grads
                .get(v)
                .unwrap_or_else(|| panic!("no grad for factor {name}"));
            let m = g
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(m.is_finite() && m > 0.0, "factor {name} grad {m}");
        }
    }
}
