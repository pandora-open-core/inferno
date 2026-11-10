//! Inferno — Fixed-point iteration and implicit derivatives for tensor equations.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};

use super::cg::{linear_cg, CgOptions, CgStats};
use super::precond::CiqPreconditioner;
use crate::linalg::jacobi_eigh;
use crate::settings;

/// A fixed point whose adjoint system the Krylov stack can solve.
pub trait FixedPointJacobian: Send + Sync {
    /// System size (the operator is `[..B, n, n]`).
    fn n(&self) -> usize;

    /// `v ↦ (I − ∂Φ/∂z)ᵀ v`, DETACHED. `v` is `[..B, n, C]`.
    fn adjoint_matvec(&self, v: &Tensor) -> Result<Tensor>;

    /// Euclidean self-adjointness — chooses CG over MINRES.
    ///
    /// Enforced by TEST, never by assertion: an operator that claims symmetry
    /// it does not have gets a wrong answer out of CG with no symptom at all.
    fn is_symmetric(&self) -> bool {
        false
    }

    /// Known null directions `[..B, n, k]`, columns mutually orthogonal.
    /// Supplying them is much better than letting a Krylov method discover
    /// them — see the module docs.
    fn null_space(&self) -> Option<Tensor> {
        None
    }

    /// The dense operator, for the small-`n` route. The default costs `n`
    /// matvecs; override it when the operator can build itself more cheaply.
    fn dense_adjoint(&self) -> Result<Tensor> {
        let n = self.n();
        let mut cols = Vec::with_capacity(n);
        for i in 0..n {
            cols.push(self.adjoint_matvec(&self.unit_probe(i)?)?);
        }
        let refs: Vec<&Tensor> = cols.iter().collect();
        // Each column is `[..B, n, 1]`; concatenating on the last axis gives
        // `[..B, n, n]` whose COLUMN `i` is `A·e_i`, i.e. exactly `A`.
        Tensor::cat(&refs, D::Minus1)
    }

    /// `e_i` shaped for [`Self::adjoint_matvec`]. Only used by the default
    /// [`Self::dense_adjoint`].
    fn unit_probe(&self, i: usize) -> Result<Tensor> {
        let n = self.n();
        let mut v = vec![0.0f64; n];
        v[i] = 1.0;
        Tensor::from_vec(v, (n, 1), &Device::Cpu)
    }
}

/// Which route [`fixed_point_adjoint`] took. Reported rather than inferred,
/// because "why is this slow" and "why is this exact" have the same answer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AdjointRoute {
    /// Dense eigendecomposition + pseudo-inverse. Exact, and handles the null
    /// space by construction rather than by deflation.
    DensePinv,
    /// Deflated CG on the matrix-free operator.
    Krylov,
}

#[derive(Clone, Debug)]
pub struct AdjointStats {
    pub route: AdjointRoute,
    pub iterations: usize,
    pub residual: f64,
    /// Relative size of the cotangent's component along the null space — how
    /// close the system came to being inconsistent.
    pub gauge_fraction: f64,
}

#[derive(Clone)]
pub struct AdjointOpts {
    pub tolerance: f64,
    pub max_iter: Option<usize>,
    pub precond: Option<Arc<CiqPreconditioner>>,
    /// Reject the solve when the cotangent's null-space component exceeds this
    /// fraction of its norm.
    ///
    /// A gauge component means `Aλ = ḡ` has NO solution, and both available
    /// failure modes are silent: CG converges to the least-squares point, and
    /// projecting first returns a confident answer to a different question. A
    /// correctly-formed cotangent from a gauge-invariant loss is orthogonal to
    /// the null space up to round-off, so anything above this is a bug
    /// upstream and should say so.
    pub max_gauge_fraction: f64,
    /// Take the Krylov route regardless of size — for tests that need to
    /// exercise it where the dense route would otherwise win.
    pub force_iterative: bool,
    /// Relative floor on eigenvalues kept by the dense pseudo-inverse.
    pub pinv_rtol: f64,
}

impl Default for AdjointOpts {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: None,
            precond: None,
            max_gauge_fraction: 1e-6,
            force_iterative: false,
            pinv_rtol: 1e-10,
        }
    }
}

/// Solve `A λ = ḡ` for a fixed-point adjoint, deflating the known null space.
///
/// The returned `λ` is ALWAYS projected off `null_space()`, whichever route
/// ran.
pub fn fixed_point_adjoint(
    jac: &dyn FixedPointJacobian,
    seed: &Tensor,
    opts: &AdjointOpts,
) -> Result<(Tensor, AdjointStats)> {
    let seed = if seed.rank() == 1 { seed.unsqueeze(D::Minus1)? } else { seed.clone() };
    let null = jac.null_space();

    // Consistency first: a gauge component makes the system unsolvable, and
    // every way of proceeding past it hides that fact.
    let gauge_fraction = match &null {
        Some(basis) => {
            let kept = project_off(&seed, basis)?;
            let lost = norm_all(&seed.sub(&kept)?)?;
            lost / norm_all(&seed)?.max(f64::MIN_POSITIVE)
        }
        None => 0.0,
    };
    if gauge_fraction > opts.max_gauge_fraction {
        return Err(candle_core::Error::Msg(format!(
            "fixed_point_adjoint: the cotangent has a {gauge_fraction:.3e} relative component \
             along the operator's null space, so the system is inconsistent and no lambda \
             solves it. Both silent options are worse than this error: CG would return the \
             least-squares point, and projecting first would answer a different question. \
             A gauge-invariant loss produces an orthogonal cotangent — check upstream, or \
             raise `max_gauge_fraction` if you know the residual is round-off."
        )));
    }
    let rhs = match &null {
        Some(basis) => project_off(&seed, basis)?,
        None => seed.clone(),
    };

    let dense_ok = !opts.force_iterative && jac.n() <= settings::max_cholesky_size::value();
    let (x, iterations, residual, route) = if dense_ok {
        let a = jac.dense_adjoint()?;
        let x = dense_pinv_solve(&a, &rhs, opts.pinv_rtol)?;
        let r = norm_all(&jac.adjoint_matvec(&x)?.sub(&rhs)?)?;
        (x, 0, r, AdjointRoute::DensePinv)
    } else {
        if !jac.is_symmetric() {
            return Err(candle_core::Error::Msg(
                "fixed_point_adjoint: the iterative route needs a self-adjoint operator (CG). \
                 A nonsymmetric adjoint needs MINRES/GMRES, which is not wired here."
                    .into(),
            ));
        }
        let cg_opts = CgOptions {
            max_iter: opts.max_iter,
            precond: opts.precond.clone(),
            ..CgOptions::for_tolerance(opts.tolerance)
        };
        // Deflate inside the matvec. `A` already annihilates the null space, so
        // `PAP == AP` mathematically; projecting both sides anyway is what
        // stops floating-point drift from re-seeding the direction CG is
        // supposed to be avoiding.
        let apply = |v: &Tensor| -> Result<Tensor> {
            let v = match &null {
                Some(b) => project_off(v, b)?,
                None => v.clone(),
            };
            let w = jac.adjoint_matvec(&v)?;
            match &null {
                Some(b) => project_off(&w, b),
                None => Ok(w),
            }
        };
        let (x, stats): (Tensor, CgStats) =
            linear_cg(apply, &rhs, None, None::<fn(&Tensor) -> Result<Tensor>>, &cg_opts)?;
        (x, stats.iterations, stats.residual_norm, AdjointRoute::Krylov)
    };

    let x = match &null {
        Some(basis) => project_off(&x, basis)?,
        None => x,
    };
    Ok((x, AdjointStats { route, iterations, residual, gauge_fraction }))
}

/// Remove the component of `v [..B, n, C]` along each column of
/// `basis [..B, n, k]` (columns assumed mutually orthogonal).
pub fn project_off(v: &Tensor, basis: &Tensor) -> Result<Tensor> {
    let basis =
        if basis.rank() == v.rank() - 1 { basis.unsqueeze(D::Minus1)? } else { basis.clone() };
    let k = basis.dim(D::Minus1)?;
    let mut out = v.clone();
    for c in 0..k {
        let q = basis.narrow(D::Minus1, c, 1)?; // [..B, n, 1]
        let den = q.sqr()?.sum_keepdim(D::Minus2)?;
        let num = out.broadcast_mul(&q)?.sum_keepdim(D::Minus2)?; // [..B, 1, C]
        out = out.broadcast_sub(&q.broadcast_mul(&num.broadcast_div(&den)?)?)?;
    }
    Ok(out)
}

fn norm_all(t: &Tensor) -> Result<f64> {
    Ok(t.sqr()?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?.sqrt())
}

/// Moore–Penrose solve of a symmetric (possibly singular) system, host-side in
/// `f64` via [`jacobi_eigh`].
///
/// Exact for the null space rather than deflating around it, which is why this
/// is the preferred route at small `n`. Two things `jacobi_eigh` requires of
/// its callers and gets here: a unit-Frobenius prescale (its convergence test
/// is ABSOLUTE) and no assumption that eigenvalues come back sorted.
/// The Moore–Penrose pseudo-inverse `A⁺` as an explicit matrix.
///
/// # Why a matrix and not a solve
///
/// [`fixed_point_adjoint`] pulls its operands to host `f64` and rebuilds the
/// answer, so it BREAKS THE AUTOGRAD GRAPH — correct for a VJP, where the
/// cotangent is already a constant, and useless for the forward-side implicit
/// correction, where the right-hand side is the tracked quantity and the
/// operator is the constant.
///
/// Returning `A⁺` lets that case be a single `matmul` against a detached
/// matrix: exact, differentiable in the right-hand side, and `O(1)` graph
/// depth. The alternative — unrolling CG on a tracked rhs — reintroduces
/// exactly the deep backward chain the implicit route exists to avoid.
///
/// Null directions are dropped at `rtol` relative to the largest eigenvalue,
/// so a singular `A` gives the least-norm solution rather than an error.
/// Requires a SYMMETRIC `A` (it goes through `jacobi_eigh`).
pub fn dense_pinv(a: &Tensor, rtol: f64) -> Result<Tensor> {
    let n = a.dim(D::Minus1)?;
    let eye = Tensor::eye(n, a.dtype(), a.device())?;
    dense_pinv_solve(a, &eye, rtol)
}

fn dense_pinv_solve(a: &Tensor, rhs: &Tensor, rtol: f64) -> Result<Tensor> {
    let n = a.dim(D::Minus1)?;
    if a.dim(D::Minus2)? != n {
        return Err(candle_core::Error::Msg("fixed_point_adjoint: operator is not square".into()));
    }
    let batch: usize = a.dims()[..a.rank() - 2].iter().product();
    let cols = rhs.dim(D::Minus1)?;

    let av: Vec<f64> = a.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
    let bv: Vec<f64> = rhs.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
    let mut out = vec![0.0f64; batch * n * cols];

    for b in 0..batch {
        let blk = &av[b * n * n..(b + 1) * n * n];
        let fro = blk.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let scaled: Vec<f64> = blk.iter().map(|v| v / fro).collect();
        let (eig, vecs) = jacobi_eigh(&scaled, n);

        let lmax = eig.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let cut = lmax * rtol;
        for c in 0..cols {
            let rhs_c: Vec<f64> = (0..n).map(|i| bv[b * n * cols + i * cols + c]).collect();
            for j in 0..n {
                if eig[j].abs() <= cut {
                    continue; // a null direction — the pseudo-inverse drops it
                }
                // Eigenvector `j` sits in COLUMN `j` of the row-major `vecs`.
                let dot: f64 = (0..n).map(|i| vecs[i * n + j] * rhs_c[i]).sum();
                let coeff = dot / (eig[j] * fro);
                for i in 0..n {
                    out[b * n * cols + i * cols + c] += coeff * vecs[i * n + j];
                }
            }
        }
    }
    Tensor::from_vec(out, rhs.shape().clone(), rhs.device())?.to_dtype(rhs.dtype())
}

#[cfg(test)]
mod tests {
    //! Verification of fixed-point iteration and implicit derivatives for tensor equations.

    use super::*;

    /// A dense operator with an optionally planted null vector, so both routes
    /// can be checked against an oracle.
    struct Dense {
        a: Tensor,
        null: Option<Tensor>,
        symmetric: bool,
    }

    impl FixedPointJacobian for Dense {
        fn n(&self) -> usize {
            self.a.dim(D::Minus1).unwrap()
        }
        fn adjoint_matvec(&self, v: &Tensor) -> Result<Tensor> {
            self.a.broadcast_matmul(v)
        }
        fn is_symmetric(&self) -> bool {
            self.symmetric
        }
        fn null_space(&self) -> Option<Tensor> {
            self.null.clone()
        }
    }

    fn lcg(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect()
    }

    /// `A = QΛQᵀ` with `λ₀ = 0`: symmetric PSD, singular on exactly one known
    /// direction.
    fn singular_spd(n: usize, seed: u64) -> (Dense, Tensor) {
        let dev = Device::Cpu;
        let raw = lcg(n * n, seed);
        let mut q: Vec<Vec<f64>> = Vec::new();
        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|i| raw[i * n + j]).collect();
            for u in &q {
                let d: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    v[i] -= d * u[i];
                }
            }
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            q.push(v);
        }
        let mut a = vec![0.0f64; n * n];
        for (j, col) in q.iter().enumerate() {
            let lam = j as f64; // λ = [0, 1, 2, …]
            for i in 0..n {
                for k in 0..n {
                    a[i * n + k] += lam * col[i] * col[k];
                }
            }
        }
        let null = Tensor::from_vec(q[0].clone(), (n, 1), &dev).unwrap();
        let a = Tensor::from_vec(a, (n, n), &dev).unwrap();
        (Dense { a, null: Some(null.clone()), symmetric: true }, null)
    }

    fn colvec(v: &[f64]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (v.len(), 1), &Device::Cpu).unwrap()
    }
    fn maxabs(t: &Tensor) -> f64 {
        t.abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    }

    /// Both routes must solve the same consistent singular system to the same
    /// minimum-norm answer. The dense one is exact; the other is deflated CG.
    #[test]
    fn the_two_routes_agree_on_a_consistent_singular_system() {
        let n = 12;
        let (op, null) = singular_spd(n, 7);
        // A consistent rhs is by definition one in the range.
        let rhs = op.adjoint_matvec(&colvec(&lcg(n, 3))).unwrap();

        let dense = AdjointOpts { tolerance: 1e-12, ..Default::default() };
        let (x_d, s_d) = fixed_point_adjoint(&op, &rhs, &dense).unwrap();
        assert_eq!(s_d.route, AdjointRoute::DensePinv);
        assert!(s_d.residual < 1e-10, "dense residual {}", s_d.residual);

        let iter = AdjointOpts { tolerance: 1e-12, force_iterative: true, ..Default::default() };
        let (x_k, s_k) = fixed_point_adjoint(&op, &rhs, &iter).unwrap();
        assert_eq!(s_k.route, AdjointRoute::Krylov);
        assert!(maxabs(&x_d.sub(&x_k).unwrap()) < 1e-6, "the two routes disagree");

        for x in [&x_d, &x_k] {
            let overlap = x.mul(&null).unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
            assert!(overlap.abs() < 1e-9, "leftover gauge component {overlap}");
        }
    }

    /// An inconsistent rhs must be REFUSED, not least-squares'd. This is the
    /// failure that otherwise looks entirely fine.
    #[test]
    fn an_inconsistent_cotangent_is_refused_by_name() {
        let n = 8;
        let (op, null) = singular_spd(n, 13);
        let good = op.adjoint_matvec(&colvec(&lcg(n, 5))).unwrap();
        let bad = good.add(&null.affine(0.5 * maxabs(&good), 0.0).unwrap()).unwrap();

        let e = fixed_point_adjoint(&op, &bad, &AdjointOpts::default()).unwrap_err().to_string();
        assert!(e.contains("inconsistent"), "unhelpful refusal: {e}");
        assert!(e.contains("null space"), "the message must name the cause: {e}");

        // …and it is a threshold, not a prohibition.
        let lax = AdjointOpts { max_gauge_fraction: 1.0, ..Default::default() };
        let (_, s) = fixed_point_adjoint(&op, &bad, &lax).unwrap();
        assert!(s.gauge_fraction > 1e-3, "the stat must report what was dropped");
    }

    /// The nonsingular case still works and needs no null space at all.
    #[test]
    fn a_nonsingular_operator_needs_no_deflation() {
        let n = 10;
        let dev = Device::Cpu;
        let m = Tensor::from_vec(lcg(n * n, 21), (n, n), &dev).unwrap();
        let a = m
            .matmul(&m.t().unwrap().contiguous().unwrap())
            .unwrap()
            .add(&Tensor::eye(n, DType::F64, &dev).unwrap())
            .unwrap();
        let op = Dense { a: a.clone(), null: None, symmetric: true };

        let x_true = colvec(&lcg(n, 33));
        let rhs = a.matmul(&x_true).unwrap();
        let (x, s) =
            fixed_point_adjoint(&op, &rhs, &AdjointOpts { tolerance: 1e-12, ..Default::default() })
                .unwrap();
        assert_eq!(s.gauge_fraction, 0.0);
        assert!(maxabs(&x.sub(&x_true).unwrap()) < 1e-8);
    }

    /// CG is valid only on a self-adjoint operator, and one that lies about it
    /// returns a wrong answer with no symptom — so the route refuses rather
    /// than trusting.
    #[test]
    fn the_iterative_route_refuses_a_nonsymmetric_operator() {
        let n = 6;
        let dev = Device::Cpu;
        let a = Tensor::from_vec(lcg(n * n, 41), (n, n), &dev).unwrap();
        let op = Dense { a, null: None, symmetric: false };
        let rhs = colvec(&lcg(n, 42));
        let e = fixed_point_adjoint(
            &op,
            &rhs,
            &AdjointOpts { force_iterative: true, ..Default::default() },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("self-adjoint"), "unhelpful refusal: {e}");
    }

    /// The default `dense_adjoint` reconstructs column by column; a flipped
    /// convention would silently transpose, which is invisible on a symmetric
    /// fixture — so this one is deliberately asymmetric.
    #[test]
    fn the_default_dense_reconstruction_is_not_transposed() {
        let n = 5;
        let dev = Device::Cpu;
        let a = Tensor::from_vec(lcg(n * n, 55), (n, n), &dev).unwrap();
        let op = Dense { a: a.clone(), null: None, symmetric: false };
        assert!(maxabs(&op.dense_adjoint().unwrap().sub(&a).unwrap()) < 1e-12);
    }
}
