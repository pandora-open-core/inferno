//! Inferno — Krylov-subspace recycling for related linear systems.

use candle_core::{DType, Result, Tensor, D};

use super::cg::{linear_cg_harvest, CgDirections, CgOptions, CgStats};
use super::precond::CiqPreconditioner;

/// Identity of the operator a subspace was harvested from.
///
/// `shape` is not redundant with `formula`: a leaf-set change (adding a `rho`
/// dampening leaf, say) changes the gradient pack layout and hence the vector
/// length, and a length mismatch is a panic in a dot product rather than a
/// merely poor guess.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RecycleKey {
    /// Structural identity of the operator — a formula hash, an epoch id,
    /// whatever the caller can produce cheaply and consistently.
    pub formula: u64,
    /// Generation counter, e.g. `StochasticControlState::ot_generation`. The
    /// optimiser mutates parameters underneath a cached subspace exactly as it
    /// does the per-step OT memo, and the discipline is the same: valid within
    /// a step, never across one.
    pub generation: u64,
    /// System size, so a shape change is a miss rather than a panic.
    pub n: usize,
}

/// An A-conjugate subspace kept between solves.
#[derive(Clone, Debug)]
pub struct RecycleSubspace {
    /// `[n, r]` — directions as columns.
    u: Tensor,
    /// `[n, r]` — `A U`, cached so the projection needs no matvec.
    au: Tensor,
    /// `[r]` — `diag(UᵀAU)`, exact because the columns are A-conjugate.
    pap: Vec<f64>,
    key: RecycleKey,
}

impl RecycleSubspace {
    /// Keep the last `r` directions of a finished solve.
    ///
    /// The LAST, not the first: early CG directions are dominated by the
    /// largest eigenvalues, which converge on their own. The slow directions —
    /// the ones worth carrying — are what the iteration is still working on
    /// when it stops.
    ///
    /// Single-column solves only. A multi-column solve builds a different
    /// direction sequence per column and there is no one subspace to keep.
    pub fn harvest(dirs: &CgDirections, key: RecycleKey, r: usize) -> Result<Option<Self>> {
        let take = r.min(dirs.p.len());
        if take == 0 || r == 0 {
            return Ok(None);
        }
        let start = dirs.p.len() - take;
        let cols: Vec<Tensor> = dirs.p[start..].to_vec();
        if cols[0].rank() != 2 || cols[0].dim(D::Minus1)? != 1 {
            return Ok(None); // batched or multi-column: no single subspace
        }
        let acols: Vec<Tensor> = dirs.ap[start..].to_vec();
        let mut pap = Vec::with_capacity(take);
        for d in &dirs.pap[start..] {
            pap.push(d.flatten_all()?.to_dtype(DType::F64)?.to_vec1::<f64>()?[0]);
        }
        // A direction with a vanishing `pᵀAp` is a null (or frozen) direction:
        // keeping it puts a division by ~0 in the projection.
        let scale = pap.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let keep: Vec<usize> =
            (0..take).filter(|&i| pap[i].abs() > 1e-12 * scale.max(1e-300)).collect();
        if keep.is_empty() {
            return Ok(None);
        }
        let u: Vec<&Tensor> = keep.iter().map(|&i| &cols[i]).collect();
        let a: Vec<&Tensor> = keep.iter().map(|&i| &acols[i]).collect();
        let pap = keep.iter().map(|&i| pap[i]).collect();
        Ok(Some(Self {
            u: Tensor::cat(&u, D::Minus1)?,
            au: Tensor::cat(&a, D::Minus1)?,
            pap,
            key,
        }))
    }

    pub fn key(&self) -> RecycleKey {
        self.key
    }

    /// Number of retained directions.
    pub fn dim(&self) -> usize {
        self.pap.len()
    }

    /// `x₀ = U (UᵀAU)⁻¹ Uᵀ b`, using the cached `AU` — no matvec.
    ///
    /// Correct as a starting point for ANY `U`; a wrong one costs iterations,
    /// never accuracy.
    pub fn initial_guess(&self, rhs: &Tensor) -> Result<Tensor> {
        let rhs2 = if rhs.rank() == 1 { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
        // Uᵀb, one column at a time so the caller may pass several.
        let utb = self.u.t()?.contiguous()?.matmul(&rhs2)?; // [r, C]
        let inv = Tensor::from_vec(
            self.pap.iter().map(|v| 1.0 / v).collect::<Vec<f64>>(),
            (self.pap.len(), 1),
            rhs2.device(),
        )?
        .to_dtype(rhs2.dtype())?;
        let coeff = utb.broadcast_mul(&inv)?; // [r, C]
        let x0 = self.u.matmul(&coeff)?;
        if rhs.rank() == 1 {
            x0.squeeze(D::Minus1)
        } else {
            Ok(x0)
        }
    }

    /// Turn the recycled directions into a SPECTRAL PRECONDITIONER.
    ///
    /// This is the half of recycling that actually changes the spectrum. A
    /// Galerkin warm start moves the starting point — it saves
    /// `log(r₀/tol)` worth of iterations and nothing more, which is why the
    /// measured gain is modest (cold 20 → repeat 16 → drifted 17). Deflating
    /// instead makes `M⁻¹A ≈ I` on the recycled directions, which changes the
    /// effective condition number.
    ///
    /// The construction, all `r × r` host arithmetic plus two tensor matmuls:
    ///
    /// ```text
    /// G  = UᵀU              → (Λ, W)  by eigendecomposition, rank-truncated
    /// Q  = U W Λ^{-1/2}     orthonormal by construction
    /// T  = QᵀAQ = Λ^{-1/2} Wᵀ diag(pᵀAp) W Λ^{-1/2}      ← uses the CACHED pᵀAp
    /// (θ, V) = eigh(T)      → Ritz values and the final basis Q V
    /// ```
    ///
    /// `diag(pᵀAp)` is exact because CG's directions are A-conjugate, so `UᵀAU`
    /// really is diagonal — that is what makes `T` free of any extra matvec.
    ///
    /// # A stale subspace is still SOUND here
    ///
    /// The module header warns that deflation "changes the operator CG sees, so
    /// staleness would be a correctness question rather than a speed one". That
    /// is true of projected/augmented Krylov methods. It is **not** true of
    /// this, because the result is an ordinary SPD preconditioner: `M = δI + Q
    /// diag(μ−δ)Qᵀ` is SPD for any orthonormal `Q` and any positive `μ, δ`,
    /// whatever operator they came from, and PCG converges against any SPD `M`.
    /// So [`RecycleKey`] stays an optimisation guard for this path too, and a
    /// stale subspace still costs iterations rather than accuracy.
    ///
    /// Returns `None` when the directions carry no usable spectrum — too few,
    /// rank-deficient, or non-positive Rayleigh quotients (which mean the
    /// operator was not positive definite on the sampled subspace).
    pub fn as_preconditioner(&self, rank_tol: f64) -> Result<Option<CiqPreconditioner>> {
        let r = self.pap.len();
        if r < 2 {
            return Ok(None);
        }
        let n = self.u.dim(D::Minus2)?;

        // G = UᵀU, host.
        let g = self.u.t()?.contiguous()?.matmul(&self.u)?;
        let gv: Vec<f64> = g.flatten_all()?.to_dtype(DType::F64)?.to_vec1()?;
        let fro = gv.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(fro.is_finite() && fro > 0.0) {
            return Ok(None);
        }
        let scaled: Vec<f64> = gv.iter().map(|v| v / fro).collect();
        let (evg, wg) = crate::linalg::jacobi_eigh(&scaled, r);

        // Rank-truncate: descending, drop below `rank_tol · λmax`.
        let mut order: Vec<usize> = (0..r).collect();
        order.sort_by(|&i, &j| evg[j].partial_cmp(&evg[i]).unwrap_or(std::cmp::Ordering::Equal));
        let lmax = evg[order[0]];
        if !(lmax.is_finite() && lmax > 0.0) {
            return Ok(None);
        }
        let keep: Vec<usize> =
            order.into_iter().filter(|&j| evg[j] > rank_tol * lmax).collect();
        let rp = keep.len();
        if rp < 2 {
            return Ok(None);
        }

        // S = W Λ^{-1/2}, [r, rp] — the whitening.
        let mut sw = vec![0.0f64; r * rp];
        for (c, &j) in keep.iter().enumerate() {
            let inv = 1.0 / (evg[j] * fro).sqrt();
            for i in 0..r {
                sw[i * rp + c] = wg[i * r + j] * inv;
            }
        }
        // T = Sᵀ diag(pap) S, [rp, rp].
        let mut t = vec![0.0f64; rp * rp];
        for a in 0..rp {
            for b in 0..rp {
                let mut acc = 0.0;
                for i in 0..r {
                    acc += sw[i * rp + a] * self.pap[i] * sw[i * rp + b];
                }
                t[a * rp + b] = acc;
            }
        }
        let tfro = t.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(tfro.is_finite() && tfro > 0.0) {
            return Ok(None);
        }
        let tscaled: Vec<f64> = t.iter().map(|v| v / tfro).collect();
        let (evt, vt) = crate::linalg::jacobi_eigh(&tscaled, rp);

        // Ascending, eigenvector columns permuted with them.
        let mut ord: Vec<usize> = (0..rp).collect();
        ord.sort_by(|&i, &j| evt[i].partial_cmp(&evt[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mu: Vec<f64> = ord.iter().map(|&j| evt[j] * tfro).collect();
        if mu.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
            return Ok(None); // not SPD on the sampled subspace
        }

        // Coefficients over U: C = S V, [r, rp].
        let mut c = vec![0.0f64; r * rp];
        for i in 0..r {
            for (slot, &j) in ord.iter().enumerate() {
                let mut acc = 0.0;
                for a in 0..rp {
                    acc += sw[i * rp + a] * vt[a * rp + j];
                }
                c[i * rp + slot] = acc;
            }
        }
        let cm = Tensor::from_vec(c, (r, rp), self.u.device())?.to_dtype(self.u.dtype())?;
        let q = self.u.matmul(&cm)?; // [n, rp], orthonormal
        let _ = n;

        // Same δ convention as the Lanczos probe: the geometric mean, because
        // δ is a SCALE and the unresolved bulk contributes max(λ/δ, δ/λ).
        let delta = (mu.iter().map(|m| m.ln()).sum::<f64>() / mu.len() as f64).exp();
        let mu_t = Tensor::from_vec(mu, (rp,), q.device())?.to_dtype(q.dtype())?;
        match CiqPreconditioner::spectral(&q, &mu_t, delta, 1e-6) {
            Ok(p) => Ok(Some(p)),
            // Every refusal in `spectral` is for a silent failure; here they
            // just mean this subspace is not worth preconditioning with, and
            // a preconditioner is an accelerator, never a dependency.
            Err(_) => Ok(None),
        }
    }

    /// Residual of the recycled guess relative to `‖b‖` — how much the
    /// subspace actually bought, reported rather than assumed.
    pub fn guess_quality(&self, rhs: &Tensor) -> Result<f64> {
        let rhs2 = if rhs.rank() == 1 { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };
        let utb = self.u.t()?.contiguous()?.matmul(&rhs2)?;
        let inv = Tensor::from_vec(
            self.pap.iter().map(|v| 1.0 / v).collect::<Vec<f64>>(),
            (self.pap.len(), 1),
            rhs2.device(),
        )?
        .to_dtype(rhs2.dtype())?;
        let ax0 = self.au.matmul(&utb.broadcast_mul(&inv)?)?;
        let num = norm(&rhs2.sub(&ax0)?)?;
        Ok(num / norm(&rhs2)?.max(f64::MIN_POSITIVE))
    }
}

fn norm(t: &Tensor) -> Result<f64> {
    Ok(t.sqr()?.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?.sqrt())
}

/// Outcome of a recycled solve, beyond the usual [`CgStats`].
#[derive(Clone, Debug)]
pub struct RecycleStats {
    /// Whether a usable subspace was found for this key.
    pub hit: bool,
    /// Directions retained afterwards.
    pub kept: usize,
}

/// [`super::cg::linear_cg`] with a recycled warm start, harvesting a fresh
/// subspace on the way out.
///
/// `slot` is read AND written: a key mismatch simply misses (the stale
/// subspace is dropped), which is the whole reason the key exists.
pub fn linear_cg_recycled<F>(
    matmul: F,
    rhs: &Tensor,
    slot: &mut Option<RecycleSubspace>,
    key: RecycleKey,
    keep: usize,
    opts: &CgOptions,
) -> Result<(Tensor, CgStats, RecycleStats)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let hit = slot.as_ref().is_some_and(|s| s.key == key);
    let x0 = if hit { Some(slot.as_ref().unwrap().initial_guess(rhs)?) } else { None };
    if !hit {
        *slot = None;
    }

    let (x, stats, dirs) = linear_cg_harvest(
        matmul,
        rhs,
        x0.as_ref(),
        None::<fn(&Tensor) -> Result<Tensor>>,
        opts,
    )?;

    // Harvest only from a COLD solve. A warm-started solve's directions
    // describe the CORRECTION to the guess, not the solution — they span
    // whatever the previous subspace missed, which is close to orthogonal to
    // the part worth keeping. Overwriting with them degrades the subspace on
    // every reuse, and the symptom is a warm start that stops helping after
    // the first hit rather than an error.
    //
    // Accumulating instead (old columns ⊕ new, GCRO-DR style) is the better
    // answer, but the two sets are not mutually A-conjugate, so `UᵀAU` stops
    // being diagonal and the O(r) projection becomes an r × r solve. Not built.
    let kept = if hit {
        slot.as_ref().map_or(0, RecycleSubspace::dim)
    } else {
        let fresh = RecycleSubspace::harvest(&dirs, key, keep)?;
        let k = fresh.as_ref().map_or(0, RecycleSubspace::dim);
        if fresh.is_some() {
            *slot = fresh;
        }
        k
    };
    Ok((x, stats, RecycleStats { hit, kept }))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::settings;
    use candle_core::Device;

    pub(super) fn lcg(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect()
    }

    /// SPD with a controlled condition number.
    pub(super) fn spd(n: usize, seed: u64, shift: f64) -> Tensor {
        let dev = Device::Cpu;
        let m = Tensor::from_vec(lcg(n * n, seed), (n, n), &dev).unwrap();
        m.matmul(&m.t().unwrap().contiguous().unwrap())
            .unwrap()
            .add(&Tensor::eye(n, DType::F64, &dev).unwrap().affine(shift, 0.0).unwrap())
            .unwrap()
    }

    /// An orthonormal basis from Gram-Schmidt on pseudo-random columns.
    pub(super) fn ortho(n: usize, seed: u64) -> Vec<Vec<f64>> {
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
        q
    }

    /// `A = Q diag(λ) Qᵀ` for an explicit spectrum.
    ///
    /// The shape that makes recycling worth anything — and the shape the OT
    /// Laplacian actually has — is a HANDFUL of small eigenvalues under a tight
    /// cluster. Those few slow directions are what CG spends its iterations on
    /// and what a small recycled subspace can capture. A random SPD spreads its
    /// spectrum instead, so no low-dimensional subspace helps and a test built
    /// on one measures nothing.
    pub(super) fn from_spectrum(lams: &[f64], seed: u64) -> Tensor {
        let n = lams.len();
        let q = ortho(n, seed);
        let mut a = vec![0.0f64; n * n];
        for (j, col) in q.iter().enumerate() {
            for i in 0..n {
                for k in 0..n {
                    a[i * n + k] += lams[j] * col[i] * col[k];
                }
            }
        }
        Tensor::from_vec(a, (n, n), &Device::Cpu).unwrap()
    }

    /// Three isolated small modes beneath a cluster near 1.
    fn clustered(n: usize, seed: u64) -> Tensor {
        let lams: Vec<f64> = (0..n)
            .map(|i| match i {
                0 => 1e-3,
                1 => 2e-3,
                2 => 4e-3,
                _ => 1.0 + 0.05 * ((i % 7) as f64) / 7.0,
            })
            .collect();
        from_spectrum(&lams, seed)
    }

    pub(super) fn colvec(v: &[f64]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (v.len(), 1), &Device::Cpu).unwrap()
    }
    pub(super) fn maxabs(t: &Tensor) -> f64 {
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
    pub(super) fn key(gen: u64, n: usize) -> RecycleKey {
        RecycleKey { formula: 0xF00D, generation: gen, n }
    }

    /// The payoff, measured the only way that means anything: ITERATION COUNT.
    ///
    /// Benchmarked with `cg_ignore_min_iter` ON. The 10-iteration floor exists
    /// so a fluke-small early residual cannot end a solve, but it also makes
    /// every count below 11 unobservable — and finishing early is the entire
    /// point of a warm start. The analogous preconditioner test in rung 4
    /// passed vacuously at 11 vs 11 for exactly this reason.
    ///
    /// Two regimes, because one number proves nothing on its own:
    ///
    /// * **Identical rhs** — the recycled span already contains the solution,
    ///   so the guess should very nearly BE the answer. This is the sharpest
    ///   available statement that the projection is reached and correct.
    /// * **Drifted rhs** — the realistic case (an ε ladder, successive Newton
    ///   steps). Still a saving, necessarily smaller.
    ///
    /// Measured on this fixture: cold 20 iterations; repeat 16, from a guess
    /// whose relative residual is 4.3e-7; drifted 17.
    ///
    /// Two honest observations about that number. The guess residual is 4e-7
    /// rather than machine precision because 20 CG iterations lose that much
    /// A-conjugacy in f64 — the projection is exact, the directions are not.
    /// And a 4e-7 start only buys four iterations because CG still has to
    /// cross the remaining four orders of magnitude at the same rate; a warm
    /// start shifts the starting point, it does not change the spectrum.
    ///
    /// So the warm start is a real but MODEST saving. The larger win the
    /// subspace could give — using it to DEFLATE, which does change the
    /// effective spectrum — needs `precond::Kind::Spectral` and is not built.
    #[test]
    fn recycling_cuts_iterations_on_a_repeated_and_on_a_drifting_rhs() {
        let _g = settings::cg_ignore_min_iter::with(true);
        let n = 40;
        // Enough distinct small modes that CG cannot finite-terminate in a
        // couple of steps, but well short of `n` so the directions stay
        // A-conjugate — see `from_spectrum`.
        let lams: Vec<f64> = (0..n)
            .map(|i| if i < 12 { 1e-2 * 1.35f64.powi(i as i32) } else { 1.0 + 0.02 * (i % 5) as f64 })
            .collect();
        let a = from_spectrum(&lams, 5);
        let opts = CgOptions::for_tolerance(1e-10);
        let mm = |v: &Tensor| a.matmul(v);
        let cold = |rhs: &Tensor| {
            crate::operators::cg::linear_cg(
                mm,
                rhs,
                None,
                None::<fn(&Tensor) -> Result<Tensor>>,
                &opts,
            )
            .unwrap()
        };

        let base = lcg(n, 11);
        let rhs = colvec(&base);
        let mut slot: Option<RecycleSubspace> = None;

        let (x0, s0) = cold(&rhs);
        let (_, w0, r0) = linear_cg_recycled(mm, &rhs, &mut slot, key(0, n), 24, &opts).unwrap();
        assert!(!r0.hit, "nothing to recycle on the first solve");
        assert_eq!(w0.iterations, s0.iterations);
        assert!(s0.iterations > 12, "fixture too easy: {} iterations", s0.iterations);
        assert!(s0.iterations < n, "fixture ran past n: {}", s0.iterations);

        // (1) Same rhs again — the span already contains the solution.
        let q = slot.as_ref().unwrap().guess_quality(&rhs).unwrap();
        let (x1, w1, r1) = linear_cg_recycled(mm, &rhs, &mut slot, key(0, n), 24, &opts).unwrap();
        assert!(r1.hit);
        println!("cold {}  repeat {} (guess residual {q:.2e})", s0.iterations, w1.iterations);
        assert!(q < 1e-3, "the recycled guess barely helped: {q:.3e}");
        assert!(
            w1.iterations < s0.iterations,
            "a repeated rhs must be cheaper, got {} vs {}",
            w1.iterations,
            s0.iterations
        );
        assert!(maxabs(&x1.sub(&x0).unwrap()) < 1e-6, "recycling changed the answer");

        // (2) A drifted rhs — a real saving, necessarily smaller.
        let drift = lcg(n, 900);
        let rhs2 =
            colvec(&base.iter().zip(&drift).map(|(b, d)| b + 0.02 * d).collect::<Vec<f64>>());
        let (x2c, s2) = cold(&rhs2);
        let (x2w, w2, r2) = linear_cg_recycled(mm, &rhs2, &mut slot, key(0, n), 24, &opts).unwrap();
        assert!(r2.hit);
        println!("cold {}  drifted-warm {}", s2.iterations, w2.iterations);
        assert!(
            w2.iterations < s2.iterations,
            "drifted warm start {} did not beat cold {}",
            w2.iterations,
            s2.iterations
        );
        assert!(maxabs(&x2w.sub(&x2c).unwrap()) < 1e-6, "recycling changed the answer");
    }

    /// The safety property. A subspace harvested from one operator, applied to
    /// a DIFFERENT one, must still give the right answer — an initial guess
    /// cannot be wrong, only unhelpful. This is what makes the key an
    /// optimisation rather than a correctness dependency.
    #[test]
    fn a_stale_subspace_costs_iterations_but_never_correctness() {
        let _g = settings::cg_ignore_min_iter::with(true);
        let n = 40;
        let a1 = spd(n, 21, 0.1);
        let a2 = spd(n, 77, 0.1); // completely unrelated operator
        let opts = CgOptions::for_tolerance(1e-10);
        let rhs = colvec(&lcg(n, 3));

        // Harvest from A₁ …
        let mut slot: Option<RecycleSubspace> = None;
        let _ = linear_cg_recycled(|v| a1.matmul(v), &rhs, &mut slot, key(0, n), 10, &opts).unwrap();
        assert!(slot.is_some());

        // … then reuse it on A₂ under the SAME key, so the guard does not save
        // us and the guess really is applied.
        let (x_stale, _, r) =
            linear_cg_recycled(|v| a2.matmul(v), &rhs, &mut slot, key(0, n), 10, &opts).unwrap();
        assert!(r.hit, "the stale subspace must actually have been used");

        let x_true = crate::operators::cg::linear_cg(
            |v: &Tensor| a2.matmul(v),
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap()
        .0;
        assert!(
            maxabs(&x_stale.sub(&x_true).unwrap()) < 1e-6,
            "a stale subspace corrupted the answer — it must only cost iterations"
        );
    }

    /// A generation bump is a miss, and a miss drops the subspace rather than
    /// using it. This is the discipline the per-step OT memo already follows:
    /// valid within a step, never across one, because the optimiser mutates
    /// the parameters underneath.
    #[test]
    fn a_generation_bump_invalidates() {
        let n = 30;
        let a = spd(n, 9, 0.1);
        let opts = CgOptions::for_tolerance(1e-8);
        let rhs = colvec(&lcg(n, 4));
        let mut slot = None;

        let (_, _, r0) =
            linear_cg_recycled(|v| a.matmul(v), &rhs, &mut slot, key(0, n), 8, &opts).unwrap();
        assert!(!r0.hit);
        let (_, _, r1) =
            linear_cg_recycled(|v| a.matmul(v), &rhs, &mut slot, key(0, n), 8, &opts).unwrap();
        assert!(r1.hit, "same key must hit");
        let (_, _, r2) =
            linear_cg_recycled(|v| a.matmul(v), &rhs, &mut slot, key(1, n), 8, &opts).unwrap();
        assert!(!r2.hit, "a generation bump must invalidate");
    }

    /// The Galerkin guess must be exact when the rhs already lies in the
    /// recycled span — the sharpest available statement that the projection
    /// (and the A-conjugacy that makes `UᵀAU` diagonal) is right.
    #[test]
    fn the_projection_is_exact_when_the_rhs_lies_in_the_span() {
        // Well-conditioned and small: CG must stop well SHORT of `n`.
        // Past `n` iterations the directions lose A-conjugacy to round-off and
        // `UᵀAU` stops being diagonal, which is a property of finite-precision
        // CG rather than of the projection — see the fixture note above.
        let n = 25;
        let a = from_spectrum(
            &(0..n).map(|i| 1.0 + 0.2 * (i as f64) / (n as f64)).collect::<Vec<f64>>(),
            13,
        );
        let opts = CgOptions::for_tolerance(1e-12);
        let rhs = colvec(&lcg(n, 6));

        let (x, st, dirs) = linear_cg_harvest(
            |v: &Tensor| a.matmul(v),
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();
        assert!(st.iterations < n, "fixture must converge short of n, got {}", st.iterations);
        // Keep EVERY direction: CG's span then contains the whole solution.
        let sub = RecycleSubspace::harvest(&dirs, key(0, n), dirs.p.len()).unwrap().unwrap();
        let guess = sub.initial_guess(&rhs).unwrap();
        assert!(
            maxabs(&guess.sub(&x).unwrap()) / maxabs(&x).max(1e-300) < 1e-6,
            "the full-span projection must reproduce the solution"
        );
        assert!(sub.guess_quality(&rhs).unwrap() < 1e-6, "and its residual must vanish");
    }

    /// `keep = 0` is a legitimate "recycling off", not a panic.
    #[test]
    fn keeping_nothing_is_a_clean_no_op() {
        let n = 20;
        let a = spd(n, 31, 0.3);
        let opts = CgOptions::for_tolerance(1e-8);
        let rhs = colvec(&lcg(n, 8));
        let mut slot = None;
        let (_, _, r) =
            linear_cg_recycled(|v| a.matmul(v), &rhs, &mut slot, key(0, n), 0, &opts).unwrap();
        assert!(!r.hit);
        assert_eq!(r.kept, 0);
        assert!(slot.is_none());
    }
}

#[cfg(test)]
mod deflation_tests {
    //! Verification of krylov-subspace recycling for related linear systems.

    use super::tests::*;
    use super::*;
    use crate::settings;

    /// **Warm start vs deflation, measured.**
    ///
    /// A Galerkin warm start moves the STARTING POINT: it saves
    /// `log(r₀/tol)` worth of iterations and cannot do more, which is why the
    /// measured gain is modest. Using the same subspace to DEFLATE changes the
    /// effective spectrum instead, so the two are not comparable in size and
    /// the point of this test is to show which mechanism is doing the work.
    #[test]
    fn deflation_beats_the_warm_start_on_the_same_subspace() {
        let _g = settings::cg_ignore_min_iter::with(true);
        let n = 60;
        // Enough distinct small modes that CG cannot finite-terminate, well
        // short of `n` so the harvested directions stay A-conjugate.
        let lams: Vec<f64> = (0..n)
            .map(|i| {
                if i < 14 {
                    1e-2 * 1.3f64.powi(i as i32)
                } else {
                    1.0 + 0.02 * (i % 5) as f64
                }
            })
            .collect();
        let a = from_spectrum(&lams, 4242);
        let opts = CgOptions::for_tolerance(1e-10);
        let rhs = colvec(&lcg(n, 17));
        let mm = |v: &Tensor| a.matmul(v);

        // Cold, to harvest.
        let (x_cold, s_cold, dirs) = linear_cg_harvest(
            mm,
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();
        let sub = RecycleSubspace::harvest(&dirs, key(0, n), 24).unwrap().unwrap();

        // (1) warm start
        let x0 = sub.initial_guess(&rhs).unwrap();
        let (x_warm, s_warm) = crate::operators::cg::linear_cg(
            mm,
            &rhs,
            Some(&x0),
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();

        // (2) deflation, same subspace
        let pre = sub.as_preconditioner(1e-10).unwrap().expect("a usable subspace");
        let o = CgOptions { precond: Some(std::sync::Arc::new(pre)), ..opts.clone() };
        let (x_def, s_def) = crate::operators::cg::linear_cg(
            mm,
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &o,
        )
        .unwrap();

        println!(
            "cold {}  warm-start {}  deflated {}",
            s_cold.iterations, s_warm.iterations, s_def.iterations
        );
        assert!(maxabs(&x_warm.sub(&x_cold).unwrap()) < 1e-6, "warm start moved the answer");
        assert!(maxabs(&x_def.sub(&x_cold).unwrap()) < 1e-6, "deflation moved the answer");
        assert!(
            s_def.iterations < s_cold.iterations,
            "deflation {} did not beat cold {}",
            s_def.iterations,
            s_cold.iterations
        );
        assert!(
            s_def.iterations < s_warm.iterations,
            "deflation {} did not beat the warm start {} — the whole point is that it \
             changes the spectrum rather than the starting point",
            s_def.iterations,
            s_warm.iterations
        );
    }

    /// The soundness claim that separates this from textbook deflation: the
    /// result is an ordinary SPD preconditioner, so a subspace harvested from a
    /// DIFFERENT operator still gives the right answer. It only costs
    /// iterations, exactly like the warm start.
    #[test]
    fn a_stale_deflation_subspace_is_still_sound() {
        let _g = settings::cg_ignore_min_iter::with(true);
        let n = 40;
        let a1 = spd(n, 21, 0.05);
        let a2 = spd(n, 77, 0.05);
        let opts = CgOptions::for_tolerance(1e-10);
        let rhs = colvec(&lcg(n, 5));

        let (_, _, dirs) = linear_cg_harvest(
            |v: &Tensor| a1.matmul(v),
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();
        let sub = RecycleSubspace::harvest(&dirs, key(0, n), 16).unwrap().unwrap();
        let Some(pre) = sub.as_preconditioner(1e-10).unwrap() else { return };

        let o = CgOptions { precond: Some(std::sync::Arc::new(pre)), ..opts.clone() };
        let (x_stale, _) = crate::operators::cg::linear_cg(
            |v: &Tensor| a2.matmul(v),
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &o,
        )
        .unwrap();
        let (x_true, _) = crate::operators::cg::linear_cg(
            |v: &Tensor| a2.matmul(v),
            &rhs,
            None,
            None::<fn(&Tensor) -> Result<Tensor>>,
            &opts,
        )
        .unwrap();
        assert!(
            maxabs(&x_stale.sub(&x_true).unwrap()) < 1e-6,
            "a deflation subspace from another operator corrupted the answer — it must \
             only ever cost iterations"
        );
    }
}
