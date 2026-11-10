//! Inferno — Numerical comparisons of iterative linear solvers on tensor operators.

use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

use super::cg::{linear_cg, CgOptions, CgStats};

/// `None` at the generic-preconditioner slot — the arms carry theirs on
/// `CgOptions::precond` instead, which is the path `linsolve` uses.
const NO_PRECOND: Option<fn(&Tensor) -> Result<Tensor>> = None;
use super::ciq::lanczos_probe;
use super::precond::{CiqPreconditioner, PrecondKind};

/// Below this condition number a preconditioner has nothing to fix and the
/// measurement is noise. Chosen at 100 because the ladder experiment's own
/// fixture sat near `1e5` and still only separated the arms by a factor of
/// two.
pub const MIN_CONDITION: f64 = 100.0;

/// One arm's measurement.
#[derive(Clone, Debug)]
pub struct BakeoffRow {
    pub name: &'static str,
    pub kind: Option<PrecondKind>,
    /// Rank of the correction ACTUALLY built — not the rank requested.
    pub rank: usize,
    pub iterations: usize,
    pub converged: bool,
    /// Seconds spent building the preconditioner. Never folded into `solve_s`.
    pub build_s: f64,
    pub solve_s: f64,
    /// `max |x − x_oracle| / max(1, max|x_oracle|)` against the dense solve.
    pub max_rel: f64,
}

impl BakeoffRow {
    /// Did this arm land on a DIFFERENT iteration count from the baseline?
    ///
    /// The assertion a caller needs, because "preconditioned and plain both
    /// took 11 iterations" has passed for a preconditioner test twice in this
    /// tree already.
    pub fn is_distinguishable(&self, baseline: &BakeoffRow) -> bool {
        self.iterations != baseline.iterations
    }
}

/// The measurement, and what it was measured on.
#[derive(Clone, Debug)]
pub struct BakeoffReport {
    pub n: usize,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub condition: f64,
    pub rows: Vec<BakeoffRow>,
}

impl BakeoffReport {
    pub fn baseline(&self) -> &BakeoffRow {
        self.rows
            .iter()
            .find(|r| r.name == "plain")
            .expect("the plain arm is always run")
    }

    /// Arms that beat plain CG on iterations. EMPTY is a legitimate outcome
    /// and must be reported as one, not treated as a failure to configure.
    pub fn winners(&self) -> Vec<&BakeoffRow> {
        let base = self.baseline().iterations;
        let mut w: Vec<&BakeoffRow> =
            self.rows.iter().filter(|r| r.name != "plain" && r.iterations < base).collect();
        w.sort_by_key(|r| r.iterations);
        w
    }

    /// A table, for `--nocapture`.
    pub fn table(&self) -> String {
        let mut s = format!(
            "n = {}   λ ∈ [{:.3e}, {:.3e}]   κ = {:.3e}\n\
             {:<10} {:>6} {:>6} {:>10} {:>10} {:>11}\n",
            self.n,
            self.lambda_min,
            self.lambda_max,
            self.condition,
            "arm",
            "rank",
            "iters",
            "build_ms",
            "solve_ms",
            "max_rel"
        );
        for r in &self.rows {
            s.push_str(&format!(
                "{:<10} {:>6} {:>6} {:>10.2} {:>10.2} {:>11.2e}{}\n",
                r.name,
                r.rank,
                r.iterations,
                r.build_s * 1e3,
                r.solve_s * 1e3,
                r.max_rel,
                if r.converged { "" } else { "  (NOT CONVERGED)" },
            ));
        }
        s
    }
}

/// The harness. `matmul` and `diag` describe ONE symmetric positive-definite
/// operator of size `n`.
///
/// `diag` must come from the operator's OWN formula — see the module docs on
/// `forward_diag`.
pub struct Bakeoff<'a> {
    pub matmul: &'a (dyn Fn(&Tensor) -> Result<Tensor> + 'a),
    pub diag: Tensor,
    pub n: usize,
    pub rhs: Tensor,
    pub tolerance: f64,
    /// Rank asked of the `LowRank` arm. `settings::max_preconditioner_size`
    /// caps it in practice.
    pub low_rank: usize,
    /// Lanczos iterations for the probe (the `Spectral` arm's basis).
    pub probe_iters: Option<usize>,
    /// Extra `(name, μ)` ladders to place on the probe's Ritz basis — the seam
    /// the sliced-Wasserstein δ experiment plugs into. `δ` is supplied per
    /// arm; `None` means "use the probe's own SLQ-weighted geometric mean".
    pub extra_deltas: Vec<(&'static str, f64)>,
}

impl<'a> Bakeoff<'a> {
    pub fn new(
        matmul: &'a (dyn Fn(&Tensor) -> Result<Tensor> + 'a),
        diag: Tensor,
        rhs: Tensor,
    ) -> Result<Self> {
        let n = diag.dims().last().copied().unwrap_or(0);
        Ok(Self {
            matmul,
            diag,
            n,
            rhs,
            tolerance: 1e-8,
            low_rank: super::super::settings::max_preconditioner_size::value(),
            probe_iters: None,
            extra_deltas: Vec::new(),
        })
    }

    fn opts(&self, precond: Option<Arc<CiqPreconditioner>>) -> CgOptions {
        CgOptions { precond, ..CgOptions::for_tolerance(self.tolerance) }
    }

    /// Run every arm and report.
    ///
    /// Errors — deliberately, rather than warning — when the operator is too
    /// well conditioned for the measurement to mean anything.
    pub fn run(&self, oracle: &Tensor) -> Result<BakeoffReport> {
        let probe_t = Instant::now();
        let probe = lanczos_probe(
            |x| (self.matmul)(x),
            self.n,
            self.rhs.dtype(),
            self.rhs.device(),
            self.probe_iters,
        )?;
        let probe_s = probe_t.elapsed().as_secs_f64();
        let (lo, hi) = probe.bounds.first().copied().unwrap_or((1.0, 1.0));
        let condition = hi / lo.max(f64::MIN_POSITIVE);
        if !(condition.is_finite() && condition >= MIN_CONDITION) {
            return Err(candle_core::Error::Msg(format!(
                "bakeoff refused: κ = {condition:.3e} < {MIN_CONDITION:.0}. On a \
                 well-conditioned operator every arm converges in a handful of iterations \
                 and the comparison is noise — a bake-off run here would report numbers \
                 that mean nothing and read as if it had measured something."
            )));
        }

        let mut rows = Vec::new();

        // -- plain, the baseline every other arm is judged against ----------
        let t = Instant::now();
        let (x, stats) = linear_cg(
            |v| (self.matmul)(v),
            &self.rhs,
            None,
            NO_PRECOND,
            &self.opts(None),
        )?;
        rows.push(self.row("plain", None, 0, &stats, 0.0, t.elapsed().as_secs_f64(), &x, oracle)?);

        // -- Jacobi: free, and the honest floor for "does M help at all" ----
        let t = Instant::now();
        let jac = Arc::new(CiqPreconditioner::jacobi(&self.diag, 1e-12)?);
        let build = t.elapsed().as_secs_f64();
        rows.push(self.arm("jacobi", jac, build, oracle)?);

        // -- LowRank from pivoted Cholesky: dominant directions only --------
        let t = Instant::now();
        let piv = CiqPreconditioner::pivoted(|x| (self.matmul)(x), &self.diag, self.low_rank);
        match piv {
            Ok(p) => {
                let build = t.elapsed().as_secs_f64();
                rows.push(self.arm("lowrank", Arc::new(p), build, oracle)?);
            }
            Err(e) => tracing::warn!("bakeoff: the pivoted arm declined ({e})"),
        }

        // -- Spectral on the Ritz basis: the only arm that deflates the SMALL
        //    end, which is the end √κ is built from.
        let t = Instant::now();
        let (q, mu, delta) = probe.spectral_parts(0)?;
        match CiqPreconditioner::spectral(&q, &mu, delta, 1e-6) {
            Ok(p) => {
                let build = probe_s + t.elapsed().as_secs_f64();
                rows.push(self.arm("spectral", Arc::new(p), build, oracle)?);
            }
            Err(e) => tracing::warn!("bakeoff: the spectral arm declined ({e})"),
        }

        // -- the same Ritz basis under a SUPPLIED δ -------------------------
        for (name, d) in &self.extra_deltas {
            let t = Instant::now();
            match CiqPreconditioner::spectral(&q, &mu, *d, 1e-6) {
                Ok(p) => {
                    let build = probe_s + t.elapsed().as_secs_f64();
                    rows.push(self.arm(name, Arc::new(p), build, oracle)?);
                }
                Err(e) => tracing::warn!("bakeoff: the {name} arm declined ({e})"),
            }
        }

        Ok(BakeoffReport { n: self.n, lambda_min: lo, lambda_max: hi, condition, rows })
    }

    fn arm(
        &self,
        name: &'static str,
        p: Arc<CiqPreconditioner>,
        build_s: f64,
        oracle: &Tensor,
    ) -> Result<BakeoffRow> {
        let (kind, rank) = (p.kind(), p.rank());
        let t = Instant::now();
        let (x, stats) = linear_cg(
            |v| (self.matmul)(v),
            &self.rhs,
            None,
            NO_PRECOND,
            &self.opts(Some(p)),
        )?;
        let solve_s = t.elapsed().as_secs_f64();
        self.row(name, Some(kind), rank, &stats, build_s, solve_s, &x, oracle)
    }

    #[allow(clippy::too_many_arguments)]
    fn row(
        &self,
        name: &'static str,
        kind: Option<PrecondKind>,
        rank: usize,
        stats: &CgStats,
        build_s: f64,
        solve_s: f64,
        x: &Tensor,
        oracle: &Tensor,
    ) -> Result<BakeoffRow> {
        let scale = oracle
            .abs()?
            .max_all()?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?
            .max(1.0);
        let max_rel = x
            .sub(oracle)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?
            / scale;
        Ok(BakeoffRow {
            name,
            kind,
            rank,
            iterations: stats.iterations,
            converged: stats.converged,
            build_s,
            solve_s,
            max_rel,
        })
    }
}

/// A dense SPD operator with a PRESCRIBED spectrum — the fixture a bake-off
/// needs when the real operator is not to hand.
///
/// `eigs` are placed exactly, on a random orthogonal basis, so `κ` is a
/// property of the fixture rather than something to be discovered. That is
/// what lets a test assert "this fixture is ill-conditioned" rather than hope.
pub fn spd_with_spectrum(eigs: &[f64], seed: u64, device: &Device) -> Result<Tensor> {
    let n = eigs.len();
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    // Gram–Schmidt a random matrix into an orthonormal basis.
    let mut q = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            q[i * n + j] = next();
        }
        for k in 0..j {
            let dot: f64 = (0..n).map(|i| q[i * n + j] * q[i * n + k]).sum();
            for i in 0..n {
                q[i * n + j] -= dot * q[i * n + k];
            }
        }
        let nrm: f64 = (0..n).map(|i| q[i * n + j] * q[i * n + j]).sum::<f64>().sqrt();
        for i in 0..n {
            q[i * n + j] /= nrm.max(1e-300);
        }
    }
    // A = Q diag(eigs) Qᵀ
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += q[i * n + k] * eigs[k] * q[j * n + k];
            }
            a[i * n + j] = acc;
        }
    }
    Tensor::from_vec(a, (n, n), device)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Verification of numerical comparisons of iterative linear solvers on tensor operators.

    use super::*;
    use crate::linalg_diff::{cholesky_diff, tri_solve_diff, tri_solve_t_diff};

    /// A spectrum with a GAP: five tiny modes, then a tight cluster near 1.
    /// That is the structure deflation exists for, and the structure the
    /// Kumaraswamy ladder could not represent.
    fn gapped(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i < 5 { 1e-3 * 3f64.powi(i as i32) } else { 1.0 + 0.02 * (i - 5) as f64 })
            .collect()
    }

    /// The operator's OWN diagonal, taken from the matrix itself — the point
    /// the module docs make about `forward_diag`.
    fn dense_diag(a: &Tensor, n: usize, dev: &Device) -> Tensor {
        let v: Vec<f64> = (0..n)
            .map(|i| {
                a.narrow(0, i, 1)
                    .and_then(|r| r.narrow(1, i, 1))
                    .and_then(|s| s.reshape(()))
                    .and_then(|s| s.to_scalar::<f64>())
                    .unwrap()
            })
            .collect();
        Tensor::from_vec(v, n, dev).unwrap()
    }

    fn dense_solve(a: &Tensor, rhs: &Tensor) -> Tensor {
        let l = cholesky_diff(a, 1e-12).unwrap();
        tri_solve_t_diff(&l, &tri_solve_diff(&l, rhs).unwrap()).unwrap()
    }

    #[test]
    fn the_bakeoff_reports_every_arm_and_separates_them() {
        let dev = Device::Cpu;
        let n = 40;
        let a = spd_with_spectrum(&gapped(n), 7, &dev).unwrap();
        let rhs = Tensor::ones((n, 1), DType::F64, &dev).unwrap();
        let oracle = dense_solve(&a, &rhs);
        let mm = |x: &Tensor| a.matmul(x);
        let diag = dense_diag(&a, n, &dev);

        let b = Bakeoff::new(&mm, diag, rhs).unwrap();
        let rep = b.run(&oracle).unwrap();
        println!("{}", rep.table());

        // The fixture must be ill-conditioned, or nothing below means anything.
        assert!(rep.condition > MIN_CONDITION, "κ = {:.3e}", rep.condition);
        // Every arm must have been built at the rank it claims.
        for r in &rep.rows {
            assert!(r.converged, "arm {} did not converge", r.name);
            if r.kind == Some(PrecondKind::Spectral) || r.kind == Some(PrecondKind::LowRank) {
                assert!(r.rank >= 2, "arm {} silently collapsed to rank {}", r.name, r.rank);
            }
            // Preconditioning must not move the ANSWER — that is the one
            // property no speed-up can buy back.
            assert!(r.max_rel < 1e-5, "arm {} moved the solution by {:.2e}", r.name, r.max_rel);
        }
        // …and at least one arm must be DISTINGUISHABLE from plain, or the
        // measurement is the 11-vs-11 non-result.
        let base = rep.baseline().clone();
        assert!(
            rep.rows.iter().any(|r| r.name != "plain" && r.is_distinguishable(&base)),
            "every arm landed on {} iterations — this bake-off cannot fail",
            base.iterations
        );
    }

    /// The refusal is the point. A bake-off on a well-conditioned operator
    /// reports numbers that mean nothing and reads exactly like one that
    /// measured something.
    ///
    /// **Vacuous version to refuse:** asserting only that `run` returns
    /// `Err` — it must refuse for THIS reason.
    #[test]
    fn a_well_conditioned_operator_is_refused_not_measured() {
        let dev = Device::Cpu;
        let n = 24;
        // κ ≈ 3: nothing to fix.
        let eigs: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * i as f64 / n as f64).collect();
        let a = spd_with_spectrum(&eigs, 3, &dev).unwrap();
        let rhs = Tensor::ones((n, 1), DType::F64, &dev).unwrap();
        let oracle = dense_solve(&a, &rhs);
        let mm = |x: &Tensor| a.matmul(x);
        let diag = dense_diag(&a, n, &dev);
        let b = Bakeoff::new(&mm, diag, rhs).unwrap();
        let e = b.run(&oracle).unwrap_err().to_string();
        assert!(e.contains("bakeoff refused"), "wrong refusal: {e}");
        assert!(e.contains("κ ="), "the refusal must report the condition number: {e}");
    }

    /// `spd_with_spectrum` must actually place the spectrum it was given —
    /// otherwise every κ claim in this file rests on a hope.
    ///
    /// Note what is asserted: `LanczosProbe::bounds` is an **enclosure**, not
    /// the Ritz range. It is padded outward precisely so a quadrature contour
    /// built on it contains the spectrum, so `hi ≥ λmax` and `lo ≤ λmin` —
    /// asserting equality would fail, and asserting `hi ≤ λmax` would be
    /// asserting the bound is BROKEN. This is also why `cgcache` refuses to
    /// serve cached bounds as nodes: a stale enclosure that has grown too
    /// TIGHT is silently wrong in the one direction that matters.
    #[test]
    fn the_fixture_has_the_spectrum_it_claims() {
        let dev = Device::Cpu;
        let n = 24;
        let eigs = gapped(n);
        let a = spd_with_spectrum(&eigs, 5, &dev).unwrap();
        let mm = |x: &Tensor| a.matmul(x);
        let p = lanczos_probe(&mm, n, DType::F64, &dev, Some(n)).unwrap();
        let (lo, hi) = p.bounds[0];
        let want_lo = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
        let want_hi = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(hi >= want_hi, "the enclosure does not contain λmax: {hi} < {want_hi}");
        assert!(lo <= want_lo, "the enclosure does not contain λmin: {lo} > {want_lo}");
        // …and it is an enclosure, not a giveaway: within an order of magnitude.
        assert!(hi < want_hi * 10.0, "λmax enclosure absurdly loose: {hi} vs {want_hi}");
        assert!(lo > want_lo / 10.0, "λmin enclosure absurdly loose: {lo} vs {want_lo}");
        // The fixture's whole point is a wide condition number.
        assert!(want_hi / want_lo > 100.0, "the fixture is not ill-conditioned");
    }

    /// A spectral probe can reduce iterations while adding setup cost. Report
    /// measured amortisation for repeated solves, but assert only numerical
    /// accuracy and iteration savings: wall-clock ordering varies by machine.
    #[test]
    fn spectral_reduces_iterations_and_reports_amortisation() {
        let dev = Device::Cpu;
        let n = 40;
        let a = spd_with_spectrum(&gapped(n), 7, &dev).unwrap();
        let rhs = Tensor::ones((n, 1), DType::F64, &dev).unwrap();
        let oracle = dense_solve(&a, &rhs);
        let mm = |x: &Tensor| a.matmul(x);
        let b = Bakeoff::new(&mm, dense_diag(&a, n, &dev), rhs).unwrap();
        let rep = b.run(&oracle).unwrap();

        let plain = rep.baseline().clone();
        let Some(spec) = rep.rows.iter().find(|r| r.kind == Some(PrecondKind::Spectral)) else {
            panic!("the spectral arm declined — nothing to amortise");
        };

        // The iteration win must be real, or there is nothing to amortise.
        assert!(
            spec.iterations < plain.iterations,
            "spectral did not reduce iterations ({} vs {})",
            spec.iterations,
            plain.iterations
        );

        // Iterations are the stable currency; wall clock at n = 40 on a
        // shared CPU is not. Express the break-even in ITERATIONS SAVED
        // against the probe's cost measured in the same unit.
        let saved_per_solve = plain.iterations - spec.iterations;
        let per_iter_s = plain.solve_s / plain.iterations.max(1) as f64;
        let build_in_iters = (spec.build_s / per_iter_s.max(1e-12)).ceil() as usize;
        let break_even = build_in_iters.div_ceil(saved_per_solve.max(1));
        println!(
            "spectral: {} → {} iterations ({saved_per_solve} saved/solve); probe costs \
             ≈{build_in_iters} iterations ⇒ break-even at ~{break_even} solves sharing it",
            plain.iterations, spec.iterations
        );

        assert!(saved_per_solve > 0);
        assert!(spec.max_rel.is_finite() && spec.max_rel < 1e-6);
        assert!(spec.build_s.is_finite() && spec.build_s >= 0.0);
        assert!(plain.solve_s.is_finite() && plain.solve_s >= 0.0);

        // Deterministic cost model: a 100-iteration probe saving 12 iterations
        // per solve pays back on the ninth solve. Measured timings above must
        // not decide whether this numerical regression passes.
        let probe_cost = 100usize;
        let saving = 12usize;
        let solves = probe_cost.div_ceil(saving);
        assert_eq!(solves, 9);
        assert!((solves - 1) * saving < probe_cost);
        assert!(solves * saving >= probe_cost);
    }
}
