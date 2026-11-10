//! Inferno — Minimum-residual iteration for symmetric indefinite linear systems.

use candle_core::{Result, Tensor, D};

use super::cg::{col_dot, col_norm, mean_scalar, safe_div, scalar_is_finite, CgOptions, ShiftedCgStats};
use crate::settings;

/// Solve `(A + σ_j I) x_j = rhs` for every `σ_j` in `shifts` via multi-shift
/// MINRES. `matmul` applies the UNSHIFTED `A`; `rhs` is `[..., N]` or
/// `[..., N, C]`.
///
/// Shifts must be finite; unlike [`super::cg::linear_cg_shifted`] they may
/// be NEGATIVE, provided each `A + σ_j I` is nonsingular — an indefinite
/// shifted system is solved, not refused.
///
/// `opts.tolerance = None` resolves to [`settings::minres_tolerance`] (not
/// `cg_tolerance` — this is the knob's intended consumer). The tolerance is
/// relative: the RHS is unit-normalised per column, same contract as
/// `linear_cg`.
pub fn linear_minres_shifted<F>(
    matmul: F,
    rhs: &Tensor,
    shifts: &[f64],
    opts: &CgOptions,
) -> Result<(Vec<Tensor>, ShiftedCgStats)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if opts.precond.is_some() {
        return Err(candle_core::Error::Msg(
            "linear_minres_shifted: a preconditioner was supplied and this solver cannot honour it. \
             No nontrivial M keeps {{M⁻¹(A + σI)}} a shifted family, so one Krylov space \
             cannot serve every shift under preconditioning — the multi-shift identity \
             the solver is built on would silently stop holding. Conjugate the operator \
             instead (`K̂ = E⁻¹KE⁻¹`, see `precond.rs`), or drop `opts.precond`."
                .to_string(),
        ));
    }
    if shifts.is_empty() {
        return Err(candle_core::Error::Msg(
            "linear_minres_shifted: needs at least one shift".into(),
        ));
    }
    if shifts.iter().any(|s| !s.is_finite()) {
        return Err(candle_core::Error::Msg(format!(
            "linear_minres_shifted: shifts must be finite, got {shifts:?}"
        )));
    }

    let is_vector = rhs.rank() == 1;
    let rhs = if is_vector { rhs.unsqueeze(D::Minus1)? } else { rhs.clone() };

    let n_rows = rhs.dim(D::Minus2)?;
    let max_iter = opts.max_iter.unwrap_or_else(settings::max_cg_iterations::value);
    let tolerance = opts.tolerance.unwrap_or_else(settings::minres_tolerance::value);
    let n_iter = if settings::terminate_cg_by_size::value() {
        max_iter.min(n_rows)
    } else {
        max_iter
    };

    // -- normalise the RHS (same contract as cg::run) --------------------------
    let rhs_norm = col_norm(&rhs)?;
    let eps_t = Tensor::full(opts.eps, rhs_norm.shape(), rhs_norm.device())?
        .to_dtype(rhs_norm.dtype())?;
    let rhs_is_zero = rhs_norm.lt(&eps_t)?;
    let rhs_norm = rhs_is_zero.where_cond(&rhs_norm.ones_like()?, &rhs_norm)?;
    let rhs_n = rhs.broadcast_div(&rhs_norm)?;
    if !scalar_is_finite(&rhs_n)? {
        return Err(candle_core::Error::Msg(
            "linear_minres_shifted: non-finite right-hand side".into(),
        ));
    }

    // -- shared Lanczos state ---------------------------------------------------
    // beta1 = 1 per column after normalisation (0 for pinned zero columns),
    // but carry it as a tensor rather than assuming: the zero-column pin
    // relies on phibar starting at exactly 0 there.
    let beta1 = col_norm(&rhs_n)?;
    let mut v = safe_div(&rhs_n.clone(), &beta1.broadcast_as(rhs_n.shape())?, opts.eps)?;
    let mut v_prev = rhs_n.zeros_like()?;
    let mut beta = beta1.clone();

    // -- per-shift Paige–Saunders state ------------------------------------------
    // Coefficients are per-column [..., 1, C]; direction vectors [..., N, C].
    struct ShiftState {
        cs: Tensor,
        sn: Tensor,
        dbar: Tensor,
        epsln: Tensor,
        phibar: Tensor,
        w_old: Tensor,
        w_older: Tensor,
        x: Tensor,
    }
    let ones_c = beta1.ones_like()?;
    let zeros_c = beta1.zeros_like()?;
    let mut states: Vec<ShiftState> = shifts
        .iter()
        .map(|_| {
            Ok(ShiftState {
                cs: ones_c.affine(-1.0, 0.0)?, // cs = −1, sn = 0: identity rotation
                sn: zeros_c.clone(),
                dbar: zeros_c.clone(),
                epsln: zeros_c.clone(),
                phibar: beta1.clone(),
                w_old: rhs_n.zeros_like()?,
                w_older: rhs_n.zeros_like()?,
                x: rhs_n.zeros_like()?,
            })
        })
        .collect::<Result<_>>()?;

    let stop_t = Tensor::full(opts.stop_updating_after, beta1.shape(), beta1.device())?
        .to_dtype(beta1.dtype())?;

    let mut tolerance_reached = false;
    let mut k_done = 0usize;

    for k in 0..n_iter {
        k_done = k + 1;

        // ---- one shared Lanczos step -------------------------------------------
        let p = matmul(&v)?;
        let alpha = col_dot(&v, &p)?;
        let r = ((&p - v.broadcast_mul(&alpha)?)? - v_prev.broadcast_mul(&beta)?)?;
        let beta_next = col_norm(&r)?;
        if !scalar_is_finite(&beta_next)? {
            return Err(candle_core::Error::Msg(
                "linear_minres_shifted: non-finite Lanczos step — broken operator".into(),
            ));
        }

        // ---- per-shift rotation + solution update -------------------------------
        let mut worst = 0.0f64;
        for (st, &sigma) in states.iter_mut().zip(shifts) {
            let alfa = alpha.affine(1.0, sigma)?; // α_k + σ

            // Freeze decision from the PRE-update residual, like cg::run's
            // has_converged (computed at the end of the previous iteration).
            // Testing the post-update φ̄ instead zeroes the very step that
            // achieves it — observed as a solve whose φ̄ reads 1e-16 while
            // the true residual is stuck at the previous iteration's level.
            let frozen = st.phibar.lt(&stop_t)?;

            let oldeps = st.epsln.clone();
            let delta = ((&st.cs * &st.dbar)? + (&st.sn * &alfa)?)?;
            let gbar = ((&st.sn * &st.dbar)? - (&st.cs * &alfa)?)?;
            st.epsln = (&st.sn * &beta_next)?;
            st.dbar = (&st.cs * &beta_next)?.affine(-1.0, 0.0)?;

            let gamma = ((gbar.sqr()? + beta_next.sqr()?)?).sqrt()?;
            st.cs = safe_div(&gbar, &gamma, opts.eps)?;
            st.sn = safe_div(&beta_next, &gamma, opts.eps)?;

            let mut phi = (&st.cs * &st.phibar)?;
            st.phibar = (&st.sn * &st.phibar)?;

            // A zero phi stops x from moving — the msMINRES analogue of the
            // zeroed alpha in cg::run.
            phi = frozen.where_cond(&phi.zeros_like()?, &phi)?;

            // w_new = (v − oldeps·w_older − delta·w_old) / γ
            let w_new = safe_div(
                &((&v - st.w_older.broadcast_mul(&oldeps)?)? - st.w_old.broadcast_mul(&delta)?)?,
                &gamma.broadcast_as(v.shape())?,
                opts.eps,
            )?;
            st.x = (&st.x + w_new.broadcast_mul(&phi)?)?;
            st.w_older = std::mem::replace(&mut st.w_old, w_new);

            worst = worst.max(mean_scalar(&st.phibar)?);
        }

        // ---- advance Lanczos ----------------------------------------------------
        let v_next = safe_div(&r, &beta_next.broadcast_as(r.shape())?, opts.eps)?;
        v_prev = std::mem::replace(&mut v, v_next);
        beta = beta_next;

        // Same fluke-floor as cg::run; MINRES's φ̄ is monotone, so once the
        // worst shift is under tolerance every shift is done for good.
        if k >= 10.min(max_iter.saturating_sub(1)) && worst < tolerance {
            tolerance_reached = true;
            break;
        }
    }

    // -- un-normalise, report ----------------------------------------------------
    let mut solutions = Vec::with_capacity(shifts.len());
    let mut norms = Vec::with_capacity(shifts.len());
    for st in &states {
        let x = st.x.broadcast_mul(&rhs_norm)?;
        solutions.push(if is_vector { x.squeeze(D::Minus1)? } else { x });
        norms.push(mean_scalar(&st.phibar)?);
    }

    let converged = tolerance_reached || n_iter == 0;
    if !converged && settings::verbose_linalg::value() {
        tracing::warn!(
            iterations = k_done,
            residual_norm = norms.iter().cloned().fold(0.0, f64::max),
            tolerance,
            "linear_minres_shifted hit its iteration budget without reaching tolerance"
        );
    }
    let stats = ShiftedCgStats { iterations: k_done, residual_norms: norms, converged };
    Ok((solutions, stats))
}

#[cfg(test)]
mod tests {
    //! Verification of minimum-residual iteration for symmetric indefinite linear systems.

    use super::*;
    use crate::operators::cg::{linear_cg_shifted, CgOptions};
    use candle_core::Device;

    /// Dense SPD `A = BᵀB + shift·I` in F64 (cg.rs's construction).
    fn spd(n: usize, shift: f64, seed: u64) -> Tensor {
        let dev = Device::Cpu;
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let b: Vec<f64> = (0..n * n).map(|_| next()).collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += b[k * n + i] * b[k * n + j];
                }
                a[i * n + j] = acc + if i == j { shift } else { 0.0 };
            }
        }
        Tensor::from_vec(a, (n, n), &dev).unwrap()
    }

    fn max_err(x: &Tensor, y: &Tensor) -> f64 {
        let a: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f64> = y.flatten_all().unwrap().to_vec1().unwrap();
        a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max)
    }

    const SHIFTS: [f64; 5] = [1e-4, 1e-2, 1.0, 10.0, 1e3];

    #[test]
    fn msminres_agrees_with_mscg_on_spd_systems() {
        let dev = Device::Cpu;
        let a = spd(24, 1.0, 7);
        let b = Tensor::from_vec(
            (0..24).map(|i| ((i * 37 % 11) as f64) - 5.0).collect::<Vec<_>>(),
            (24, 1),
            &dev,
        )
        .unwrap();
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xm, sm) = linear_minres_shifted(|x| a.matmul(x), &b, &SHIFTS, &opts).unwrap();
        let (xc, _) = linear_cg_shifted(|x| a.matmul(x), &b, &SHIFTS, &opts).unwrap();
        assert!(sm.converged, "{sm:?}");
        for ((m, c), s) in xm.iter().zip(&xc).zip(&SHIFTS) {
            let err = max_err(m, c);
            assert!(err < 1e-6, "shift {s}: msMINRES vs msCG differ by {err:.3e}");
        }
    }

    #[test]
    fn msminres_matches_a_dense_solve() {
        let dev = Device::Cpu;
        let a = spd(16, 0.5, 3);
        let b = Tensor::from_vec(
            (0..16).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>(),
            (16, 1),
            &dev,
        )
        .unwrap();
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xs, _) = linear_minres_shifted(|x| a.matmul(x), &b, &SHIFTS, &opts).unwrap();
        let eye = Tensor::eye(16, candle_core::DType::F64, &dev).unwrap();
        for (x, &s) in xs.iter().zip(&SHIFTS) {
            let a_s = (&a + eye.affine(s, 0.0).unwrap()).unwrap();
            let want = crate::linalg::cholesky_solve(&a_s, &b, 0.0).unwrap();
            let err = max_err(x, &want);
            assert!(err < 1e-6, "shift {s}: msMINRES vs dense differ by {err:.3e}");
        }
    }

    /// The capability msCG structurally lacks: an INDEFINITE shifted system
    /// (diagonal −1 entry survives the shift). Exact solutions are
    /// elementwise, so the oracle needs no factorisation.
    #[test]
    fn msminres_solves_an_indefinite_system_that_mscg_cannot() {
        let dev = Device::Cpu;
        let d: Vec<f64> = vec![-1.0, 0.5, 0.9, 1.4, 2.0, 2.7, 3.5, 4.4];
        let n = d.len();
        let diag = Tensor::from_vec(d.clone(), (n, 1), &dev).unwrap();
        let dmat = |x: &Tensor| -> Result<Tensor> { x.broadcast_mul(&diag) };
        let b = Tensor::from_vec((0..n).map(|i| 1.0 + i as f64).collect::<Vec<_>>(), (n, 1), &dev)
            .unwrap();
        // −1 + 0.3 = −0.7: the shifted system is indefinite but nonsingular.
        let shifts = [0.3, 2.0];
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_minres_shifted(dmat, &b, &shifts, &opts).unwrap();
        assert!(stats.converged, "{stats:?}");
        for (x, &s) in xs.iter().zip(&shifts) {
            let got: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
            for (i, (gi, di)) in got.iter().zip(&d).enumerate() {
                let want = (1.0 + i as f64) / (di + s);
                assert!(
                    (gi - want).abs() < 1e-6 * want.abs().max(1.0),
                    "shift {s} row {i}: {gi} vs {want}"
                );
            }
        }
        // And the honesty cross-check: msCG on the same indefinite problem
        // must NOT reach this accuracy (its safe_div zeroes the negative-
        // curvature steps). If this ever starts passing, the doc claims
        // above are stale.
        let (xc, _) = linear_cg_shifted(dmat, &b, &shifts, &opts).unwrap();
        let cg_err: f64 = xc
            .iter()
            .zip(&shifts)
            .map(|(x, &s)| {
                let got: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
                got.iter()
                    .zip(&d)
                    .enumerate()
                    .map(|(i, (gi, di))| (gi - (1.0 + i as f64) / (di + s)).abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        assert!(cg_err > 1e-3, "msCG unexpectedly solved the indefinite system (err {cg_err:.3e})");
    }

    #[test]
    fn msminres_solves_batched_multi_rhs() {
        let dev = Device::Cpu;
        let blocks: Vec<Tensor> =
            (0..3).map(|s| spd(16, 0.5 + s as f64, 20 + s as u64)).collect();
        let a = Tensor::stack(&blocks, 0).unwrap();
        let b = Tensor::from_vec(
            (0..3 * 16 * 2).map(|i| ((i * 29 % 17) as f64) / 7.0 - 1.0).collect::<Vec<_>>(),
            (3, 16, 2),
            &dev,
        )
        .unwrap();
        let shifts = [1e-2, 1.0, 50.0];
        let opts = CgOptions { max_iter: Some(400), ..CgOptions::for_tolerance(1e-12) };
        let (xm, stats) =
            linear_minres_shifted(|x| a.broadcast_matmul(x), &b, &shifts, &opts).unwrap();
        assert!(stats.converged);
        let (xc, _) =
            linear_cg_shifted(|x| a.broadcast_matmul(x), &b, &shifts, &opts).unwrap();
        for ((m, c), s) in xm.iter().zip(&xc).zip(&shifts) {
            assert_eq!(m.dims(), &[3, 16, 2]);
            let err = max_err(m, c);
            assert!(err < 1e-6, "shift {s}: batched msMINRES vs msCG differ by {err:.3e}");
        }
    }

    #[test]
    fn msminres_reports_nonconvergence_honestly() {
        let dev = Device::Cpu;
        let n = 32;
        let d: Vec<f64> =
            (0..n).map(|i| 1e-6f64.powf(1.0 - i as f64 / (n - 1) as f64)).collect();
        let diag = Tensor::from_vec(d, (n, 1), &dev).unwrap();
        let dmat = |x: &Tensor| -> Result<Tensor> { x.broadcast_mul(&diag) };
        let b = Tensor::from_vec(vec![1.0f64; n], (n, 1), &dev).unwrap();
        let opts = CgOptions { max_iter: Some(5), ..CgOptions::for_tolerance(1e-12) };
        let (xs, stats) = linear_minres_shifted(dmat, &b, &[1e-8, 1e-2], &opts).unwrap();
        assert!(!stats.converged);
        for x in &xs {
            let v: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
            assert!(v.iter().all(|q| q.is_finite()), "nonconverged solve leaked non-finites");
        }
    }

    #[test]
    fn msminres_refuses_empty_or_nonfinite_shifts() {
        let dev = Device::Cpu;
        let a = spd(8, 1.0, 5);
        let b = Tensor::from_vec(vec![1.0f64; 8], (8, 1), &dev).unwrap();
        assert!(linear_minres_shifted(|x| a.matmul(x), &b, &[], &CgOptions::default()).is_err());
        assert!(
            linear_minres_shifted(|x| a.matmul(x), &b, &[1.0, f64::NAN], &CgOptions::default())
                .is_err()
        );
    }
}
