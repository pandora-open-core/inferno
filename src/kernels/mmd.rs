//! Inferno — Differentiable maximum mean discrepancy with RBF and inverse-multiquadric kernels.
//!
//! Inputs are sample matrices `[n, features]` with matching floating dtype and
//! device. Gradients reach sample values; bandwidths are fixed scalar values.

use super::sq_pdist;
use candle_core::{Result as CResult, Tensor};

#[derive(Copy, Clone, Debug)]
pub enum KernelChoice {
    /// k(x, y) = exp(-||x-y||² / (2 σ²))
    Rbf,
    /// k(x, y) = (c² + ||x-y||²)^(-1/2). σ here plays the role of c.
    Imq,
}

/// Silverman-style median-heuristic bandwidth. Falls back to `1.0` for
/// degenerate (constant) batches.
///
/// `σ = median(‖xᵢ − xⱼ‖) / √2` over the STRICTLY off-diagonal pairs. Both
/// details matter and both were previously wrong: the mean of squared
/// distances is dragged by outliers (the whole point of the median
/// heuristic is that it isn't), and the `i == j` entries are identically
/// zero, so including them pulls the bandwidth toward zero at a rate that
/// grows with `1/n`. A too-small σ makes the RBF kernel a delta function,
/// every off-diagonal kernel value underflows, and MMD collapses to a
/// constant — which reads as "converged" rather than as a bug.
///
/// The sort is CPU-side: this is called once per eval on an `[n, n]` matrix
/// with `n` = the sample budget (hundreds), so `O(n² log n)` is free
/// relative to the rollout that produced the samples.
pub fn silverman_bandwidth(x: &Tensor) -> CResult<f64> {
    let d = sq_pdist(x, x)?; // [n, n], squared distances
    let n = d.dim(0)?;
    if n < 2 {
        return Ok(1.0);
    }

    let flat = d
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut off_diag: Vec<f32> = Vec::with_capacity(n * (n - 1));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag.push(flat[i * n + j]);
            }
        }
    }

    off_diag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Even count (n² − n is always even), so the median is the mean of the
    // two central order statistics.
    let mid = off_diag.len() / 2;
    let median_sq = 0.5 * (off_diag[mid - 1] as f64 + off_diag[mid] as f64);

    // A constant batch has every pairwise distance at zero — there is no
    // scale to recover, so hand back the documented fallback rather than a
    // degenerate σ.
    if !median_sq.is_finite() || median_sq <= 1e-12 {
        return Ok(1.0);
    }

    // Median-heuristic adjustment: /√2 is the Schölkopf convention.
    Ok((median_sq.sqrt() / std::f64::consts::SQRT_2).max(1e-6))
}

/// Single-bandwidth biased MMD² with the chosen kernel.
pub fn mmd_rbf(x: &Tensor, y: &Tensor, sigma: f64) -> CResult<Tensor> {
    mmd_with(x, y, sigma, KernelChoice::Rbf, /*unbiased=*/ false)
}

pub fn mmd_imq(x: &Tensor, y: &Tensor, c: f64) -> CResult<Tensor> {
    mmd_with(x, y, c, KernelChoice::Imq, /*unbiased=*/ false)
}

/// Multi-bandwidth RBF MMD² — `Σ_σ MMD²_σ(x, y)`. When `bandwidths` is
/// empty, uses `silverman_bandwidth(x)` × {0.25, 0.5, 1, 2, 4}.
pub fn mmd_rbf_multi(x: &Tensor, y: &Tensor, bandwidths: &[f64]) -> CResult<Tensor> {
    let grid: Vec<f64> = if bandwidths.is_empty() {
        let base = silverman_bandwidth(x)?;
        vec![0.25 * base, 0.5 * base, base, 2.0 * base, 4.0 * base]
    } else {
        bandwidths.to_vec()
    };

    let mut acc: Option<Tensor> = None;
    for s in grid {
        let term = mmd_rbf(x, y, s)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    Ok(acc.expect("at least one bandwidth"))
}

fn kernel_from_sq(d: &Tensor, sigma: f64, choice: KernelChoice) -> CResult<Tensor> {
    match choice {
        KernelChoice::Rbf => {
            // exp(-d / (2 σ²))
            let inv = -1.0 / (2.0 * sigma * sigma);
            (d * inv)?.exp()
        }
        KernelChoice::Imq => {
            // (c² + d)^(-1/2)
            let c2 = sigma * sigma;
            let sum = (d + c2)?;
            // 1/sqrt: rsqrt = pow(-0.5). Candle has `powf`.
            sum.powf(-0.5)
        }
    }
}

fn mmd_with(
    x: &Tensor,
    y: &Tensor,
    sigma: f64,
    choice: KernelChoice,
    unbiased: bool,
) -> CResult<Tensor> {
    let dxx = sq_pdist(x, x)?;
    let dyy = sq_pdist(y, y)?;
    let dxy = sq_pdist(x, y)?;
    let kxx = kernel_from_sq(&dxx, sigma, choice)?;
    let kyy = kernel_from_sq(&dyy, sigma, choice)?;
    let kxy = kernel_from_sq(&dxy, sigma, choice)?;

    if unbiased {
        // U-statistic: drop the diagonals from K_xx, K_yy. Candle doesn't
        // expose `.diag()` directly, so we extract via an identity mask.
        let n = kxx.dim(0)?;
        let m = kyy.dim(0)?;
        let nf = n as f64;
        let mf = m as f64;
        let eye_xx = Tensor::eye(n, kxx.dtype(), kxx.device())?;
        let eye_yy = Tensor::eye(m, kyy.dtype(), kyy.device())?;
        let diag_xx = kxx.mul(&eye_xx)?.sum_all()?;
        let diag_yy = kyy.mul(&eye_yy)?.sum_all()?;
        let sum_xx = kxx.sum_all()?;
        let sum_yy = kyy.sum_all()?;
        let sum_xy = kxy.mean_all()?;

        let mean_xx = (sum_xx.sub(&diag_xx)? / (nf * (nf - 1.0)))?;
        let mean_yy = (sum_yy.sub(&diag_yy)? / (mf * (mf - 1.0)))?;
        let two_xy = (sum_xy * 2.0_f64)?;
        let mmd = mean_xx.add(&mean_yy)?.sub(&two_xy)?;
        return Ok(mmd);
    }

    // V-statistic (biased): mean over all pairs including diagonals.
    let mxx = kxx.mean_all()?;
    let myy = kyy.mean_all()?;
    let mxy = kxy.mean_all()?;
    let two_mxy = (mxy * 2.0_f64)?;
    let mmd = mxx.add(&myy)?.sub(&two_mxy)?;
    Ok(mmd)
}

// ---------------------------------------------------------------------------
// Convenience: f64 scalar extraction
// ---------------------------------------------------------------------------

/// Cast a rank-0 tensor to an `f64`. Used by the worker to flatten an MMD
/// tensor into the `EnsembleShard::mmd_to_target` field.
pub fn scalar_f64(t: &Tensor) -> CResult<f64> {
    // Force-cast through f32 to avoid dtype mismatches; the result is
    // always small enough that the precision loss doesn't matter for an
    // ensemble-selection metric.
    let v = t.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
    Ok(v as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Verification of differentiable maximum mean discrepancy with rbf and inverse-multiquadric kernels.

    use super::*;
    use candle_core::{DType, Device};

    /// Build `[n, d]` from a row-major vec.
    fn t(rows: &[&[f32]], dev: &Device) -> Tensor {
        let n = rows.len();
        let d = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_vec(flat, (n, d), dev).unwrap()
    }

    /// The regression test for the mean→median fix. A tight cluster plus one
    /// far outlier: the median bandwidth must track the CLUSTER scale, while
    /// a mean-of-squared-distances estimator is dragged toward the outlier.
    #[test]
    fn silverman_is_robust_to_outliers() {
        let dev = Device::Cpu;
        let x = t(
            &[
                &[0.0, 0.0],
                &[1.0, 0.0],
                &[0.0, 1.0],
                &[1.0, 1.0],
                &[1000.0, 1000.0], // outlier
            ],
            &dev,
        );
        let h = silverman_bandwidth(&x).unwrap();

        // Cluster pairwise distances are 1 or √2, so the median distance sits
        // in [1, √2] and σ = median/√2 lands near [0.7, 1.0].
        assert!(
            h > 0.5 && h < 1.5,
            "median bandwidth {h} should track the cluster scale"
        );

        // Sanity: the mean-of-squares estimator this replaced would be
        // dominated by the outlier's ~2e6 squared distances.
        let d = sq_pdist(&x, &x).unwrap();
        let mean_based = (d.mean_all().unwrap().to_scalar::<f32>().unwrap() as f64)
            .max(1e-12)
            .sqrt()
            / std::f64::consts::SQRT_2;
        assert!(
            mean_based > 100.0 * h,
            "mean-based estimator {mean_based} should be wildly larger than median-based {h}"
        );
    }

    /// Pins diagonal-exclusion by exact arithmetic rather than by a
    /// tolerance, because the two answers are far apart and easy to name.
    ///
    /// Points {0, 1, 2, 3} on a line. The 12 strictly-off-diagonal squared
    /// distances are {1×6, 4×4, 9×2}; sorted, the two central entries are
    /// 1 and 4, so the median is 2.5 and σ = √2.5/√2 ≈ 1.1180.
    ///
    /// Including the 4 zero-valued diagonal entries gives 16 values whose two
    /// central entries are both 1 — median 1, σ = 1/√2 ≈ 0.7071. Asserting
    /// the former therefore fails loudly if the diagonal ever creeps back in.
    #[test]
    fn silverman_excludes_the_diagonal() {
        let dev = Device::Cpu;
        let x = t(&[&[0.0], &[1.0], &[2.0], &[3.0]], &dev);
        let h = silverman_bandwidth(&x).unwrap();

        let expect_excluded = (2.5f64).sqrt() / std::f64::consts::SQRT_2; // ≈ 1.1180
        let expect_included = 1.0f64 / std::f64::consts::SQRT_2; // ≈ 0.7071
        assert!(
            (h - expect_excluded).abs() < 1e-9,
            "expected the diagonal-excluded median σ {expect_excluded}, got {h} \
             (the diagonal-included value would be {expect_included})"
        );
    }

    /// The consequence of excluding the diagonal: padding a batch with
    /// duplicate points must not collapse the bandwidth toward zero the way
    /// n zero-distance diagonal entries would.
    #[test]
    fn silverman_is_stable_under_batch_growth() {
        let dev = Device::Cpu;
        // Same 4-point geometry, tiled 10× — the off-diagonal distribution
        // gains duplicate-pair zeros, but the diagonal contributes none.
        let base: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32]).collect();
        let br: Vec<&[f32]> = base.iter().map(|v| v.as_slice()).collect();
        let h4 = silverman_bandwidth(&t(&br, &dev)).unwrap();

        let tiled: Vec<Vec<f32>> = (0..40).map(|i| vec![(i % 4) as f32]).collect();
        let tr: Vec<&[f32]> = tiled.iter().map(|v| v.as_slice()).collect();
        let h40 = silverman_bandwidth(&t(&tr, &dev)).unwrap();

        // Both stay on the same order of magnitude — the duplicate-pair zeros
        // shift the median by one order statistic, not by a factor of n.
        assert!(h4 > 0.0 && h40 > 0.0);
        assert!(
            h4 / h40 < 2.0 && h40 / h4 < 2.0,
            "bandwidth should not collapse as n grows: {h4} (n=4) vs {h40} (n=40)"
        );
    }

    #[test]
    fn silverman_constant_batch_falls_back_to_one() {
        let dev = Device::Cpu;
        let x = t(&[&[3.0, 3.0], &[3.0, 3.0], &[3.0, 3.0]], &dev);
        assert_eq!(silverman_bandwidth(&x).unwrap(), 1.0);
    }

    #[test]
    fn silverman_single_row_falls_back_to_one() {
        let dev = Device::Cpu;
        let x = t(&[&[1.0, 2.0]], &dev);
        assert_eq!(silverman_bandwidth(&x).unwrap(), 1.0);
    }

    /// Deterministic ring — a stand-in for two-moons that needs no RNG.
    fn ring(n: usize, radius: f32, shift: f32, dev: &Device) -> Tensor {
        let mut v = Vec::with_capacity(n * 2);
        for i in 0..n {
            let a = (i as f32) / (n as f32) * std::f32::consts::TAU;
            v.push(radius * a.cos() + shift);
            v.push(radius * a.sin());
        }
        Tensor::from_vec(v, (n, 2), dev).unwrap()
    }

    #[test]
    fn mmd_against_self_is_near_zero() {
        let dev = Device::Cpu;
        let x = ring(64, 1.0, 0.0, &dev);
        let m = scalar_f64(&mmd_rbf_multi(&x, &x, &[]).unwrap()).unwrap();
        assert!(m.abs() < 1e-5, "MMD(x, x) should vanish, got {m}");
    }

    /// The property the early-stop signal depends on: MMD must increase
    /// monotonically as the two distributions separate.
    #[test]
    fn mmd_is_monotone_in_the_shift() {
        let dev = Device::Cpu;
        let x = ring(64, 1.0, 0.0, &dev);
        let shifts = [0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0];
        let mut prev = f64::NEG_INFINITY;
        for s in shifts {
            let y = ring(64, 1.0, s, &dev);
            // Fix the bandwidth grid off x alone so the comparison isn't
            // confounded by a per-pair bandwidth that moves with the shift.
            let base = silverman_bandwidth(&x).unwrap();
            let grid: Vec<f64> = [0.25, 0.5, 1.0, 2.0, 4.0]
                .iter()
                .map(|f| f * base)
                .collect();
            let m = scalar_f64(&mmd_rbf_multi(&x, &y, &grid).unwrap()).unwrap();
            assert!(
                m > prev,
                "MMD should grow with the shift: at {s} got {m}, previous {prev}"
            );
            prev = m;
        }
    }

    #[test]
    fn mmd_multi_matches_explicit_grid() {
        let dev = Device::Cpu;
        let x = ring(32, 1.0, 0.0, &dev);
        let y = ring(32, 1.0, 0.5, &dev);
        let base = silverman_bandwidth(&x).unwrap();
        let grid: Vec<f64> = [0.25, 0.5, 1.0, 2.0, 4.0]
            .iter()
            .map(|f| f * base)
            .collect();
        let a = scalar_f64(&mmd_rbf_multi(&x, &y, &[]).unwrap()).unwrap();
        let b = scalar_f64(&mmd_rbf_multi(&x, &y, &grid).unwrap()).unwrap();
        assert!(
            (a - b).abs() < 1e-6,
            "empty bandwidths must expand to the Silverman grid: {a} vs {b}"
        );
    }

    #[test]
    fn scalar_f64_round_trips() {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![2.5f32], 1, &dev)
            .unwrap()
            .reshape(())
            .unwrap();
        assert!((scalar_f64(&v).unwrap() - 2.5).abs() < 1e-9);
        let d = Tensor::from_vec(vec![2.5f64], 1, &dev)
            .unwrap()
            .reshape(())
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        assert!((scalar_f64(&d).unwrap() - 2.5).abs() < 1e-6);
    }
}
