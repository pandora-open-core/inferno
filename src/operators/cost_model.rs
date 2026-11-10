//! Inferno — Expression working-set estimates and matrix-multiplication factoring costs.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Result;

use super::symbolic::{infer, Expr, LazyTensor};

/// Whether to apply the factoring rewrite.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum FactorPolicy {
    /// Leave every squared distance in its direct, maximally-accurate form.
    #[default]
    Never,
    /// Factor where the cost model predicts a win.
    Auto,
    /// Factor every eligible site, model or not. Benchmarking and tests.
    Always,
}

/// Relative op costs, per element visited.
///
/// Transcendentals are an order dearer than an add on every backend that
/// matters, and lumping them with `Add` is what makes a naive node-count
/// model prefer the wrong form. `MatDot` is discounted because its work is a
/// GEMM: one BLAS call with blocked, vectorised, cache-resident inner loops,
/// against candle's broadcast-materialise-then-reduce pipeline.
const W_ELEMENTWISE: f64 = 1.0;
const W_TRANSCENDENTAL: f64 = 6.0;
const W_LEAF: f64 = 0.5;
const W_GEMM: f64 = 0.15;

/// Feature widths below which factoring is refused regardless of the model.
///
/// The model counts arithmetic but not the fixed cost of *reaching* BLAS —
/// shape juggling, a contiguity copy, the call itself. Below a handful of
/// features that overhead dominates the arithmetic it saves, and rather than
/// fudge the weights to encode it, the guard is explicit.
const MIN_FACTOR_DIM: usize = 4;

/// Fractional improvement required before rewriting, so a formula does not
/// churn for a rounding-level "win".
const MARGIN: f64 = 0.05;

fn node_weight(e: &Expr) -> f64 {
    match e {
        Expr::Leaf { .. } => W_LEAF,
        Expr::Scalar(_) | Expr::EyeIdentity { .. } => 0.0,
        Expr::Exp(_) | Expr::Log(_) | Expr::Sqrt(_) | Expr::Recip(_) | Expr::Pow(..) => {
            W_TRANSCENDENTAL
        }
        // `powf` is a libm call per element, not a multiply.
        Expr::PowF(..) => W_TRANSCENDENTAL,
        Expr::Div(..) => W_TRANSCENDENTAL,
        Expr::MatDot(..) => W_GEMM,
        _ => W_ELEMENTWISE,
    }
}

/// Weighted work on the pair grid, per pair element, in the large-tile limit.
///
/// Only nodes depending on BOTH axes contribute: an I-only node costs
/// `O(s·d)` against the grid's `O(s²)`, so its share vanishes as the tile
/// grows. Nodes are ptr-deduplicated, matching what the per-tile memo
/// actually evaluates.
pub fn pair_work(e: &Arc<Expr>) -> Result<f64> {
    let mut seen: HashMap<*const Expr, ()> = HashMap::new();
    let mut total = 0.0;
    walk_work(e, &mut seen, &mut total)?;
    Ok(total)
}

fn walk_work(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, total: &mut f64) -> Result<()> {
    if seen.insert(Arc::as_ptr(e), ()).is_some() {
        return Ok(());
    }
    let m = infer(e)?;
    if m.uses_i && m.uses_j {
        *total += node_weight(e) * m.dim.max(1) as f64;
    }
    match e.as_ref() {
        Expr::Leaf { .. } | Expr::Scalar(_) | Expr::EyeIdentity { .. } => {}
        Expr::Neg(a)
        | Expr::Exp(a)
        | Expr::Log(a)
        | Expr::Sqrt(a)
        | Expr::Recip(a)
        | Expr::Sum(a)
        | Expr::PowF(a, _)
        | Expr::MaxC(a, _)
        | Expr::GtC(a, _)
        | Expr::Affine { e: a, .. } => walk_work(a, seen, total)?,
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::MatDot(a, b) => {
            walk_work(a, seen, total)?;
            walk_work(b, seen, total)?;
        }
    }
    Ok(())
}

/// Rewrite every eligible squared distance in `root` into its GEMM-backed
/// expansion. Returns the formula unchanged under
/// [`FactorPolicy::Never`], or where no site qualifies.
///
/// The result is NOT interned — run it through
/// [`LazyTensor::simplify`](super::symbolic::LazyTensor::simplify) afterwards
/// to hash-cons the new nodes (`factor` on [`LazyTensor`] does this for you).
pub fn factor_sq_dist(root: &Arc<Expr>, policy: FactorPolicy) -> Result<Arc<Expr>> {
    if policy == FactorPolicy::Never {
        return Ok(root.clone());
    }
    let mut memo: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
    rewrite(root, policy, &mut memo)
}

/// `(a, b)` when `e` is a squared distance between an I-pure and a J-pure
/// operand — the two spellings the codebase actually produces:
/// `Sum(PowF(Sub(a, b), 2))` and `Sum(Mul(x, x))` with `x = Sub(a, b)` the
/// SAME node (which is what CSE leaves behind for `&d * &d`).
fn as_sq_dist(e: &Arc<Expr>) -> Result<Option<(Arc<Expr>, Arc<Expr>)>> {
    let Expr::Sum(inner) = e.as_ref() else {
        return Ok(None);
    };
    let diff = match inner.as_ref() {
        Expr::PowF(d, p) if *p == 2.0 => d,
        Expr::Mul(x, y) if Arc::ptr_eq(x, y) => x,
        _ => return Ok(None),
    };
    let Expr::Sub(a, b) = diff.as_ref() else {
        return Ok(None);
    };
    let (ma, mb) = (infer(a)?, infer(b)?);
    // Purity is what makes the cross term a matrix product at all.
    if !(ma.uses_i && !ma.uses_j) || !(mb.uses_j && !mb.uses_i) {
        return Ok(None);
    }
    if ma.dim != mb.dim || ma.dim < MIN_FACTOR_DIM {
        return Ok(None);
    }
    Ok(Some((a.clone(), b.clone())))
}

/// `max(‖a‖² − 2⟨a,b⟩ + ‖b‖², 0)`.
fn expand(a: &Arc<Expr>, b: &Arc<Expr>) -> Arc<Expr> {
    let al = LazyTensor::from_arc(a.clone());
    let bl = LazyTensor::from_arc(b.clone());
    // `Mul(a, a)` rather than `PowF(a, 2)`: same value, one multiply instead
    // of a libm call, and the shared operand collapses in the per-tile memo.
    let na = (&al * &al).sum_last();
    let nb = (&bl * &bl).sum_last();
    let cross = al.mat_dot(&bl).affine(-2.0, 0.0);
    (&(&na + &cross) + &nb).max_c(0.0).expr().clone()
}

fn rewrite(
    e: &Arc<Expr>,
    policy: FactorPolicy,
    memo: &mut HashMap<*const Expr, Arc<Expr>>,
) -> Result<Arc<Expr>> {
    if let Some(done) = memo.get(&Arc::as_ptr(e)) {
        return Ok(done.clone());
    }
    let out = match as_sq_dist(e)? {
        Some((a, b)) => {
            // Recurse into the operands first — a nested squared distance
            // inside `a` is its own site.
            let a = rewrite(&a, policy, memo)?;
            let b = rewrite(&b, policy, memo)?;
            let candidate = expand(&a, &b);
            let take = match policy {
                FactorPolicy::Always => true,
                FactorPolicy::Never => false,
                FactorPolicy::Auto => pair_work(&candidate)? < pair_work(e)? * (1.0 - MARGIN),
            };
            if take {
                candidate
            } else {
                rebuild(e, policy, memo)?
            }
        }
        None => rebuild(e, policy, memo)?,
    };
    memo.insert(Arc::as_ptr(e), out.clone());
    Ok(out)
}

fn rebuild(
    e: &Arc<Expr>,
    p: FactorPolicy,
    memo: &mut HashMap<*const Expr, Arc<Expr>>,
) -> Result<Arc<Expr>> {
    let mut r = |c: &Arc<Expr>, memo: &mut HashMap<*const Expr, Arc<Expr>>| rewrite(c, p, memo);
    Ok(match e.as_ref() {
        Expr::Leaf { .. } | Expr::Scalar(_) | Expr::EyeIdentity { .. } => e.clone(),
        Expr::Neg(a) => Arc::new(Expr::Neg(r(a, memo)?)),
        Expr::Exp(a) => Arc::new(Expr::Exp(r(a, memo)?)),
        Expr::Log(a) => Arc::new(Expr::Log(r(a, memo)?)),
        Expr::Sqrt(a) => Arc::new(Expr::Sqrt(r(a, memo)?)),
        Expr::Recip(a) => Arc::new(Expr::Recip(r(a, memo)?)),
        Expr::Sum(a) => Arc::new(Expr::Sum(r(a, memo)?)),
        Expr::PowF(a, q) => Arc::new(Expr::PowF(r(a, memo)?, *q)),
        Expr::MaxC(a, c) => Arc::new(Expr::MaxC(r(a, memo)?, *c)),
        Expr::GtC(a, c) => Arc::new(Expr::GtC(r(a, memo)?, *c)),
        Expr::Affine { e: a, mul, add } => Arc::new(Expr::Affine {
            e: r(a, memo)?,
            mul: *mul,
            add: *add,
        }),
        Expr::Add(a, b) => Arc::new(Expr::Add(r(a, memo)?, r(b, memo)?)),
        Expr::Sub(a, b) => Arc::new(Expr::Sub(r(a, memo)?, r(b, memo)?)),
        Expr::Mul(a, b) => Arc::new(Expr::Mul(r(a, memo)?, r(b, memo)?)),
        Expr::Div(a, b) => Arc::new(Expr::Div(r(a, memo)?, r(b, memo)?)),
        Expr::Pow(a, b) => Arc::new(Expr::Pow(r(a, memo)?, r(b, memo)?)),
        Expr::MatDot(a, b) => Arc::new(Expr::MatDot(r(a, memo)?, r(b, memo)?)),
    })
}

#[cfg(test)]
mod tests {
    //! Verification of expression working-set estimates and matrix-multiplication factoring costs.

    use super::*;
    use crate::operators::symbolic::BlockCfg;
    use candle_core::{DType, Device, Tensor, Var, D};

    fn one_shot() -> BlockCfg {
        BlockCfg {
            force_dense: true,
            ..Default::default()
        }
    }

    fn pts(n: usize, d: usize, seed: usize) -> Tensor {
        Tensor::from_vec(
            (0..n * d)
                .map(|i| (((i * 37 + seed * 11) % 23) as f64) / 7.0 - 1.5)
                .collect::<Vec<_>>(),
            (n, d),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// Whether the DAG contains a `MatDot` — the direct signal that the
    /// rewrite fired, rather than a node-count proxy (`factor` re-interns,
    /// so counts move for reasons unrelated to factoring).
    fn has_mat_dot(e: &Arc<Expr>) -> bool {
        match e.as_ref() {
            Expr::MatDot(..) => true,
            Expr::Leaf { .. } | Expr::Scalar(_) | Expr::EyeIdentity { .. } => false,
            Expr::Neg(a)
            | Expr::Exp(a)
            | Expr::Log(a)
            | Expr::Sqrt(a)
            | Expr::Recip(a)
            | Expr::Sum(a)
            | Expr::PowF(a, _)
            | Expr::MaxC(a, _)
            | Expr::GtC(a, _)
            | Expr::Affine { e: a, .. } => has_mat_dot(a),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => has_mat_dot(a) || has_mat_dot(b),
        }
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f64 {
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

    /// `‖(xᵢ − yⱼ)·w‖²` — the real `cost_lazy` shape.
    fn sq_dist(x: &Tensor, y: &Tensor, w: &Tensor) -> LazyTensor {
        let wp = LazyTensor::pm(w);
        let xi = &LazyTensor::vi(x) * &wp;
        let yj = &LazyTensor::vj(y) * &wp;
        (&(&xi - &yj)).powf(2.0).sum_last()
    }

    #[test]
    fn factoring_matches_the_direct_form() {
        let (x, y) = (pts(11, 8, 1), pts(7, 8, 2));
        let w = Tensor::full(0.9f64, (1, 8), &Device::Cpu).unwrap();
        let f = sq_dist(&x, &y, &w);
        let g = f.factor(FactorPolicy::Always).unwrap();
        let (a, b) = (
            f.to_dense(&one_shot()).unwrap(),
            g.to_dense(&one_shot()).unwrap(),
        );
        assert_eq!(a.dims(), b.dims());
        let d = max_abs(&a, &b);
        assert!(d < 1e-12, "factored form moved the value by {d:.3e}");
    }

    /// The cancellation case, stated as a bound rather than hidden: a SELF
    /// pair has an exactly-zero diagonal, and the expansion must not produce
    /// a negative one — that is what the `MaxC` clamp is for.
    #[test]
    fn factoring_keeps_the_self_pair_diagonal_non_negative() {
        for dtype in [DType::F32, DType::F64] {
            let x = pts(24, 16, 5).to_dtype(dtype).unwrap();
            let w = Tensor::full(1.3f64, (1, 16), &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let g = sq_dist(&x, &x, &w).factor(FactorPolicy::Always).unwrap();
            let dense = g.to_dense(&one_shot()).unwrap();
            let diag = crate::operators::dense_diagonal(&dense)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap();
            assert!(
                diag.iter().all(|v| *v >= 0.0),
                "{dtype:?}: negative self-distance on the diagonal: {diag:?}"
            );
            // And the whole matrix stays non-negative.
            let min = dense
                .min_all()
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            assert!(min >= 0.0, "{dtype:?}: negative squared distance {min}");
        }
    }

    /// The f32 error budget, pinned. If this drifts, the doc's error analysis
    /// is wrong and the `gemm_factoring_enabled` default needs revisiting.
    #[test]
    fn f32_factoring_error_stays_within_the_documented_budget() {
        let (x, y) = (pts(32, 32, 3), pts(32, 32, 9));
        let w = Tensor::full(1.0f64, (1, 32), &Device::Cpu).unwrap();
        let direct = sq_dist(&x, &y, &w).to_dense(&one_shot()).unwrap();
        let (x32, y32) = (
            x.to_dtype(DType::F32).unwrap(),
            y.to_dtype(DType::F32).unwrap(),
        );
        let w32 = w.to_dtype(DType::F32).unwrap();
        let got = sq_dist(&x32, &y32, &w32)
            .factor(FactorPolicy::Always)
            .unwrap()
            .to_dense(&one_shot())
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let err = max_abs(&direct, &got);
        assert!(
            err < 1e-3,
            "f32 factoring absolute error {err:.3e} exceeds the budget"
        );
    }

    #[test]
    fn gradients_survive_the_factoring() {
        let (x, y) = (pts(9, 8, 4), pts(6, 8, 7));
        let wv = Var::from_tensor(&Tensor::full(0.7f64, (1, 8), &Device::Cpu).unwrap()).unwrap();
        let loss_of = |g: &LazyTensor| -> Tensor {
            g.to_dense(&one_shot())
                .unwrap()
                .exp()
                .unwrap()
                .sum_all()
                .unwrap()
        };
        let direct = sq_dist(&x, &y, wv.as_tensor()).affine(-1.0, 0.0);
        let factored = direct.factor(FactorPolicy::Always).unwrap();

        let gd = loss_of(&direct)
            .backward()
            .unwrap()
            .get(wv.as_tensor())
            .unwrap()
            .to_vec2::<f64>()
            .unwrap();
        let gf = loss_of(&factored)
            .backward()
            .unwrap()
            .get(wv.as_tensor())
            .expect("factored form dropped w from the graph")
            .to_vec2::<f64>()
            .unwrap();
        for (a, b) in gd[0].iter().zip(gf[0].iter()) {
            assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "grad {a} vs {b}");
        }
    }

    #[test]
    fn the_model_prefers_the_gemm_at_width_and_declines_at_d_below_the_guard() {
        let w = Tensor::full(1.0f64, (1, 32), &Device::Cpu).unwrap();
        let wide = sq_dist(&pts(64, 32, 1), &pts(64, 32, 2), &w);
        let auto = wide.factor(FactorPolicy::Auto).unwrap();
        assert!(has_mat_dot(auto.expr()), "Auto declined a d=32 site");
        assert!(pair_work(auto.expr()).unwrap() < pair_work(wide.expr()).unwrap());
        // d = 2 is below MIN_FACTOR_DIM: the site is not even eligible.
        let w2 = Tensor::full(1.0f64, (1, 2), &Device::Cpu).unwrap();
        let thin = sq_dist(&pts(64, 2, 1), &pts(64, 2, 2), &w2);
        let thin_auto = thin.factor(FactorPolicy::Auto).unwrap();
        assert!(
            !has_mat_dot(thin_auto.expr()),
            "Auto factored a d=2 site despite the MIN_FACTOR_DIM guard"
        );
    }

    #[test]
    fn a_mixed_operand_is_refused() {
        // `a` depends on BOTH axes, so the cross term is not a matrix
        // product and there is nothing to lower it to.
        let (x, y) = (pts(8, 8, 1), pts(8, 8, 2));
        let mixed = &LazyTensor::vi(&x) * &LazyTensor::vj(&y); // pair-valued
        let f = (&(&mixed - &LazyTensor::vj(&y))).powf(2.0).sum_last();
        let g = f.factor(FactorPolicy::Always).unwrap();
        assert!(!has_mat_dot(g.expr()), "an impure operand was factored");
    }

    /// `MatDot` refuses an operand that is not axis-pure — the invariant the
    /// rewrite relies on, enforced independently of the rewrite.
    #[test]
    fn mat_dot_rejects_impure_operands() {
        let (x, y) = (pts(8, 4, 1), pts(8, 4, 2));
        let pair = &LazyTensor::vi(&x) * &LazyTensor::vj(&y);
        assert!(pair.mat_dot(&LazyTensor::vj(&y)).infer().is_err());
        assert!(LazyTensor::vi(&x).mat_dot(&pair).infer().is_err());
        // Mismatched contraction widths are refused too.
        let z = pts(8, 5, 3);
        assert!(LazyTensor::vi(&x)
            .mat_dot(&LazyTensor::vj(&z))
            .infer()
            .is_err());
    }

    /// `MatDot` is `Sum(Mul(a, b))` with BLAS doing the contraction — the
    /// forward oracle for the node itself, independent of the rewrite.
    #[test]
    fn mat_dot_matches_the_elementwise_contraction() {
        let (x, y) = (pts(13, 6, 1), pts(9, 6, 2));
        let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
        let want = (&xi * &yj).sum_last().to_dense(&one_shot()).unwrap();
        let got = xi.mat_dot(&yj).to_dense(&one_shot()).unwrap();
        assert_eq!(got.dims(), &[13, 9]);
        assert!(max_abs(&want, &got) < 1e-12);
        // …and under tiling, including tiles that do not divide the grid.
        let tiled = xi
            .mat_dot(&yj)
            .to_dense(&BlockCfg::with_tiles(5, 4))
            .unwrap();
        assert!(max_abs(&want, &tiled) < 1e-12);
    }

    #[test]
    fn mat_dot_carries_a_batch_axis() {
        let x = Tensor::rand(0f64, 1f64, (3, 7, 5), &Device::Cpu).unwrap();
        let y = Tensor::rand(0f64, 1f64, (3, 4, 5), &Device::Cpu).unwrap();
        let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
        let want = (&xi * &yj).sum_last().to_dense(&one_shot()).unwrap();
        let got = xi.mat_dot(&yj).to_dense(&one_shot()).unwrap();
        assert_eq!(got.dims(), &[3, 7, 4]);
        assert!(max_abs(&want, &got) < 1e-12);
    }

    /// A `Pm` operand broadcasts against a batched one — the batch-rank
    /// mismatch `broadcast_matmul` alone would refuse.
    #[test]
    fn mat_dot_broadcasts_a_shared_operand_across_a_batch() {
        let x = Tensor::rand(0f64, 1f64, (3, 7, 5), &Device::Cpu).unwrap();
        let y = Tensor::rand(0f64, 1f64, (4, 5), &Device::Cpu).unwrap();
        let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
        let want = (&xi * &yj).sum_last().to_dense(&one_shot()).unwrap();
        let got = xi.mat_dot(&yj).to_dense(&one_shot()).unwrap();
        assert_eq!(got.dims(), &[3, 7, 4]);
        assert!(max_abs(&want, &got) < 1e-12);
    }

    /// The point of the node: the `[bi, bj, d]` intermediate never exists.
    /// Peak allocation is checked through the footprint model the scheduler
    /// shares, so a regression here also moves the tile sizes.
    #[test]
    fn factoring_removes_the_feature_axis_from_the_pair_grid() {
        let w = Tensor::full(1.0f64, (1, 32), &Device::Cpu).unwrap();
        let f = sq_dist(&pts(512, 32, 1), &pts(512, 32, 2), &w);
        let g = f.factor(FactorPolicy::Always).unwrap();
        let before = crate::operators::schedule::footprint(f.expr()).unwrap();
        let after = crate::operators::schedule::footprint(g.expr()).unwrap();
        // Before: the difference and its square are both dim-32 PAIR nodes.
        assert!(before.pair >= 64, "fixture drifted: {before:?}");
        // After: the only pair-dependent nodes are the MatDot, its affine,
        // two adds and the clamp — five nodes, every one a pair-SCALAR. The
        // feature width has moved entirely onto the i/j margins.
        assert!(
            after.pair <= 8,
            "the feature axis still reaches the pair grid: {after:?}"
        );
        assert!(
            after.i_only > 64 && after.j_only > 64,
            "margins lost the features: {after:?}"
        );
        // Which buys a much larger tile under the same budget: the grid term
        // shrinks 13×, and the side grows until the (now dominant) linear
        // margin terms bind — measured ~3.3×.
        let budget = 1 << 20;
        let (sb, sa) = (before.square_side(budget), after.square_side(budget));
        assert!(
            sa > 3 * sb,
            "tile side {sb} → {sa} is a smaller win than expected"
        );
    }

    /// Measured, not modelled. `cargo test -p pandorapurple --lib
    /// gemm_factoring_benchmark -- --ignored --nocapture`.
    ///
    /// Ignored because a wall-clock assertion is a flaky test on shared CI,
    /// but the number is the entire justification for the rung, so it lives
    /// here rather than in a scratch file.
    #[test]
    #[ignore = "benchmark"]
    fn gemm_factoring_benchmark() {
        use std::time::Instant;
        let cfg = BlockCfg::default();
        for (m, d) in [(256usize, 32usize), (1024, 32), (2048, 64)] {
            let w = Tensor::full(1.0f64, (1, d), &Device::Cpu).unwrap();
            let x = Tensor::rand(0f32, 1f32, (m, d), &Device::Cpu).unwrap();
            let y = Tensor::rand(0f32, 1f32, (m, d), &Device::Cpu).unwrap();
            let w = w.to_dtype(DType::F32).unwrap();
            let direct = sq_dist(&x, &y, &w);
            let factored = direct.factor(FactorPolicy::Always).unwrap();

            let run = |f: &LazyTensor| {
                let t0 = Instant::now();
                for _ in 0..5 {
                    f.logsumexp_reduce(crate::operators::ReduceAxis::J, &cfg)
                        .unwrap();
                }
                t0.elapsed().as_secs_f64() / 5.0
            };
            let _ = run(&direct); // warm
            let (a, b) = (run(&direct), run(&factored));
            println!(
                "m={m} d={d}: direct {:.1} ms, factored {:.1} ms  →  {:.2}x",
                a * 1e3,
                b * 1e3,
                a / b
            );
        }
    }

    #[test]
    fn factoring_preserves_the_leaves_and_the_inferred_extents() {
        let w = Tensor::full(1.0f64, (1, 8), &Device::Cpu).unwrap();
        let f = sq_dist(&pts(17, 8, 1), &pts(11, 8, 2), &w);
        let g = f.factor(FactorPolicy::Always).unwrap();
        let (mf, mg) = (f.infer().unwrap(), g.infer().unwrap());
        assert_eq!(
            (mf.m, mf.n),
            (mg.m, mg.n),
            "factoring changed the pair extents"
        );
        assert_eq!(mg.dim, 1);
        assert_eq!(
            f.leaves().len(),
            g.leaves().len(),
            "factoring changed the leaf pack"
        );
    }

    /// The whole reason `Sum(Mul(x, x))` is a recognised spelling: it is what
    /// CSE leaves behind for the `&d * &d` idiom.
    #[test]
    fn the_shared_mul_spelling_is_recognised_after_cse() {
        let (x, y) = (pts(10, 8, 1), pts(10, 8, 2));
        let d = &LazyTensor::vi(&x) - &LazyTensor::vj(&y);
        let f = (&d * &d).sum_last().simplify().unwrap();
        let g = f.factor(FactorPolicy::Always).unwrap();
        assert!(has_mat_dot(g.expr()), "the Mul spelling was not recognised");
        assert!(
            max_abs(
                &f.to_dense(&one_shot()).unwrap(),
                &g.to_dense(&one_shot()).unwrap()
            ) < 1e-12
        );
        let _ = D::Minus1;
    }
}
