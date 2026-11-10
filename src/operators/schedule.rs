//! Inferno — Tile scheduling from expression working sets and caller-supplied memory budgets.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use candle_core::{DType, Result};

use super::symbolic::{infer, Expr};

/// What one tile of a formula costs to hold, split by how each term scales.
///
/// Units are ELEMENTS of the formula's dtype, per `(bi, bj)` tile, already
/// multiplied by the batch extent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Σ dim over nodes that depend on BOTH axes — the `s²` term.
    pub pair: usize,
    /// Σ dim over nodes that depend on `i` only — linear in `s`.
    pub i_only: usize,
    /// Σ dim over nodes that depend on `j` only — linear in `s`.
    pub j_only: usize,
    /// Σ dim over tile-invariant nodes — constant.
    pub constant: usize,
    /// Product of the batch dims; every tile carries them.
    pub batch: usize,
    /// Distinct nodes in the DAG (what the memo keys).
    pub nodes: usize,
    pub dtype: DType,
}

impl Footprint {
    /// Working set in ELEMENTS for a `bi × bj` tile.
    pub fn cells(&self, bi: usize, bj: usize) -> usize {
        self.batch.saturating_mul(
            self.pair
                .saturating_mul(bi)
                .saturating_mul(bj)
                .saturating_add(self.i_only.saturating_mul(bi))
                .saturating_add(self.j_only.saturating_mul(bj))
                .saturating_add(self.constant),
        )
    }

    /// Working set in bytes for a `bi × bj` tile.
    pub fn bytes(&self, bi: usize, bj: usize) -> usize {
        self.cells(bi, bj)
            .saturating_mul(self.dtype.size_in_bytes().max(1))
    }

    /// Largest square side `s` with `bytes(s, s) <= budget`.
    ///
    /// Exact root of `pair·s² + (i_only + j_only)·s + constant = budget_cells`
    /// rather than the old `sqrt(budget / cell)`, which ignored the linear and
    /// constant terms entirely and so over-shot on formulas with fat I-side
    /// or J-side intermediates.
    pub fn square_side(&self, budget_bytes: usize) -> usize {
        let esz = self.dtype.size_in_bytes().max(1);
        let batch = self.batch.max(1);
        let avail = (budget_bytes / esz / batch) as f64 - self.constant as f64;
        if avail <= 0.0 {
            return 1;
        }
        let a = self.pair as f64;
        let b = (self.i_only + self.j_only) as f64;
        let s = if a > 0.0 {
            // Positive root of a·s² + b·s − avail = 0.
            (-b + (b * b + 4.0 * a * avail).sqrt()) / (2.0 * a)
        } else if b > 0.0 {
            avail / b
        } else {
            // Nothing scales with the tile — any size fits.
            return usize::MAX;
        };
        if !s.is_finite() || s < 1.0 {
            1
        } else {
            s as usize
        }
    }
}

/// Measure a formula's per-tile working set.
///
/// Walks DISTINCT nodes (ptr-deduped, matching the memo's keying) and buckets
/// each by its axis dependence. A malformed subtree surfaces here as an
/// `infer` error rather than at tile time.
pub fn footprint(e: &Arc<Expr>) -> Result<Footprint> {
    let root = infer(e)?;
    let batch: usize = root.batch.iter().product::<usize>().max(1);
    let mut fp = Footprint {
        pair: 0,
        i_only: 0,
        j_only: 0,
        constant: 0,
        batch,
        nodes: 0,
        dtype: root.dtype,
    };
    let mut seen: HashMap<*const Expr, ()> = HashMap::new();
    walk(e, &mut seen, &mut fp)?;
    Ok(fp)
}

/// Measure SEVERAL formulas that share one sweep — and therefore one per-tile
/// memo. Nodes are deduplicated ACROSS formulas, which is exactly what the
/// shared memo does, so a cost subtree common to a fused pair is counted once
/// rather than twice.
pub fn footprint_many<'a>(exprs: impl Iterator<Item = &'a Arc<Expr>>) -> Result<Footprint> {
    let mut seen: HashMap<*const Expr, ()> = HashMap::new();
    let mut fp: Option<Footprint> = None;
    for e in exprs {
        let root = infer(e)?;
        let batch: usize = root.batch.iter().product::<usize>().max(1);
        let acc = fp.get_or_insert(Footprint {
            pair: 0,
            i_only: 0,
            j_only: 0,
            constant: 0,
            batch,
            nodes: 0,
            dtype: root.dtype,
        });
        acc.batch = acc.batch.max(batch);
        walk(e, &mut seen, acc)?;
    }
    fp.ok_or_else(|| candle_core::Error::Msg("schedule: no formulas to measure".into()))
}

fn walk(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, fp: &mut Footprint) -> Result<()> {
    if seen.insert(Arc::as_ptr(e), ()).is_some() {
        return Ok(()); // DAG-shared: one memo entry, counted once
    }
    let m = infer(e)?;
    fp.nodes += 1;
    let d = m.dim.max(1);
    match (m.uses_i, m.uses_j) {
        (true, true) => fp.pair += d,
        (true, false) => fp.i_only += d,
        (false, true) => fp.j_only += d,
        (false, false) => fp.constant += d,
    }
    for child in children(e) {
        walk(child, seen, fp)?;
    }
    Ok(())
}

/// A node's children — the one place the `Expr` shape is enumerated for
/// traversal in this module.
fn children(e: &Arc<Expr>) -> Vec<&Arc<Expr>> {
    match e.as_ref() {
        Expr::Leaf { .. } | Expr::Scalar(_) | Expr::EyeIdentity { .. } => vec![],
        Expr::Neg(a)
        | Expr::Exp(a)
        | Expr::Log(a)
        | Expr::Sqrt(a)
        | Expr::Recip(a)
        | Expr::Sum(a)
        | Expr::PowF(a, _)
        | Expr::MaxC(a, _)
        | Expr::GtC(a, _)
        | Expr::Affine { e: a, .. } => vec![a],
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::MatDot(a, b) => vec![a, b],
    }
}

// ---------------------------------------------------------------------------
// Autotune hook
// ---------------------------------------------------------------------------

/// An override for the analytic tile choice.
///
/// Returning `None` defers to the analytic policy, so a tuner can specialise
/// only the shapes it has actually measured.
pub trait TileTuner: Send + Sync {
    fn tiles(
        &self,
        fp: &Footprint,
        m: usize,
        n: usize,
        budget_bytes: usize,
    ) -> Option<(usize, usize)>;
}

static TILE_TUNER: OnceLock<Arc<dyn TileTuner>> = OnceLock::new();

/// Register the process-wide tile tuner. Set once, at startup.
pub fn set_tile_tuner(t: Arc<dyn TileTuner>) -> std::result::Result<(), &'static str> {
    TILE_TUNER
        .set(t)
        .map_err(|_| "tile tuner already registered")
}

pub fn tile_tuner() -> Option<&'static Arc<dyn TileTuner>> {
    TILE_TUNER.get()
}

#[cfg(test)]
mod tests {
    //! Verification of tile scheduling from expression working sets and caller-supplied memory budgets.

    use super::*;
    use crate::operators::symbolic::{BlockCfg, LazyTensor};
    use candle_core::{Device, Tensor};

    fn pts(n: usize, d: usize) -> Tensor {
        Tensor::zeros((n, d), DType::F32, &Device::Cpu).unwrap()
    }

    /// `‖xᵢ − xⱼ‖²` — the shape whose footprint the flat model under-counts.
    fn sq_dist(m: usize, n: usize, d: usize) -> LazyTensor {
        let x = pts(m, d);
        let y = pts(n, d);
        (&(&LazyTensor::vi(&x) - &LazyTensor::vj(&y)))
            .powf(2.0)
            .sum_last()
    }

    #[test]
    fn footprint_buckets_nodes_by_axis_dependence() {
        let f = sq_dist(8, 8, 4);
        let fp = footprint(f.expr()).unwrap();
        // Vi leaf → i_only(4); Vj leaf → j_only(4); Sub, PowF → pair(4 each);
        // Sum → pair(1).
        assert_eq!(fp.i_only, 4, "{fp:?}");
        assert_eq!(fp.j_only, 4, "{fp:?}");
        assert_eq!(fp.pair, 4 + 4 + 1, "{fp:?}");
        assert_eq!(fp.constant, 0, "{fp:?}");
        assert_eq!(fp.nodes, 5, "{fp:?}");
    }

    /// The bug this rung fixes: the flat `root_dim` model sizes the tile from
    /// the reduction's dim 1 and misses every dim-`d` intermediate the memo
    /// holds. On a real kernel formula that is an order of magnitude.
    #[test]
    fn flat_root_dim_model_understates_the_working_set() {
        let f = sq_dist(4096, 4096, 32);
        let fp = footprint(f.expr()).unwrap();
        // What the memo actually holds per pair element vs what the root says.
        assert_eq!(fp.pair, 32 + 32 + 1);
        let budget = 8 << 20;
        let side = fp.square_side(budget);
        let flat = BlockCfg {
            budget_bytes: budget,
            ..Default::default()
        }
        .tiles(4096, 4096, 1, DType::F32)
        .0;
        assert!(
            side * 4 < flat,
            "footprint side {side} should be far below the flat model's {flat}"
        );
        assert!(
            fp.bytes(side, side) <= budget,
            "chosen tile exceeds the budget"
        );
        assert!(
            fp.bytes(side + 1, side + 1) > budget,
            "side {side} is not maximal"
        );
    }

    #[test]
    fn square_side_respects_the_linear_and_constant_terms() {
        // A formula with a fat I-side term and a thin pair term: the linear
        // term is what the old sqrt model dropped.
        let x = pts(1000, 64);
        let y = pts(1000, 1);
        let big_i = LazyTensor::vi(&x).exp().sum_last(); // i_only 64 + 1
        let f = &big_i * &LazyTensor::vj(&y);
        let fp = footprint(f.expr()).unwrap();
        assert!(fp.i_only >= 64, "{fp:?}");
        let budget = 1 << 16;
        let s = fp.square_side(budget);
        assert!(
            fp.bytes(s, s) <= budget && fp.bytes(s + 1, s + 1) > budget,
            "{fp:?} s={s}"
        );
    }

    #[test]
    fn batch_extent_multiplies_the_footprint() {
        let x = Tensor::zeros((3, 16, 4), DType::F32, &Device::Cpu).unwrap();
        let y = Tensor::zeros((3, 16, 4), DType::F32, &Device::Cpu).unwrap();
        let f = (&(&LazyTensor::vi(&x) - &LazyTensor::vj(&y)))
            .powf(2.0)
            .sum_last();
        let fp = footprint(f.expr()).unwrap();
        assert_eq!(fp.batch, 3);
        let flat = sq_dist(16, 16, 4);
        let fp1 = footprint(flat.expr()).unwrap();
        assert_eq!(fp.cells(4, 4), 3 * fp1.cells(4, 4));
    }

    #[test]
    fn a_formula_with_no_tile_scaling_takes_any_side() {
        let p = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        let f = LazyTensor::pm(&p).exp();
        let fp = footprint(f.expr()).unwrap();
        assert_eq!(fp.pair, 0);
        assert_eq!(fp.i_only, 0);
        assert_eq!(fp.j_only, 0);
        assert_eq!(fp.square_side(1024), usize::MAX);
    }
}
