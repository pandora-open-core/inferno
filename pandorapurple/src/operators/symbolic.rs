//! Inferno — Lazy pairwise tensor formulas with tiled evaluation and differentiable reductions.
//!
//! Expressions combine row, column and shared parameter leaves. Tiled evaluation
//! bounds temporary storage, and tracked reductions retain Candle gradients.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use candle_core::{DType, Device, Error, Result, Tensor, D};

#[path = "schema.rs"]
mod schema;
pub use schema::{Axis, Expr, Meta, ReduceAxis, broadcast_batch, infer};

/// A symbolic pair-expression. Cheap to clone (one `Arc` bump).
#[derive(Clone)]
pub struct LazyTensor {
    expr: Arc<Expr>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl LazyTensor {
    /// A row variable (`Vi`) from `[..B, M, d]`.
    pub fn vi(t: &Tensor) -> Self {
        Self::leaf(t, Axis::I)
    }
    /// A column variable (`Vj`) from `[..B, N, d]`.
    pub fn vj(t: &Tensor) -> Self {
        Self::leaf(t, Axis::J)
    }
    /// A parameter (`Pm`) from `[..B, 1, d]`. **Learnable
    /// hyperparameters belong here** — that is what keeps them in the graph.
    pub fn pm(t: &Tensor) -> Self {
        Self::leaf(t, Axis::P)
    }
    /// A dense pair leaf from `[..B, M, N]` (see
    /// [`Axis::IJ`]). Pair-scalar: behaves as a dim-1 formula.
    pub fn vij(t: &Tensor) -> Self {
        Self::leaf(t, Axis::IJ)
    }

    fn leaf(t: &Tensor, axis: Axis) -> Self {
        Self { expr: Arc::new(Expr::Leaf { data: t.clone(), axis }) }
    }

    /// A literal. For anything learnable use [`pm`](Self::pm) instead.
    pub fn scalar(v: f64) -> Self {
        Self { expr: Arc::new(Expr::Scalar(v)) }
    }

    /// Symbolic `δ·I` — never allocates an `[M, N]` eye.
    pub fn eye_identity(delta: f64) -> Self {
        Self { expr: Arc::new(Expr::EyeIdentity { delta }) }
    }

    fn wrap(e: Expr) -> Self {
        Self { expr: Arc::new(e) }
    }

    /// Wrap an existing node WITHOUT copying — the seam `genred`'s symbolic
    /// differentiation uses to build gradient formulas that SHARE subtrees
    /// with the primal (e.g. `Exp`'s derivative reuses the primal `Exp`
    /// node, so the per-tile memo evaluates it once).
    pub(crate) fn from_arc(expr: Arc<Expr>) -> Self {
        Self { expr }
    }

    pub fn expr(&self) -> &Arc<Expr> {
        &self.expr
    }

    // -- unary ------------------------------------------------------------

    /// `.sum(-1)` — contract the feature axis.
    pub fn sum_last(&self) -> Self {
        Self::wrap(Expr::Sum(self.expr.clone()))
    }
    pub fn exp(&self) -> Self {
        Self::wrap(Expr::Exp(self.expr.clone()))
    }
    pub fn log(&self) -> Self {
        Self::wrap(Expr::Log(self.expr.clone()))
    }
    pub fn sqrt(&self) -> Self {
        Self::wrap(Expr::Sqrt(self.expr.clone()))
    }
    pub fn recip(&self) -> Self {
        Self::wrap(Expr::Recip(self.expr.clone()))
    }
    pub fn neg(&self) -> Self {
        Self::wrap(Expr::Neg(self.expr.clone()))
    }
    pub fn powf(&self, p: f64) -> Self {
        Self::wrap(Expr::PowF(self.expr.clone(), p))
    }
    /// Tensor exponent — `exp(e · log self)`.
    pub fn pow(&self, e: &Self) -> Self {
        Self::wrap(Expr::Pow(self.expr.clone(), e.expr.clone()))
    }
    pub fn affine(&self, mul: f64, add: f64) -> Self {
        Self::wrap(Expr::Affine { e: self.expr.clone(), mul, add })
    }
    /// Hard floor `max(self, c)` — see [`Expr::MaxC`].
    pub fn max_c(&self, c: f64) -> Self {
        Self::wrap(Expr::MaxC(self.expr.clone(), c))
    }
    /// Indicator `1.0 if self > c else 0.0` — see [`Expr::GtC`].
    pub fn gt_c(&self, c: f64) -> Self {
        Self::wrap(Expr::GtC(self.expr.clone(), c))
    }
    /// `Σ_d selfᵢ[d]·otherⱼ[d]` as one GEMM-backed node — see
    /// [`Expr::MatDot`]. `self` must be I-pure and `other` J-pure; the check
    /// runs in `infer`, not here (builders stay infallible).
    pub fn mat_dot(&self, other: &Self) -> Self {
        Self::wrap(Expr::MatDot(self.expr.clone(), other.expr.clone()))
    }

    /// Deep copy with every leaf `.detach()`ed — used for the untracked
    /// first pass of [`logsumexp_reduce`], so pass 1 allocates no tape.
    pub fn detach_leaves(&self) -> Self {
        Self { expr: detach_expr(&self.expr) }
    }

    /// Every `Leaf` NODE's tensor, in stable depth-first left-to-right
    /// first-visit order, deduplicated by node identity (a DAG-shared node
    /// appears once; two distinct nodes wrapping the SAME tensor appear
    /// twice — deliberately, so that packing the list and routing a
    /// cotangent back through `Tensor::cat`'s VJP sums the two
    /// contributions into the one underlying tensor, which is exactly the
    /// chain rule for a parameter used in two places).
    ///
    /// This reads the leaves used by symbolic reductions; [`with_leaves`](Self::with_leaves) is
    /// the "write" half. The two use the SAME traversal, so index `k` in
    /// one is index `k` in the other.
    pub fn leaves(&self) -> Vec<Tensor> {
        let mut seen: HashMap<*const Expr, ()> = HashMap::new();
        let mut out = Vec::new();
        collect_leaves(&self.expr, &mut seen, &mut out);
        out
    }

    /// The same formula with leaf `k` (in [`leaves`](Self::leaves) order)
    /// replaced by `new[k]`. Shapes and dtypes must match; DAG sharing is
    /// preserved (a node reached twice is substituted once).
    pub fn with_leaves(&self, new: &[Tensor]) -> Result<Self> {
        let expected = self.leaves().len();
        if new.len() != expected {
            return Err(candle_core::Error::Msg(format!(
                "with_leaves: formula has {expected} leaves, got {} replacements",
                new.len()
            )));
        }
        let mut memo: HashMap<*const Expr, Arc<Expr>> = HashMap::new();
        let mut cursor = 0usize;
        let expr = substitute_leaves(&self.expr, new, &mut cursor, &mut memo)?;
        Ok(Self { expr })
    }

    /// Validate and infer shapes. Runs once at the head of every reduction.
    pub fn infer(&self) -> Result<Meta> {
        infer(&self.expr)
    }

    /// CSE + algebraic rewriting (see [`super::rewrite`]): hash-conses the
    /// DAG so structurally equal subtrees share one node (the per-tile memo
    /// then evaluates them once), folds constants, eliminates identities and
    /// fuses affine chains. Value- and shape-metadata-preserving.
    ///
    /// NOTE: may MERGE duplicate leaves (same tensor, same axis), shrinking
    /// [`leaves`](Self::leaves) — take `leaves()` from the simplified
    /// formula, never mix pre- and post-simplify leaf indexing.
    pub fn simplify(&self) -> Result<Self> {
        Ok(Self { expr: super::rewrite::simplify_expr(&self.expr)? })
    }

    /// Evaluate every MAXIMAL tile-invariant operation subtree once and
    /// splice the result back in as a `Pm` leaf.
    ///
    /// A subtree with no `Vi`/`Vj` beneath it is constant across every tile
    /// of a sweep, yet the per-tile memo re-evaluates it per tile (the memo
    /// dies with the tile). Hoisting evaluates it ONCE — with TRACKED ops,
    /// deliberately: the spliced leaf is a computed tensor, so when a
    /// `Genred` packs the hoisted formula's leaves, gradients route from
    /// the pack through the hoist chain to the original parameters via
    /// candle's own tape. Hoisting is also what keeps derivative formulas
    /// small: the VJP differentiates against ONE leaf instead of re-deriving
    /// the whole parameter chain (which would otherwise make every
    /// backward formula re-embeds the full hyperparameter mixing).
    ///
    /// NOT pure (it evaluates tensors), which is why it is a separate pass
    /// from [`simplify`](Self::simplify) and is invoked by evaluation entry
    /// points rather than folded into rewriting.
    pub fn hoist_invariant(&self) -> Result<Self> {
        let mut memo = HashMap::new();
        Ok(Self { expr: hoist_expr(&self.expr, &mut memo)? })
    }

    /// Rewrite squared distances into their GEMM-backed expansion, then
    /// re-intern — see [`super::cost_model`].
    ///
    /// Kept OUT of [`simplify`](Self::simplify) because it changes values
    /// (catastrophic cancellation near `a ≈ b`); the error analysis and the
    /// clamp that bounds it are documented on the `cost_model` module.
    pub fn factor(&self, policy: super::cost_model::FactorPolicy) -> Result<Self> {
        let e = super::cost_model::factor_sq_dist(&self.expr, policy)?;
        Self { expr: e }.simplify()
    }

    /// [`factor`](Self::factor) under the process-wide policy: `Auto` when
    /// [`crate::settings::gemm_factoring_enabled`] is set, else `Never`.
    /// This is what production evaluation paths call, so flipping the
    /// setting reaches every consumer at once.
    pub fn factor_by_settings(&self) -> Result<Self> {
        use super::cost_model::FactorPolicy;
        let p = if crate::settings::gemm_factoring_enabled::value() {
            FactorPolicy::Auto
        } else {
            FactorPolicy::Never
        };
        if p == FactorPolicy::Never {
            return Ok(self.clone());
        }
        self.factor(p)
    }

    /// Number of DISTINCT nodes in the DAG (ptr-deduped) — the quantity CSE
    /// shrinks and the per-tile memo pays for.
    pub fn node_count(&self) -> usize {
        fn walk(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>) {
            if seen.insert(Arc::as_ptr(e), ()).is_some() {
                return;
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
                | Expr::Affine { e: a, .. } => walk(a, seen),
                Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Div(a, b)
                | Expr::Pow(a, b)
                | Expr::MatDot(a, b) => {
                    walk(a, seen);
                    walk(b, seen);
                }
            }
        }
        let mut seen = HashMap::new();
        walk(&self.expr, &mut seen);
        seen.len()
    }
}

fn collect_leaves(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, out: &mut Vec<Tensor>) {
    if seen.insert(Arc::as_ptr(e), ()).is_some() {
        return; // DAG-shared node: already visited
    }
    match e.as_ref() {
        Expr::Leaf { data, .. } => out.push(data.clone()),
        Expr::Scalar(_) | Expr::EyeIdentity { .. } => {}
        Expr::Neg(a)
        | Expr::Exp(a)
        | Expr::Log(a)
        | Expr::Sqrt(a)
        | Expr::Recip(a)
        | Expr::Sum(a)
        | Expr::PowF(a, _)
        | Expr::MaxC(a, _)
        | Expr::GtC(a, _)
        | Expr::Affine { e: a, .. } => collect_leaves(a, seen, out),
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::MatDot(a, b) => {
            collect_leaves(a, seen, out);
            collect_leaves(b, seen, out);
        }
    }
}

fn substitute_leaves(
    e: &Arc<Expr>,
    new: &[Tensor],
    cursor: &mut usize,
    memo: &mut HashMap<*const Expr, Arc<Expr>>,
) -> Result<Arc<Expr>> {
    if let Some(done) = memo.get(&Arc::as_ptr(e)) {
        return Ok(done.clone());
    }
    let rebuilt = match e.as_ref() {
        Expr::Leaf { data, axis } => {
            let r = &new[*cursor];
            *cursor += 1;
            if r.dims() != data.dims() || r.dtype() != data.dtype() {
                return Err(candle_core::Error::Msg(format!(
                    "with_leaves: leaf {} expects {:?} {:?}, got {:?} {:?}",
                    *cursor - 1,
                    data.dims(),
                    data.dtype(),
                    r.dims(),
                    r.dtype()
                )));
            }
            Arc::new(Expr::Leaf { data: r.clone(), axis: *axis })
        }
        Expr::Scalar(v) => Arc::new(Expr::Scalar(*v)),
        Expr::EyeIdentity { delta } => Arc::new(Expr::EyeIdentity { delta: *delta }),
        Expr::Neg(a) => Arc::new(Expr::Neg(substitute_leaves(a, new, cursor, memo)?)),
        Expr::Exp(a) => Arc::new(Expr::Exp(substitute_leaves(a, new, cursor, memo)?)),
        Expr::Log(a) => Arc::new(Expr::Log(substitute_leaves(a, new, cursor, memo)?)),
        Expr::Sqrt(a) => Arc::new(Expr::Sqrt(substitute_leaves(a, new, cursor, memo)?)),
        Expr::Recip(a) => Arc::new(Expr::Recip(substitute_leaves(a, new, cursor, memo)?)),
        Expr::Sum(a) => Arc::new(Expr::Sum(substitute_leaves(a, new, cursor, memo)?)),
        Expr::PowF(a, p) => Arc::new(Expr::PowF(substitute_leaves(a, new, cursor, memo)?, *p)),
        Expr::MaxC(a, c) => Arc::new(Expr::MaxC(substitute_leaves(a, new, cursor, memo)?, *c)),
        Expr::GtC(a, c) => Arc::new(Expr::GtC(substitute_leaves(a, new, cursor, memo)?, *c)),
        Expr::Affine { e: a, mul, add } => Arc::new(Expr::Affine {
            e: substitute_leaves(a, new, cursor, memo)?,
            mul: *mul,
            add: *add,
        }),
        Expr::Add(a, b) => Arc::new(Expr::Add(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
        Expr::Sub(a, b) => Arc::new(Expr::Sub(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
        Expr::Mul(a, b) => Arc::new(Expr::Mul(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
        Expr::Div(a, b) => Arc::new(Expr::Div(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
        Expr::Pow(a, b) => Arc::new(Expr::Pow(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
        Expr::MatDot(a, b) => Arc::new(Expr::MatDot(
            substitute_leaves(a, new, cursor, memo)?,
            substitute_leaves(b, new, cursor, memo)?,
        )),
    };
    memo.insert(Arc::as_ptr(e), rebuilt.clone());
    Ok(rebuilt)
}

fn detach_expr(e: &Arc<Expr>) -> Arc<Expr> {
    let mut memo = HashMap::new();
    detach_expr_memo(e, &mut memo)
}

/// Hoist several formulas through ONE memo, so an invariant subtree shared
/// across formulas is evaluated once and spliced as the SAME leaf node —
/// which keeps the fused-sweep pack deduplication and the per-tile memo
/// sharing intact.
pub(crate) fn hoist_many(exprs: &[Arc<Expr>]) -> Result<Vec<Arc<Expr>>> {
    let mut memo = HashMap::new();
    exprs.iter().map(|e| hoist_expr(e, &mut memo)).collect()
}

fn hoist_expr(
    e: &Arc<Expr>,
    memo: &mut HashMap<*const Expr, Arc<Expr>>,
) -> Result<Arc<Expr>> {
    if let Some(done) = memo.get(&Arc::as_ptr(e)) {
        return Ok(done.clone());
    }
    // Leaves and literals carry no per-tile work — nothing to hoist. A
    // malformed subtree keeps its shape so the evaluator reports it.
    let is_op = !matches!(e.as_ref(), Expr::Leaf { .. } | Expr::Scalar(_));
    let out = match (is_op, infer(e)) {
        (true, Ok(m)) if !m.uses_i && !m.uses_j => {
            // Tile-invariant operation subtree: evaluate ONCE on a 1×1 tile
            // (tracked — see hoist_invariant's doc) and splice as a Pm leaf.
            let mut tile_memo: Memo = HashMap::new();
            let t = eval_block(e, &m, &(0..1), &(0..1), &mut tile_memo)?;
            let t = t.squeeze(t.rank() - 2)?; // [.., 1(i), 1(j), d] → [.., 1, d]
            Arc::new(Expr::Leaf { data: t, axis: Axis::P })
        }
        _ => match e.as_ref() {
            Expr::Leaf { .. } | Expr::Scalar(_) | Expr::EyeIdentity { .. } => e.clone(),
            Expr::Neg(a) => Arc::new(Expr::Neg(hoist_expr(a, memo)?)),
            Expr::Exp(a) => Arc::new(Expr::Exp(hoist_expr(a, memo)?)),
            Expr::Log(a) => Arc::new(Expr::Log(hoist_expr(a, memo)?)),
            Expr::Sqrt(a) => Arc::new(Expr::Sqrt(hoist_expr(a, memo)?)),
            Expr::Recip(a) => Arc::new(Expr::Recip(hoist_expr(a, memo)?)),
            Expr::Sum(a) => Arc::new(Expr::Sum(hoist_expr(a, memo)?)),
            Expr::PowF(a, p) => Arc::new(Expr::PowF(hoist_expr(a, memo)?, *p)),
            Expr::MaxC(a, c) => Arc::new(Expr::MaxC(hoist_expr(a, memo)?, *c)),
            Expr::GtC(a, c) => Arc::new(Expr::GtC(hoist_expr(a, memo)?, *c)),
            Expr::Affine { e: a, mul, add } => {
                Arc::new(Expr::Affine { e: hoist_expr(a, memo)?, mul: *mul, add: *add })
            }
            Expr::Add(a, b) => {
                Arc::new(Expr::Add(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
            Expr::Sub(a, b) => {
                Arc::new(Expr::Sub(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
            Expr::Mul(a, b) => {
                Arc::new(Expr::Mul(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
            Expr::Div(a, b) => {
                Arc::new(Expr::Div(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
            Expr::Pow(a, b) => {
                Arc::new(Expr::Pow(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
            Expr::MatDot(a, b) => {
                Arc::new(Expr::MatDot(hoist_expr(a, memo)?, hoist_expr(b, memo)?))
            }
        },
    };
    memo.insert(Arc::as_ptr(e), out.clone());
    Ok(out)
}

/// Detach several formulas through ONE memo, so DAG sharing survives —
/// within a formula (`&d * &d` stays one subtree) and ACROSS formulas
/// (fused reductions rely on the shared cost subtree staying one `Arc`, or
/// the per-tile memo cannot deduplicate its evaluation).
pub(crate) fn detach_many(exprs: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
    let mut memo = HashMap::new();
    exprs.iter().map(|e| detach_expr_memo(e, &mut memo)).collect()
}

fn detach_expr_memo(e: &Arc<Expr>, memo: &mut HashMap<*const Expr, Arc<Expr>>) -> Arc<Expr> {
    if let Some(done) = memo.get(&Arc::as_ptr(e)) {
        return done.clone();
    }
    let rebuilt = match e.as_ref() {
        Expr::Leaf { data, axis } => Expr::Leaf { data: data.detach(), axis: *axis },
        Expr::Scalar(v) => Expr::Scalar(*v),
        Expr::EyeIdentity { delta } => Expr::EyeIdentity { delta: *delta },
        Expr::Neg(a) => Expr::Neg(detach_expr_memo(a, memo)),
        Expr::Exp(a) => Expr::Exp(detach_expr_memo(a, memo)),
        Expr::Log(a) => Expr::Log(detach_expr_memo(a, memo)),
        Expr::Sqrt(a) => Expr::Sqrt(detach_expr_memo(a, memo)),
        Expr::Recip(a) => Expr::Recip(detach_expr_memo(a, memo)),
        Expr::Sum(a) => Expr::Sum(detach_expr_memo(a, memo)),
        Expr::PowF(a, p) => Expr::PowF(detach_expr_memo(a, memo), *p),
        Expr::MaxC(a, c) => Expr::MaxC(detach_expr_memo(a, memo), *c),
        Expr::GtC(a, c) => Expr::GtC(detach_expr_memo(a, memo), *c),
        Expr::Affine { e, mul, add } => {
            Expr::Affine { e: detach_expr_memo(e, memo), mul: *mul, add: *add }
        }
        Expr::Add(a, b) => Expr::Add(detach_expr_memo(a, memo), detach_expr_memo(b, memo)),
        Expr::Sub(a, b) => Expr::Sub(detach_expr_memo(a, memo), detach_expr_memo(b, memo)),
        Expr::Mul(a, b) => Expr::Mul(detach_expr_memo(a, memo), detach_expr_memo(b, memo)),
        Expr::Div(a, b) => Expr::Div(detach_expr_memo(a, memo), detach_expr_memo(b, memo)),
        Expr::Pow(a, b) => Expr::Pow(detach_expr_memo(a, memo), detach_expr_memo(b, memo)),
        Expr::MatDot(a, b) => {
            Expr::MatDot(detach_expr_memo(a, memo), detach_expr_memo(b, memo))
        }
    };
    let out = Arc::new(rebuilt);
    memo.insert(Arc::as_ptr(e), out.clone());
    out
}

// -- operator overloads ------------------------------------------------------
//
// Builders are infallible so these can exist; every shape error surfaces from
// `infer()` at reduction time, naming the node and both dims.

macro_rules! bin_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl std::ops::$trait<&LazyTensor> for &LazyTensor {
            type Output = LazyTensor;
            fn $method(self, rhs: &LazyTensor) -> LazyTensor {
                LazyTensor::wrap(Expr::$variant(self.expr.clone(), rhs.expr.clone()))
            }
        }
        impl std::ops::$trait<LazyTensor> for LazyTensor {
            type Output = LazyTensor;
            fn $method(self, rhs: LazyTensor) -> LazyTensor {
                LazyTensor::wrap(Expr::$variant(self.expr, rhs.expr))
            }
        }
        impl std::ops::$trait<&LazyTensor> for LazyTensor {
            type Output = LazyTensor;
            fn $method(self, rhs: &LazyTensor) -> LazyTensor {
                LazyTensor::wrap(Expr::$variant(self.expr, rhs.expr.clone()))
            }
        }
        impl std::ops::$trait<f64> for &LazyTensor {
            type Output = LazyTensor;
            fn $method(self, rhs: f64) -> LazyTensor {
                LazyTensor::wrap(Expr::$variant(
                    self.expr.clone(),
                    Arc::new(Expr::Scalar(rhs)),
                ))
            }
        }
    };
}
bin_op!(Add, add, Add);
bin_op!(Sub, sub, Sub);
bin_op!(Mul, mul, Mul);
bin_op!(Div, div, Div);

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tiling policy
// ---------------------------------------------------------------------------

/// Tile-size policy. Budget-driven rather than a fixed tile: at the sizes
/// this crate actually runs (`K_ATOMS = 30`, `D = 32` — a 115 KB tile),
/// tiling must NOT fire, or it is pure overhead on top of a matrix that fits
/// comfortably.
#[derive(Clone, Debug)]
pub struct BlockCfg {
    /// Working-set budget in bytes for one tile.
    ///
    /// Defaults to the machine's usable L2 rather than to a constant — see
    /// [`Default`](BlockCfg::default) for why that level and not L1.
    pub budget_bytes: usize,
    /// Explicit row-tile override. Tests only.
    pub bi: Option<usize>,
    /// Explicit column-tile override. Tests only.
    pub bj: Option<usize>,
    /// Force one-shot materialisation — the oracle path.
    pub force_dense: bool,
}

impl Default for BlockCfg {
    /// # The budget is the machine's, not a constant
    ///
    /// This was `8 << 20` — a number with no provenance, and one that happens
    /// to be larger than the L2 of every machine in the fleet and smaller than
    /// their L3. A tile sized to it therefore "fits the budget" while
    /// streaming from the last level on every pass, which is the same as
    /// having no tiling policy at all.
    ///
    /// L2 and not L1 because of what this budget covers: the per-tile memo
    /// holds EVERY node's tile live at once, not just the root's, so the
    /// working set here is a formula's worth of tiles rather than one. The
    /// register-level blocking that targets L1 lives inside the kernels
    /// (`dk_tensor::pairwise`), one level down, where the working set really
    /// is a single tile.
    ///
    /// At the sizes this crate actually runs — `K_ATOMS = 30`, `D = 32`, a
    /// 115 KB tile — this still does not tile, which is the behaviour the
    /// module doc above asks for and the reason the change is safe.
    fn default() -> Self {
        Self {
            budget_bytes: dk_tensor::cache_model::CacheBudget::global().usable_l2(),
            bi: None,
            bj: None,
            force_dense: false,
        }
    }
}

impl BlockCfg {
    pub fn with_tiles(bi: usize, bj: usize) -> Self {
        Self { bi: Some(bi.max(1)), bj: Some(bj.max(1)), ..Default::default() }
    }

    /// Chosen `(bi, bj)` for a specific FORMULA — the production policy.
    ///
    /// Sizes the tile from the formula's measured working set
    /// ([`super::schedule::footprint`]) rather than from the root's feature
    /// dim, because the per-tile memo holds every node's tile live at once.
    /// Falls back to [`tiles`](Self::tiles) if the formula cannot be
    /// measured — the caller is about to `infer` it anyway and will report
    /// the real error.
    pub(crate) fn tiles_for(&self, e: &Arc<Expr>, m: usize, n: usize, meta: &Meta) -> (usize, usize) {
        if let (Some(bi), Some(bj)) = (self.bi, self.bj) {
            return (bi.min(m).max(1), bj.min(n).max(1));
        }
        let Ok(fp) = super::schedule::footprint(e) else {
            return self.tiles(m, n, meta.dim, meta.dtype);
        };
        if let Some(t) = super::schedule::tile_tuner() {
            if let Some((bi, bj)) = t.tiles(&fp, m, n, self.budget_bytes) {
                return (bi.clamp(1, m.max(1)), bj.clamp(1, n.max(1)));
            }
        }
        if self.force_dense || fp.bytes(m.max(1), n.max(1)) <= self.budget_bytes {
            return (m.max(1), n.max(1));
        }
        let side = fp.square_side(self.budget_bytes);
        (side.min(m).max(1), side.min(n).max(1))
    }

    /// Chosen `(bi, bj)` from a flat `dim` estimate.
    ///
    /// Kept for callers that have no formula in hand (and as the fallback
    /// inside [`tiles_for`](Self::tiles_for)). It under-counts any formula
    /// with dim-`d` intermediates — prefer `tiles_for`.
    pub(crate) fn tiles(&self, m: usize, n: usize, dim: usize, dtype: DType) -> (usize, usize) {
        if let (Some(bi), Some(bj)) = (self.bi, self.bj) {
            return (bi.min(m).max(1), bj.min(n).max(1));
        }
        let esz = dtype.size_in_bytes().max(1);
        let cell = dim.max(1) * esz;
        if self.force_dense || m.saturating_mul(n).saturating_mul(cell) <= self.budget_bytes {
            return (m.max(1), n.max(1));
        }
        let side = ((self.budget_bytes / cell) as f64).sqrt() as usize;
        let side = side.clamp(1, m.max(n).max(1));
        (side.min(m).max(1), side.min(n).max(1))
    }
}

// ---------------------------------------------------------------------------
// Block evaluation
// ---------------------------------------------------------------------------

/// Per-tile memo. Keyed on node identity so a DAG evaluates as a DAG.
///
/// Threaded as `&mut` rather than living behind a `RefCell` on the node: with
/// interior mutability `Expr` stops being `Sync`, and the `KernelBackend` /
/// `KernelEval` bounds then break at a distance with an unhelpful error.
pub(crate) type Memo = HashMap<*const Expr, Tensor>;

/// Evaluate one `(i, j)` tile → `[..B, bi, bj, dim]`.
pub fn eval_block(
    e: &Arc<Expr>,
    meta: &Meta,
    i: &Range<usize>,
    j: &Range<usize>,
    memo: &mut Memo,
) -> Result<Tensor> {
    let key = Arc::as_ptr(e);
    if let Some(t) = memo.get(&key) {
        return Ok(t.clone());
    }
    let out = eval_block_uncached(e, meta, i, j, memo)?;
    memo.insert(key, out.clone());
    Ok(out)
}

fn eval_block_uncached(
    e: &Arc<Expr>,
    meta: &Meta,
    i: &Range<usize>,
    j: &Range<usize>,
    memo: &mut Memo,
) -> Result<Tensor> {
    match e.as_ref() {
        Expr::Leaf { data, axis } => {
            let rank = data.rank();
            let row = rank - 2;
            match axis {
                // [..B, M, d] → [..B, bi, d] → [..B, bi, 1, d]
                Axis::I => data.narrow(row, i.start, i.len())?.unsqueeze(rank - 1),
                // [..B, N, d] → [..B, bj, d] → [..B, 1, bj, d]
                Axis::J => data.narrow(row, j.start, j.len())?.unsqueeze(row),
                // [..B, 1, d] → [..B, 1, 1, d]
                Axis::P => data.unsqueeze(row),
                // Dense pair [..B, M, N] → [..B, bi, bj] → [..B, bi, bj, 1]
                Axis::IJ => data
                    .narrow(rank - 2, i.start, i.len())?
                    .narrow(rank - 1, j.start, j.len())?
                    .unsqueeze(rank)
                    .and_then(|t| t.contiguous()),
            }
        }
        Expr::Scalar(v) => {
            let shape = tile_shape(meta, 1, 1, 1);
            Tensor::full(*v, shape, &meta.device)?.to_dtype(meta.dtype)
        }
        Expr::EyeIdentity { delta } => eye_tile(meta, i, j, *delta),
        Expr::Neg(a) => eval_block(a, meta, i, j, memo)?.neg(),
        Expr::Exp(a) => eval_block(a, meta, i, j, memo)?.exp(),
        Expr::Log(a) => eval_block(a, meta, i, j, memo)?.log(),
        Expr::Sqrt(a) => eval_block(a, meta, i, j, memo)?.sqrt(),
        Expr::Recip(a) => eval_block(a, meta, i, j, memo)?.recip(),
        Expr::PowF(a, p) => eval_block(a, meta, i, j, memo)?.powf(*p),
        Expr::Affine { e, mul, add } => eval_block(e, meta, i, j, memo)?.affine(*mul, *add),
        Expr::Sum(a) => eval_block(a, meta, i, j, memo)?.sum_keepdim(D::Minus1),
        Expr::MaxC(a, c) => {
            let t = eval_block(a, meta, i, j, memo)?;
            // Bit-parity with the dense idiom `t.maximum(t.affine(0, c))`.
            let floor = t.affine(0.0, *c)?;
            t.maximum(&floor)
        }
        Expr::GtC(a, c) => {
            let t = eval_block(a, meta, i, j, memo)?;
            let thresh = t.affine(0.0, *c)?;
            t.gt(&thresh)?.to_dtype(t.dtype())
        }
        Expr::Add(a, b) => {
            let (x, y) = (eval_block(a, meta, i, j, memo)?, eval_block(b, meta, i, j, memo)?);
            x.broadcast_add(&y)
        }
        Expr::Sub(a, b) => {
            let (x, y) = (eval_block(a, meta, i, j, memo)?, eval_block(b, meta, i, j, memo)?);
            x.broadcast_sub(&y)
        }
        Expr::Mul(a, b) => {
            let (x, y) = (eval_block(a, meta, i, j, memo)?, eval_block(b, meta, i, j, memo)?);
            x.broadcast_mul(&y)
        }
        Expr::Div(a, b) => {
            let (x, y) = (eval_block(a, meta, i, j, memo)?, eval_block(b, meta, i, j, memo)?);
            x.broadcast_div(&y)
        }
        // aᵇ = exp(b · log a). The only lowering that is not bit-exact
        // against a `.powf()` reference — see the tolerance note in tests.
        Expr::Pow(a, b) => {
            let x = eval_block(a, meta, i, j, memo)?.log()?;
            let y = eval_block(b, meta, i, j, memo)?;
            x.broadcast_mul(&y)?.exp()
        }
        // The whole point: contract `d` with BLAS instead of building the
        // `[bi, bj, d]` product first. The operands arrive from `eval_block`
        // already shaped for the pair grid — `[..B, bi, 1, d]` and
        // `[..B, 1, bj, d]` — so dropping each one's singleton pair axis
        // recovers the un-paired matrices the GEMM wants.
        Expr::MatDot(a, b) => {
            let ta = eval_block(a, meta, i, j, memo)?;
            let tb = eval_block(b, meta, i, j, memo)?;
            let rank = ta.rank();
            let left = ta.squeeze(rank - 2)?; // [..B, bi, d]
            let right = tb.squeeze(rank - 3)?; // [..B, bj, d]
            // Lift both to the formula's batch shape before the matmul:
            // `broadcast_matmul` broadcasts leading dims, but only when the
            // ranks already agree, and an operand may carry fewer.
            let left = broadcast_to_batch(&left, meta, i.len())?;
            let right = broadcast_to_batch(&right, meta, j.len())?;
            left.contiguous()?
                .matmul(&right.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
                .unsqueeze(D::Minus1)
        }
    }
}

/// Expand a `[..b, s, d]` operand to the formula's full batch shape, so two
/// operands with different (broadcastable) batch ranks can meet in a matmul.
fn broadcast_to_batch(t: &Tensor, meta: &Meta, s: usize) -> Result<Tensor> {
    let d = t.dims()[t.rank() - 1];
    let mut want = meta.batch.clone();
    want.push(s);
    want.push(d);
    if t.dims() == want.as_slice() {
        return Ok(t.clone());
    }
    t.broadcast_as(want)
}

fn tile_shape(meta: &Meta, bi: usize, bj: usize, dim: usize) -> Vec<usize> {
    let mut s = meta.batch.clone();
    s.push(bi);
    s.push(bj);
    s.push(dim);
    s
}

/// The `δ·I` tile.
///
/// Two shortcuts the Python's `exp(−1e9·(dᵢ−dⱼ)²)` spelling cannot take:
/// a tile that misses the diagonal is constant-folded to zeros, and the
/// overlapping case is written as an exact 0/1 pattern rather than relying
/// on f32 underflow.
///
/// The index compared is the GLOBAL one (`i.start + local`). Using the
/// tile-local index instead silently puts an identity on every diagonal
/// tile's own diagonal, which is the single easiest bug to introduce here.
fn eye_tile(meta: &Meta, i: &Range<usize>, j: &Range<usize>, delta: f64) -> Result<Tensor> {
    let shape = tile_shape(meta, i.len(), j.len(), 1);
    if i.end <= j.start || j.end <= i.start {
        return Ok(Tensor::zeros(shape, meta.dtype, &meta.device)?);
    }
    // Built in f64 and narrowed by `to_dtype`, NOT in f32. `delta as f32`
    // quantises the jitter before the formula's dtype gets a say, so an F64
    // formula saw 0.0010000000474974513 instead of 1e-3 — 4.7e-11 of error
    // sitting on the diagonal, which is exactly where a generated kernel
    // emitting an exact `(i == j) ? delta : 0` would appear to disagree with
    // the candle path. F32 formulas are unaffected: rounding 1e-3 to f32 once
    // gives the same bits either way.
    let mut v = vec![0f64; i.len() * j.len()];
    for (r, gi) in i.clone().enumerate() {
        if gi >= j.start && gi < j.end {
            v[r * j.len() + (gi - j.start)] = delta;
        }
    }
    Tensor::from_vec(v, (i.len(), j.len(), 1), &meta.device)?
        .to_dtype(meta.dtype)?
        .broadcast_as(shape)
}

// ---------------------------------------------------------------------------
// Reductions — the only things that actually evaluate
// ---------------------------------------------------------------------------

impl LazyTensor {
    fn resolved(&self) -> Result<(Meta, usize, usize)> {
        let meta = self.infer()?;
        let m = meta.m.ok_or_else(|| {
            Error::Msg("symbolic: formula has no Vi leaf, so the row count is unknown".into())
        })?;
        let n = meta.n.ok_or_else(|| {
            Error::Msg("symbolic: formula has no Vj leaf, so the column count is unknown".into())
        })?;
        Ok((meta, m, n))
    }

    /// Evaluate one tile and drop the trailing singleton — `[..B, bi, bj]`.
    fn eval_scalar_tile(
        &self,
        meta: &Meta,
        i: &Range<usize>,
        j: &Range<usize>,
    ) -> Result<Tensor> {
        let mut memo = Memo::new();
        let t = eval_block(&self.expr, meta, i, j, &mut memo)?;
        if meta.dim != 1 {
            return Err(Error::Msg(format!(
                "symbolic: reduction needs a pair-scalar formula (dim 1), got dim {} — \
                 did you forget .sum_last()?",
                meta.dim
            )));
        }
        t.squeeze(D::Minus1)
    }

    /// Materialise the full `[..B, M, N]` matrix — the escape hatch, and the
    /// oracle the blocked paths are tested against.
    pub fn to_dense(&self, cfg: &BlockCfg) -> Result<Tensor> {
        let (meta, m, n) = self.resolved()?;
        let (bi, bj) = cfg.tiles_for(&self.expr, m, n, &meta);

        let mut rows = Vec::new();
        let mut i0 = 0;
        while i0 < m {
            let i = i0..(i0 + bi).min(m);
            let mut cols = Vec::new();
            let mut j0 = 0;
            while j0 < n {
                let j = j0..(j0 + bj).min(n);
                cols.push(self.eval_scalar_tile(&meta, &i, &j)?);
                j0 = j.end;
            }
            let refs: Vec<&Tensor> = cols.iter().collect();
            rows.push(Tensor::cat(&refs, D::Minus1)?);
            i0 = i.end;
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, D::Minus2)
    }

    /// Plain sum over `axis`.
    pub fn sum_reduce(&self, axis: ReduceAxis, cfg: &BlockCfg) -> Result<Tensor> {
        let (meta, m, n) = self.resolved()?;
        let (bi, bj) = cfg.tiles_for(&self.expr, m, n, &meta);
        match axis {
            ReduceAxis::J => {
                let mut rows = Vec::new();
                let mut i0 = 0;
                while i0 < m {
                    let i = i0..(i0 + bi).min(m);
                    let mut parts = Vec::new();
                    let mut j0 = 0;
                    while j0 < n {
                        let j = j0..(j0 + bj).min(n);
                        parts.push(self.eval_scalar_tile(&meta, &i, &j)?.sum(D::Minus1)?);
                        j0 = j.end;
                    }
                    rows.push(stack_sum(&parts)?);
                    i0 = i.end;
                }
                let refs: Vec<&Tensor> = rows.iter().collect();
                Tensor::cat(&refs, D::Minus1)
            }
            ReduceAxis::I => {
                let mut cols = Vec::new();
                let mut j0 = 0;
                while j0 < n {
                    let j = j0..(j0 + bj).min(n);
                    let mut parts = Vec::new();
                    let mut i0 = 0;
                    while i0 < m {
                        let i = i0..(i0 + bi).min(m);
                        parts.push(self.eval_scalar_tile(&meta, &i, &j)?.sum(D::Minus2)?);
                        i0 = i.end;
                    }
                    cols.push(stack_sum(&parts)?);
                    j0 = j.end;
                }
                let refs: Vec<&Tensor> = cols.iter().collect();
                Tensor::cat(&refs, D::Minus1)
            }
        }
    }

    /// `log Σ exp(·)` over `axis`, without materialising the pair matrix.
    ///
    /// Two passes. Pass 1 takes the max from a leaf-detached copy of the
    /// formula, so it allocates no tape; pass 2 accumulates
    /// `(tile − m).exp().sum()` tracked.
    ///
    /// **Detaching the max is exact, not an approximation.**
    /// `LSE(a) = m + log Σ exp(a − m)` holds for *any* `m`, so `∂out/∂m ≡ 0`
    /// analytically — this is the same argument `softmin_tracked` already
    /// relies on. It is a different thing from `barycenter.rs`'s truncated
    /// implicit differentiation, which really does drop a series tail.
    ///
    /// Accumulation is `stack + sum`, never a chained `add`: chaining would
    /// make the backward graph `O(n_blocks)` deep, which is exactly how
    /// `sinkhorn_coupling_t` overflows the stack.
    pub fn logsumexp_reduce(&self, axis: ReduceAxis, cfg: &BlockCfg) -> Result<Tensor> {
        let (meta, m, n) = self.resolved()?;



        let (bi, bj) = cfg.tiles_for(&self.expr, m, n, &meta);
        let detached = self.detach_leaves();
        let red = match axis {
            ReduceAxis::J => D::Minus1,
            ReduceAxis::I => D::Minus2,
        };

        // Iterate the KEPT axis outermost; reduce over the other.
        let (kept_n, kept_b, other_n, other_b) = match axis {
            ReduceAxis::J => (m, bi, n, bj),
            ReduceAxis::I => (n, bj, m, bi),
        };

        let mut out = Vec::new();
        let mut k0 = 0;
        while k0 < kept_n {
            let kr = k0..(k0 + kept_b).min(kept_n);
            let ranges: Vec<Range<usize>> = {
                let mut v = Vec::new();
                let mut o0 = 0;
                while o0 < other_n {
                    let r = o0..(o0 + other_b).min(other_n);
                    o0 = r.end;
                    v.push(r);
                }
                v
            };
            let pick = |o: &Range<usize>| -> (Range<usize>, Range<usize>) {
                match axis {
                    ReduceAxis::J => (kr.clone(), o.clone()),
                    ReduceAxis::I => (o.clone(), kr.clone()),
                }
            };

            // -- pass 1: running max, fully detached --------------------
            let mut mx: Option<Tensor> = None;
            for o in &ranges {
                let (i, j) = pick(o);
                let t = detached.eval_scalar_tile(&meta, &i, &j)?.max_keepdim(red)?;
                mx = Some(match mx {
                    Some(prev) => prev.maximum(&t)?,
                    None => t,
                });
            }
            let mx = mx.ok_or_else(|| Error::Msg("symbolic: empty reduction axis".into()))?;

            // -- pass 2: tracked accumulation ---------------------------
            let mut parts = Vec::with_capacity(ranges.len());
            for o in &ranges {
                let (i, j) = pick(o);
                let t = self.eval_scalar_tile(&meta, &i, &j)?;
                parts.push(t.broadcast_sub(&mx)?.exp()?.sum(red)?);
            }
            let total = stack_sum(&parts)?;
            out.push(total.log()?.add(&mx.squeeze(red)?)?);
            k0 = kr.end;
        }
        let refs: Vec<&Tensor> = out.iter().collect();
        Tensor::cat(&refs, D::Minus1)
    }

    /// `K @ rhs` in row tiles. Reuses the same block discipline as
    /// [`LazyKernelOperator::matmul`](super::LazyKernelOperator) so BLAS
    /// still does the inner product.
    pub fn matmul(&self, rhs: &Tensor, cfg: &BlockCfg) -> Result<Tensor> {
        let (meta, m, n) = self.resolved()?;
        let (bi, _) = cfg.tiles_for(&self.expr, m, n, &meta);
        let full_j = 0..n;
        let mut blocks = Vec::new();
        let mut i0 = 0;
        while i0 < m {
            let i = i0..(i0 + bi).min(m);
            let kb = self.eval_scalar_tile(&meta, &i, &full_j)?; // [..B, bi, N]
            blocks.push(kb.broadcast_matmul(rhs)?);
            i0 = i.end;
        }
        let refs: Vec<&Tensor> = blocks.iter().collect();
        Tensor::cat(&refs, D::Minus2)
    }

    /// `diag(K)` — evaluates only the tiles the diagonal passes through.
    pub fn diagonal(&self, cfg: &BlockCfg) -> Result<Tensor> {
        let (meta, m, n) = self.resolved()?;
        let k = m.min(n);
        let (bi, _) = cfg.tiles_for(&self.expr, m, n, &meta);
        let mut parts = Vec::new();
        let mut i0 = 0;
        while i0 < k {
            let r = i0..(i0 + bi).min(k);
            // Square tile straddling the diagonal, then its own diagonal.
            let tile = self.eval_scalar_tile(&meta, &r, &r)?;
            parts.push(super::dense_diagonal(&tile)?);
            i0 = r.end;
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, D::Minus1)
    }
}

/// Sum a list of partial results with `O(1)` backward depth.
///
/// The chained alternative (`acc = acc.add(&p)`) is `O(len)` deep, which is
/// the precise mechanism by which `sinkhorn_coupling_t`'s 400-iteration
/// unroll overflows a 2 MiB stack on `backward()`.
pub(crate) fn stack_sum(parts: &[Tensor]) -> Result<Tensor> {
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::stack(&refs, 0)?.sum(0)
}

/// Compile-time proof that a formula can cross the `Send + Sync` bounds the
/// kernel seams require. If someone reaches for `Rc`/`RefCell` inside
/// [`Expr`], this stops compiling here rather than at a distant call site.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<Expr>>();
    assert_send_sync::<LazyTensor>();
};

#[cfg(test)]
mod tests {
    //! Verification of lazy pairwise tensor formulas with tiled evaluation and differentiable reductions.

    use super::*;
    use candle_core::Var;

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

    fn one_shot() -> BlockCfg {
        BlockCfg { force_dense: true, ..Default::default() }
    }

    // ---- tile-invariant hoisting ----------------------------------------

    #[test]
    fn hoisting_collapses_pm_chains_and_preserves_values() {
        let dev = candle_core::Device::Cpu;
        let x = Tensor::randn(0f64, 1f64, (6, 2), &dev).unwrap();
        let a = Tensor::full(0.4f64, (1, 1), &dev).unwrap();
        let b = Tensor::full(1.5f64, (1, 1), &dev).unwrap();
        let c = Tensor::full(0.2f64, (1, 1), &dev).unwrap();
        // γ = a·b + c is tile-invariant; the exp(−d²·γ) around it is not.
        let gamma = &(&LazyTensor::pm(&a) * &LazyTensor::pm(&b)) + &LazyTensor::pm(&c);
        let diff = &LazyTensor::vi(&x) - &LazyTensor::vj(&x);
        let k = (&(&diff * &diff).sum_last() * &gamma).affine(-1.0, 0.0).exp();

        let hoisted = k.hoist_invariant().unwrap();
        assert!(
            hoisted.node_count() < k.node_count(),
            "no nodes hoisted: {} vs {}",
            hoisted.node_count(),
            k.node_count()
        );
        // The a·b+c chain (3 leaves) becomes ONE computed Pm leaf.
        assert_eq!(k.leaves().len(), 5, "fixture drifted");
        assert_eq!(hoisted.leaves().len(), 3, "expected [x, x, hoisted-γ]");
        assert!(
            max_abs(&hoisted.to_dense(&one_shot()).unwrap(), &k.to_dense(&one_shot()).unwrap())
                < 1e-15
        );
    }

    // ---- leaf substitution ---------------------------------------------

    #[test]
    fn leaves_and_with_leaves_roundtrip_and_preserve_sharing() {
        let dev = candle_core::Device::Cpu;
        let x = Tensor::randn(0f64, 1f64, (5, 2), &dev).unwrap();
        let g = Tensor::full(0.7f64, (1, 1), &dev).unwrap();
        let xi = LazyTensor::vi(&x);
        let xj = LazyTensor::vj(&x);
        let diff = &xi - &xj;
        // `&diff * &diff` shares ONE subtree: its leaves must be counted
        // once each, not twice.
        let d2 = (&diff * &diff).sum_last();
        let k = (&d2 * &LazyTensor::pm(&g)).affine(-1.0, 0.0).exp();

        let leaves = k.leaves();
        assert_eq!(leaves.len(), 3, "expected [x@vi, x@vj, gamma], got {}", leaves.len());
        assert_eq!(leaves[0].dims(), &[5, 2]);
        assert_eq!(leaves[1].dims(), &[5, 2]);
        assert_eq!(leaves[2].dims(), &[1, 1]);

        // Identity substitution: bit-for-bit the same evaluation.
        let same = k.with_leaves(&leaves).unwrap();
        assert!(max_abs(&same.to_dense(&one_shot()).unwrap(), &k.to_dense(&one_shot()).unwrap()) == 0.0);

        // Real substitution: doubling gamma equals rebuilding with 2·gamma.
        let g2 = g.affine(2.0, 0.0).unwrap();
        let swapped = k
            .with_leaves(&[leaves[0].clone(), leaves[1].clone(), g2.clone()])
            .unwrap();
        let rebuilt = {
            let xi = LazyTensor::vi(&x);
            let xj = LazyTensor::vj(&x);
            let diff = &xi - &xj;
            let d2 = (&diff * &diff).sum_last();
            (&d2 * &LazyTensor::pm(&g2)).affine(-1.0, 0.0).exp()
        };
        assert!(
            max_abs(&swapped.to_dense(&one_shot()).unwrap(), &rebuilt.to_dense(&one_shot()).unwrap())
                < 1e-14
        );
    }

    #[test]
    fn with_leaves_refuses_wrong_count_and_wrong_shape() {
        let dev = candle_core::Device::Cpu;
        let x = Tensor::randn(0f64, 1f64, (4, 2), &dev).unwrap();
        let k = (&LazyTensor::vi(&x) - &LazyTensor::vj(&x)).powf(2.0).sum_last();
        let leaves = k.leaves();
        assert_eq!(leaves.len(), 2);
        assert!(k.with_leaves(&leaves[..1]).is_err(), "wrong count accepted");
        let bad = Tensor::randn(0f64, 1f64, (3, 2), &dev).unwrap();
        assert!(k.with_leaves(&[leaves[0].clone(), bad]).is_err(), "wrong shape accepted");
    }

    /// `‖(xᵢ − xⱼ)·w‖²` — the shape of the real cost formula.
    fn sq_dist(x1: &Tensor, x2: &Tensor, w: &Tensor) -> LazyTensor {
        let wp = LazyTensor::pm(w);
        let xi = &LazyTensor::vi(x1) * &wp;
        let xj = &LazyTensor::vj(x2) * &wp;
        (&(&xi - &xj)).powf(2.0).sum_last()
    }

    /// Dense reference for `sq_dist`.
    fn sq_dist_dense(x1: &Tensor, x2: &Tensor, w: &Tensor) -> Tensor {
        let a = x1.broadcast_mul(w).unwrap();
        let b = x2.broadcast_mul(w).unwrap();
        a.unsqueeze(1)
            .unwrap()
            .broadcast_sub(&b.unsqueeze(0).unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
    }

    // -- forward equivalence ----------------------------------------------

    #[test]
    fn matches_the_dense_reference() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (13, 5), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (9, 5), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 5), &dev).unwrap();

        let got = sq_dist(&x1, &x2, &w).to_dense(&one_shot()).unwrap();
        let want = sq_dist_dense(&x1, &x2, &w);
        assert_eq!(got.dims(), &[13, 9]);
        assert!(max_abs(&got, &want) < 1e-5, "{}", max_abs(&got, &want));
    }

    /// Tiling is a memory strategy, not an approximation. `sum_last` is
    /// intra-tile (it contracts the feature axis), so blocking cannot
    /// perturb it at all — this one is exact.
    #[test]
    fn tiling_does_not_change_sum_last() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (17, 4), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (11, 4), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 4), &dev).unwrap();
        let f = sq_dist(&x1, &x2, &w);

        let a = f.to_dense(&one_shot()).unwrap();
        // deliberately non-divisible tiles
        let b = f.to_dense(&BlockCfg::with_tiles(5, 3)).unwrap();
        assert_eq!(max_abs(&a, &b), 0.0, "sum_last is intra-tile; must be bitwise");
    }

    #[test]
    fn batch_axis_broadcasts() {
        let dev = Device::Cpu;
        // x1 carries a leading latent axis, x2 does not — the case that
        // forces `fan()`'s materialising broadcast_as today.
        let x1 = Tensor::randn(0f32, 1f32, (3, 6, 4), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (1, 5, 4), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 1, 4), &dev).unwrap();
        let out = sq_dist(&x1, &x2, &w).to_dense(&one_shot()).unwrap();
        assert_eq!(out.dims(), &[3, 6, 5]);
    }

    // -- reductions --------------------------------------------------------

    #[test]
    fn logsumexp_matches_dense_and_survives_tiling() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (12, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (10, 3), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 3), &dev).unwrap();
        let f = sq_dist(&x1, &x2, &w).neg();

        let dense = f.to_dense(&one_shot()).unwrap();
        let want = {
            let m = dense.max_keepdim(D::Minus1).unwrap();
            dense
                .broadcast_sub(&m)
                .unwrap()
                .exp()
                .unwrap()
                .sum(D::Minus1)
                .unwrap()
                .log()
                .unwrap()
                .add(&m.squeeze(D::Minus1).unwrap())
                .unwrap()
        };

        let a = f.logsumexp_reduce(ReduceAxis::J, &one_shot()).unwrap();
        let b = f.logsumexp_reduce(ReduceAxis::J, &BlockCfg::with_tiles(5, 4)).unwrap();
        assert_eq!(a.dims(), &[12]);
        assert!(max_abs(&a, &want) < 1e-5, "one-shot vs dense: {}", max_abs(&a, &want));
        // Reduction ACROSS j changes f32 summation order — relative, not bitwise.
        assert!(max_abs(&a, &b) < 1e-5, "tiled vs one-shot: {}", max_abs(&a, &b));
    }

    #[test]
    fn logsumexp_over_i_reduces_the_other_axis() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (7, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (9, 3), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 3), &dev).unwrap();
        let f = sq_dist(&x1, &x2, &w).neg();
        let out = f.logsumexp_reduce(ReduceAxis::I, &BlockCfg::with_tiles(3, 4)).unwrap();
        assert_eq!(out.dims(), &[9]);

        let dense = f.to_dense(&one_shot()).unwrap();
        let m = dense.max_keepdim(D::Minus2).unwrap();
        let want = dense
            .broadcast_sub(&m).unwrap().exp().unwrap().sum(D::Minus2).unwrap()
            .log().unwrap().add(&m.squeeze(D::Minus2).unwrap()).unwrap();
        assert!(max_abs(&out, &want) < 1e-5);
    }

    #[test]
    fn sum_reduce_matches_dense_on_both_axes() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f32, 1f32, (8, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f32, 1f32, (6, 3), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 3), &dev).unwrap();
        let f = sq_dist(&x1, &x2, &w);
        let dense = f.to_dense(&one_shot()).unwrap();

        let sj = f.sum_reduce(ReduceAxis::J, &BlockCfg::with_tiles(3, 4)).unwrap();
        assert!(max_abs(&sj, &dense.sum(D::Minus1).unwrap()) < 1e-4);
        let si = f.sum_reduce(ReduceAxis::I, &BlockCfg::with_tiles(3, 4)).unwrap();
        assert!(max_abs(&si, &dense.sum(D::Minus2).unwrap()) < 1e-4);
    }

    #[test]
    fn matmul_and_diagonal_match_dense() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (11, 3), &dev).unwrap();
        let w = Tensor::rand(0.5f32, 1.5f32, (1, 3), &dev).unwrap();
        let f = sq_dist(&x, &x, &w).neg().exp();
        let dense = f.to_dense(&one_shot()).unwrap();

        let rhs = Tensor::randn(0f32, 1f32, (11, 2), &dev).unwrap();
        let got = f.matmul(&rhs, &BlockCfg::with_tiles(4, 11)).unwrap();
        assert!(max_abs(&got, &dense.matmul(&rhs).unwrap()) < 1e-5);

        let d = f.diagonal(&BlockCfg::with_tiles(4, 4)).unwrap();
        assert_eq!(d.dims(), &[11]);
        assert!(max_abs(&d, &super::super::dense_diagonal(&dense).unwrap()) < 1e-5);
    }

    // -- the symbolic identity --------------------------------------------

    /// Pins the global-vs-local index. A tile-local index would put an
    /// identity on every diagonal tile's own diagonal; the non-square shape
    /// and straddling tiles are chosen to expose exactly that.
    #[test]
    fn eye_identity_is_exact_on_straddling_tiles() {
        let dev = Device::Cpu;
        let (m, n) = (30usize, 17usize);
        let x1 = Tensor::zeros((m, 2), DType::F32, &dev).unwrap();
        let x2 = Tensor::zeros((n, 2), DType::F32, &dev).unwrap();
        let w = Tensor::ones((1, 2), DType::F32, &dev).unwrap();
        let f = &sq_dist(&x1, &x2, &w) - &LazyTensor::eye_identity(1e-4);

        let want = {
            let mut v = vec![0f32; m * n];
            for i in 0..m.min(n) {
                v[i * n + i] = -1e-4;
            }
            Tensor::from_vec(v, (m, n), &dev).unwrap()
        };

        let a = f.to_dense(&one_shot()).unwrap();
        assert_eq!(max_abs(&a, &want), 0.0, "one-shot eye must be exact");
        // tiles that straddle, miss, and contain the diagonal
        let b = f.to_dense(&BlockCfg::with_tiles(7, 5)).unwrap();
        assert_eq!(max_abs(&b, &want), 0.0, "tiled eye must be exact — global index?");
    }

    // -- gradients ---------------------------------------------------------

    /// The whole reason this layer exists: a hyperparameter enters as a `pm`
    /// leaf and keeps its gradient. Nothing here can be a host scalar.
    #[test]
    fn gradient_reaches_a_pm_parameter() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f64, 1f64, (9, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f64, 1f64, (7, 3), &dev).unwrap();
        let w = Var::from_tensor(&Tensor::ones((1, 3), DType::F64, &dev).unwrap()).unwrap();

        let f = sq_dist(&x1, &x2, w.as_tensor()).neg();
        let loss = f.logsumexp_reduce(ReduceAxis::J, &BlockCfg::with_tiles(4, 3)).unwrap()
            .sum_all().unwrap();
        let g = loss.backward().unwrap();
        let gw = g.get(&w).expect("gradient must reach the pm leaf");
        let mag = gw.abs().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(mag.is_finite() && mag > 0.0, "pm gradient {mag}");
    }

    /// Central finite differences in F64 — the instrument for gradient
    /// correctness, since there is no reference implementation to diff
    /// against (the point is that new gradients appear).
    #[test]
    fn logsumexp_gradient_matches_finite_differences() {
        let dev = Device::Cpu;
        let x1 = Tensor::randn(0f64, 1f64, (6, 3), &dev).unwrap();
        let x2 = Tensor::randn(0f64, 1f64, (5, 3), &dev).unwrap();
        let w0 = Tensor::rand(0.5f64, 1.5f64, (1, 3), &dev).unwrap();
        let cfg = BlockCfg::with_tiles(2, 2);

        let loss_of = |w: &Tensor| -> f64 {
            sq_dist(&x1, &x2, w)
                .neg()
                .logsumexp_reduce(ReduceAxis::J, &cfg)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };

        let wv = Var::from_tensor(&w0).unwrap();
        let auto = {
            let l = sq_dist(&x1, &x2, wv.as_tensor())
                .neg()
                .logsumexp_reduce(ReduceAxis::J, &cfg)
                .unwrap()
                .sum_all()
                .unwrap();
            l.backward().unwrap().get(&wv).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap()
        };

        let h = 1e-6;
        let base: Vec<f64> = w0.flatten_all().unwrap().to_vec1().unwrap();
        for k in 0..base.len() {
            let bump = |sign: f64| {
                let mut v = base.clone();
                v[k] += sign * h;
                loss_of(&Tensor::from_vec(v, (1, base.len()), &dev).unwrap())
            };
            let fd = (bump(1.0) - bump(-1.0)) / (2.0 * h);
            let rel = (auto[k] - fd).abs() / fd.abs().max(1e-8);
            assert!(rel < 1e-5, "grad[{k}]: auto {} vs fd {fd} (rel {rel})", auto[k]);
        }
    }

    /// `Pow` with a TENSOR exponent — the node that exists so `rq_alpha`
    /// need not be scalarised. Gradient must reach the exponent itself.
    #[test]
    fn tensor_exponent_carries_gradient() {
        let dev = Device::Cpu;
        let x1 = Tensor::rand(1f64, 2f64, (5, 2), &dev).unwrap();
        let x2 = Tensor::rand(1f64, 2f64, (4, 2), &dev).unwrap();
        let w = Tensor::ones((1, 2), DType::F64, &dev).unwrap();
        let alpha = Var::from_tensor(&Tensor::full(1.5f64, (1, 1), &dev).unwrap()).unwrap();

        let base = sq_dist(&x1, &x2, &w).affine(1.0, 1.0);
        let f = base.pow(&LazyTensor::pm(alpha.as_tensor()).neg());
        let loss = f.sum_reduce(ReduceAxis::J, &one_shot()).unwrap().sum_all().unwrap();
        let g = loss.backward().unwrap();
        let ga = g.get(&alpha).expect("gradient must reach the tensor exponent");
        let mag = ga.abs().unwrap().sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(mag.is_finite() && mag > 0.0, "alpha gradient {mag}");
    }

    /// Backward depth must not grow with the tile count. Chained `add`
    /// accumulation would be O(n_blocks) deep — the mechanism by which
    /// `sinkhorn_coupling_t` overflows the stack.
    #[test]
    fn backward_depth_is_independent_of_tile_count() {
        let handle = std::thread::Builder::new()
            .stack_size(512 << 10) // deliberately small
            .spawn(|| {
                let dev = Device::Cpu;
                let x1 = Tensor::randn(0f32, 1f32, (256, 3), &dev).unwrap();
                let x2 = Tensor::randn(0f32, 1f32, (4096, 3), &dev).unwrap();
                let w = Var::from_tensor(&Tensor::ones((1, 3), DType::F32, &dev).unwrap()).unwrap();
                // 4096/16 = 256 j-tiles
                let cfg = BlockCfg::with_tiles(256, 16);
                let loss = sq_dist(&x1, &x2, w.as_tensor())
                    .neg()
                    .logsumexp_reduce(ReduceAxis::J, &cfg)
                    .unwrap()
                    .sum_all()
                    .unwrap();
                let g = loss.backward().unwrap();
                assert!(g.get(&w).is_some());
            })
            .unwrap();
        handle.join().expect("backward overflowed the stack — chained accumulation?");
    }

    // -- validation --------------------------------------------------------

    #[test]
    fn dim_mismatch_is_reported_with_both_dims() {
        let dev = Device::Cpu;
        let a = LazyTensor::vi(&Tensor::zeros((4, 32), DType::F32, &dev).unwrap());
        let b = LazyTensor::vj(&Tensor::zeros((3, 8), DType::F32, &dev).unwrap());
        let e = (&a * &b).sum_last().to_dense(&one_shot()).unwrap_err().to_string();
        assert!(e.contains("Mul dim mismatch"), "{e}");
        assert!(e.contains("32") && e.contains('8'), "should name both dims: {e}");
    }

    #[test]
    fn reduction_requires_a_pair_scalar() {
        let dev = Device::Cpu;
        let a = LazyTensor::vi(&Tensor::zeros((4, 5), DType::F32, &dev).unwrap());
        let b = LazyTensor::vj(&Tensor::zeros((3, 5), DType::F32, &dev).unwrap());
        // no .sum_last() — dim is still 5
        let e = (&a - &b).to_dense(&one_shot()).unwrap_err().to_string();
        assert!(e.contains("sum_last"), "error should say what to do: {e}");
    }

    #[test]
    fn pm_leaf_must_have_a_singleton_pair_axis() {
        let dev = Device::Cpu;
        let bad = LazyTensor::pm(&Tensor::zeros((4, 5), DType::F32, &dev).unwrap());
        assert!(bad.infer().is_err());
    }

    #[test]
    fn missing_vi_or_vj_is_reported() {
        let dev = Device::Cpu;
        let only_i = LazyTensor::vi(&Tensor::zeros((4, 5), DType::F32, &dev).unwrap()).sum_last();
        let e = only_i.to_dense(&one_shot()).unwrap_err().to_string();
        assert!(e.contains("Vj"), "{e}");
    }

    // -- tiling policy -----------------------------------------------------

    /// At the sizes this crate actually runs, tiling must not fire.
    #[test]
    fn small_problems_are_evaluated_in_one_shot() {
        let cfg = BlockCfg::default();
        // K_ATOMS=30, D=32, f32 → 115 KB, far under the 8 MiB budget
        assert_eq!(cfg.tiles(30, 30, 32, DType::F32), (30, 30));
        // a genuinely large grid does tile
        let (bi, bj) = cfg.tiles(100_000, 100_000, 32, DType::F32);
        assert!(bi < 100_000 && bj < 100_000, "expected tiling, got {bi}x{bj}");
        assert!(bi * bj * 32 * 4 <= cfg.budget_bytes, "tile exceeds the budget");
    }
}
