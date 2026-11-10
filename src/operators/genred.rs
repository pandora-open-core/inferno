//! Inferno — Tiled symbolic sum and log-sum-exp reductions with symbolic vector-Jacobian products.
//!
//! Forward evaluation avoids retaining per-tile activations; backward evaluates
//! symbolic vector-Jacobian products with the same tiling discipline.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor, D};

use super::ciq::pack_leaves;
use super::symbolic::{
    eval_block, infer, stack_sum, Axis, BlockCfg, Expr, LazyTensor, Memo, Meta, ReduceAxis,
};

#[path = "reduction_schema.rs"]
mod reduction_schema;
pub use reduction_schema::ReduceOp;

/// A reduction over a pair formula, as a value.
#[derive(Clone)]
pub struct Genred {
    pub formula: LazyTensor,
    pub op: ReduceOp,
    /// The axis CONTRACTED (same convention as [`ReduceAxis`]): `J` keeps
    /// rows, `I` keeps columns.
    pub axis: ReduceAxis,
    pub cfg: BlockCfg,
}

impl Genred {
    pub fn new(formula: LazyTensor, op: ReduceOp, axis: ReduceAxis) -> Self {
        Self {
            formula,
            op,
            axis,
            cfg: BlockCfg::default(),
        }
    }
    pub fn with_cfg(mut self, cfg: BlockCfg) -> Self {
        self.cfg = cfg;
        self
    }

    /// Evaluate with NO tape — leaves detached, simplified (CSE + rewrites,
    /// see [`super::rewrite`]), tile-invariant subtrees hoisted, tiled,
    /// `O(tile)` memory.
    pub fn eval_detached(&self) -> Result<Tensor> {
        let f = self.formula.simplify()?.hoist_invariant()?.detach_leaves();
        match self.op {
            ReduceOp::Sum => sum_reduce_vec(&f, self.axis, &self.cfg, None),
            ReduceOp::LogSumExp => f.logsumexp_reduce(self.axis, &self.cfg),
        }
    }

    /// Evaluate as ONE differentiable node: forward tape-free, gradients to
    /// every leaf (and through them to whatever built the leaves) via the
    /// symbolic VJP evaluated in the backward.
    ///
    /// The formula is simplified FIRST, and the gradient pack is built from
    /// the SIMPLIFIED formula's leaves — CSE may merge duplicate leaves
    /// (same tensor, same axis), in which case the merged node's cotangent
    /// bucket accumulates what `cat`'s VJP would previously have summed.
    pub fn eval(&self) -> Result<Tensor> {
        // simplify → hoist: hoisted subtrees become COMPUTED leaves in the
        // pack, so their gradients route through candle's tape outside the
        // CustomOp, and the symbolic VJP differentiates against one leaf
        // instead of the whole hyperparameter chain.
        let g = Genred {
            formula: self.formula.simplify()?.hoist_invariant()?,
            op: self.op,
            axis: self.axis,
            cfg: self.cfg.clone(),
        };
        let leaves = collect_leaf_nodes(g.formula.expr());
        if leaves.is_empty() {
            // Nothing to differentiate toward — the detached value IS the
            // gradient-complete answer.
            return g.eval_detached();
        }
        let tensors: Vec<Tensor> = leaves.iter().map(|l| l.data.clone()).collect();
        let (packed, shapes) = pack_leaves(&tensors)?;
        packed
            .contiguous()?
            .apply_op1(GenredOp { genred: g, shapes })
    }

    /// Per-leaf gradients as VALUES that themselves carry a tape — so
    /// `grad(grad(·))` works and Hessian-vector products are reachable.
    ///
    /// This is the piece candle cannot give us. Its `backward()` is single
    /// pass and hands back a plain `GradStore`, and nothing computed inside a
    /// `CustomOp::bwd` lands on a tape at all — so [`GenredOp::bwd`]'s
    /// gradients are values and nothing more. The symbolic VJP has no such
    /// limit: [`vjp`] returns each cotangent as a FORMULA, and the rule set
    /// is closed over `Expr`, so a gradient reduction is just another
    /// [`Genred`]. Evaluating those here with `eval` rather than
    /// `eval_detached` is the whole trick — each returned tensor is an
    /// independent `CustomOp1` whose own backward re-enters `vjp`, giving the
    /// next order with `O(1)` tape per level.
    ///
    /// `seed` is the output cotangent, shaped like this reduction's output.
    /// It must be LIVE (not detached) if you want `∂/∂seed` — which is
    /// exactly what makes an HVP a directional derivative:
    ///
    /// ```text
    /// g  = L.grad(&ones)             // tracked, one per leaf
    /// s  = Σ_i ⟨g_i, v_i⟩            // scalar, tracked
    /// Hv = s.backward()              // re-enters vjp on the gradient formulas
    /// ```
    ///
    /// Returns leaves and gradients aligned pairwise, in `leaves()` order of
    /// the SIMPLIFIED formula — the same order [`eval`](Self::eval) packs, so
    /// CSE may have merged duplicate leaves, and that merged node's bucket
    /// accumulates what `cat`'s VJP would otherwise have summed.
    ///
    /// Cost note: every gradient is a fresh tiled reduction over the same
    /// grid, so this is `O(leaves)` sweeps. The hoist and the cons table make
    /// the shared subtrees free, not the sweeps.
    pub fn grad(&self, seed: &Tensor) -> Result<GenredGrad> {
        // Normalise exactly as `eval` does, so the leaf order matches.
        let g = Genred {
            formula: self.formula.simplify()?.hoist_invariant()?,
            op: self.op,
            axis: self.axis,
            cfg: self.cfg.clone(),
        };
        let meta = g.formula.infer()?;
        let (m, n) = (
            meta.m
                .ok_or_else(|| candle_core::Error::Msg("genred::grad: no Vi leaf".into()))?,
            meta.n
                .ok_or_else(|| candle_core::Error::Msg("genred::grad: no Vj leaf".into()))?,
        );

        // The kept axis carries the output index, so the output cotangent
        // enters the gradient formulas as a leaf on that axis.
        let kept_leaf = |t: &Tensor| -> LazyTensor {
            match g.axis {
                ReduceAxis::J => LazyTensor::vi(t),
                ReduceAxis::I => LazyTensor::vj(t),
            }
        };

        let root_cot = match g.op {
            ReduceOp::Sum => kept_leaf(&seed.contiguous()?),
            ReduceOp::LogSumExp => {
                // ∂/∂F log Σ exp F = softmax: G ⊙ exp(F − out).
                //
                // `out` is the TRACKED forward, where `bwd` uses a detached
                // `res`. That difference is the point: the softmax depends on
                // the output, so a detached one would silently drop a term
                // from every SECOND derivative while leaving the first exact.
                let out = g.eval()?;
                let gl = kept_leaf(&seed.unsqueeze(D::Minus1)?.contiguous()?);
                let ol = kept_leaf(&out.unsqueeze(D::Minus1)?.contiguous()?);
                &gl * &(&g.formula - &ol).exp()
            }
        };

        let buckets = vjp(g.formula.expr(), root_cot)?;
        let leaves = collect_leaf_nodes(g.formula.expr());

        let mut grads = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            grads.push(tracked_leaf_grad(
                buckets.get(&leaf.ptr),
                leaf,
                m,
                n,
                &g.cfg,
            )?);
        }
        Ok(GenredGrad {
            leaves: leaves.iter().map(|l| l.data.clone()).collect(),
            grads,
        })
    }
}

/// The result of [`Genred::grad`]: leaves and their gradients, aligned
/// pairwise and in the simplified formula's `leaves()` order.
///
/// **These are per-leaf-NODE gradients, not per-tensor.** One tensor used at
/// two axes — `vi(x)` and `vj(x)`, the usual shape of a symmetric kernel —
/// is two distinct leaf nodes and appears TWICE, each carrying only its own
/// partial. The total derivative with respect to that tensor is their sum,
/// which is what `Tensor::cat`'s VJP does for you on the
/// [`eval`](Genred::eval) path. Use [`for_tensor`](Self::for_tensor) rather
/// than indexing, or you will compare a partial against a total.
#[derive(Clone, Debug)]
pub struct GenredGrad {
    /// Leaf tensors of the SIMPLIFIED formula.
    pub leaves: Vec<Tensor>,
    /// One gradient per leaf — same order, same shape, and TRACKED.
    pub grads: Vec<Tensor>,
}

impl GenredGrad {
    /// Total gradient with respect to `t`: the SUM over every leaf node
    /// holding that tensor. `None` if the formula does not mention it.
    pub fn for_tensor(&self, t: &Tensor) -> Option<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (leaf, g) in self.leaves.iter().zip(&self.grads) {
            if leaf.id() != t.id() {
                continue;
            }
            acc = Some(match acc {
                Some(a) => (a + g).ok()?,
                None => g.clone(),
            });
        }
        acc
    }

    /// `Σ_i ⟨gᵢ, vᵢ⟩` — the scalar whose gradient is the Hessian-vector
    /// product. `directions` must align with [`leaves`](Self::leaves).
    ///
    /// Kept here rather than left to callers because the pairing is the easy
    /// thing to get wrong: `leaves()` order is the SIMPLIFIED formula's, not
    /// the order the caller wrote its leaves in.
    pub fn directional(&self, directions: &[Tensor]) -> Result<Tensor> {
        if directions.len() != self.grads.len() {
            return Err(candle_core::Error::Msg(format!(
                "GenredGrad::directional: {} directions for {} leaves",
                directions.len(),
                self.grads.len()
            )));
        }
        let mut acc: Option<Tensor> = None;
        for (g, v) in self.grads.iter().zip(directions) {
            let term = (g * v)?.sum_all()?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        acc.ok_or_else(|| candle_core::Error::Msg("GenredGrad::directional: no leaves".into()))
    }
}

// ---------------------------------------------------------------------------
// Vector-valued tiled sum reduction
// ---------------------------------------------------------------------------

/// `Σ_axis F` for a formula of ANY feature dim — the generalisation of the
/// pair-scalar `sum_reduce` that [`ReduceOp::Sum`] needs. Output
/// `[..B, kept, d]`. `fallback_extent` supplies `(m, n)` when the formula
/// itself lacks a `Vi`/`Vj` leaf (a gradient bucket can be constant along
/// the contracted axis — the sum then multiplies by the extent, which the
/// injected ones-leaf provides).
fn sum_reduce_vec(
    f: &LazyTensor,
    axis: ReduceAxis,
    cfg: &BlockCfg,
    fallback_extent: Option<(usize, usize)>,
) -> Result<Tensor> {
    // Inject a ones-leaf for any missing axis so the tile loop has extents
    // and the constant-along-axis case sums to `extent × value`.
    let mut f = f.clone();
    let mut meta = f.infer()?;
    if let Some((m, n)) = fallback_extent {
        if meta.m.is_none() {
            let ones = Tensor::ones((m, 1), meta.dtype, &meta.device)?;
            f = &f * &LazyTensor::vi(&ones);
            meta = f.infer()?;
        }
        if meta.n.is_none() {
            let ones = Tensor::ones((n, 1), meta.dtype, &meta.device)?;
            f = &f * &LazyTensor::vj(&ones);
            meta = f.infer()?;
        }
    }
    let m = meta.m.ok_or_else(|| {
        candle_core::Error::Msg("genred: formula has no Vi leaf and no fallback extent".into())
    })?;
    let n = meta.n.ok_or_else(|| {
        candle_core::Error::Msg("genred: formula has no Vj leaf and no fallback extent".into())
    })?;

    let (bi, bj) = cfg.tiles_for(f.expr(), m, n, &meta);

    let tile = |i: &Range<usize>, j: &Range<usize>| -> Result<Tensor> {
        let mut memo: Memo = HashMap::new();
        eval_block(f.expr(), &meta, i, j, &mut memo) // [..B, bi, bj, d]
    };

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
                    let t = tile(&i, &j)?;
                    parts.push(t.sum(t.rank() - 2)?); // contract bj → [..B, bi, d]
                    j0 = j.end;
                }
                rows.push(stack_sum(&parts)?);
                i0 = i.end;
            }
            let refs: Vec<&Tensor> = rows.iter().collect();
            Tensor::cat(&refs, D::Minus2)
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
                    let t = tile(&i, &j)?;
                    parts.push(t.sum(t.rank() - 3)?); // contract bi → [..B, bj, d]
                    i0 = i.end;
                }
                cols.push(stack_sum(&parts)?);
                j0 = j.end;
            }
            let refs: Vec<&Tensor> = cols.iter().collect();
            Tensor::cat(&refs, D::Minus2)
        }
    }
}

// ---------------------------------------------------------------------------
// Symbolic VJP over the formula DAG
// ---------------------------------------------------------------------------

struct LeafInfo {
    ptr: *const Expr,
    data: Tensor,
    axis: Axis,
}

/// Leaf NODES in `leaves()` order (DFS first-visit, ptr-deduped) — must stay
/// aligned with `LazyTensor::leaves` / `pack_leaves`.
fn collect_leaf_nodes(e: &Arc<Expr>) -> Vec<LeafInfo> {
    fn walk(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, out: &mut Vec<LeafInfo>) {
        if seen.insert(Arc::as_ptr(e), ()).is_some() {
            return;
        }
        match e.as_ref() {
            Expr::Leaf { data, axis } => out.push(LeafInfo {
                ptr: Arc::as_ptr(e),
                data: data.clone(),
                axis: *axis,
            }),
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
            | Expr::Affine { e: a, .. } => walk(a, seen, out),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b)
            | Expr::MatDot(a, b) => {
                walk(a, seen, out);
                walk(b, seen, out);
            }
        }
    }
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    walk(e, &mut seen, &mut out);
    out
}

/// Post-order node list, ptr-deduped — reversed, it is a valid
/// root-first order for reverse-mode accumulation on the DAG.
fn topo_postorder(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    fn walk(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, out: &mut Vec<Arc<Expr>>) {
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
            | Expr::Affine { e: a, .. } => walk(a, seen, out),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b)
            | Expr::MatDot(a, b) => {
                walk(a, seen, out);
                walk(b, seen, out);
            }
        }
        out.push(e.clone());
    }
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    walk(e, &mut seen, &mut out);
    out
}

fn dim_of(e: &Arc<Expr>) -> Result<usize> {
    Ok(infer(e)?.dim)
}

/// Accumulate a cotangent contribution onto a node, collapsing a broadcast
/// (`contribution dim d > node dim 1` ⇒ `sum_last`) — the VJP of implicit
/// feature broadcasting.
fn add_cot(
    map: &mut HashMap<*const Expr, LazyTensor>,
    node: &Arc<Expr>,
    contrib: LazyTensor,
) -> Result<()> {
    let node_dim = dim_of(node)?;
    let contrib_dim = infer(contrib.expr())?.dim;
    let contrib = if node_dim == 1 && contrib_dim > 1 {
        contrib.sum_last()
    } else {
        contrib
    };
    match map.remove(&Arc::as_ptr(node)) {
        Some(prev) => {
            map.insert(Arc::as_ptr(node), &prev + &contrib);
        }
        None => {
            map.insert(Arc::as_ptr(node), contrib);
        }
    }
    Ok(())
}

/// Reverse-mode over the DAG: root cotangent in, per-LEAF cotangent
/// FORMULAS out (keyed by leaf node ptr). Every rule reuses primal subtrees
/// by `Arc` where the derivative allows it.
fn vjp(root: &Arc<Expr>, root_cot: LazyTensor) -> Result<HashMap<*const Expr, LazyTensor>> {
    let order = topo_postorder(root);
    let mut cot: HashMap<*const Expr, LazyTensor> = HashMap::new();
    cot.insert(Arc::as_ptr(root), root_cot);
    let mut leaf_cot: HashMap<*const Expr, LazyTensor> = HashMap::new();

    for node in order.iter().rev() {
        let Some(ct) = cot.remove(&Arc::as_ptr(node)) else {
            continue;
        };
        let this = LazyTensor::from_arc(node.clone());
        match node.as_ref() {
            Expr::Leaf { .. } => {
                match leaf_cot.remove(&Arc::as_ptr(node)) {
                    Some(prev) => {
                        leaf_cot.insert(Arc::as_ptr(node), &prev + &ct);
                    }
                    None => {
                        leaf_cot.insert(Arc::as_ptr(node), ct);
                    }
                };
            }
            Expr::Scalar(_) | Expr::EyeIdentity { .. } => {}
            Expr::Neg(a) => add_cot(&mut cot, a, ct.neg())?,
            Expr::Add(a, b) => {
                add_cot(&mut cot, a, ct.clone())?;
                add_cot(&mut cot, b, ct)?;
            }
            Expr::Sub(a, b) => {
                add_cot(&mut cot, a, ct.clone())?;
                add_cot(&mut cot, b, ct.neg())?;
            }
            Expr::Mul(a, b) => {
                add_cot(&mut cot, a, &ct * &LazyTensor::from_arc(b.clone()))?;
                add_cot(&mut cot, b, &ct * &LazyTensor::from_arc(a.clone()))?;
            }
            Expr::Div(a, b) => {
                let bl = LazyTensor::from_arc(b.clone());
                add_cot(&mut cot, a, &ct / &bl)?;
                // d(a/b)/db = −(a/b)/b — reuses the primal quotient node.
                add_cot(&mut cot, b, &(&ct * &this).neg() / &bl)?;
            }
            Expr::Affine { e: a, mul, .. } => add_cot(&mut cot, a, ct.affine(*mul, 0.0))?,
            Expr::PowF(a, p) => {
                let al = LazyTensor::from_arc(a.clone());
                add_cot(&mut cot, a, (&ct * &al.powf(p - 1.0)).affine(*p, 0.0))?;
            }
            Expr::Pow(a, b) => {
                // node = exp(b·log a):  ∂/∂a = node·b/a,  ∂/∂b = node·log a.
                let al = LazyTensor::from_arc(a.clone());
                let bl = LazyTensor::from_arc(b.clone());
                add_cot(&mut cot, a, &(&(&ct * &this) * &bl) / &al)?;
                add_cot(&mut cot, b, &(&ct * &this) * &al.log())?;
            }
            Expr::Exp(a) => add_cot(&mut cot, a, &ct * &this)?, // reuses primal Exp
            Expr::Log(a) => add_cot(&mut cot, a, &ct / &LazyTensor::from_arc(a.clone()))?,
            // d max(a,c)/da = 1 above the floor, 0 below (corner takes the
            // floor side — candle `maximum` parity).
            Expr::MaxC(a, c) => {
                let ind = LazyTensor::from_arc(a.clone()).gt_c(*c);
                add_cot(&mut cot, a, &ct * &ind)?;
            }
            // Piecewise-constant: derivative identically zero.
            Expr::GtC(..) => {}
            Expr::Sqrt(a) => add_cot(&mut cot, a, (&ct / &this).affine(0.5, 0.0))?,
            Expr::Recip(a) => add_cot(&mut cot, a, (&(&ct * &this) * &this).neg())?,
            // Feature contraction: the dim-1 cotangent broadcasts back over
            // the child's features — the algebra's broadcasting does it.
            Expr::Sum(a) => add_cot(&mut cot, a, ct)?,
            // `MatDot ≡ Sum(Mul(a, b))`, so its VJP is exactly the Mul rule
            // composed with Sum's: the dim-1 cotangent broadcasts back over
            // the contracted features and multiplies the other operand.
            //
            // The contributions are PAIR formulas (`ct` is pair-shaped),
            // which `eval_leaf_grad` then reduces over the operand's free
            // axis. So the backward pass does NOT keep the forward's GEMM
            // shape — it pays the `[bi, bj, d]` tile the forward avoided.
            // That is the right trade here: the detached Sinkhorn loop runs
            // this formula tens to hundreds of times per solve and the
            // backward once.
            Expr::MatDot(a, b) => {
                add_cot(&mut cot, a, &ct * &LazyTensor::from_arc(b.clone()))?;
                add_cot(&mut cot, b, &ct * &LazyTensor::from_arc(a.clone()))?;
            }
        }
    }
    Ok(leaf_cot)
}

// ---------------------------------------------------------------------------
// The differentiable node
// ---------------------------------------------------------------------------

struct GenredOp {
    genred: Genred,
    shapes: Vec<Vec<usize>>,
}

/// Hand a CUDA-resident result back to candle as `(CudaStorage, Shape)`.
///
/// The detached evaluators are plain candle ops over the formula's leaves, so
/// on a CUDA device their output ALREADY lives in device memory — the whole job
/// here is to hand ownership back through the `CustomOp` boundary, which is why
/// this is `try_clone` and not a copy through the host. Compare `cpu_fwd`,
/// which round-trips a `Vec<f64>`: that is unavoidable there (the boundary
/// wants a `CpuStorage` it owns), and unnecessary here.
///
/// This is the ONLY thing standing between the tracked symbolic path and a GPU:
/// `CustomOp1`'s default `cuda_fwd` returns `Err("no cuda implementation")`
/// (`vendor/candle/candle-core/src/custom_op.rs:17`), so without it every
/// `Genred::eval()` / `FusedSweep::eval()` consumer — `softmin_lazy`, the
/// Sinkhorn dual pairs, `matmul_tracked`, `ibp_step_symbolic` — fails outright
/// on a CUDA device.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_storage_of(y: &Tensor, who: &str) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::backend::BackendStorage;
    let y = y.contiguous()?;
    let shape = y.shape().clone();
    let (store, layout) = y.storage_and_layout();
    match &*store {
        candle_core::Storage::Cuda(cs) => Ok((cs.try_clone(layout)?, shape)),
        _ => Err(candle_core::Error::Msg(format!(
            "{who}: cuda_fwd was reached but the detached evaluation produced \
             non-CUDA storage — a leaf is on the wrong device"
        ))),
    }
}

impl CustomOp1 for GenredOp {
    fn name(&self) -> &'static str {
        "genred"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, _l1: &Layout) -> Result<(CpuStorage, Shape)> {
        // The packed argument's values are the leaves already inside the
        // formula — it exists as the graph edge for the parameter cotangent.
        let as_f32 = matches!(s1, CpuStorage::F32(_));
        let y = self.genred.eval_detached()?;
        let shape = y.shape().clone();
        let host: Vec<f64> = y
            .to_dtype(candle_core::DType::F64)?
            .flatten_all()?
            .to_vec1()?;
        Ok((crate::linalg_diff::storage_from_f64(host, as_f32), shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _s1: &candle_core::CudaStorage,
        _l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_storage_of(&self.genred.eval_detached()?, "genred")
    }

    fn bwd(&self, packed: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let g = &self.genred;
        let meta = g.formula.infer()?;
        let (m, n) = (
            meta.m
                .ok_or_else(|| candle_core::Error::Msg("genred: no Vi leaf".into()))?,
            meta.n
                .ok_or_else(|| candle_core::Error::Msg("genred: no Vj leaf".into()))?,
        );
        // The kept axis carries the output index, so the output cotangent
        // enters the gradient formulas as a leaf on that axis.
        let kept_leaf = |t: &Tensor| -> LazyTensor {
            match g.axis {
                ReduceAxis::J => LazyTensor::vi(t),
                ReduceAxis::I => LazyTensor::vj(t),
            }
        };

        let root_cot = match g.op {
            ReduceOp::Sum => kept_leaf(&grad_res.detach().contiguous()?),
            ReduceOp::LogSumExp => {
                // ∂/∂F log Σ exp F = softmax: G ⊙ exp(F − out).
                let gl = kept_leaf(&grad_res.detach().unsqueeze(D::Minus1)?.contiguous()?);
                let ol = kept_leaf(&res.detach().unsqueeze(D::Minus1)?.contiguous()?);
                &gl * &(&g.formula - &ol).exp()
            }
        };

        let buckets = vjp(g.formula.expr(), root_cot)?;
        let leaves = collect_leaf_nodes(g.formula.expr());

        let mut flat = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            flat.push(eval_leaf_grad(buckets.get(&leaf.ptr), leaf, m, n, &g.cfg)?.flatten_all()?);
        }
        let packed_bar = Tensor::cat(&flat, 0)?.to_dtype(packed.dtype())?;
        Ok(Some(packed_bar))
    }
}

/// Evaluate one leaf's accumulated cotangent FORMULA to its gradient tensor
/// — the shared tail of every symbolic backward ([`GenredOp`] and
/// [`FusedSweepOp`]).
fn eval_leaf_grad(
    bucket: Option<&LazyTensor>,
    leaf: &LeafInfo,
    m: usize,
    n: usize,
    cfg: &BlockCfg,
) -> Result<Tensor> {
    let Some(bucket) = bucket else {
        return leaf.data.zeros_like();
    };
    // Simplify + hoist the DERIVED formula too: the VJP rules emit
    // affine/neg chains and re-built products that the rewriter folds, the
    // hash-consing re-shares with the primal subtrees, and the hoist
    // evaluates once per bucket sweep instead of once per tile.
    let bucket = bucket.simplify()?.hoist_invariant()?.detach_leaves();

    // A dense pair leaf's gradient is the pair cotangent ITSELF — no
    // reduction, just materialise the bucket over the grid.
    if leaf.axis == Axis::IJ {
        return reconcile_to_leaf(bucket.to_dense(cfg)?, leaf, false);
    }

    let gt = match leaf.axis {
        // Σ over the OTHER pair axis lands on the leaf's own index; a Pm
        // leaf is summed over both.
        Axis::I => sum_reduce_vec(&bucket, ReduceAxis::J, cfg, Some((m, n)))?,
        Axis::J => sum_reduce_vec(&bucket, ReduceAxis::I, cfg, Some((m, n)))?,
        Axis::P => {
            sum_reduce_vec(&bucket, ReduceAxis::J, cfg, Some((m, n)))?.sum_keepdim(D::Minus2)?
        }
        Axis::IJ => unreachable!("handled above"),
    };
    reconcile_to_leaf(gt, leaf, true)
}

/// The reduce axis a leaf's cotangent bucket is contracted over: the OTHER
/// pair axis, so the sum lands on the leaf's own index. A `Pm` leaf has no
/// index and is summed over both — this returns the first, and the caller
/// applies the trailing `sum_keepdim`.
fn bucket_reduce_axis(axis: Axis) -> ReduceAxis {
    match axis {
        Axis::J => ReduceAxis::I,
        // `I` and `P` both start by contracting J.
        Axis::I | Axis::P => ReduceAxis::J,
        Axis::IJ => ReduceAxis::J, // unused; IJ never reduces
    }
}

/// Shape the reduced cotangent back onto its leaf: undo the feature
/// broadcast, then the batch broadcast. Pure candle ops, so it composes into
/// a tracked path unchanged — which is what lets [`eval_leaf_grad`] and
/// [`Genred::grad`] share it and never drift on the fiddly part.
fn reconcile_to_leaf(mut gt: Tensor, leaf: &LeafInfo, feature_expand: bool) -> Result<Tensor> {
    let want = leaf.data.dims();
    // A dim-1 leaf used at dim d was summed by `add_cot`; a dim-d leaf used
    // at dim 1 has to broadcast back out here.
    if feature_expand {
        let have = gt.dims()[gt.rank() - 1];
        let d_leaf = want[want.len() - 1];
        if have == 1 && d_leaf > 1 {
            let mut target = gt.dims().to_vec();
            *target.last_mut().unwrap() = d_leaf;
            gt = gt.broadcast_as(target)?.contiguous()?;
        }
    }
    while gt.rank() > want.len() {
        gt = gt.sum(0)?;
    }
    let have_dims = gt.dims().to_vec();
    for (ax, (&w, &h)) in want.iter().zip(&have_dims).enumerate() {
        if w == 1 && h > 1 {
            gt = gt.sum_keepdim(ax)?;
        }
    }
    Ok(gt)
}

/// Give a formula both pair axes, injecting a ones-leaf for any it lacks.
///
/// [`sum_reduce_vec`] does this itself from `fallback_extent`, but
/// [`Genred::eval`] reaches it through `eval_detached`, which passes `None`.
/// A cotangent bucket is routinely constant along one axis (it never
/// mentions a `Vi`/`Vj` leaf), so without this a tracked gradient node dies
/// with "no Vi leaf and no fallback extent". The injected leaf is a fresh
/// non-`Var` tensor, so its own cotangent bucket goes nowhere.
fn with_pair_extents(f: &LazyTensor, m: usize, n: usize) -> Result<LazyTensor> {
    let mut f = f.clone();
    let meta = f.infer()?;
    if meta.m.is_none() {
        let ones = Tensor::ones((m, 1), meta.dtype, &meta.device)?;
        f = &f * &LazyTensor::vi(&ones);
    }
    let meta = f.infer()?;
    if meta.n.is_none() {
        let ones = Tensor::ones((n, 1), meta.dtype, &meta.device)?;
        f = &f * &LazyTensor::vj(&ones);
    }
    Ok(f)
}

/// [`eval_leaf_grad`]'s tracked twin: the same cotangent bucket, evaluated as
/// a fresh [`Genred`] node instead of a bare tiled reduction, so the result
/// carries a tape and can be differentiated AGAIN.
///
/// The two differ only in how the bucket is evaluated — the formula itself
/// comes from the same [`vjp`] — and they share [`reconcile_to_leaf`].
fn tracked_leaf_grad(
    bucket: Option<&LazyTensor>,
    leaf: &LeafInfo,
    m: usize,
    n: usize,
    cfg: &BlockCfg,
) -> Result<Tensor> {
    // A leaf the root does not depend on has gradient exactly zero, at every
    // order — so a detached constant is the CORRECT answer here, not a
    // missing tape.
    let Some(bucket) = bucket else {
        return leaf.data.zeros_like();
    };

    if leaf.axis == Axis::IJ {
        // `to_dense` materialises `[M, N]`, so unlike every other arm this
        // one puts the whole pair grid on the tape. Supported rather than
        // refused because `cost_lazy`'s dense-cost OT genuinely uses `Vij`,
        // but be aware second order over a dense-pair leaf is `O(MN)` memory
        // where the rest of this file is `O(tile)`.
        return reconcile_to_leaf(bucket.to_dense(cfg)?, leaf, false);
    }

    // NO `detach_leaves` — that is the entire difference from the detached
    // path, and getting it wrong FAILS SILENTLY: `Genred::eval` short-
    // circuits to `eval_detached` when the leaf set is empty, so a detached
    // bucket still yields a perfectly good value with no tape and no error.
    let f = with_pair_extents(bucket, m, n)?;
    let node = Genred::new(f, ReduceOp::Sum, bucket_reduce_axis(leaf.axis)).with_cfg(cfg.clone());
    let mut gt = node.eval()?;
    if leaf.axis == Axis::P {
        gt = gt.sum_keepdim(D::Minus2)?;
    }
    reconcile_to_leaf(gt, leaf, true)
}

// ---------------------------------------------------------------------------
// FusedSweep — several reductions of MIXED op over one pair grid, one sweep
// ---------------------------------------------------------------------------

/// One reduction inside a [`FusedSweep`]: a formula, the op that contracts
/// it, and the axis contracted.
#[derive(Clone)]
pub struct FusedItem {
    pub formula: LazyTensor,
    pub op: ReduceOp,
    /// The axis CONTRACTED (same convention as [`ReduceAxis`]).
    pub axis: ReduceAxis,
}

impl FusedItem {
    /// `Σ_axis F` — vector-valued, any feature dim.
    pub fn sum(formula: LazyTensor, axis: ReduceAxis) -> Self {
        Self {
            formula,
            op: ReduceOp::Sum,
            axis,
        }
    }
    /// `log Σ_axis exp F` — pair-scalar formulas only.
    pub fn lse(formula: LazyTensor, axis: ReduceAxis) -> Self {
        Self {
            formula,
            op: ReduceOp::LogSumExp,
            axis,
        }
    }
}

/// Several reductions of POSSIBLY DIFFERENT op over formulas sharing ONE
/// `(m, n)` pair grid, evaluated in a single scheduled tile sweep.
///
/// The point: formulas that share subexpressions (Sinkhorn's paired `f`/`g`
/// softmins share the whole COST subtree; a transport cost shares it with
/// the potentials that produced it) evaluate the shared piece ONCE per tile,
/// because each tile's memo is shared across the fused formulas. Two
/// supporting passes make that sharing real end to end:
/// [`crate::operators::rewrite::simplify_many`] interns all formulas in one
/// hash-cons table (equal subtrees become one `Arc` even across formulas),
/// and `detach_many` detaches through one memo (a per-formula detach would
/// split the shared `Arc` back apart).
///
/// **Scheduling** — the reason ops can mix at all. `Sum` is a single-pass
/// accumulator; `LogSumExp` needs two (a running max, then exp-sums against
/// it). So the sweep runs pass 1 over every item — accumulating `Sum` items
/// to completion and `LogSumExp` items' maxes — and pass 2 over the
/// `LogSumExp` items only. A sweep with no `LogSumExp` item makes ONE grid
/// pass ([`FusedSweep::passes`]) — and note a `Sum` item pays ONE tile
/// evaluation even in a two-pass sweep, because pass 2 skips it.
///
/// Unfused, `R` reductions cost `R` separate sweeps each re-evaluating the
/// shared subtree; fused, at most 2 passes with one evaluation per tile.
///
/// Requirements, validated: `LogSumExp` formulas pair-scalar (dim 1 — `Sum`
/// formulas may carry any feature dim), same `m`/`n`/batch/dtype/device (the
/// per-tile memo may hand a `Scalar`/`EyeIdentity` tile evaluated under one
/// formula's meta to another, and those depend on batch/dtype/device only).
pub struct FusedSweep {
    pub items: Vec<FusedItem>,
    pub cfg: BlockCfg,
}

/// Jointly simplified + hoisted items, their metas, the shared grid, and
/// each item's predicted output shape.
struct Prepared {
    items: Vec<FusedItem>,
    metas: Vec<Meta>,
    m: usize,
    n: usize,
    out_shapes: Vec<Vec<usize>>,
}

impl FusedSweep {
    pub fn new(items: Vec<FusedItem>) -> Self {
        Self {
            items,
            cfg: BlockCfg::default(),
        }
    }
    pub fn with_cfg(mut self, cfg: BlockCfg) -> Self {
        self.cfg = cfg;
        self
    }

    /// Grid passes this sweep makes: 2 when any item is a `LogSumExp`, else 1.
    pub fn passes(&self) -> usize {
        Self::passes_for(self.items.iter().map(|it| it.op))
    }

    /// Passes a sweep over `ops` would make — the scheduling rule, exposed
    /// so callers (and tests) can reason about cost without building one.
    pub fn passes_for(ops: impl IntoIterator<Item = ReduceOp>) -> usize {
        if ops.into_iter().any(|op| op == ReduceOp::LogSumExp) {
            2
        } else {
            1
        }
    }

    /// Jointly simplified formulas + validated common grid + output shapes.
    fn prepare(&self) -> Result<Prepared> {
        if self.items.is_empty() {
            return Err(candle_core::Error::Msg("fused sweep: no reductions".into()));
        }
        let roots: Vec<Arc<Expr>> = self
            .items
            .iter()
            .map(|it| it.formula.expr().clone())
            .collect();
        let simplified = crate::operators::rewrite::simplify_many(&roots)?;
        // Joint hoist: an invariant subtree shared across the fused formulas
        // is evaluated once and spliced as the SAME leaf node.
        let hoisted = crate::operators::symbolic::hoist_many(&simplified)?;
        let items: Vec<FusedItem> = hoisted
            .into_iter()
            .zip(self.items.iter())
            .map(|(e, it)| FusedItem {
                formula: LazyTensor::from_arc(e),
                op: it.op,
                axis: it.axis,
            })
            .collect();

        let metas: Vec<Meta> = items
            .iter()
            .map(|it| it.formula.infer())
            .collect::<Result<Vec<_>>>()?;
        let m0 = &metas[0];
        let (m, n) = (
            m0.m.ok_or_else(|| candle_core::Error::Msg("fused sweep: no Vi leaf".into()))?,
            m0.n.ok_or_else(|| candle_core::Error::Msg("fused sweep: no Vj leaf".into()))?,
        );
        for (it, mt) in items.iter().zip(&metas) {
            if it.op == ReduceOp::LogSumExp && mt.dim != 1 {
                return Err(candle_core::Error::Msg(format!(
                    "fused sweep: LogSumExp needs a pair-scalar formula, got dim {}",
                    mt.dim
                )));
            }
            if mt.m != Some(m) || mt.n != Some(n) || mt.batch != m0.batch || mt.dtype != m0.dtype {
                return Err(candle_core::Error::Msg(
                    "fused sweep: all reductions must share one (m, n) grid, batch and dtype"
                        .into(),
                ));
            }
        }
        let out_shapes = items
            .iter()
            .zip(&metas)
            .map(|(it, mt)| out_shape(mt, it, m, n))
            .collect();
        Ok(Prepared {
            items,
            metas,
            m,
            n,
            out_shapes,
        })
    }

    /// Evaluate with NO tape. `Sum` items come back `[..B, kept, d]`,
    /// `LogSumExp` items `[..B, kept]`.
    pub fn eval_detached(&self) -> Result<Vec<Tensor>> {
        let p = self.prepare()?;
        fused_sweep(&detach_items(&p.items), p.m, p.n, &self.cfg, &p.out_shapes)
    }

    /// Evaluate as ONE differentiable node over the UNION leaf pack of all
    /// fused formulas; the backward derives each reduction's own cotangent
    /// (`Sum`: the output cotangent as a kept-axis leaf; `LogSumExp`: the
    /// softmax cotangent), merges the per-leaf buckets across reductions by
    /// node pointer, and evaluates each merged bucket once.
    pub fn eval(&self) -> Result<Vec<Tensor>> {
        let p = self.prepare()?;
        let leaves = collect_leaf_nodes_many(p.items.iter().map(|it| it.formula.expr()));
        if leaves.is_empty() {
            // No leaf ⇒ no Vi/Vj ⇒ `prepare` already rejected it; kept as
            // the total guard so the pack below is never empty.
            return fused_sweep(&detach_items(&p.items), p.m, p.n, &self.cfg, &p.out_shapes);
        }
        let tensors: Vec<Tensor> = leaves.iter().map(|l| l.data.clone()).collect();
        let (packed, _shapes) = pack_leaves(&tensors)?;
        // Outputs have MIXED rank (`Sum` carries a feature axis, `LogSumExp`
        // does not), so the node returns them flattened and concatenated and
        // the split below reshapes — `narrow` + `reshape` both carry gradient.
        let flat = packed.contiguous()?.apply_op1(FusedSweepOp {
            items: p.items,
            cfg: self.cfg.clone(),
            out_shapes: p.out_shapes.clone(),
            m: p.m,
            n: p.n,
        })?;
        let mut out = Vec::with_capacity(p.out_shapes.len());
        let mut off = 0usize;
        for s in &p.out_shapes {
            let len = numel(s);
            out.push(flat.narrow(0, off, len)?.reshape(s.clone())?);
            off += len;
        }
        Ok(out)
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>()
}

/// The shape a reduction lands on: the formula's batch prefix, the kept
/// extent, and — for `Sum` only — the formula's feature dim.
fn out_shape(meta: &Meta, it: &FusedItem, m: usize, n: usize) -> Vec<usize> {
    let mut s = meta.batch.clone();
    s.push(match it.axis {
        ReduceAxis::J => m,
        ReduceAxis::I => n,
    });
    if it.op == ReduceOp::Sum {
        s.push(meta.dim);
    }
    s
}

/// Detach every item's leaves through ONE memo, so a subtree shared across
/// formulas stays one `Arc` on the detached side too.
fn detach_items(items: &[FusedItem]) -> Vec<FusedItem> {
    let roots: Vec<Arc<Expr>> = items.iter().map(|it| it.formula.expr().clone()).collect();
    crate::operators::symbolic::detach_many(&roots)
        .into_iter()
        .zip(items)
        .map(|(e, it)| FusedItem {
            formula: LazyTensor::from_arc(e),
            op: it.op,
            axis: it.axis,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// FusedLse — the all-LogSumExp special case, kept as the named entry point
// ---------------------------------------------------------------------------

/// Several LogSumExp reductions over one pair grid — [`FusedSweep`] with
/// every item a `LogSumExp`. The two-pass schedule and every guarantee
/// (shared per-tile memo, union leaf pack, merged cotangent buckets) are
/// [`FusedSweep`]'s; this is the ergonomic constructor for the case that
/// motivated fusion (Sinkhorn's Jacobi potential pairs).
pub struct FusedLse {
    pub items: Vec<(LazyTensor, ReduceAxis)>,
    pub cfg: BlockCfg,
}

impl FusedLse {
    pub fn new(items: Vec<(LazyTensor, ReduceAxis)>) -> Self {
        Self {
            items,
            cfg: BlockCfg::default(),
        }
    }
    pub fn with_cfg(mut self, cfg: BlockCfg) -> Self {
        self.cfg = cfg;
        self
    }

    fn sweep(&self) -> FusedSweep {
        FusedSweep {
            items: self
                .items
                .iter()
                .map(|(f, a)| FusedItem::lse(f.clone(), *a))
                .collect(),
            cfg: self.cfg.clone(),
        }
    }

    /// Evaluate with NO tape.
    pub fn eval_detached(&self) -> Result<Vec<Tensor>> {
        self.sweep().eval_detached()
    }

    /// Evaluate as ONE differentiable node — see [`FusedSweep::eval`].
    pub fn eval(&self) -> Result<Vec<Tensor>> {
        self.sweep().eval()
    }
}

/// Leaf nodes of SEVERAL formulas through one seen-set: the union pack, in
/// formula order, each shared node once.
fn collect_leaf_nodes_many<'a>(roots: impl Iterator<Item = &'a Arc<Expr>>) -> Vec<LeafInfo> {
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    fn walk(e: &Arc<Expr>, seen: &mut HashMap<*const Expr, ()>, out: &mut Vec<LeafInfo>) {
        if seen.insert(Arc::as_ptr(e), ()).is_some() {
            return;
        }
        match e.as_ref() {
            Expr::Leaf { data, axis } => out.push(LeafInfo {
                ptr: Arc::as_ptr(e),
                data: data.clone(),
                axis: *axis,
            }),
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
            | Expr::Affine { e: a, .. } => walk(a, seen, out),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b)
            | Expr::MatDot(a, b) => {
                walk(a, seen, out);
                walk(b, seen, out);
            }
        }
    }
    for r in roots {
        walk(r, &mut seen, &mut out);
    }
    out
}

/// Tiles for a sweep that shares ONE per-tile memo across several formulas,
/// so the working set is the UNION of their footprints — a shared cost
/// subtree counted once, not once per formula.
///
/// Explicit `bi`/`bj` overrides win outright: they are the only way a test
/// can force a multi-tile sweep on a grid that would otherwise fit the
/// budget in one shot.
fn sweep_tiles(
    items: &[FusedItem],
    m: usize,
    n: usize,
    cfg: &BlockCfg,
    dtype: DType,
) -> (usize, usize) {
    if let (Some(bi), Some(bj)) = (cfg.bi, cfg.bj) {
        return (bi.min(m).max(1), bj.min(n).max(1));
    }
    let Ok(fp) = super::schedule::footprint_many(items.iter().map(|it| it.formula.expr())) else {
        return cfg.tiles(m, n, 1, dtype);
    };
    if let Some((bi, bj)) =
        super::schedule::tile_tuner().and_then(|t| t.tiles(&fp, m, n, cfg.budget_bytes))
    {
        return (bi.clamp(1, m.max(1)), bj.clamp(1, n.max(1)));
    }
    if cfg.force_dense || fp.bytes(m.max(1), n.max(1)) <= cfg.budget_bytes {
        return (m.max(1), n.max(1));
    }
    let s = fp.square_side(cfg.budget_bytes);
    (s.min(m).max(1), s.min(n).max(1))
}

/// Tile evaluations performed by [`fused_sweep`], per thread — the schedule
/// probe. Thread-local rather than a global counter so the assertion is
/// race-free under the test harness's one-thread-per-test execution.
#[cfg(test)]
thread_local! {
    pub(crate) static TILE_EVALS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn note_tile_eval() {
    #[cfg(test)]
    TILE_EVALS.with(|c| c.set(c.get() + 1));
}

/// The scheduled fused sweep.
///
/// Pass 1 visits every tile once and, from ONE shared memo per tile, both
/// accumulates the `Sum` items to completion and takes the `LogSumExp`
/// items' running maxes (row-max for axis-J, col-max for axis-I). Pass 2
/// re-visits the tiles for the `LogSumExp` items only, accumulating exp-sums
/// against those now-fixed maxes — and is SKIPPED entirely when no item is a
/// `LogSumExp`.
///
/// Formulas must be jointly interned — sharing works through the per-tile
/// memo, which is keyed by node pointer. `expect_shapes` is the caller's
/// predicted output shape per item, checked here so the flat packing in
/// [`FusedSweep::eval`] can never silently mis-split.
fn fused_sweep(
    items: &[FusedItem],
    m: usize,
    n: usize,
    cfg: &BlockCfg,
    expect_shapes: &[Vec<usize>],
) -> Result<Vec<Tensor>> {
    let metas: Vec<Meta> = items
        .iter()
        .map(|it| it.formula.infer())
        .collect::<Result<Vec<_>>>()?;
    let (bi, bj) = sweep_tiles(items, m, n, cfg, metas[0].dtype);
    let blocks = |total: usize, step: usize| -> Vec<Range<usize>> {
        let mut v = Vec::new();
        let mut o = 0;
        while o < total {
            let r = o..(o + step).min(total);
            o = r.end;
            v.push(r);
        }
        v
    };
    let i_blocks = blocks(m, bi);
    let j_blocks = blocks(n, bj);

    // `[..B, bi, bj, d]` — the raw tile, feature axis intact (`Sum` needs it).
    let tile = |r: usize, i: &Range<usize>, j: &Range<usize>, memo: &mut Memo| -> Result<Tensor> {
        note_tile_eval();
        eval_block(items[r].formula.expr(), &metas[r], i, j, memo)
    };
    let kept_blocks = |r: usize| -> usize {
        if items[r].axis == ReduceAxis::J {
            i_blocks.len()
        } else {
            j_blocks.len()
        }
    };

    // Per item, per kept block: `Sum` partials, and the `LogSumExp` running max.
    let mut sums: Vec<Vec<Vec<Tensor>>> = (0..items.len())
        .map(|r| vec![Vec::new(); kept_blocks(r)])
        .collect();
    let mut maxes: Vec<Vec<Option<Tensor>>> = (0..items.len())
        .map(|r| vec![None; kept_blocks(r)])
        .collect();

    // ---- pass 1: Sum to completion, LogSumExp running maxes ----------------
    for (ib, i) in i_blocks.iter().enumerate() {
        for (jb, j) in j_blocks.iter().enumerate() {
            let mut memo: Memo = HashMap::new();
            for r in 0..items.len() {
                let t = tile(r, i, j, &mut memo)?;
                match items[r].op {
                    ReduceOp::Sum => {
                        // Contract the reduced pair axis, keep the features.
                        let (slot, red) = match items[r].axis {
                            ReduceAxis::J => (ib, t.rank() - 2),
                            ReduceAxis::I => (jb, t.rank() - 3),
                        };
                        sums[r][slot].push(t.sum(red)?);
                    }
                    ReduceOp::LogSumExp => {
                        let t = t.squeeze(D::Minus1)?; // [..B, bi, bj]
                        let (slot, mx) = match items[r].axis {
                            ReduceAxis::J => (ib, t.max_keepdim(D::Minus1)?), // [..B, bi, 1]
                            ReduceAxis::I => (jb, t.max_keepdim(D::Minus2)?), // [..B, 1, bj]
                        };
                        maxes[r][slot] = Some(match maxes[r][slot].take() {
                            Some(prev) => prev.maximum(&mx)?,
                            None => mx,
                        });
                    }
                }
            }
        }
    }

    // ---- pass 2: exp-sums against the fixed maxes — LogSumExp items only ---
    let lse: Vec<usize> = (0..items.len())
        .filter(|&r| items[r].op == ReduceOp::LogSumExp)
        .collect();
    let mut parts: Vec<Vec<Vec<Tensor>>> = (0..items.len())
        .map(|r| vec![Vec::new(); kept_blocks(r)])
        .collect();
    if !lse.is_empty() {
        for (ib, i) in i_blocks.iter().enumerate() {
            for (jb, j) in j_blocks.iter().enumerate() {
                let mut memo: Memo = HashMap::new();
                for &r in &lse {
                    let t = tile(r, i, j, &mut memo)?.squeeze(D::Minus1)?;
                    match items[r].axis {
                        ReduceAxis::J => {
                            let mx = maxes[r][ib].as_ref().expect("pass 1 filled");
                            parts[r][ib].push(t.broadcast_sub(mx)?.exp()?.sum(D::Minus1)?);
                        }
                        ReduceAxis::I => {
                            let mx = maxes[r][jb].as_ref().expect("pass 1 filled");
                            parts[r][jb].push(t.broadcast_sub(mx)?.exp()?.sum(D::Minus2)?);
                        }
                    }
                }
            }
        }
    }

    // ---- combine per kept block, cat over blocks ---------------------------
    let mut out = Vec::with_capacity(items.len());
    for (r, it) in items.iter().enumerate() {
        let y = match it.op {
            ReduceOp::Sum => {
                // `stack_sum` per block keeps backward depth O(1) in the
                // tile count — the invariant the whole tiling layer rests on.
                let pieces: Vec<Tensor> = sums[r]
                    .iter()
                    .map(|p| stack_sum(p))
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&Tensor> = pieces.iter().collect();
                Tensor::cat(&refs, D::Minus2)?
            }
            ReduceOp::LogSumExp => {
                let red = match it.axis {
                    ReduceAxis::J => D::Minus1,
                    ReduceAxis::I => D::Minus2,
                };
                let mut pieces = Vec::with_capacity(parts[r].len());
                for (kb, p) in parts[r].iter().enumerate() {
                    let total = stack_sum(p)?;
                    let mx = maxes[r][kb].as_ref().expect("pass 1 filled").squeeze(red)?;
                    pieces.push(total.log()?.add(&mx)?);
                }
                let refs: Vec<&Tensor> = pieces.iter().collect();
                Tensor::cat(&refs, D::Minus1)?
            }
        };
        if y.dims() != expect_shapes[r].as_slice() {
            return Err(candle_core::Error::Msg(format!(
                "fused sweep: reduction {r} landed on {:?}, predicted {:?}",
                y.dims(),
                expect_shapes[r]
            )));
        }
        out.push(y);
    }
    Ok(out)
}

struct FusedSweepOp {
    items: Vec<FusedItem>, // jointly simplified + hoisted, LIVE leaves
    cfg: BlockCfg,
    out_shapes: Vec<Vec<usize>>,
    m: usize,
    n: usize,
}

impl FusedSweepOp {
    /// Slice one item's `[..]`-shaped block out of the flat packed output.
    fn slice(&self, flat: &Tensor, r: usize) -> Result<Tensor> {
        let off: usize = self.out_shapes[..r].iter().map(|s| numel(s)).sum();
        flat.narrow(0, off, numel(&self.out_shapes[r]))?
            .reshape(self.out_shapes[r].clone())
    }

    /// The detached sweep, flat-concatenated. Outputs have MIXED rank (a `Sum`
    /// item is vector-valued, an LSE item is pair-scalar), so the node returns
    /// one flat buffer and `eval` narrows it back — shared by both forward
    /// arms so the two cannot disagree about the packing.
    fn flat_detached(&self) -> Result<Tensor> {
        let items = detach_items(&self.items);
        let outs = fused_sweep(&items, self.m, self.n, &self.cfg, &self.out_shapes)?;
        let flats: Vec<Tensor> = outs
            .iter()
            .map(|t| t.flatten_all())
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = flats.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

impl CustomOp1 for FusedSweepOp {
    fn name(&self) -> &'static str {
        "fused_sweep"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, _l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let as_f32 = matches!(s1, CpuStorage::F32(_));
        let cat = self.flat_detached()?;
        let shape = cat.shape().clone();
        let host: Vec<f64> = cat
            .to_dtype(candle_core::DType::F64)?
            .flatten_all()?
            .to_vec1()?;
        Ok((crate::linalg_diff::storage_from_f64(host, as_f32), shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _s1: &candle_core::CudaStorage,
        _l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_storage_of(&self.flat_detached()?, "fused_sweep")
    }

    fn bwd(&self, packed: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Split the flat output/cotangent back per reduction, derive each
        // one's own cotangent formula, and MERGE the per-leaf buckets across
        // reductions by node pointer — the jointly-interned DAGs make the
        // pointers line up.
        let (res, grad_res) = (res.detach(), grad_res.detach());
        let mut merged: HashMap<*const Expr, LazyTensor> = HashMap::new();
        for (r, it) in self.items.iter().enumerate() {
            let g_r = self.slice(&grad_res, r)?.contiguous()?;
            let kept_leaf = |t: &Tensor| -> LazyTensor {
                match it.axis {
                    ReduceAxis::J => LazyTensor::vi(t),
                    ReduceAxis::I => LazyTensor::vj(t),
                }
            };
            let root_cot = match it.op {
                // The output cotangent enters on the kept axis, already
                // carrying the feature dim the formula reduces over.
                ReduceOp::Sum => kept_leaf(&g_r),
                // ∂/∂F log Σ exp F = softmax: G ⊙ exp(F − out).
                ReduceOp::LogSumExp => {
                    let o_r = self.slice(&res, r)?.contiguous()?;
                    let gl = kept_leaf(&g_r.unsqueeze(D::Minus1)?.contiguous()?);
                    let ol = kept_leaf(&o_r.unsqueeze(D::Minus1)?.contiguous()?);
                    &gl * &(&it.formula - &ol).exp()
                }
            };
            for (ptr, bucket) in vjp(it.formula.expr(), root_cot)? {
                match merged.remove(&ptr) {
                    Some(prev) => {
                        merged.insert(ptr, &prev + &bucket);
                    }
                    None => {
                        merged.insert(ptr, bucket);
                    }
                }
            }
        }

        let leaves = collect_leaf_nodes_many(self.items.iter().map(|it| it.formula.expr()));
        let mut flat = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            flat.push(
                eval_leaf_grad(merged.get(&leaf.ptr), leaf, self.m, self.n, &self.cfg)?
                    .flatten_all()?,
            );
        }
        let packed_bar = Tensor::cat(&flat, 0)?.to_dtype(packed.dtype())?;
        Ok(Some(packed_bar))
    }
}

#[cfg(test)]
mod tests {
    //! Verification of tiled symbolic sum and log-sum-exp reductions with symbolic vector-jacobian products.

    use super::*;
    use candle_core::{DType, Device, Var};

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

    fn pts(n: usize, d: usize, scale: f64) -> Tensor {
        Tensor::from_vec(
            (0..n * d)
                .map(|i| ((i * 31 % 17) as f64) / 8.0 * scale - 1.0)
                .collect::<Vec<_>>(),
            (n, d),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// RBF pair formula `exp(−γ‖xᵢ−xⱼ‖²)` (+ optional δI).
    fn rbf(x: &Tensor, gamma: &Tensor, delta: Option<f64>) -> LazyTensor {
        let xi = LazyTensor::vi(x);
        let xj = LazyTensor::vj(x);
        let diff = &xi - &xj;
        let d2 = (&diff * &diff).sum_last();
        let k = (&d2 * &LazyTensor::pm(gamma)).affine(-1.0, 0.0).exp();
        match delta {
            Some(d) => &k + &LazyTensor::eye_identity(d),
            None => k,
        }
    }

    #[test]
    fn sum_genred_matches_matmul_and_sum_reduce() {
        let n = 14;
        let x = pts(n, 2, 1.0);
        let g = Tensor::full(0.6f64, (1, 1), &Device::Cpu).unwrap();
        let k = rbf(&x, &g, Some(0.3));
        let cfg = BlockCfg::default();

        // Vector-valued: Σ_j k_ij · rhs_j == the tiled matmul.
        let rhs = pts(n, 3, 0.7);
        let node = Genred::new(&k * &LazyTensor::vj(&rhs), ReduceOp::Sum, ReduceAxis::J);
        assert!(
            max_abs(
                &node.eval_detached().unwrap(),
                &k.matmul(&rhs, &cfg).unwrap()
            ) < 1e-12
        );

        // Pair-scalar over J == sum_reduce (modulo the trailing singleton).
        let node = Genred::new(k.clone(), ReduceOp::Sum, ReduceAxis::J);
        let got = node.eval_detached().unwrap().squeeze(D::Minus1).unwrap();
        assert!(max_abs(&got, &k.sum_reduce(ReduceAxis::J, &cfg).unwrap()) < 1e-12);

        // And over I.
        let node = Genred::new(k.clone(), ReduceOp::Sum, ReduceAxis::I);
        let got = node.eval_detached().unwrap().squeeze(D::Minus1).unwrap();
        assert!(max_abs(&got, &k.sum_reduce(ReduceAxis::I, &cfg).unwrap()) < 1e-12);
    }

    #[test]
    fn lse_genred_matches_logsumexp_reduce() {
        let n = 12;
        let x = pts(n, 2, 1.3);
        let g = Tensor::full(0.9f64, (1, 1), &Device::Cpu).unwrap();
        // −γ d² directly (no exp): a well-scaled LSE input.
        let xi = LazyTensor::vi(&x);
        let xj = LazyTensor::vj(&x);
        let diff = &xi - &xj;
        let f = (&(&diff * &diff).sum_last() * &LazyTensor::pm(&g)).affine(-1.0, 0.0);
        for axis in [ReduceAxis::J, ReduceAxis::I] {
            let node = Genred::new(f.clone(), ReduceOp::LogSumExp, axis);
            let want = f.logsumexp_reduce(axis, &BlockCfg::default()).unwrap();
            assert!(max_abs(&node.eval_detached().unwrap(), &want) < 1e-12);
        }
    }

    /// THE core test: the symbolic VJP against candle's tape over the very
    /// same tiled evaluation. Every leaf, both paths, near machine
    /// precision — any rule transcription error shows here first.
    #[test]
    fn symbolic_vjp_matches_the_tape() {
        let n = 10;
        let x0 = pts(n, 2, 1.0);
        let g0 = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let r0 = pts(n, 2, 0.9);
        let w = pts(n, 2, 1.1); // fixed cotangent seed

        let grads_of = |symbolic: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let xv = Var::from_tensor(&x0).unwrap();
            let gv = Var::from_tensor(&g0).unwrap();
            let rv = Var::from_tensor(&r0).unwrap();
            let k = rbf(xv.as_tensor(), gv.as_tensor(), Some(0.3));
            let y = if symbolic {
                Genred::new(
                    &k * &LazyTensor::vj(rv.as_tensor()),
                    ReduceOp::Sum,
                    ReduceAxis::J,
                )
                .eval()
                .unwrap()
            } else {
                k.matmul(rv.as_tensor(), &BlockCfg::default()).unwrap()
            };
            let loss = y.mul(&w).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let take = |v: &Var| -> Vec<f64> {
                grads
                    .get(v.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            (take(&xv), take(&gv), take(&rv))
        };

        let (sx, sg, sr) = grads_of(true);
        let (tx, tg, tr) = grads_of(false);
        for (name, s, t) in [("x", &sx, &tx), ("gamma", &sg, &tg), ("rhs", &sr, &tr)] {
            for (i, (a, b)) in s.iter().zip(t.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                    "{name}[{i}]: symbolic {a} vs tape {b}"
                );
            }
        }
    }

    // -- Genred::grad — gradients as differentiable VALUES -------------------

    /// `grad()` must reproduce exactly what the tape produces. Same fixture
    /// and same seed as `symbolic_vjp_matches_the_tape`, so a divergence
    /// between the two paths shows up as a difference against a known-good
    /// reference rather than against itself.
    #[test]
    fn genred_grad_values_match_the_tape() {
        let n = 10;
        let x0 = pts(n, 2, 1.0);
        let g0 = Tensor::full(0.7f64, (1, 1), &Device::Cpu).unwrap();
        let r0 = pts(n, 2, 0.9);
        let w = pts(n, 2, 1.1);

        // Tape reference.
        let xv = Var::from_tensor(&x0).unwrap();
        let gv = Var::from_tensor(&g0).unwrap();
        let rv = Var::from_tensor(&r0).unwrap();
        let k = rbf(xv.as_tensor(), gv.as_tensor(), Some(0.3));
        let f = &k * &LazyTensor::vj(rv.as_tensor());
        let y = Genred::new(f.clone(), ReduceOp::Sum, ReduceAxis::J)
            .eval()
            .unwrap();
        let tape = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();

        // `grad()` with the same seed `w`.
        let node = Genred::new(f, ReduceOp::Sum, ReduceAxis::J);
        let gg = node.grad(&w).unwrap();
        assert_eq!(gg.leaves.len(), gg.grads.len());

        // `for_tensor`, not indexing: `x` is used at BOTH axes, so it is two
        // leaf nodes each holding a partial. Comparing either one alone
        // against the tape's total is the mistake this API exists to stop.
        for (name, v) in [("x", &xv), ("gamma", &gv), ("rhs", &rv)] {
            let got: Vec<f64> = gg
                .for_tensor(v.as_tensor())
                .unwrap_or_else(|| panic!("{name} missing from grad()"))
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let want: Vec<f64> = tape
                .get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(got.len(), want.len(), "{name}");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                    "{name}[{i}]: grad() {a} vs tape {b}"
                );
            }
        }

        // And the per-node split really is a split: `x` occupies two leaves.
        let x_nodes = gg
            .leaves
            .iter()
            .filter(|l| l.id() == xv.as_tensor().id())
            .count();
        assert_eq!(
            x_nodes, 2,
            "x is used at both axes, so it is two leaf nodes"
        );
    }

    /// The one that matters: `grad()`'s output must carry a TAPE.
    ///
    /// Values alone cannot catch this. `Genred::eval` short-circuits to
    /// `eval_detached` on an empty leaf set, so a `grad()` that detached its
    /// buckets would return perfectly correct numbers, no error, and no way
    /// to differentiate again — which is the entire capability being added.
    #[test]
    fn genred_grad_output_is_itself_differentiable() {
        let n = 6;
        let x0 = pts(n, 2, 1.0);
        let g0 = Tensor::full(0.5f64, (1, 1), &Device::Cpu).unwrap();
        let w = Tensor::ones((n, 1), DType::F64, &Device::Cpu).unwrap();

        let gv = Var::from_tensor(&g0).unwrap();
        let k = rbf(&x0, gv.as_tensor(), Some(0.3));
        let gg = Genred::new(k, ReduceOp::Sum, ReduceAxis::J)
            .grad(&w)
            .unwrap();

        // Contract to a scalar and differentiate the GRADIENT — second order.
        let dirs: Vec<Tensor> = gg.leaves.iter().map(|l| l.ones_like().unwrap()).collect();
        let s = gg.directional(&dirs).unwrap();
        let second = s.backward().unwrap();

        let d2 = second
            .get(gv.as_tensor())
            .expect("no second-order gradient reached gamma — grad() returned detached values");
        let v: f64 = d2.flatten_all().unwrap().to_vec1::<f64>().unwrap()[0];
        assert!(
            v.is_finite() && v != 0.0,
            "second derivative is {v}, expected finite non-zero"
        );
    }

    /// The rung-7 gate: the SECOND derivative must be right, not merely
    /// present. Checked two ways, because each catches what the other misses.
    ///
    /// `L(γ) = Σ_ij exp(−γ·d²ᵢⱼ)`, so `∂L/∂γ = −Σ d²·exp(−γd²)` and
    /// `∂²L/∂γ² = Σ d⁴·exp(−γd²)` in closed form — that pins the VALUE.
    /// Central-differencing `grad()` itself pins that the second derivative
    /// is the derivative OF THE FIRST as this code computes it, which an
    /// analytic check alone would not (both could be right about a function
    /// nobody is actually evaluating). F64 throughout: f32 central
    /// differences on a second derivative are hopeless.
    #[test]
    fn genred_grad_second_derivative_matches_analytic_and_finite_differences() {
        let n = 7;
        let x = pts(n, 2, 1.0);
        let dev = Device::Cpu;
        let w = Tensor::ones((n, 1), DType::F64, &dev).unwrap();
        let gamma0 = 0.6f64;

        // ∂L/∂γ at an arbitrary γ, via grad().
        let dl_dgamma = |g: f64| -> f64 {
            let gt = Tensor::full(g, (1, 1), &dev).unwrap();
            let node = Genred::new(rbf(&x, &gt, None), ReduceOp::Sum, ReduceAxis::J);
            let gg = node.grad(&w).unwrap();
            gg.for_tensor(&gt)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()[0]
        };

        // Second order: differentiate grad()'s OUTPUT.
        let gv = Var::from_tensor(&Tensor::full(gamma0, (1, 1), &dev).unwrap()).unwrap();
        let node = Genred::new(rbf(&x, gv.as_tensor(), None), ReduceOp::Sum, ReduceAxis::J);
        let gg = node.grad(&w).unwrap();
        let g_scalar = gg.for_tensor(gv.as_tensor()).unwrap().sum_all().unwrap();
        let second = g_scalar.backward().unwrap();
        let got: f64 = second
            .get(gv.as_tensor())
            .expect("grad() output was not differentiable")
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()[0];

        // Analytic: Σ_ij d⁴ exp(−γ d²).
        let xs: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
        let mut analytic = 0.0;
        for i in 0..n {
            for j in 0..n {
                let d2 = (0..2)
                    .map(|c| {
                        let t = xs[i * 2 + c] - xs[j * 2 + c];
                        t * t
                    })
                    .sum::<f64>();
                analytic += d2 * d2 * (-gamma0 * d2).exp();
            }
        }
        assert!(
            (got - analytic).abs() < 1e-8 * (1.0 + analytic.abs()),
            "second derivative {got} vs analytic {analytic}"
        );

        // …and against central differences of the first derivative.
        let h = 1e-5;
        let fd = (dl_dgamma(gamma0 + h) - dl_dgamma(gamma0 - h)) / (2.0 * h);
        assert!(
            (got - fd).abs() < 1e-5 * (1.0 + fd.abs()),
            "second derivative {got} vs central difference {fd}"
        );
    }

    /// A cotangent bucket that is constant along a pair axis has no `Vi`/`Vj`
    /// leaf of its own, and `Genred::eval` — unlike `sum_reduce_vec` — has no
    /// `fallback_extent` to fall back on. Without the injected ones-leaf this
    /// dies with "no Vi leaf and no fallback extent".
    ///
    /// A `Pm`-only formula is the sharpest case: EVERY bucket is
    /// axis-constant.
    #[test]
    fn genred_grad_handles_buckets_that_are_constant_along_an_axis() {
        let n = 5;
        let dev = Device::Cpu;
        let a = Tensor::full(1.3f64, (1, 1), &dev).unwrap();
        let av = Var::from_tensor(&a).unwrap();
        let w = Tensor::ones((n, 1), DType::F64, &dev).unwrap();

        // The pair grid comes from the vi/vj leaves; the differentiated
        // parameter is Pm-only, so its cotangent mentions neither axis.
        let x = pts(n, 1, 1.0);
        let f = &(&LazyTensor::vi(&x) + &LazyTensor::vj(&x)) * &LazyTensor::pm(av.as_tensor());
        let gg = Genred::new(f, ReduceOp::Sum, ReduceAxis::J)
            .grad(&w)
            .unwrap();

        let idx = gg
            .leaves
            .iter()
            .position(|l| l.id() == av.as_tensor().id())
            .expect("the Pm leaf should be present");
        let g: f64 = gg.grads[idx]
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()[0];
        // d/da Σ_ij (xi + xj)·a = Σ_ij (xi + xj) = 2n·Σx.
        let xs: Vec<f64> = x.flatten_all().unwrap().to_vec1().unwrap();
        let want: f64 = 2.0 * (n as f64) * xs.iter().sum::<f64>();
        assert!(
            (g - want).abs() < 1e-9 * (1.0 + want.abs()),
            "Pm grad {g} vs {want}"
        );
    }

    /// FD gradcheck for the LSE node (softmax cotangent path), both leaves.
    #[test]
    fn lse_genred_gradcheck() {
        let n = 8;
        let x0 = pts(n, 2, 1.0);
        let g0 = Tensor::full(0.5f64, (1, 1), &Device::Cpu).unwrap();
        let w = pts(n, 1, 1.0).squeeze(D::Minus1).unwrap();

        let loss_of = |xt: &Tensor, gt: &Tensor| -> Result<f64> {
            let xi = LazyTensor::vi(xt);
            let xj = LazyTensor::vj(xt);
            let diff = &xi - &xj;
            let f = (&(&diff * &diff).sum_last() * &LazyTensor::pm(gt)).affine(-1.0, 0.0);
            let y = Genred::new(f, ReduceOp::LogSumExp, ReduceAxis::J).eval()?;
            y.mul(&w)?.sum_all()?.to_scalar::<f64>()
        };

        let xv = Var::from_tensor(&x0).unwrap();
        let gv = Var::from_tensor(&g0).unwrap();
        let xi = LazyTensor::vi(xv.as_tensor());
        let xj = LazyTensor::vj(xv.as_tensor());
        let diff = &xi - &xj;
        let f = (&(&diff * &diff).sum_last() * &LazyTensor::pm(gv.as_tensor())).affine(-1.0, 0.0);
        let y = Genred::new(f, ReduceOp::LogSumExp, ReduceAxis::J)
            .eval()
            .unwrap();
        let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let gx: Vec<f64> = grads
            .get(xv.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let gg: Vec<f64> = grads
            .get(gv.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let h = 1e-6;
        let fd = |var: &Var, f: &dyn Fn(&Tensor) -> Result<f64>| -> Vec<f64> {
            let base: Vec<f64> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
            let dims = var.shape().dims().to_vec();
            (0..base.len())
                .map(|i| {
                    let mut up = base.clone();
                    up[i] += h;
                    let mut dn = base.clone();
                    dn[i] -= h;
                    let lu = f(&Tensor::from_vec(up, dims.clone(), &Device::Cpu).unwrap()).unwrap();
                    let ld = f(&Tensor::from_vec(dn, dims.clone(), &Device::Cpu).unwrap()).unwrap();
                    (lu - ld) / (2.0 * h)
                })
                .collect()
        };
        let fdx = fd(&xv, &|t| loss_of(t, gv.as_tensor()));
        let fdg = fd(&gv, &|t| loss_of(xv.as_tensor(), t));
        for (i, (a, b)) in gx.iter().zip(&fdx).enumerate() {
            assert!(
                (a - b).abs() < 1e-6 * (1.0 + b.abs()),
                "x[{i}]: {a} vs FD {b}"
            );
        }
        for (i, (a, b)) in gg.iter().zip(&fdg).enumerate() {
            assert!(
                (a - b).abs() < 1e-6 * (1.0 + b.abs()),
                "gamma[{i}]: {a} vs FD {b}"
            );
        }
    }

    /// Fusion parity: two LSE reductions sharing a cost subtree, run fused
    /// (one sweep, shared tile memos, one CustomOp, merged buckets) vs as
    /// two independent Genred nodes. Values and every gradient — including
    /// the SHARED leaves x and γ, whose fused gradient is the merged-bucket
    /// path — must agree to near machine precision.
    #[test]
    fn fused_lse_matches_two_single_genreds() {
        let n_pts = 11;
        let x0 = pts(n_pts, 2, 1.0);
        let g0 = Tensor::full(0.65f64, (1, 1), &Device::Cpu).unwrap();
        let u0 = pts(n_pts, 1, 0.5);
        let v0 = pts(n_pts, 1, 0.4);
        let w_f = pts(n_pts, 1, 1.1).squeeze(D::Minus1).unwrap();
        let w_g = pts(n_pts, 1, 0.9).squeeze(D::Minus1).unwrap();

        let grads_of = |fused: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            let xv = Var::from_tensor(&x0).unwrap();
            let gv = Var::from_tensor(&g0).unwrap();
            let uv = Var::from_tensor(&u0).unwrap();
            let vv = Var::from_tensor(&v0).unwrap();
            // Shared cost: −γ‖xi−xj‖², built ONCE and referenced by both.
            let xi = LazyTensor::vi(xv.as_tensor());
            let xj = LazyTensor::vj(xv.as_tensor());
            let diff = &xi - &xj;
            let c =
                (&(&diff * &diff).sum_last() * &LazyTensor::pm(gv.as_tensor())).affine(-1.0, 0.0);
            let f1 = &c + &LazyTensor::vj(uv.as_tensor()); // reduce J
            let f2 = &c + &LazyTensor::vi(vv.as_tensor()); // reduce I
            let (o1, o2) = if fused {
                let outs = FusedLse::new(vec![(f1, ReduceAxis::J), (f2, ReduceAxis::I)])
                    .with_cfg(BlockCfg::with_tiles(4, 3))
                    .eval()
                    .unwrap();
                (outs[0].clone(), outs[1].clone())
            } else {
                (
                    Genred::new(f1, ReduceOp::LogSumExp, ReduceAxis::J)
                        .with_cfg(BlockCfg::with_tiles(4, 3))
                        .eval()
                        .unwrap(),
                    Genred::new(f2, ReduceOp::LogSumExp, ReduceAxis::I)
                        .with_cfg(BlockCfg::with_tiles(4, 3))
                        .eval()
                        .unwrap(),
                )
            };
            let loss = (o1.mul(&w_f).unwrap().sum_all().unwrap()
                + o2.mul(&w_g).unwrap().sum_all().unwrap())
            .unwrap();
            let grads = loss.backward().unwrap();
            let take = |v: &Var| -> Vec<f64> {
                grads
                    .get(v.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            let vals: Vec<f64> = o1.flatten_all().unwrap().to_vec1().unwrap();
            (vals, take(&xv), take(&gv), take(&uv), take(&vv))
        };

        let (val_f, x_f, g_f, u_f, v_f) = grads_of(true);
        let (val_s, x_s, g_s, u_s, v_s) = grads_of(false);
        for (name, a, b) in [
            ("o1", &val_f, &val_s),
            ("x", &x_f, &x_s),
            ("gamma", &g_f, &g_s),
            ("u", &u_f, &u_s),
            ("v", &v_f, &v_s),
        ] {
            assert_eq!(a.len(), b.len(), "{name}: length mismatch");
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (p - q).abs() < 1e-10 * (1.0 + q.abs()),
                    "{name}[{i}]: fused {p} vs single {q}"
                );
            }
        }
    }

    /// The dense-pair leaf end-to-end: a DENSE cost tensor rides as a Vij
    /// leaf through an LSE Genred (the Sinkhorn softmin shape), and its
    /// gradient — the softmax cotangent materialised `to_dense` — must
    /// match the tape route (tracked tiled logsumexp on the same formula).
    /// Also pins MaxC inside a formula: values match candle `maximum`, and
    /// entries BELOW the floor contribute exactly zero gradient.
    #[test]
    fn vij_dense_cost_lse_matches_the_tape_and_maxc_gates_gradients() {
        let (m, n) = (7, 5);
        let c0 = pts(m, n, 1.0); // dense [M, N] "cost"
        let g0 = pts(n, 1, 0.5);
        let w = pts(m, 1, 1.0).squeeze(D::Minus1).unwrap();
        // Floor chosen so SOME cost entries sit below it (pts spans ~[-1, 1]).
        let floor = -0.2f64;

        let grads_of = |symbolic: bool| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let cv = Var::from_tensor(&c0).unwrap();
            let gv = Var::from_tensor(&g0).unwrap();
            let f = &LazyTensor::vij(cv.as_tensor())
                .max_c(floor)
                .affine(-1.0, 0.0)
                + &LazyTensor::vj(gv.as_tensor());
            let y = if symbolic {
                Genred::new(f, ReduceOp::LogSumExp, ReduceAxis::J)
                    .with_cfg(BlockCfg::with_tiles(3, 2))
                    .eval()
                    .unwrap()
            } else {
                f.logsumexp_reduce(ReduceAxis::J, &BlockCfg::with_tiles(3, 2))
                    .unwrap()
            };
            let vals: Vec<f64> = y.flatten_all().unwrap().to_vec1().unwrap();
            let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
            let take = |v: &Var| -> Vec<f64> {
                grads
                    .get(v.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            (vals, take(&cv), take(&gv))
        };

        let (val_s, c_s, g_s) = grads_of(true);
        let (val_t, c_t, g_t) = grads_of(false);
        for (name, a, b) in [
            ("value", &val_s, &val_t),
            ("cost", &c_s, &c_t),
            ("g", &g_s, &g_t),
        ] {
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (p - q).abs() < 1e-10 * (1.0 + q.abs()),
                    "{name}[{i}]: genred {p} vs tape {q}"
                );
            }
        }
        // The corner-side pin: every cost entry below the floor is gated to
        // exactly zero gradient by GtC.
        let host: Vec<f64> = c0.flatten_all().unwrap().to_vec1().unwrap();
        let mut below = 0;
        for (i, (&ci, &gi)) in host.iter().zip(&c_s).enumerate() {
            if ci < floor {
                below += 1;
                assert!(gi == 0.0, "cost[{i}] = {ci} < floor but grad {gi} != 0");
            }
        }
        assert!(
            below > 0,
            "fixture has no below-floor entries — floor pin is vacuous"
        );
    }

    /// The hoist claim end-to-end: a formula whose bandwidth is a CHAIN of
    /// pm parameters (γ = a·b + c) gets the chain evaluated once and packed
    /// as a computed leaf — and the gradients to a, b, c must still equal
    /// the tape route, because they flow through the hoist chain OUTSIDE
    /// the CustomOp via cat's VJP.
    #[test]
    fn genred_gradients_flow_through_hoisted_parameter_chains() {
        let n = 9;
        let x0 = pts(n, 2, 1.0);
        let r0 = pts(n, 2, 0.7);
        let w = pts(n, 2, 1.1);

        let grads_of = |symbolic: bool| -> Vec<Vec<f64>> {
            let av =
                Var::from_tensor(&Tensor::full(0.4f64, (1, 1), &Device::Cpu).unwrap()).unwrap();
            let bv =
                Var::from_tensor(&Tensor::full(1.5f64, (1, 1), &Device::Cpu).unwrap()).unwrap();
            let cv =
                Var::from_tensor(&Tensor::full(0.2f64, (1, 1), &Device::Cpu).unwrap()).unwrap();
            let gamma = &(&LazyTensor::pm(av.as_tensor()) * &LazyTensor::pm(bv.as_tensor()))
                + &LazyTensor::pm(cv.as_tensor());
            let xi = LazyTensor::vi(&x0);
            let xj = LazyTensor::vj(&x0);
            let diff = &xi - &xj;
            let k = (&(&diff * &diff).sum_last() * &gamma)
                .affine(-1.0, 0.0)
                .exp();
            let y = if symbolic {
                Genred::new(&k * &LazyTensor::vj(&r0), ReduceOp::Sum, ReduceAxis::J)
                    .eval()
                    .unwrap()
            } else {
                k.matmul(&r0, &BlockCfg::default()).unwrap()
            };
            let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
            [&av, &bv, &cv]
                .iter()
                .map(|v| {
                    grads
                        .get(v.as_tensor())
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap()
                })
                .collect()
        };
        let s = grads_of(true);
        let t = grads_of(false);
        for (name, (a, b)) in ["a", "b", "c"].iter().zip(s.iter().zip(t.iter())) {
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (p - q).abs() < 1e-10 * (1.0 + q.abs()),
                    "{name}[{i}]: hoisted-genred {p} vs tape {q}"
                );
            }
        }
    }

    /// CSE merges duplicate leaves (same tensor, same axis) into ONE pack
    /// entry whose bucket accumulates both DAG paths — the gradient must
    /// equal the tape's, which routes each un-merged use separately and
    /// sums through candle.
    #[test]
    fn merged_duplicate_leaves_still_route_full_gradients() {
        let n = 9;
        let x0 = pts(n, 2, 1.0);
        let r0 = pts(n, 2, 0.8);
        let w = pts(n, 2, 1.2);

        let grads_of = |symbolic: bool| -> Vec<f64> {
            let xv = Var::from_tensor(&x0).unwrap();
            // vi(x) deliberately built TWICE (separate leaf nodes, same
            // tensor): k = exp(−‖xi−xj‖²) + ⟨xi', xj⟩·0.1 with xi' a second
            // vi(x) node. CSE merges them; the tape path keeps them apart.
            let xi_a = LazyTensor::vi(xv.as_tensor());
            let xi_b = LazyTensor::vi(xv.as_tensor());
            let xj = LazyTensor::vj(xv.as_tensor());
            let diff = &xi_a - &xj;
            let k = &(&diff * &diff).sum_last().affine(-1.0, 0.0).exp()
                + &(&xi_b * &xj).sum_last().affine(0.1, 0.0);
            let y = if symbolic {
                Genred::new(&k * &LazyTensor::vj(&r0), ReduceOp::Sum, ReduceAxis::J)
                    .eval()
                    .unwrap()
            } else {
                k.matmul(&r0, &BlockCfg::default()).unwrap()
            };
            let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
            grads
                .get(xv.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let s = grads_of(true);
        let t = grads_of(false);
        for (i, (a, b)) in s.iter().zip(&t).enumerate() {
            assert!(
                (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                "merged-leaf grad[{i}]: symbolic {a} vs tape {b}"
            );
        }
    }

    /// MIXED-OP FUSION, the core test: a vector-valued `Sum` and two `LSE`
    /// reductions over the SAME grid, all three sharing one cost subtree,
    /// run as one fused sweep vs as three independent `Genred` nodes.
    /// Values and every gradient — including the shared `x`/`γ`, whose fused
    /// gradient is the merged-bucket path across reductions of DIFFERENT op
    /// — must agree to near machine precision.
    #[test]
    fn fused_sweep_mixes_sum_and_lse_against_three_single_genreds() {
        let n_pts = 11;
        let x0 = pts(n_pts, 2, 1.0);
        let g0 = Tensor::full(0.65f64, (1, 1), &Device::Cpu).unwrap();
        let u0 = pts(n_pts, 1, 0.5);
        let v0 = pts(n_pts, 1, 0.4);
        let rhs = pts(n_pts, 3, 0.7); // makes the Sum item vector-valued (d = 3)
        let w_s = pts(n_pts, 3, 1.0);
        let w_f = pts(n_pts, 1, 1.1).squeeze(D::Minus1).unwrap();
        let w_g = pts(n_pts, 1, 0.9).squeeze(D::Minus1).unwrap();
        let tiles = BlockCfg::with_tiles(4, 3);

        type Grads = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
        let grads_of = |fused: bool| -> Grads {
            let xv = Var::from_tensor(&x0).unwrap();
            let gv = Var::from_tensor(&g0).unwrap();
            let uv = Var::from_tensor(&u0).unwrap();
            let vv = Var::from_tensor(&v0).unwrap();
            // Shared cost −γ‖xi−xj‖², built ONCE and referenced by all three.
            let xi = LazyTensor::vi(xv.as_tensor());
            let xj = LazyTensor::vj(xv.as_tensor());
            let diff = &xi - &xj;
            let c =
                (&(&diff * &diff).sum_last() * &LazyTensor::pm(gv.as_tensor())).affine(-1.0, 0.0);
            let f_sum = &c.exp() * &LazyTensor::vj(&rhs); // Sum over J, dim 3
            let f1 = &c + &LazyTensor::vj(uv.as_tensor()); // LSE over J
            let f2 = &c + &LazyTensor::vi(vv.as_tensor()); // LSE over I
            let (o_s, o1, o2) = if fused {
                let outs = FusedSweep::new(vec![
                    FusedItem::sum(f_sum, ReduceAxis::J),
                    FusedItem::lse(f1, ReduceAxis::J),
                    FusedItem::lse(f2, ReduceAxis::I),
                ])
                .with_cfg(tiles.clone())
                .eval()
                .unwrap();
                (outs[0].clone(), outs[1].clone(), outs[2].clone())
            } else {
                let one = |f, op, ax| {
                    Genred::new(f, op, ax)
                        .with_cfg(tiles.clone())
                        .eval()
                        .unwrap()
                };
                (
                    one(f_sum, ReduceOp::Sum, ReduceAxis::J),
                    one(f1, ReduceOp::LogSumExp, ReduceAxis::J),
                    one(f2, ReduceOp::LogSumExp, ReduceAxis::I),
                )
            };
            assert_eq!(o_s.dims(), &[n_pts, 3], "Sum item lost its feature axis");
            assert_eq!(o1.dims(), &[n_pts], "LSE item grew a feature axis");
            let loss = ((o_s.mul(&w_s).unwrap().sum_all().unwrap()
                + o1.mul(&w_f).unwrap().sum_all().unwrap())
            .unwrap()
                + o2.mul(&w_g).unwrap().sum_all().unwrap())
            .unwrap();
            let grads = loss.backward().unwrap();
            let take = |v: &Var| -> Vec<f64> {
                grads
                    .get(v.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            };
            (
                o_s.flatten_all().unwrap().to_vec1().unwrap(),
                o1.flatten_all().unwrap().to_vec1().unwrap(),
                take(&xv),
                take(&gv),
                take(&uv),
                take(&vv),
            )
        };

        let (vs_f, v1_f, x_f, g_f, u_f, v_f) = grads_of(true);
        let (vs_s, v1_s, x_s, g_s, u_s, v_s) = grads_of(false);
        for (name, a, b) in [
            ("sum_value", &vs_f, &vs_s),
            ("lse_value", &v1_f, &v1_s),
            ("x", &x_f, &x_s),
            ("gamma", &g_f, &g_s),
            ("u", &u_f, &u_s),
            ("v", &v_f, &v_s),
        ] {
            assert_eq!(a.len(), b.len(), "{name}: length mismatch");
            for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (p - q).abs() < 1e-10 * (1.0 + q.abs()),
                    "{name}[{i}]: fused {p} vs single {q}"
                );
            }
        }
    }

    /// The SCHEDULE, pinned by counting tile evaluations: a `Sum`-only sweep
    /// makes ONE grid pass (one evaluation per item per tile); adding an LSE
    /// item makes it two — and the `Sum` item still only pays one, because
    /// pass 2 visits the LSE items alone. Getting this wrong is invisible in
    /// values and doubles the dominant cost, so it is asserted directly.
    #[test]
    fn sweep_schedules_one_pass_for_sum_and_two_for_lse() {
        let n = 8;
        let x = pts(n, 2, 1.0);
        let g = Tensor::full(0.5f64, (1, 1), &Device::Cpu).unwrap();
        let rhs = pts(n, 1, 0.6);
        let tiles = BlockCfg::with_tiles(4, 4); // 2 x 2 = 4 tiles
        let n_tiles = 4;

        let build = || {
            let xi = LazyTensor::vi(&x);
            let xj = LazyTensor::vj(&x);
            let diff = &xi - &xj;
            (&(&diff * &diff).sum_last() * &LazyTensor::pm(&g)).affine(-1.0, 0.0)
        };
        let evals_for = |items: Vec<FusedItem>| -> usize {
            TILE_EVALS.with(|c| c.set(0));
            FusedSweep::new(items)
                .with_cfg(tiles.clone())
                .eval_detached()
                .unwrap();
            TILE_EVALS.with(|c| c.get())
        };

        // Two Sum items: one pass, so 2 items x 4 tiles.
        let two_sums = vec![
            FusedItem::sum(&build().exp() * &LazyTensor::vj(&rhs), ReduceAxis::J),
            FusedItem::sum(&build().exp() * &LazyTensor::vi(&rhs), ReduceAxis::I),
        ];
        assert_eq!(FusedSweep::passes_for([ReduceOp::Sum, ReduceOp::Sum]), 1);
        assert_eq!(evals_for(two_sums), 2 * n_tiles);

        // One Sum + one LSE: two passes, but the Sum is only visited in pass 1.
        let mixed = vec![
            FusedItem::sum(&build().exp() * &LazyTensor::vj(&rhs), ReduceAxis::J),
            FusedItem::lse(build(), ReduceAxis::J),
        ];
        assert_eq!(
            FusedSweep::passes_for([ReduceOp::Sum, ReduceOp::LogSumExp]),
            2
        );
        assert_eq!(evals_for(mixed), n_tiles + 2 * n_tiles);

        // Two LSE items: two passes over both — the pre-existing FusedLse cost.
        let two_lse = vec![
            FusedItem::lse(build(), ReduceAxis::J),
            FusedItem::lse(build(), ReduceAxis::I),
        ];
        assert_eq!(evals_for(two_lse), 2 * (2 * n_tiles));
    }

    /// A `Sum`-only fused sweep must equal `Genred`'s own `Sum` path exactly
    /// — same tiling, same `stack_sum` accumulation order — so fusing a
    /// transport cost alongside potentials cannot perturb its value.
    #[test]
    fn sum_only_sweep_matches_genred_sum_bitwise() {
        let n = 13;
        let x = pts(n, 2, 1.0);
        let g = Tensor::full(0.6f64, (1, 1), &Device::Cpu).unwrap();
        let rhs = pts(n, 3, 0.7);
        let tiles = BlockCfg::with_tiles(5, 4);
        let k = rbf(&x, &g, Some(0.3));
        let f = &k * &LazyTensor::vj(&rhs);

        let fused = FusedSweep::new(vec![FusedItem::sum(f.clone(), ReduceAxis::J)])
            .with_cfg(tiles.clone())
            .eval_detached()
            .unwrap();
        let single = Genred::new(f, ReduceOp::Sum, ReduceAxis::J)
            .with_cfg(tiles)
            .eval_detached()
            .unwrap();
        assert_eq!(fused[0].dims(), single.dims());
        assert_eq!(
            max_abs(&fused[0], &single),
            0.0,
            "Sum fusion perturbed the value"
        );
    }

    /// Mixed ops with a BATCH prefix: the flat output packing has to split
    /// and reshape correctly when the items' ranks differ AND both carry
    /// leading batch dims.
    #[test]
    fn fused_sweep_handles_batched_mixed_ranks() {
        let (b, m, n, d) = (2usize, 6usize, 5usize, 3usize);
        let xi = Tensor::rand(0f64, 1f64, (b, m, d), &Device::Cpu).unwrap();
        let xj = Tensor::rand(0f64, 1f64, (b, n, d), &Device::Cpu).unwrap();
        let rhs = Tensor::rand(0f64, 1f64, (b, n, 2), &Device::Cpu).unwrap();
        let diff = &LazyTensor::vi(&xi) - &LazyTensor::vj(&xj);
        let c = (&diff * &diff).sum_last().affine(-1.0, 0.0);
        let tiles = BlockCfg::with_tiles(4, 3);

        let outs = FusedSweep::new(vec![
            FusedItem::sum(&c.exp() * &LazyTensor::vj(&rhs), ReduceAxis::J),
            FusedItem::lse(c.clone(), ReduceAxis::I),
        ])
        .with_cfg(tiles.clone())
        .eval()
        .unwrap();
        assert_eq!(outs[0].dims(), &[b, m, 2]);
        assert_eq!(outs[1].dims(), &[b, n]);

        let want_sum = Genred::new(
            &c.exp() * &LazyTensor::vj(&rhs),
            ReduceOp::Sum,
            ReduceAxis::J,
        )
        .with_cfg(tiles.clone())
        .eval_detached()
        .unwrap();
        let want_lse = Genred::new(c, ReduceOp::LogSumExp, ReduceAxis::I)
            .with_cfg(tiles)
            .eval_detached()
            .unwrap();
        assert!(max_abs(&outs[0], &want_sum) < 1e-12);
        assert!(max_abs(&outs[1], &want_lse) < 1e-12);
    }

    /// The depth guarantee, for the MIXED sweep: forward + backward over
    /// many tiles on a deliberately small stack. `Sum` accumulates per block
    /// with `stack_sum`, not a chained `add`, so backward depth is O(1) in
    /// the tile count — the same invariant `Genred` pins for itself.
    #[test]
    fn fused_sweep_backward_survives_many_tiles_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let n = 192;
                let x0 = pts(n, 2, 1.0);
                let g0 = Tensor::full(0.4f64, (1, 1), &Device::Cpu).unwrap();
                let xv = Var::from_tensor(&x0).unwrap();
                let k = rbf(xv.as_tensor(), &g0, Some(0.2));
                let rhs = pts(n, 1, 0.8);
                let outs = FusedSweep::new(vec![
                    FusedItem::sum(&k * &LazyTensor::vj(&rhs), ReduceAxis::J),
                    FusedItem::lse(k.log(), ReduceAxis::J),
                ])
                .with_cfg(BlockCfg::with_tiles(16, 16)) // 144 tiles
                .eval()
                .unwrap();
                let loss = (outs[0].sum_all().unwrap() + outs[1].sum_all().unwrap()).unwrap();
                let grads = loss.backward().unwrap();
                let g: Vec<f64> = grads
                    .get(xv.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert!(g.iter().all(|v| v.is_finite()));
                assert!(g.iter().any(|v| v.abs() > 0.0));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// An LSE item with a feature dim is a mistake the sweep must reject —
    /// a `Sum` neighbour makes dim > 1 legal for the sweep as a whole, so
    /// the per-item check is the only thing standing between a caller and a
    /// silently-wrong `squeeze`.
    #[test]
    fn lse_item_with_a_feature_dim_is_rejected() {
        let n = 5;
        let x = pts(n, 2, 1.0);
        let vec_formula = &LazyTensor::vi(&x) - &LazyTensor::vj(&x); // dim 2
        let err = FusedSweep::new(vec![
            FusedItem::sum(vec_formula.clone(), ReduceAxis::J),
            FusedItem::lse(vec_formula, ReduceAxis::J),
        ])
        .eval_detached()
        .unwrap_err()
        .to_string();
        assert!(err.contains("pair-scalar"), "unexpected error: {err}");
    }

    /// The memory claim, pinned the same way the tiling layer pins its own
    /// depth guarantee: forward + backward across MANY tiles on a thread
    /// with a deliberately small stack. The symbolic backward is fresh
    /// tiled evaluation — no deep graph, no per-tile tape.
    #[test]
    fn genred_backward_survives_many_tiles_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(|| {
                let n = 192;
                let x0 = pts(n, 2, 1.0);
                let g0 = Tensor::full(0.4f64, (1, 1), &Device::Cpu).unwrap();
                let xv = Var::from_tensor(&x0).unwrap();
                let k = rbf(xv.as_tensor(), &g0, Some(0.2));
                let rhs = pts(n, 1, 0.8);
                let node = Genred::new(&k * &LazyTensor::vj(&rhs), ReduceOp::Sum, ReduceAxis::J)
                    .with_cfg(BlockCfg::with_tiles(16, 16)); // 144 tiles
                let y = node.eval().unwrap();
                let grads = y.sum_all().unwrap().backward().unwrap();
                let g: Vec<f64> = grads
                    .get(xv.as_tensor())
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert!(g.iter().all(|v| v.is_finite()));
                assert!(g.iter().any(|v| v.abs() > 0.0));
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
