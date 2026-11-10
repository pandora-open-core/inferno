//! Inferno — Shared axis and reduction schemas for symbolic pairwise formulas.

use candle_core::{DType, Device, Error, Result, Tensor};
use std::sync::Arc;
/// Which axis of the `(i, j)` pair grid a leaf varies along:
/// `Vi` / `Vj` / `Pm`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Axis {
    /// Varies down the rows: `[..B, M, d]`.
    I,
    /// Varies across the columns: `[..B, N, d]`.
    J,
    /// A parameter, broadcast over both: `[..B, 1, d]`.
    P,
    /// A DENSE pair leaf: `[..B, M, N]`, pair-scalar (implicit dim 1).
    ///
    /// Accepts a materialised pair matrix as a leaf
    /// so that dense-cost OT (a `[K0,K1]` cost from a per-pair Riccati
    /// solve, say) can still ride the symbolic reductions; its gradient is
    /// the accumulated pair cotangent evaluated `to_dense` rather than
    /// reduced.
    IJ,
}

/// A node in the formula DAG.
///
/// `Arc` rather than `Rc` because a formula gets stored inside operators that
/// cross `Send + Sync` bounds (`KernelFn`, `KernelEval`, `KernelBackend` are
/// all `Sync`). The atomic refcount is paid once per node at *build* time,
/// never per tile.
pub enum Expr {
    Leaf {
        data: Tensor,
        axis: Axis,
    },
    Scalar(f64),
    Neg(Arc<Expr>),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    /// `mul · e + add` — fused because the code this replaces uses
    /// `.affine()` everywhere, and fusing keeps the tile graph shallow.
    Affine {
        e: Arc<Expr>,
        mul: f64,
        add: f64,
    },
    /// Constant exponent.
    PowF(Arc<Expr>, f64),
    /// **Tensor** exponent, lowered to `exp(b · log a)`. Needed because
    /// `rq_alpha` is learnable — a constant exponent is exactly what forces
    /// the gradient-severing scalar extraction.
    Pow(Arc<Expr>, Arc<Expr>),
    Exp(Arc<Expr>),
    Log(Arc<Expr>),
    Sqrt(Arc<Expr>),
    Recip(Arc<Expr>),
    /// `.sum(-1)` with keepdim — contracts the feature axis, leaving dim 1
    /// so the result still broadcasts against other pair-scalars.
    Sum(Arc<Expr>),
    /// Symbolic `δ·I`. The Python spells this `δ·exp(−1e9·(dᵢ−dⱼ)²)`, which
    /// is exact-by-underflow; as a node we can emit the exact 0/1 pattern
    /// and constant-fold tiles that miss the diagonal entirely.
    EyeIdentity {
        delta: f64,
    },
    /// Hard floor `max(e, c)` — bit-parity with candle `maximum`. Its
    /// derivative is the indicator [`Expr::GtC`]; the subgradient at the
    /// corner takes the FLOOR side (zero), matching candle's `maximum`
    /// backward. The floors this exists for (1e-30 marginal guards, log
    /// floors) are numerical safety rails, not learnable regions.
    MaxC(Arc<Expr>, f64),
    /// Elementwise indicator `1.0 if e > c else 0.0`. Piecewise-constant,
    /// so its own derivative is zero — it exists to EXPRESS `MaxC`'s VJP
    /// (and doubles as a mask primitive).
    GtC(Arc<Expr>, f64),
    /// `Σ_d aᵢ[d]·bⱼ[d]` — the inner product of an **I-pure** left operand
    /// with a **J-pure** right one, as ONE pair-scalar node.
    ///
    /// Semantically identical to `Sum(Mul(a, b))`, but that spelling makes
    /// the evaluator materialise a `[bi, bj, d]` product before contracting
    /// it. Here both operands stay in their un-paired `[.., s, d]` form and
    /// the contraction is a single `broadcast_matmul` — BLAS does the work
    /// and the `d` axis never reaches the pair grid. This is the node that
    /// makes the `‖x−y‖² → ‖x‖² − 2⟨x,y⟩ + ‖y‖²` factoring pay
    /// (see [`super::cost_model`]); on its own it is just a cheaper dot.
    ///
    /// Purity is a HARD requirement, checked in [`infer`]: `a` may not
    /// depend on `j`, nor `b` on `i`. A mixed operand cannot be written as a
    /// matrix factor at all, so there is nothing to lower it to.
    MatDot(Arc<Expr>, Arc<Expr>),
}

/// Inferred metadata for a formula — computed once, before any `O(M·N)` work.
#[derive(Clone, Debug)]
pub struct Meta {
    /// Trailing (feature) dimension.
    pub dim: usize,
    pub uses_i: bool,
    pub uses_j: bool,
    /// Right-aligned broadcast of every leaf's batch shape.
    pub batch: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    /// Rows, from the I-leaves. `None` when the formula has none.
    pub m: Option<usize>,
    /// Columns, from the J-leaves.
    pub n: Option<usize>,
}

/// Right-aligned numpy broadcast of two batch shapes.
pub fn broadcast_batch(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut out = vec![0usize; len];
    for k in 0..len {
        let av = if k < len - a.len() {
            1
        } else {
            a[k - (len - a.len())]
        };
        let bv = if k < len - b.len() {
            1
        } else {
            b[k - (len - b.len())]
        };
        out[k] = match (av, bv) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(Error::Msg(format!(
                    "symbolic: batch shapes {a:?} and {b:?} do not broadcast ({x} vs {y})"
                )));
            }
        };
    }
    Ok(out)
}

fn combine_dim(op: &str, a: usize, b: usize) -> Result<usize> {
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (1, y) => Ok(y),
        (x, 1) => Ok(x),
        (x, y) => Err(Error::Msg(format!(
            "symbolic: {op} dim mismatch {x} vs {y}"
        ))),
    }
}

pub fn infer(e: &Arc<Expr>) -> Result<Meta> {
    match e.as_ref() {
        Expr::Leaf { data, axis } => {
            let rank = data.rank();
            if rank < 2 {
                return Err(Error::Msg(format!(
                    "symbolic: leaf needs rank >= 2 ([..B, S, d]), got {:?}",
                    data.dims()
                )));
            }
            let dims = data.dims();
            if *axis == Axis::IJ {
                // Dense pair leaf [..B, M, N] — pair-scalar (dim 1).
                return Ok(Meta {
                    dim: 1,
                    uses_i: true,
                    uses_j: true,
                    batch: dims[..rank - 2].to_vec(),
                    dtype: data.dtype(),
                    device: data.device().clone(),
                    m: Some(dims[rank - 2]),
                    n: Some(dims[rank - 1]),
                });
            }
            let s = dims[rank - 2];
            let d = dims[rank - 1];
            if *axis == Axis::P && s != 1 {
                return Err(Error::Msg(format!(
                    "symbolic: Pm leaf must have a singleton pair axis, got {dims:?}"
                )));
            }
            Ok(Meta {
                dim: d,
                uses_i: *axis == Axis::I,
                uses_j: *axis == Axis::J,
                batch: dims[..rank - 2].to_vec(),
                dtype: data.dtype(),
                device: data.device().clone(),
                m: (*axis == Axis::I).then_some(s),
                n: (*axis == Axis::J).then_some(s),
            })
        }
        // Constants carry no dtype/device opinion; they inherit from a
        // sibling during combination. A formula that is ONLY constants has
        // nothing to infer from and is rejected at reduce time.
        Expr::Scalar(_) | Expr::EyeIdentity { .. } => Ok(Meta {
            dim: 1,
            uses_i: matches!(e.as_ref(), Expr::EyeIdentity { .. }),
            uses_j: matches!(e.as_ref(), Expr::EyeIdentity { .. }),
            batch: vec![],
            dtype: DType::F32,
            device: Device::Cpu,
            m: None,
            n: None,
        }),
        Expr::Neg(a) | Expr::Exp(a) | Expr::Log(a) | Expr::Sqrt(a) | Expr::Recip(a) => infer(a),
        Expr::MaxC(a, _) | Expr::GtC(a, _) => infer(a),
        Expr::Affine { e, .. } => infer(e),
        Expr::PowF(a, _) => infer(a),
        Expr::Sum(a) => {
            let mut m = infer(a)?;
            m.dim = 1;
            Ok(m)
        }
        // `Σ_d aᵢ[d]·bⱼ[d]`: contracts the feature axis (dim 1, like `Sum`)
        // and is a pair quantity by construction. Both operands must have
        // the SAME feature width — there is no broadcasting inside a
        // contraction — and each must be pure in its own axis, or it cannot
        // be written as a matrix factor.
        Expr::MatDot(a, b) => {
            let ma = infer(a)?;
            let mb = infer(b)?;
            if ma.uses_j {
                return Err(Error::Msg(
                    "symbolic: MatDot left operand depends on j — it must be I-pure".into(),
                ));
            }
            if mb.uses_i {
                return Err(Error::Msg(
                    "symbolic: MatDot right operand depends on i — it must be J-pure".into(),
                ));
            }
            if ma.dim != mb.dim {
                return Err(Error::Msg(format!(
                    "symbolic: MatDot contracts {} against {} features",
                    ma.dim, mb.dim
                )));
            }
            let real_a = has_leaf(a);
            Ok(Meta {
                dim: 1,
                uses_i: ma.uses_i,
                uses_j: mb.uses_j,
                batch: broadcast_batch(&ma.batch, &mb.batch)?,
                dtype: if real_a { ma.dtype } else { mb.dtype },
                device: if real_a {
                    ma.device.clone()
                } else {
                    mb.device.clone()
                },
                m: ma.m,
                n: mb.n,
            })
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            let op = match e.as_ref() {
                Expr::Add(..) => "Add",
                Expr::Sub(..) => "Sub",
                Expr::Mul(..) => "Mul",
                Expr::Div(..) => "Div",
                _ => "Pow",
            };
            let ma = infer(a)?;
            let mb = infer(b)?;
            let real_a = has_leaf(a);
            let real_b = has_leaf(b);
            Ok(Meta {
                dim: combine_dim(op, ma.dim, mb.dim)?,
                uses_i: ma.uses_i || mb.uses_i,
                uses_j: ma.uses_j || mb.uses_j,
                batch: broadcast_batch(&ma.batch, &mb.batch)?,
                // Prefer the side that actually has a tensor leaf.
                dtype: if real_a { ma.dtype } else { mb.dtype },
                device: if real_a {
                    ma.device.clone()
                } else {
                    mb.device.clone()
                },
                m: match (ma.m, mb.m) {
                    (Some(x), Some(y)) if x != y => {
                        return Err(Error::Msg(format!("symbolic: {op} row count {x} vs {y}")));
                    }
                    (x, y) => x.or(y),
                },
                n: match (ma.n, mb.n) {
                    (Some(x), Some(y)) if x != y => {
                        return Err(Error::Msg(format!("symbolic: {op} col count {x} vs {y}")));
                    }
                    (x, y) => x.or(y),
                },
            })
            .map(|mut meta| {
                if !real_a && real_b {
                    meta.dtype = mb.dtype;
                    meta.device = mb.device.clone();
                }
                meta
            })
        }
    }
}

fn has_leaf(e: &Arc<Expr>) -> bool {
    match e.as_ref() {
        Expr::Leaf { .. } => true,
        Expr::Scalar(_) | Expr::EyeIdentity { .. } => false,
        Expr::MaxC(a, _)
        | Expr::GtC(a, _)
        | Expr::Neg(a)
        | Expr::Exp(a)
        | Expr::Log(a)
        | Expr::Sqrt(a)
        | Expr::Recip(a)
        | Expr::Sum(a)
        | Expr::PowF(a, _) => has_leaf(a),
        Expr::Affine { e, .. } => has_leaf(e),
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Pow(a, b)
        | Expr::MatDot(a, b) => has_leaf(a) || has_leaf(b),
    }
}

/// Which axis a reduction contracts.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ReduceAxis {
    /// Contract rows → `[..B, N]`.
    I,
    /// Contract columns → `[..B, M]`.
    J,
}
