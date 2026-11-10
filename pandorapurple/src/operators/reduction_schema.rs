//! Inferno — Shared operation identifiers for symbolic tensor reductions.

/// Which reduction a [`Genred`] performs over the contracted axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// `out_k = Σ_over F` — vector-valued (`[..B, kept, d]`). With the
    /// formula `k(x_i, x_j) · Vj(rhs)` this IS the kernel matmul.
    Sum,
    /// `out_k = log Σ_over exp F` — pair-scalar formulas only
    /// (`[..B, kept]`), matching `logsumexp_reduce`.
    LogSumExp,
}

