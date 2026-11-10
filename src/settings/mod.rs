//! Inferno — Scoped numerical settings controlling tensor operators and solver behaviour.
//! Scoped overrides are thread-local; pass values explicitly into worker threads.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Sentinel for "no process default set". `u64::MAX` decodes to a NaN as
/// f64 and to an absurd count as usize, so it can never collide with a
/// value anyone would deliberately store.
const UNSET: u64 = u64::MAX;

/// Element width a setting can be keyed on.
///
/// Mirrors the `float`/`double`/`half` split of Python's
/// `_dtype_value_context`. The point of keying jitter on dtype at all is
/// that the right amount is a function of machine epsilon — an f32 matrix
/// needs roughly `ε_f32/ε_f64 ≈ 10⁹` times more of it, and one number
/// cannot serve both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Machine epsilon at this width.
    pub fn eps(self) -> f64 {
        match self {
            // f16 has a 10-bit mantissa: 2^-10.
            Self::F16 => 9.765625e-4,
            Self::F32 => f32::EPSILON as f64,
            Self::F64 => f64::EPSILON,
        }
    }
}

impl TryFrom<candle_core::DType> for Dtype {
    type Error = candle_core::DType;

    /// Integer and bf16 tensors have no meaningful jitter, so they are
    /// refused rather than silently mapped onto the nearest float width.
    fn try_from(d: candle_core::DType) -> Result<Self, Self::Error> {
        match d {
            candle_core::DType::F16 => Ok(Self::F16),
            candle_core::DType::F32 => Ok(Self::F32),
            candle_core::DType::F64 => Ok(Self::F64),
            other => Err(other),
        }
    }
}

/// Encoding for values held in an [`AtomicU64`].
pub trait Storable: Copy {
    fn encode(self) -> u64;
    fn decode(bits: u64) -> Self;
}

impl Storable for f64 {
    #[inline]
    fn encode(self) -> u64 {
        self.to_bits()
    }
    #[inline]
    fn decode(bits: u64) -> Self {
        f64::from_bits(bits)
    }
}

impl Storable for usize {
    #[inline]
    fn encode(self) -> u64 {
        self as u64
    }
    #[inline]
    fn decode(bits: u64) -> Self {
        bits as usize
    }
}

impl Storable for bool {
    #[inline]
    fn encode(self) -> u64 {
        self as u64
    }
    #[inline]
    fn decode(bits: u64) -> Self {
        bits != 0
    }
}

/// Declares a single-valued setting — the analogue of Python's
/// `_value_context`.
macro_rules! value_setting {
    ($(#[$meta:meta])* $name:ident : $ty:ty = $default:expr) => {
        $(#[$meta])*
        pub mod $name {
            use super::*;

            /// Compiled-in value, used when nothing has overridden it.
            pub const BUILTIN: $ty = $default;

            static PROCESS: AtomicU64 = AtomicU64::new(UNSET);
            thread_local! {
                static SCOPED: Cell<Option<u64>> = const { Cell::new(None) };
            }

            /// Resolve: thread override, else process default, else built-in.
            pub fn value() -> $ty {
                if let Some(bits) = SCOPED.with(|c| c.get()) {
                    return <$ty as Storable>::decode(bits);
                }
                match PROCESS.load(Ordering::Relaxed) {
                    UNSET => BUILTIN,
                    bits => <$ty as Storable>::decode(bits),
                }
            }

            /// True when neither tier has overridden the built-in.
            pub fn is_default() -> bool {
                SCOPED.with(|c| c.get()).is_none() && PROCESS.load(Ordering::Relaxed) == UNSET
            }

            /// Process-wide. Visible from every thread, rayon workers
            /// included. Intended for startup, not for per-call scoping.
            pub fn set_default(v: $ty) {
                PROCESS.store(Storable::encode(v), Ordering::Relaxed);
            }

            /// Clear the process default, restoring [`BUILTIN`].
            pub fn reset_default() {
                PROCESS.store(UNSET, Ordering::Relaxed);
            }

            /// Override on THIS THREAD until the returned guard drops.
            ///
            /// Bind it — `let _g = ...with(x);`. Writing `let _ = ...with(x);`
            /// drops the guard immediately and the override never applies.
            #[must_use = "the override lasts only as long as this guard is alive"]
            pub fn with(v: $ty) -> Guard {
                let prev = SCOPED.with(|c| c.replace(Some(Storable::encode(v))));
                Guard { prev }
            }

            /// Restores the previous thread-local value on drop, so nested
            /// guards unwind exactly.
            pub struct Guard {
                prev: Option<u64>,
            }

            impl Drop for Guard {
                fn drop(&mut self) {
                    SCOPED.with(|c| c.set(self.prev));
                }
            }
        }
    };
}

/// Declares a boolean setting — Python's `_feature_flag`.
macro_rules! flag_setting {
    ($(#[$meta:meta])* $name:ident = $default:expr) => {
        value_setting! {
            $(#[$meta])*
            $name: bool = $default
        }
    };
}

/// Declares a per-dtype setting — Python's `_dtype_value_context`.
macro_rules! dtype_setting {
    ($(#[$meta:meta])* $name:ident : $ty:ty = { f16: $h:expr, f32: $f:expr, f64: $d:expr }) => {
        $(#[$meta])*
        pub mod $name {
            use super::*;

            pub const BUILTIN_F16: $ty = $h;
            pub const BUILTIN_F32: $ty = $f;
            pub const BUILTIN_F64: $ty = $d;

            pub const fn builtin(dtype: Dtype) -> $ty {
                match dtype {
                    Dtype::F16 => BUILTIN_F16,
                    Dtype::F32 => BUILTIN_F32,
                    Dtype::F64 => BUILTIN_F64,
                }
            }

            static PROCESS: [AtomicU64; 3] = [
                AtomicU64::new(UNSET),
                AtomicU64::new(UNSET),
                AtomicU64::new(UNSET),
            ];
            thread_local! {
                static SCOPED: [Cell<Option<u64>>; 3] =
                    const { [Cell::new(None), Cell::new(None), Cell::new(None)] };
            }

            const fn idx(d: Dtype) -> usize {
                match d {
                    Dtype::F16 => 0,
                    Dtype::F32 => 1,
                    Dtype::F64 => 2,
                }
            }

            pub fn value(dtype: Dtype) -> $ty {
                let i = idx(dtype);
                if let Some(bits) = SCOPED.with(|s| s[i].get()) {
                    return <$ty as Storable>::decode(bits);
                }
                match PROCESS[i].load(Ordering::Relaxed) {
                    UNSET => builtin(dtype),
                    bits => <$ty as Storable>::decode(bits),
                }
            }

            pub fn is_default(dtype: Dtype) -> bool {
                let i = idx(dtype);
                SCOPED.with(|s| s[i].get()).is_none()
                    && PROCESS[i].load(Ordering::Relaxed) == UNSET
            }

            /// Process-wide, for one dtype. See the module docs on why this
            /// tier exists separately from [`with`].
            pub fn set_default(dtype: Dtype, v: $ty) {
                PROCESS[idx(dtype)].store(Storable::encode(v), Ordering::Relaxed);
            }

            pub fn reset_default(dtype: Dtype) {
                PROCESS[idx(dtype)].store(UNSET, Ordering::Relaxed);
            }

            /// Override one dtype on this thread until the guard drops.
            #[must_use = "the override lasts only as long as this guard is alive"]
            pub fn with(dtype: Dtype, v: $ty) -> Guard {
                let i = idx(dtype);
                let prev = SCOPED.with(|s| s[i].replace(Some(Storable::encode(v))));
                Guard { i, prev }
            }

            pub struct Guard {
                i: usize,
                prev: Option<u64>,
            }

            impl Drop for Guard {
                fn drop(&mut self) {
                    SCOPED.with(|s| s[self.i].set(self.prev));
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// The settings themselves
// ---------------------------------------------------------------------------

dtype_setting! {
    /// Jitter added to the diagonal by [`crate::linalg::psd_safe_cholesky`]
    /// when a factorization fails.
    ///
    /// Defaults match gpytorch: `1e-6` for f32, `1e-8` for f64. The f16
    /// value has no gpytorch counterpart and is set from the same
    /// `ε`-proportionality the other two follow.
    ///
    /// This is the knob for a matrix that is PSD in exact arithmetic and
    /// indefinite by a hair in floating point — a Gram matrix of nearly
    /// collinear vectors, which is exactly what `SᵀS` becomes when an
    /// optimiser's steps stop turning.
    cholesky_jitter: f64 = { f16: 1e-3, f32: 1e-6, f64: 1e-8 }
}

value_setting! {
    /// How many escalating jitter attempts [`crate::linalg::psd_safe_cholesky`]
    /// makes before giving up. Each attempt multiplies the jitter by ten,
    /// so three tries spans three orders of magnitude.
    cholesky_max_tries: usize = 3
}

dtype_setting! {
    /// Floor on a returned variance. Below this, a variance is numerical
    /// noise around zero and may be negative — which is not a variance.
    min_variance: f64 = { f16: 1e-3, f32: 1e-6, f64: 1e-10 }
}

dtype_setting! {
    /// Jitter for Cholesky factorizations in VARIATIONAL models.
    ///
    /// Two orders LARGER than [`cholesky_jitter`], and that gap is the
    /// point rather than an inconsistency: a variational covariance is a
    /// free parameter the optimiser is actively pushing toward the
    /// boundary of the PSD cone, so it sits near-singular by construction
    /// for much of training. An exact-inference kernel matrix is only ever
    /// near-singular by accident.
    ///
    /// Keep them separate. Raising `cholesky_jitter` to what the
    /// variational path needs would blur every exact solve in the tree.
    variational_cholesky_jitter: f64 = { f16: 1e-2, f32: 1e-4, f64: 1e-6 }
}

value_setting! {
    /// Gauss–Hermite quadrature points for a latent GP likelihood.
    num_gauss_hermite_locs: usize = 20
}

value_setting! {
    /// Quadrature points for contour integral quadrature (`K^{±1/2} b`).
    num_contour_quadrature: usize = 15
}

value_setting! {
    /// Relative residual tolerance for the CIQ contour's shifted solves —
    /// consumed by [`crate::operators::ciq`] for BOTH multi-shift solvers
    /// ([`crate::operators::minres::linear_minres_shifted`] and
    /// [`crate::operators::cg::linear_cg_shifted`]); the name keeps the
    /// gpytorch lineage.
    minres_tolerance: f64 = 1e-4
}

value_setting! {
    /// Below this size, skip pivoted-Cholesky preconditioning entirely —
    /// building the preconditioner would cost more than it saves.
    min_preconditioning_size: usize = 2000
}

value_setting! {
    /// Rank of the pivoted-Cholesky preconditioner. 0 disables it.
    max_preconditioner_size: usize = 15
}

value_setting! {
    /// Relative noise added to a tridiagonal diagonal before
    /// eigendecomposition, so the square root of an eigenvalue is stable.
    tridiagonal_jitter: f64 = 1e-6
}

value_setting! {
    /// Size below which a dense Cholesky is preferred to an iterative
    /// method.
    max_cholesky_size: usize = 800
}

flag_setting! {
    /// Route [`crate::operators::LinearOperator::sqrt_matmul`] /
    /// `inv_sqrt_matmul` through contour-integral quadrature
    /// ([`crate::operators::ciq`]) when the operator exceeds
    /// [`max_cholesky_size`]. OFF by default: below the size threshold the
    /// dense Cholesky path is used regardless, so flipping this changes
    /// nothing at current model sizes — it exists so large-M callers opt in
    /// deliberately.
    ///
    /// A dedicated flag rather than `num_contour_quadrature = 0` as a
    /// disable sentinel: Q must keep meaning "how many nodes" for explicit
    /// [`crate::operators::ciq::CiqOptions`] callers even while the trait
    /// routing is off.
    ciq_enabled = false
}

flag_setting! {
    /// Route the [`crate::operators::LinearOperator`] solve family
    /// (`cholesky_solve` / `inv_quad` / `log_abs_det` /
    /// `inv_quad_logdet`) through the differentiable Krylov nodes in
    /// [`crate::operators::linsolve`] when the operator exceeds
    /// [`max_cholesky_size`] — same opt-in contract as [`ciq_enabled`],
    /// OFF by default so behaviour below the threshold (and everywhere,
    /// until a caller flips this) is bit-for-bit the dense Cholesky path.
    ///
    /// Two semantic changes on the routed arm, both deliberate:
    /// `log_abs_det` becomes a STOCHASTIC Lanczos-quadrature estimate
    /// (probe count [`num_trace_samples`]), and batch dims of operator and
    /// rhs must match exactly (the dense path broadcasts).
    iterative_solves_enabled = false
}

flag_setting! {
    /// Rewrite symbolic squared distances into `‖x‖² − 2⟨x,y⟩ + ‖y‖²`, so
    /// the feature axis is contracted by a GEMM instead of being
    /// materialised on the pair grid
    /// ([`crate::operators::cost_model`]). OFF by default because — unlike
    /// every rewrite in [`crate::operators::rewrite`] — this one CHANGES
    /// VALUES: the expansion is the textbook catastrophic-cancellation case
    /// near `x ≈ y`, costing `≈ ε·(‖x‖² + ‖y‖²)` absolute rather than
    /// `≈ ε·‖x−y‖²`.
    ///
    /// The error is bounded (the result is clamped at 0, and in f32 it is
    /// the same order as the log-sum-exp reduction error already present),
    /// but "same order as existing error" is a judgement about a particular
    /// consumer, not a property of the rewrite — so it is opted into, per
    /// the [`ciq_enabled`] contract, rather than assumed.
    gemm_factoring_enabled = false
}

value_setting! {
    /// Probe vectors drawn per stochastic-Lanczos-quadrature
    /// log-determinant on the routed solve path. gpytorch's
    /// `num_trace_samples` default.
    num_trace_samples: usize = 10
}

value_setting! {
    /// Ceiling on [`crate::operators::cg::linear_cg`] iterations.
    ///
    /// Raising this rarely buys accuracy — CG converges in a number of
    /// iterations set by the condition number, not by the budget. If a
    /// solve is not accurate enough, lower [`cg_tolerance`] or precondition
    /// it; a larger ceiling mostly just spends longer being wrong.
    max_cg_iterations: usize = 1000
}

value_setting! {
    /// Relative residual tolerance at which CG stops.
    ///
    /// The gpytorch default of `1` looks absurd until you notice the
    /// residual is measured against a RHS that CG normalised to unit norm
    /// first, and that the consumer is a stochastic gradient estimate
    /// rather than a solve anyone reads. Tighten it for a solve whose
    /// value is used directly.
    cg_tolerance: f64 = 1.0
}

value_setting! {
    /// Ceiling on the Lanczos tridiagonal size built alongside a CG solve
    /// (for stochastic Lanczos quadrature). Cannot exceed
    /// [`max_cg_iterations`] — you cannot get more Lanczos coefficients
    /// than you ran iterations.
    max_lanczos_quadrature_iterations: usize = 20
}

flag_setting! {
    /// Cap CG at `n` iterations for an `n × n` system.
    ///
    /// In exact arithmetic CG converges in at most `n` steps, so this is
    /// the theoretically-justified cap. It is OFF by default because in
    /// floating point the Krylov basis loses orthogonality and a restarted
    /// or over-run CG genuinely does keep improving past `n`.
    terminate_cg_by_size = false
}

flag_setting! {
    /// Let a CG solve exit as soon as it meets tolerance, instead of running
    /// a minimum of 10 iterations first.
    ///
    /// The floor exists so a fluke-small early residual cannot end a solve
    /// before CG has built any Krylov space worth the name — gpytorch does
    /// the same. The cost is that EVERY converged solve pays at least 11
    /// iterations no matter how fast it really converged, so a solve that is
    /// genuinely done in one step still spends ten wasted matvecs.
    ///
    /// That matters in two places. A well-preconditioned solve is exactly
    /// the case that converges immediately, so the floor hides most of the
    /// preconditioner's payoff and makes iteration counts below ~11
    /// unobservable — benchmark with this ON. And an operator whose matvec
    /// is expensive (a tiled symbolic kernel, a Hessian-vector product) pays
    /// the ten wasted steps in full.
    ///
    /// OFF by default: the floor is a real safeguard, and turning it off
    /// trades robustness on an ill-conditioned or noisy operator for speed
    /// on a well-conditioned one. Turn it on when you know the operator.
    cg_ignore_min_iter = false
}

flag_setting! {
    /// Log whenever an expensive or numerically delicate linear-algebra
    /// routine runs — in particular, every time a Cholesky needs jitter and
    /// how much it needed.
    ///
    /// Off by default; the jitter escalation path is the one worth watching
    /// when a model trains fine at f64 and diverges at f32.
    verbose_linalg = false
}

flag_setting! {
    /// Run the extra consistency checks that are cheap in isolation and
    /// expensive in a hot loop.
    debug = true
}

#[cfg(test)]
mod tests {
    //! Verification of scoped numerical settings controlling tensor operators and solver behaviour.

    use super::*;

    #[test]
    fn resolution_falls_through_the_three_tiers() {
        cholesky_max_tries::reset_default();
        assert!(cholesky_max_tries::is_default());
        assert_eq!(cholesky_max_tries::value(), cholesky_max_tries::BUILTIN);

        cholesky_max_tries::set_default(7);
        assert_eq!(cholesky_max_tries::value(), 7);
        assert!(!cholesky_max_tries::is_default());

        {
            let _g = cholesky_max_tries::with(11);
            assert_eq!(cholesky_max_tries::value(), 11);
        }
        // guard dropped: back to the process default, NOT the built-in
        assert_eq!(cholesky_max_tries::value(), 7);

        cholesky_max_tries::reset_default();
        assert_eq!(cholesky_max_tries::value(), cholesky_max_tries::BUILTIN);
    }

    /// A guard restores what was there before it, not the built-in — which
    /// is the whole difference between nesting correctly and clobbering.
    #[test]
    fn nested_guards_unwind_exactly() {
        assert_eq!(tridiagonal_jitter::value(), 1e-6);
        {
            let _a = tridiagonal_jitter::with(1e-3);
            assert_eq!(tridiagonal_jitter::value(), 1e-3);
            {
                let _b = tridiagonal_jitter::with(1e-1);
                assert_eq!(tridiagonal_jitter::value(), 1e-1);
            }
            assert_eq!(
                tridiagonal_jitter::value(),
                1e-3,
                "inner guard clobbered the outer"
            );
        }
        assert_eq!(tridiagonal_jitter::value(), 1e-6);
    }

    /// Each dtype is an independent slot. Setting the f32 jitter must not
    /// move the f64 one — they differ by two orders for a reason.
    #[test]
    fn dtype_slots_are_independent() {
        assert_eq!(cholesky_jitter::value(Dtype::F32), 1e-6);
        assert_eq!(cholesky_jitter::value(Dtype::F64), 1e-8);

        let _g = cholesky_jitter::with(Dtype::F32, 1e-2);
        assert_eq!(cholesky_jitter::value(Dtype::F32), 1e-2);
        assert_eq!(
            cholesky_jitter::value(Dtype::F64),
            1e-8,
            "f64 slot moved with f32"
        );
    }

    /// The documented limitation, pinned so it cannot regress into a
    /// surprise: `with` is thread-local, `set_default` is not.
    #[test]
    fn scoped_overrides_do_not_leak_across_threads_but_defaults_do() {
        max_cholesky_size::reset_default();
        let _g = max_cholesky_size::with(1);
        assert_eq!(max_cholesky_size::value(), 1);

        let seen = std::thread::spawn(max_cholesky_size::value).join().unwrap();
        assert_eq!(
            seen,
            max_cholesky_size::BUILTIN,
            "a scoped override escaped its thread"
        );

        max_cholesky_size::set_default(64);
        let seen = std::thread::spawn(max_cholesky_size::value).join().unwrap();
        assert_eq!(seen, 64, "a process default failed to reach another thread");
        max_cholesky_size::reset_default();
    }

    #[test]
    fn flags_round_trip() {
        assert!(debug::value());
        {
            let _g = debug::with(false);
            assert!(!debug::value());
        }
        assert!(debug::value());

        assert!(!verbose_linalg::value());
    }

    #[test]
    fn dtype_maps_from_candle_and_refuses_the_rest() {
        assert_eq!(Dtype::try_from(candle_core::DType::F32), Ok(Dtype::F32));
        assert_eq!(Dtype::try_from(candle_core::DType::F64), Ok(Dtype::F64));
        assert_eq!(Dtype::try_from(candle_core::DType::F16), Ok(Dtype::F16));
        assert!(Dtype::try_from(candle_core::DType::U32).is_err());
        assert!(Dtype::try_from(candle_core::DType::BF16).is_err());
    }

    /// Jitter is keyed on dtype because the right amount tracks machine
    /// epsilon; if that ordering ever inverts, the defaults are wrong.
    #[test]
    fn jitter_defaults_track_machine_epsilon() {
        assert!(Dtype::F64.eps() < Dtype::F32.eps());
        assert!(Dtype::F32.eps() < Dtype::F16.eps());
        assert!(cholesky_jitter::value(Dtype::F64) < cholesky_jitter::value(Dtype::F32));
        assert!(cholesky_jitter::value(Dtype::F32) < cholesky_jitter::value(Dtype::F16));
    }

    #[test]
    fn storable_round_trips_every_type() {
        for v in [0.0f64, 1e-8, -3.5, f64::MAX] {
            assert_eq!(f64::decode(Storable::encode(v)), v);
        }
        for v in [0usize, 1, 800, usize::MAX / 2] {
            assert_eq!(usize::decode(Storable::encode(v)), v);
        }
        for v in [true, false] {
            assert_eq!(bool::decode(Storable::encode(v)), v);
        }
    }
}
