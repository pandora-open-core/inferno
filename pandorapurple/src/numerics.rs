//! Inferno — Numerical helpers for stable scalar and tensor computations.

use candle_core::{DType, Device, Result, Tensor, D};

/// Shared finite-replacement bound for pred/target sanitisers. The Python
/// `elbo.py` landscape scrub and the per-term sanitisers reference this
/// single value (`deepsecrets.numerics.FINITE_BOUND`).
pub const FINITE_BOUND: f64 = 1e4;

// ---------------------------------------------------------------- clamps ---

/// C^∞ saturating bound — `mid + half·tanh((x − mid)/half)` over `(low,
/// high)`. Identity near the centre, gradient nonzero everywhere — the
/// GEOC-safe replacement for a hard clamp on symmetric/narrow
/// gradient-carrying ranges. 1:1 with the Python.
pub fn smooth_clamp(x: &Tensor, low: f64, high: f64) -> Result<Tensor> {
    let mid = 0.5 * (low + high);
    let half = 0.5 * (high - low);
    x.affine(1.0 / half, -mid / half)?.tanh()?.affine(half, mid)
}

/// Softplus floor: `x + w·softplus((low − x)/w)`. ≈ identity for `x ≫ low`,
/// saturates smoothly to `low` below it. `width` defaults to
/// `max(|low|, 1e-3)` — near-hard for tiny floors like variance eps, gentle
/// for O(1) bounds.
///
/// Anchored at `x`, not at `low`: algebraically identical to
/// `low + w·softplus((x−low)/w)`, but the far-inside branch is `x + ~0`
/// instead of `low + (x−low)` — exact identity, no rounding against `low`.
pub fn smooth_clamp_min(x: &Tensor, low: f64, width: Option<f64>) -> Result<Tensor> {
    let w = width.unwrap_or_else(|| low.abs().max(1e-3));
    let s = crate::model::softplus(&x.affine(-1.0 / w, low / w)?)?;
    x.add(&s.affine(w, 0.0)?)
}

/// Tensor-floor variant of [`smooth_clamp_min`] — the Python function's
/// `low`-is-a-tensor branch, for a PER-ELEMENT floor (see the Python
/// `stochastics.utils.staggered_var_floor`). Default width is the
/// per-element `max(|low|, 1e-3)`.
pub fn smooth_clamp_min_t(x: &Tensor, low: &Tensor, width: Option<f64>) -> Result<Tensor> {
    let w = match width {
        Some(w) => low.zeros_like()?.affine(0.0, w)?,
        None => low.abs()?.maximum(&low.zeros_like()?.affine(0.0, 1e-3)?)?,
    };
    let arg = low.broadcast_sub(x)?.broadcast_div(&w)?;
    let s = crate::model::softplus(&arg)?;
    x.broadcast_add(&s.broadcast_mul(&w)?)
}

/// Softplus ceiling: `x − w·softplus((x − high)/w)`. ≈ identity for
/// `x ≪ high`, saturates smoothly to `high` above it. `width` defaults to
/// `max(|high|·0.01, 1e-3)` (1% of the ceiling — the softplus tail must
/// underflow at the bottom of the range; see the Python doc for the float32
/// argument).
///
/// Anchored at `x`: the algebraically identical `high − w·softplus((high−x)/w)`
/// is catastrophic in f32 for huge `high` (`ulp(1e8) = 8` annihilates every
/// value below ~8).
pub fn smooth_clamp_max(x: &Tensor, high: f64, width: Option<f64>) -> Result<Tensor> {
    let w = width.unwrap_or_else(|| (high.abs() * 0.01).max(1e-3));
    let s = crate::model::softplus(&x.affine(1.0 / w, -high / w)?)?;
    x.sub(&s.affine(w, 0.0)?)
}

/// Two-sided softplus clamp: identity in the INTERIOR (unlike the tanh
/// form), smooth saturation at both edges. Use for wide positive ranges
/// (variances, stds) where a tanh clamp would distort healthy values.
///
/// When `width` is `None` each side computes its OWN default (the Python
/// behavior): the floor uses `max(|low|, 1e-3)`, the ceiling
/// `max(|high|·0.01, 1e-3)`.
pub fn smooth_clamp_range(x: &Tensor, low: f64, high: f64, width: Option<f64>) -> Result<Tensor> {
    smooth_clamp_max(&smooth_clamp_min(x, low, width)?, high, width)
}

/// Smoothly map `u ∈ [0, 1]` into `(eps, 1 − eps)`: `eps + (1 − 2eps)·u`.
/// Replaces `.clamp(eps, 1-eps)` on unit-interval draws (clamp kills the
/// gradient at the boundary). 1:1 with the Python.
pub fn affine_unit(u: &Tensor, eps: f64) -> Result<Tensor> {
    u.affine(1.0 - 2.0 * eps, eps)
}

// ------------------------------------------------------------ sanitisers ---

/// NaN → `nan_v`, then CLAMP into `[neginf, posinf]`.
///
/// # This is not `torch.nan_to_num`
///
/// Torch's replaces `±inf` and leaves every finite value alone; this clamps
/// unconditionally. For the trust-region caller ([`sanitize_target`]) the
/// clamp IS the intent and the two agree on finite input because the bounds
/// are wide. Use [`replace_nonfinite`] wherever the Python passes-through
/// finite values (e.g. [`finite_pred`]). Same helper as
/// `stochastics::loss_terms::nan_to_num`, duplicated privately so this
/// module stays dependency-light (the Python module's stated contract).
fn nan_to_num_clamp(x: &Tensor, nan_v: f64, posinf: f64, neginf: f64) -> Result<Tensor> {
    let is_nan = x.ne(x)?;
    let filled = is_nan.where_cond(&x.zeros_like()?.affine(0.0, nan_v)?, x)?;
    filled
        .minimum(&filled.zeros_like()?.affine(0.0, posinf)?)?
        .maximum(&filled.zeros_like()?.affine(0.0, neginf)?)
}

/// `torch.nan_to_num(x, nan, posinf, neginf)` — replace the three non-finite
/// classes, pass finite values through UNCHANGED (gradients of replaced
/// elements are zeroed; healthy elements keep theirs).
fn replace_nonfinite(x: &Tensor, nan_v: f64, posinf: f64, neginf: f64) -> Result<Tensor> {
    let fill = |v: f64| x.zeros_like()?.affine(0.0, v);
    let inf = fill(f64::INFINITY)?;
    let out = x.ne(x)?.where_cond(&fill(nan_v)?, x)?;
    let out = out.eq(&inf)?.where_cond(&fill(posinf)?, &out)?;
    out.eq(&inf.neg()?)?.where_cond(&fill(neginf)?, &out)
}

/// Finite, optionally trust-regioned regression/teacher target.
///
/// NaN replacement precedes any bounding (NaN rides through clamp). The norm
/// clip rescales whole rows rather than truncating per-coordinate, so a
/// spiked target keeps its direction while losing its magnitude.
///
/// # Divergences from the Python (kept: this is the moved
/// `stochastics::loss_terms::sanitize_target` behavior, which callers
/// depend on)
/// * The non-finite pass CLAMPS finite values into `[-bound, bound]` too
///   (torch's `nan_to_num` leaves finite values alone). The two agree
///   whenever finite targets sit inside the bound — the intended regime.
/// * The rescale factor is the HARD `max(‖t‖/max_norm, 1)`, not the Python's
///   smooth `smooth_clamp_min(n, max_norm, width=0.05·max_norm)` — a kink at
///   the trust-region boundary instead of a smooth shoulder.
///
/// The old call path `loss_terms::sanitize_target(t, max_norm)` is a thin
/// wrapper for `sanitize_target(t, Some(max_norm), max_norm)` — i.e. it also
/// used `max_norm` as the finite bound, which this signature now makes
/// explicit.
pub fn sanitize_target(target: &Tensor, max_norm: Option<f64>, bound: f64) -> Result<Tensor> {
    let t = nan_to_num_clamp(target, 0.0, bound, -bound)?;
    match max_norm {
        None => Ok(t),
        Some(mn) => {
            let norms = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [.., 1]
            let scale = norms
                .affine(1.0 / mn, 0.0)?
                .maximum(&norms.affine(0.0, 1.0)?)?; // max(‖t‖/mn, 1)
            t.broadcast_div(&scale)
        }
    }
}

/// Prediction-side sanitiser: non-finite elements are replaced (and their
/// gradients zeroed); healthy elements pass through with values and
/// gradients untouched. No bound is applied to finite values — the model
/// must stay trainable. 1:1 with the Python (`finite_pred`).
pub fn finite_pred(pred: &Tensor, bound: f64) -> Result<Tensor> {
    replace_nonfinite(pred, 0.0, bound, -bound)
}

/// Squared matching residual with an optional Huber bound `δ` on the
/// residual gradient (`None` → exact L2).
///
/// # Divergences from the Python (kept: this is the moved
/// `stochastics::loss_terms::matching_residual` behavior, which callers
/// depend on)
/// * ELEMENTWISE, same shape as the inputs — the Python sums over the
///   trailing dim (`.sum(-1)`); Rust callers do their own reduction.
/// * The Huber branch is `r²` / `δ(2|r| − δ)` — consistent with `pow(2)` in
///   the quadratic region, i.e. exactly 2× torch's `F.huber_loss` (whose
///   quadratic region carries a ½ factor). The Python doc notes the NORM_KEYS
///   EMA absorbs the constant either way.
pub fn matching_residual(pred: &Tensor, target: &Tensor, huber_delta: Option<f64>) -> Result<Tensor> {
    let r = pred.sub(target)?;
    match huber_delta {
        None => r.sqr(),
        Some(d) => {
            // Huber: r² where |r| ≤ δ, else δ(2|r| − δ) — bounded gradient.
            let abs_r = r.abs()?;
            let quad = r.sqr()?;
            let lin = abs_r.affine(2.0 * d, -d * d)?;
            let small = abs_r.le(&abs_r.affine(0.0, d)?)?;
            small.where_cond(&quad, &lin)
        }
    }
}

// ------------------------------------------------------------- time grid ---

/// THE uniform time grid — one generator for every clock in the model:
/// `base_time + delta·time_stretch·k` for `k ∈ [0, horizon)`, reversed when
/// `reverse` (`k = 0` becomes the LARGEST lag).
///
/// `delta` is REQUIRED and has no default on purpose (see the Python doc at
/// `deepsecrets.numerics.build_time_grid`): the training/GP clock and the Itô
/// window clock differ in exactly this one number, and under the derived
/// data clock the spacing is physical and load-bearing — a caller inheriting
/// someone else's spacing silently is the failure mode this consolidation
/// exists to close.
///
/// The grid is computed in f64 and cast to `dtype` last (the Python default
/// dtype is float64).
pub fn build_time_grid(
    horizon: usize,
    delta: f64,
    base_time: f64,
    time_stretch: f64,
    reverse: bool,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if horizon < 1 {
        candle_core::bail!("horizon must be >= 1; got {horizon}");
    }
    let mut steps = Tensor::arange(0f64, horizon as f64, device)?;
    if reverse {
        steps = steps.affine(-1.0, (horizon - 1) as f64)?;
    }
    steps.affine(delta * time_stretch, base_time)?.to_dtype(dtype)
}

// ------------------------------------------------------------------ tests ---

#[cfg(test)]
mod tests {
    //! Verification of numerical helpers for stable scalar and tensor computations.

    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    fn scalar(v: f64) -> Tensor {
        Tensor::new(&[v], &dev()).unwrap()
    }

    fn val(t: &Tensor) -> f64 {
        t.flatten_all().unwrap().to_vec1::<f64>().unwrap()[0]
    }

    #[test]
    fn smooth_clamp_matches_python_tanh_form() {
        // Python: mid=1, half=3 → 1 + 3·tanh((0−1)/3) = 0.03546178740509698
        let y = smooth_clamp(&scalar(0.0), -2.0, 4.0).unwrap();
        assert!((val(&y) - 0.03546178740509698).abs() < 1e-12);
        // near the centre ≈ identity
        let y = smooth_clamp(&scalar(1.0), -2.0, 4.0).unwrap();
        assert!((val(&y) - 1.0).abs() < 1e-12);
        // saturates at the bound (tanh(33) rounds to exactly 1.0 in f64)
        let y = smooth_clamp(&scalar(100.0), -2.0, 4.0).unwrap();
        assert!(val(&y) <= 4.0 && val(&y) > 3.9999);
    }

    #[test]
    fn smooth_clamp_min_default_width_and_values() {
        // low=1.0 → default w = max(|1|, 1e-3) = 1.0
        // x=1:  1 + softplus(0) = 1 + ln 2
        let y = smooth_clamp_min(&scalar(1.0), 1.0, None).unwrap();
        assert!((val(&y) - 1.6931471805599454).abs() < 1e-12);
        // x=-2: -2 + softplus(3) = 1.048587351573742
        let y = smooth_clamp_min(&scalar(-2.0), 1.0, None).unwrap();
        assert!((val(&y) - 1.048587351573742).abs() < 1e-12);
        // far inside: exact identity (anchored at x)
        let y = smooth_clamp_min(&scalar(50.0), 1.0, None).unwrap();
        assert_eq!(val(&y), 50.0);
        // explicit width
        let y = smooth_clamp_min(&scalar(0.0), 0.0, Some(2.0)).unwrap();
        assert!((val(&y) - 2.0 * std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn smooth_clamp_max_default_width_and_values() {
        // high=2.0 → default w = max(0.02, 1e-3) = 0.02
        // x=2: 2 − 0.02·softplus(0) = 1.986137056388801
        let y = smooth_clamp_max(&scalar(2.0), 2.0, None).unwrap();
        assert!((val(&y) - 1.986137056388801).abs() < 1e-12);
        // beyond: saturates to ~high;  x=3 → 3 − 0.02·softplus(50) = 2.0
        let y = smooth_clamp_max(&scalar(3.0), 2.0, None).unwrap();
        assert!((val(&y) - 2.0).abs() < 1e-9);
        // far below: exact identity
        let y = smooth_clamp_max(&scalar(-5.0), 2.0, None).unwrap();
        assert_eq!(val(&y), -5.0);
    }

    #[test]
    fn smooth_clamp_range_is_identity_in_the_interior() {
        let y = smooth_clamp_range(&scalar(5.0), 0.001, 100.0, None).unwrap();
        assert!((val(&y) - 5.0).abs() < 1e-9);
        let y = smooth_clamp_range(&scalar(-3.0), 0.5, 2.0, Some(0.01)).unwrap();
        assert!((val(&y) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn smooth_clamp_min_tensor_floor_is_per_element() {
        let x = Tensor::new(&[0.0f64, 0.0], &dev()).unwrap();
        let low = Tensor::new(&[1.0f64, -1.0], &dev()).unwrap();
        let y = smooth_clamp_min_t(&x, &low, None).unwrap();
        let v = y.to_vec1::<f64>().unwrap();
        // element 0: w=1 → 0 + softplus(1) = 1.3132616875182228
        assert!((v[0] - 1.3132616875182228).abs() < 1e-12);
        // element 1: w=1 → 0 + softplus(-1) = 0.31326168751822286
        assert!((v[1] - 0.31326168751822286).abs() < 1e-12);
    }

    #[test]
    fn affine_unit_maps_unit_interval_into_open_band() {
        assert!((val(&affine_unit(&scalar(0.5), 0.1).unwrap()) - 0.5).abs() < 1e-15);
        assert!((val(&affine_unit(&scalar(0.0), 0.1).unwrap()) - 0.1).abs() < 1e-15);
        assert!((val(&affine_unit(&scalar(1.0), 0.1).unwrap()) - 0.9).abs() < 1e-15);
    }

    #[test]
    fn sanitize_target_kills_nonfinite_and_caps_row_norm() {
        let t = Tensor::new(&[[3.0f64, 4.0], [f64::NAN, f64::INFINITY]], &dev()).unwrap();
        let s = sanitize_target(&t, Some(1.0), FINITE_BOUND).unwrap();
        let v = s.to_vec2::<f64>().unwrap();
        // row 0: norm 5 > 1 → rescaled to unit norm keeping direction
        assert!((v[0][0] - 0.6).abs() < 1e-12 && (v[0][1] - 0.8).abs() < 1e-12);
        // row 1: nan→0, inf→bound, then unit-normed → (0, 1)
        assert!((v[1][0]).abs() < 1e-12 && (v[1][1] - 1.0).abs() < 1e-12);
        // no trust region: bound clamp only (divergence: clamps finite too)
        let s = sanitize_target(&t, None, 10.0).unwrap();
        let v = s.to_vec2::<f64>().unwrap();
        assert_eq!(v[0], vec![3.0, 4.0]);
        assert_eq!(v[1], vec![0.0, 10.0]);
    }

    #[test]
    fn finite_pred_passes_finite_values_through_unchanged() {
        let t = Tensor::new(
            &[2e4f64, -2e4, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            &dev(),
        )
        .unwrap();
        let v = finite_pred(&t, FINITE_BOUND).unwrap().to_vec1::<f64>().unwrap();
        // finite values beyond the bound are NOT clamped (torch nan_to_num)
        assert_eq!(v, vec![2e4, -2e4, 0.0, 1e4, -1e4]);
    }

    #[test]
    fn matching_residual_is_elementwise_with_doubled_huber() {
        let p = Tensor::new(&[3.0f64, 0.5], &dev()).unwrap();
        let t = Tensor::new(&[0.0f64, 0.0], &dev()).unwrap();
        // L2: elementwise squares, NOT summed over the trailing dim
        let v = matching_residual(&p, &t, None).unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(v, vec![9.0, 0.25]);
        // Huber δ=1: |3| > 1 → 1·(2·3 − 1) = 5 (2× torch's 2.5); |0.5| ≤ 1 → 0.25
        let v = matching_residual(&p, &t, Some(1.0)).unwrap().to_vec1::<f64>().unwrap();
        assert_eq!(v, vec![5.0, 0.25]);
    }

    #[test]
    fn build_time_grid_matches_python_formula() {
        let g = build_time_grid(4, 0.5, 1.0, 2.0, false, &dev(), DType::F64).unwrap();
        assert_eq!(g.to_vec1::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let g = build_time_grid(4, 0.5, 1.0, 2.0, true, &dev(), DType::F64).unwrap();
        assert_eq!(g.to_vec1::<f64>().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
        let g = build_time_grid(1, 0.1, 0.25, 1.0, true, &dev(), DType::F32).unwrap();
        assert_eq!(g.to_vec1::<f32>().unwrap(), vec![0.25f32]);
        assert!(build_time_grid(0, 0.5, 0.0, 1.0, false, &dev(), DType::F64).is_err());
    }

    #[test]
    fn public_model_helpers_resolve() {
        // model/utils re-exports
        let y = crate::model::smooth_clamp(&scalar(0.0), -2.0, 4.0).unwrap();
        assert!((val(&y) - 0.03546178740509698).abs() < 1e-12);
        let _ = crate::model::affine_unit(&scalar(0.5), 0.1).unwrap();

    }
}
