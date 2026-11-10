//! Blocked, register-tiled pairwise kernels.
//!
//! One contraction shape covers almost every hot loop on the transport path:
//!
//! ```text
//!   out[i, j] = f( sum_c (a[i, c] - b[j, c])^2 )      a [M, D], b [N, D]
//! ```
//!
//! A codebook's ground cost is that with `M = N = K`. A corpus's
//! responsibilities are that with a softmax for `f`. An atom's displacement
//! statistics are that with a weighted moment accumulation on top. Each was
//! written independently, each materialised the `[M, N, D]` intermediate the
//! contraction exists to avoid, and each walked it in whatever order the
//! broadcast happened to produce.
//!
//! # The three things this file does differently
//!
//! **It never forms `[M, N, D]`.** The `D` axis is contracted inside the
//! register tile, so the largest thing that exists is the `MR x LANES` tile of
//! accumulators.
//!
//! **It holds the support resident.** `b` arrives already
//! [`PackedSupport`]-packed — dim-major, so `LANES` support columns at a fixed
//! feature are one vector load rather than a gather — and the loop order keeps
//! one packed chunk in L1 while the whole of `a` streams past it. Which chunk
//! size that is comes from [`CacheBudget`], not from a constant.
//!
//! **It runs four accumulator chains.** [`MR`] independent `f32x8`
//! accumulators, so the next iteration's multiply issues while the previous
//! addition is still in flight. One chain would run at the add's latency
//! rather than its throughput, which on this shape is a factor of about four
//! and is invisible in a flop count.
//!
//! # Exactness, and why it is the default
//!
//! [`Accum::Exact`] performs `acc + t*t` as two rounded operations, in feature
//! order, which is **bit-for-bit** what the broadcast-then-sum formulation
//! produces. That is not fastidiousness: these kernels are being dropped in
//! under caches whose tests assert a cached value equals an uncached one
//! exactly, and a fused multiply-add — one rounding instead of two — would
//! quietly break that equality everywhere at once.
//!
//! [`Accum::Fused`] is the FMA form, available for callers who are not
//! standing behind such a promise. It is faster and it is not bit-identical;
//! both facts are the caller's to weigh, so it is never the default.
//!
//! The same choice arises for `exp` in the row softmax and does NOT resolve
//! the same way — see [`ExpMode`], where the exact form is measurably the
//! majority of the total cost rather than a rounding error on it.

use wide::f32x8;

use crate::cache_model::{CacheBudget, LANES, MR};
use crate::error::{Error, Result};
use crate::panel::{PackedSupport, Panel, PanelRef};

/// How the squared-distance accumulation rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Accum {
    /// `acc + t*t`, two roundings — bit-identical to the broadcast form.
    #[default]
    Exact,
    /// `fma(t, t, acc)`, one rounding — faster, and NOT bit-identical.
    Fused,
}

#[inline(always)]
fn step<const FUSED: bool>(acc: f32x8, t: f32x8) -> f32x8 {
    if FUSED {
        t.mul_add(t, acc)
    } else {
        acc + t * t
    }
}

// ---------------------------------------------------------------------------
// The micro-kernel
// ---------------------------------------------------------------------------

/// One `rows x LANES` tile of squared distances, contracted over all `d`.
///
/// `rows <= MR`. Written against a fixed-size accumulator array so the common
/// full tile unrolls cleanly; the ragged tail costs a predictable branch per
/// feature and happens at most once per streaming block.
#[inline(always)]
fn micro_tile<const FUSED: bool>(
    a: &[f32],
    lda: usize,
    i0: usize,
    rows: usize,
    b: &PackedSupport,
    j0: usize,
    out: &mut [f32],
    ldc: usize,
) {
    let d = b.d();
    let bs = b.stride();
    let bd = b.as_slice();
    let mut acc = [f32x8::ZERO; MR];

    for c in 0..d {
        // The one load the whole layout exists for: LANES support columns at a
        // fixed feature, contiguous.
        let bv = load8(&bd[c * bs + j0..]);
        for r in 0..rows {
            let av = f32x8::splat(a[(i0 + r) * lda + c]);
            acc[r] = step::<FUSED>(acc[r], av - bv);
        }
    }

    for r in 0..rows {
        let dst = &mut out[(i0 + r) * ldc + j0..(i0 + r) * ldc + j0 + LANES];
        dst.copy_from_slice(&acc[r].to_array());
    }
}

#[inline(always)]
fn store8(dst: &mut [f32], v: f32x8) {
    dst[..LANES].copy_from_slice(&v.to_array());
}

#[inline(always)]
fn load8(s: &[f32]) -> f32x8 {
    let arr: [f32; LANES] = s[..LANES].try_into().expect("slice shorter than a vector");
    f32x8::from(arr)
}

// ---------------------------------------------------------------------------
// sqdist
// ---------------------------------------------------------------------------

/// `out[i, j] = sum_c (a[i, c] - b[j, c])^2`, blocked to the machine's L1.
///
/// `out` is `[a.rows(), out_stride]` with `out_stride >= b.stride()`; the
/// columns from `b.n()` to `b.stride()` are the packed support's poisoned
/// padding and carry enormous finite values by design — see
/// [`crate::panel`]. Callers that hand the result on must carry `b.n()` with
/// it, not `out_stride`.
pub fn sqdist_into<'a>(
    a: impl Into<PanelRef<'a>>,
    b: &PackedSupport,
    out: &mut [f32],
    out_stride: usize,
    accum: Accum,
) -> Result<()> {
    let a = a.into();
    if a.cols != b.d() {
        return Err(Error::Msg(format!(
            "sqdist: streaming panel is {}-dimensional, support is {}-dimensional",
            a.cols,
            b.d()
        )));
    }
    if out_stride < b.stride() {
        return Err(Error::Msg(format!(
            "sqdist: out_stride {out_stride} is narrower than the padded support {}",
            b.stride()
        )));
    }
    if out.len() < a.rows * out_stride {
        return Err(Error::Msg(format!(
            "sqdist: output needs {} values, got {}",
            a.rows * out_stride,
            out.len()
        )));
    }
    match accum {
        Accum::Exact => sqdist_blocked::<false>(a, b, out, out_stride),
        Accum::Fused => sqdist_blocked::<true>(a, b, out, out_stride),
    }
    Ok(())
}

#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw",
    "x86_64+avx2+fma",
    "x86_64+sse4.1",
    "aarch64+neon",
))]
fn sqdist_blocked<const FUSED: bool>(
    a: PanelRef<'_>,
    b: &PackedSupport,
    out: &mut [f32],
    ldc: usize,
) {
    let (m, lda) = (a.rows, a.stride);
    let ad = a.data;
    let t = CacheBudget::global().pair_tiling(m, b.n(), b.d(), std::mem::size_of::<f32>());

    // Support chunk outermost: it is the operand every streamed row re-reads,
    // so it is the one that must stay resident while the stream goes past. The
    // reverse order would evict it once per streaming block.
    let mut j0 = 0;
    while j0 < b.stride() {
        let j_end = (j0 + t.nc_padded).min(b.stride());
        let mut i0 = 0;
        while i0 < m {
            let i_end = (i0 + t.mc).min(m);
            let mut i = i0;
            while i < i_end {
                let rows = MR.min(i_end - i);
                let mut j = j0;
                while j < j_end {
                    micro_tile::<FUSED>(ad, lda, i, rows, b, j, out, ldc);
                    j += LANES;
                }
                i += MR;
            }
            i0 += t.mc;
        }
        j0 += t.nc_padded;
    }
}

/// Convenience wrapper: dense row-major `[m, n]` squared distances from two
/// unpacked row-major operands.
///
/// Packs internally, so it is the right entry point for a one-shot call and
/// the wrong one for a hot loop — anything called per message should cache the
/// [`PackedSupport`] and call [`sqdist_into`].
pub fn sqdist_dense(
    a_rows: &[f32],
    m: usize,
    b_rows: &[f32],
    n: usize,
    d: usize,
    accum: Accum,
) -> Result<Vec<f32>> {
    let a = Panel::from_rows(a_rows, m, d)?;
    let b = PackedSupport::pack(b_rows, n, d)?;
    let stride = b.stride();
    let mut scratch = vec![0.0f32; m * stride];
    sqdist_into(&a, &b, &mut scratch, stride, accum)?;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        out[i * n..(i + 1) * n].copy_from_slice(&scratch[i * stride..i * stride + n]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Row softmax
// ---------------------------------------------------------------------------

/// Which `exp` the row softmax uses.
///
/// # This is the one place the exactness argument does not go the same way
///
/// [`Accum::Exact`] costs almost nothing, so it is the default and the case
/// for it is easy. `exp` is different, and the difference is measured rather
/// than assumed: on the production shape (4096 points, 30 atoms, 16 features)
/// the distance kernel takes 0.18 ms and the scalar-libm softmax on top of it
/// takes a further 0.48 ms. The transcendental is **the majority of the
/// work**, not the few percent a flop count suggests — there are 30x fewer of
/// them than there are multiply-adds, and each costs roughly 20 cycles against
/// a fraction of one.
///
/// So both are offered and neither is obviously right:
///
/// * [`Libm`](Self::Libm) is `f32::exp`, in row order — bit-identical to the
///   tensor formulation, which is what lets these kernels be dropped in under
///   a cache that promises a cached value equals an uncached one.
/// * [`Vector`](Self::Vector) is a polynomial across eight lanes. It differs
///   in the last few bits from two independent causes — the polynomial itself,
///   and the changed summation order of the partition function — and it is
///   materially faster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExpMode {
    /// Scalar `f32::exp`, row order. Bit-identical to the tensor form.
    #[default]
    Libm,
    /// Eight-lane polynomial. Faster, and NOT bit-identical.
    Vector,
}

/// Written into padding columns before the row maximum is taken.
///
/// Finite rather than `-inf`: a lane holding `-inf` reaches the vector `exp`
/// as `-inf - mx`, and a polynomial approximation is not obliged to return
/// zero for that — it may return a NaN, which would then poison the partition
/// function. Large and negative underflows cleanly under either `exp`.
const MASK_LOGIT: f32 = -1.0e30;

/// In-place `softmax(-buf / temperature)` over the first `valid` columns of
/// each row, with the padding columns set to exactly zero.
///
/// Under [`ExpMode::Libm`] the operation order matches the tensor formulation
/// term for term — scale, row max, shift, `exp`, clamp the partition function
/// at `1e-30`, divide — so a caller swapping one for the other sees identical
/// bits, not merely a close answer.
pub fn softmax_neg_scaled_rows(
    buf: &mut [f32],
    rows: usize,
    valid: usize,
    stride: usize,
    temperature: f64,
    exp: ExpMode,
) -> Result<()> {
    if valid == 0 || stride < valid {
        return Err(Error::Msg(format!(
            "softmax: {valid} valid columns in a stride of {stride}"
        )));
    }
    if buf.len() < rows * stride {
        return Err(Error::Msg(format!(
            "softmax: needs {} values, got {}",
            rows * stride,
            buf.len()
        )));
    }
    if exp == ExpMode::Vector && stride % LANES != 0 {
        return Err(Error::Msg(format!(
            "softmax: the vector path needs a stride that is a whole number of \
             vectors, got {stride}"
        )));
    }
    let scale = (-1.0 / temperature.max(1e-12)) as f32;
    match exp {
        ExpMode::Libm => softmax_libm(buf, rows, valid, stride, scale),
        ExpMode::Vector => softmax_vector(buf, rows, valid, stride, scale),
    }
    Ok(())
}

fn softmax_libm(buf: &mut [f32], rows: usize, valid: usize, stride: usize, scale: f32) {
    for i in 0..rows {
        let row = &mut buf[i * stride..(i + 1) * stride];
        // The padding columns hold squared distances to a poisoned support.
        // Zeroing them here is what makes the mask explicit rather than
        // leaving it to underflow — belt as well as braces.
        for v in row[valid..].iter_mut() {
            *v = 0.0;
        }
        let mut mx = f32::NEG_INFINITY;
        for v in row[..valid].iter_mut() {
            *v *= scale;
            if *v > mx {
                mx = *v;
            }
        }
        let mut sum = 0.0f32;
        for v in row[..valid].iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        // A point far from every atom underflows the whole row to zero, and
        // the unclamped divide turns that into 0/0 and a silent NaN field.
        let z = sum.max(1e-30);
        for v in row[..valid].iter_mut() {
            *v /= z;
        }
    }
}

#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw",
    "x86_64+avx2+fma",
    "x86_64+sse4.1",
    "aarch64+neon",
))]
fn softmax_vector(buf: &mut [f32], rows: usize, valid: usize, stride: usize, scale: f32) {
    let sv = f32x8::splat(scale);
    let full = valid / LANES * LANES; // whole vectors entirely inside the valid span
    for i in 0..rows {
        let row = &mut buf[i * stride..(i + 1) * stride];

        // Pass 1 — scale and row-maximum together. Scaling the padding too
        // would multiply the poison by a large negative number and turn it
        // into the row maximum, which is the opposite of masking it, so the
        // mask is written afterwards and never enters the max.
        let mut mv = f32x8::splat(f32::NEG_INFINITY);
        let mut j = 0;
        while j < full {
            let v = load8(&row[j..]) * sv;
            store8(&mut row[j..], v);
            mv = mv.max(v);
            j += LANES;
        }
        let mut mx = mv.to_array().iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in row[full..valid].iter_mut() {
            *v *= scale;
            if *v > mx {
                mx = *v;
            }
        }
        for v in row[valid..].iter_mut() {
            *v = MASK_LOGIT;
        }

        // Pass 2 — exp and partition function together, over the whole padded
        // row. The mask lanes contribute `exp(-1e30 - mx)`, which underflows
        // to zero, so they cost lanes rather than correctness.
        let mxv = f32x8::splat(mx);
        let mut acc = f32x8::ZERO;
        let mut j = 0;
        while j < stride {
            let v = (load8(&row[j..]) - mxv).exp();
            store8(&mut row[j..], v);
            acc += v;
            j += LANES;
        }
        // Zeroed explicitly rather than trusting the polynomial's behaviour on
        // a huge negative argument. The sum above already included whatever it
        // returned, so this is belt on top of the underflow, not instead of it.
        let mut trailing = 0.0f32;
        for v in row[valid..].iter_mut() {
            trailing += *v;
            *v = 0.0;
        }
        let z = (acc.reduce_add() - trailing).max(1e-30);

        // Pass 3 — divide.
        let zv = f32x8::splat(z);
        let mut j = 0;
        while j < stride {
            let v = load8(&row[j..]) / zv;
            store8(&mut row[j..], v);
            j += LANES;
        }
    }
}

/// `[m, stride]` soft assignment of `z` to a packed support.
///
/// # Fused in L1-sized row blocks, and that is the whole optimisation
///
/// The obvious composition — all the distances, then the softmax — was
/// measured at roughly a quarter the speed of the distance kernel alone, and
/// the reason is not arithmetic. At the production shape the `[m, stride]`
/// result is half a megabyte; the softmax makes five further passes over it
/// (scale, max, `exp`, sum, divide), and every one of those streams from L3
/// because the buffer left L1 the moment it was written. Blocking the two
/// together so a band of rows is softmaxed while it is still resident turns
/// five memory-bound passes into five cache-resident ones.
///
/// `temperature` is ABSOLUTE, in the units of the support's own geometry. A
/// caller that does not know that geometry wants the relative form — the
/// measured failure of the absolute one is that it underflows to a hard
/// nearest-neighbour assignment wearing a softmax's clothes.
pub fn responsibilities_into(
    z: &Panel,
    support: &PackedSupport,
    temperature: f64,
    out: &mut [f32],
    out_stride: usize,
    accum: Accum,
    exp: ExpMode,
) -> Result<()> {
    if out_stride < support.stride() {
        return Err(Error::Msg(format!(
            "responsibilities: out_stride {out_stride} is narrower than the padded \
             support {}",
            support.stride()
        )));
    }
    if out.len() < z.rows() * out_stride {
        return Err(Error::Msg(format!(
            "responsibilities: output needs {} values, got {}",
            z.rows() * out_stride,
            out.len()
        )));
    }
    let rows = z.rows();
    // The fusion block. Distances for these rows are computed and immediately
    // consumed while they are still in L1; the softmax's five passes then hit
    // cache rather than main memory.
    let block = CacheBudget::global()
        .pair_tiling(rows, support.n(), support.d(), std::mem::size_of::<f32>())
        .mc
        .max(1);
    let mut i0 = 0;
    while i0 < rows {
        let n = block.min(rows - i0);
        let view = z.rows_view(i0, n)?;
        let dst = &mut out[i0 * out_stride..(i0 + n) * out_stride];
        sqdist_into(view, support, dst, out_stride, accum)?;
        softmax_neg_scaled_rows(dst, n, support.n(), out_stride, temperature, exp)?;
        i0 += n;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagonal Mahalanobis and the mixture responsibilities built on it
// ---------------------------------------------------------------------------

/// One `rows x LANES` tile of `sum_c (a[i,c] - mu[j,c])^2 / var[j,c]`.
///
/// Divides rather than multiplying by a reciprocal, deliberately: the tensor
/// formulation is `diff.sqr().broadcast_div(var).sum(-1)`, and `x * (1/v)` is
/// not `x / v` in the last bit. A reciprocal would be faster and would break
/// the equality that lets this replace the tensor path underneath a test.
///
/// There is no fused variant here for the same reason. The tile's cost is the
/// vector divide, not the accumulate, so folding the add into an FMA would buy
/// a rounding difference and no measurable speed.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn micro_tile_maha(
    a: &[f32],
    lda: usize,
    i0: usize,
    rows: usize,
    mu: &PackedSupport,
    var: &PackedSupport,
    j0: usize,
    out: &mut [f32],
    ldc: usize,
) {
    let d = mu.d();
    let bs = mu.stride();
    let (md, vd) = (mu.as_slice(), var.as_slice());
    let mut acc = [f32x8::ZERO; MR];

    for c in 0..d {
        let bv = load8(&md[c * bs + j0..]);
        let vv = load8(&vd[c * bs + j0..]);
        for r in 0..rows {
            let av = f32x8::splat(a[(i0 + r) * lda + c]);
            let t = av - bv;
            acc[r] = acc[r] + (t * t) / vv;
        }
    }
    for r in 0..rows {
        let dst = &mut out[(i0 + r) * ldc + j0..(i0 + r) * ldc + j0 + LANES];
        dst.copy_from_slice(&acc[r].to_array());
    }
}

/// `out[i, j] = sum_c (a[i,c] - mu[j,c])^2 / var[j,c]` — the diagonal
/// Mahalanobis distance a diagonal Gaussian mixture is built on.
///
/// `mu` must be packed with the default position poison and `var` with
/// [`PackedSupport::pack_with_pad`] and a padding of `1.0`; see that method for
/// why the two padding values differ.
pub fn mahalanobis_diag_into<'a>(
    a: impl Into<PanelRef<'a>>,
    mu: &PackedSupport,
    var: &PackedSupport,
    out: &mut [f32],
    out_stride: usize,
) -> Result<()> {
    let a = a.into();
    if a.cols != mu.d() {
        return Err(Error::Msg(format!(
            "mahalanobis: points are {}-dimensional, means are {}-dimensional",
            a.cols,
            mu.d()
        )));
    }
    if mu.n() != var.n() || mu.d() != var.d() || mu.stride() != var.stride() {
        return Err(Error::Msg(format!(
            "mahalanobis: means are {}x{} and variances {}x{} — they must be packed \
             identically",
            mu.n(),
            mu.d(),
            var.n(),
            var.d()
        )));
    }
    if out_stride < mu.stride() || out.len() < a.rows * out_stride {
        return Err(Error::Msg(format!(
            "mahalanobis: output must be {}x{} or wider, got {} values at stride \
             {out_stride}",
            a.rows,
            mu.stride(),
            out.len()
        )));
    }
    maha_blocked(a, mu, var, out, out_stride);
    Ok(())
}

#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw",
    "x86_64+avx2+fma",
    "x86_64+sse4.1",
    "aarch64+neon",
))]
fn maha_blocked<'a>(
    a: PanelRef<'a>,
    mu: &PackedSupport,
    var: &PackedSupport,
    out: &mut [f32],
    ldc: usize,
) {
    let (m, lda) = (a.rows, a.stride);
    let ad = a.data;
    let t = CacheBudget::global().pair_tiling(m, mu.n(), mu.d(), std::mem::size_of::<f32>());
    let mut j0 = 0;
    while j0 < mu.stride() {
        let j_end = (j0 + t.nc_padded).min(mu.stride());
        let mut i0 = 0;
        while i0 < m {
            let i_end = (i0 + t.mc).min(m);
            let mut i = i0;
            while i < i_end {
                let rows = MR.min(i_end - i);
                let mut j = j0;
                while j < j_end {
                    micro_tile_maha(ad, lda, i, rows, mu, var, j, out, ldc);
                    j += LANES;
                }
                i += MR;
            }
            i0 += t.mc;
        }
        j0 += t.nc_padded;
    }
}

/// `[m, stride]` posterior responsibilities under a diagonal Gaussian mixture:
/// `softmax_j( -0.5*quad[i,j] - half_logdet[j] + log_w[j] )`.
///
/// `half_logdet` and `log_w` are passed separately rather than pre-summed
/// because the tensor formulation applies them as two separate rounded
/// operations, in that order, and folding them on the host would change the
/// last bit of every logit.
///
/// # One deliberate difference from the tensor path
///
/// The partition function is clamped at `1e-30`. `candle_nn::ops::softmax`
/// does not clamp, so a row whose logits all underflow becomes `0/0` there and
/// stays finite here. The two agree bit for bit on every row that is not
/// already a NaN in the tensor form.
#[allow(clippy::too_many_arguments)]
pub fn gmm_responsibilities_into(
    z: &Panel,
    mu: &PackedSupport,
    var: &PackedSupport,
    half_logdet: &[f32],
    log_w: &[f32],
    out: &mut [f32],
    out_stride: usize,
) -> Result<()> {
    let k = mu.n();
    if half_logdet.len() < k || log_w.len() < k {
        return Err(Error::Msg(format!(
            "mixture responsibilities: need {k} log-determinants and weights"
        )));
    }
    if out_stride < mu.stride() || out.len() < z.rows() * out_stride {
        return Err(Error::Msg(format!(
            "mixture responsibilities: output must be {}x{} or wider",
            z.rows(),
            mu.stride()
        )));
    }
    let rows = z.rows();
    // Same fusion as `responsibilities_into`: a band of rows is turned into
    // logits and normalised while it is still in L1, rather than the whole
    // matrix being written and then walked four more times.
    let block = CacheBudget::global()
        .pair_tiling(rows, k, mu.d(), std::mem::size_of::<f32>())
        .mc
        .max(1);
    let mut i0 = 0;
    while i0 < rows {
        let n = block.min(rows - i0);
        let view = z.rows_view(i0, n)?;
        let dst = &mut out[i0 * out_stride..(i0 + n) * out_stride];
        mahalanobis_diag_into(view, mu, var, dst, out_stride)?;
        for i in 0..n {
            let row = &mut dst[i * out_stride..(i + 1) * out_stride];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..k {
                // The tensor form's exact order: scale, subtract the
                // half-log-determinant, add the log weight.
                let v = row[j] * -0.5 - half_logdet[j] + log_w[j];
                row[j] = v;
                if v > mx {
                    mx = v;
                }
            }
            for v in row[k..].iter_mut() {
                *v = 0.0;
            }
            let mut sum = 0.0f32;
            for v in row[..k].iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            let zsum = sum.max(1e-30);
            for v in row[..k].iter_mut() {
                *v /= zsum;
            }
        }
        i0 += n;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reductions over a pairwise matrix
// ---------------------------------------------------------------------------

/// Mean over rows of `sqrt(min_{j != i} c[i, j])`.
///
/// The codebook's own length unit: the mean nearest-neighbour distance, from a
/// matrix of SQUARED distances. Reads only the valid `k x k` block of a
/// possibly padded buffer, which is the whole reason it takes a stride.
pub fn nn_spacing_from_sqdist(c: &[f32], k: usize, stride: usize) -> Result<f64> {
    if k == 0 {
        return Err(Error::Msg("nn_spacing: an empty support has no spacing".into()));
    }
    if k < 2 {
        // One atom has no neighbour, so there is no geometry to measure. The
        // caller's lengths are then in raw units, which is the only honest
        // answer.
        return Ok(1.0);
    }
    if c.len() < k * stride || stride < k {
        return Err(Error::Msg(format!(
            "nn_spacing: needs a {k}x{k} block in a stride of {stride}, got {} values",
            c.len()
        )));
    }
    let mut total = 0.0f64;
    for i in 0..k {
        let row = &c[i * stride..i * stride + k];
        let mut best = f32::INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if j != i && v < best {
                best = v;
            }
        }
        total += (best.max(0.0) as f64).sqrt();
    }
    Ok(total / k as f64)
}

/// `acc[j, c] += sum_i resp[i, j] * z[i, c]` and `w[j] += sum_i resp[i, j]`.
///
/// The weighted centroid of the points assigned to each support column — a
/// `[valid, d]` accumulation that the broadcast formulation writes as a
/// per-column loop over the whole point set, contracting nothing and touching
/// `z` once per column instead of once per point.
///
/// # Order, and therefore exactness
///
/// `i` is the outer loop, so every accumulator receives its terms in
/// increasing point order — the same order a per-column loop produces, and
/// therefore the same `f64` sum bit for bit. The inner loop over `c` is a
/// plain `axpy` over contiguous memory, which is the shape an auto-vectoriser
/// handles without help; writing it with explicit `f64` vectors would gain
/// nothing and would fix the width at the source.
pub fn weighted_centroid_accumulate(
    resp: &[f32],
    resp_stride: usize,
    z: PanelRef<'_>,
    valid: usize,
    acc: &mut [f64],
    w_tot: &mut [f64],
) -> Result<()> {
    let d = z.cols;
    if resp.len() < z.rows * resp_stride || resp_stride < valid {
        return Err(Error::Msg(format!(
            "weighted centroid: needs {}x{valid} responsibilities in a stride of \
             {resp_stride}",
            z.rows
        )));
    }
    if acc.len() < valid * d || w_tot.len() < valid {
        return Err(Error::Msg(format!(
            "weighted centroid: accumulators must hold {valid}x{d} and {valid}"
        )));
    }
    accumulate_centroids(resp, resp_stride, z.data, z.rows, z.stride, d, valid, acc, w_tot);
    Ok(())
}

#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw",
    "x86_64+avx2+fma",
    "x86_64+sse4.1",
    "aarch64+neon",
))]
#[allow(clippy::too_many_arguments)]
fn accumulate_centroids(
    resp: &[f32],
    resp_stride: usize,
    z: &[f32],
    rows: usize,
    z_stride: usize,
    d: usize,
    valid: usize,
    acc: &mut [f64],
    w_tot: &mut [f64],
) {
    for i in 0..rows {
        let zrow = &z[i * z_stride..i * z_stride + d];
        let rrow = &resp[i * resp_stride..i * resp_stride + valid];
        for (j, &rw) in rrow.iter().enumerate() {
            let w = rw as f64;
            if w <= 0.0 {
                continue;
            }
            w_tot[j] += w;
            let a = &mut acc[j * d..(j + 1) * d];
            for (av, &zv) in a.iter_mut().zip(zrow) {
                *av += w * zv as f64;
            }
        }
    }
}

/// Responsibility-weighted displacement moments, accumulated in `f64`.
///
/// For each support column `j`: total weight, weight-times-distance, and
/// weight-times-distance-squared, where distance is
/// `sqrt(sqdist) * inv_spacing`.
///
/// # Precision
///
/// Distances are computed in `f32` and the moments accumulated in `f64`. The
/// widening is deliberately on the accumulator and not on the distance: the
/// inputs are `f32`, so a `f64` subtraction of two of them recovers no
/// information the `f32` one lost — whereas summing `N` weighted terms is
/// exactly where a `f32` accumulator does lose, and does so worse the more
/// data there is.
pub fn weighted_displacement_moments(
    resp: &[f32],
    sqdist: &[f32],
    rows: usize,
    valid: usize,
    stride: usize,
    inv_spacing: f64,
    mass: &mut [f64],
    disp_sum: &mut [f64],
    disp_sq: &mut [f64],
) -> Result<()> {
    if mass.len() < valid || disp_sum.len() < valid || disp_sq.len() < valid {
        return Err(Error::Msg(format!(
            "displacement moments: accumulators must hold {valid} entries"
        )));
    }
    if resp.len() < rows * stride || sqdist.len() < rows * stride || stride < valid {
        return Err(Error::Msg(format!(
            "displacement moments: needs {rows}x{valid} in a stride of {stride}"
        )));
    }
    for i in 0..rows {
        let r = &resp[i * stride..i * stride + valid];
        let q = &sqdist[i * stride..i * stride + valid];
        for j in 0..valid {
            let w = r[j] as f64;
            if w <= 0.0 {
                continue;
            }
            let dist = (q[j].max(0.0) as f64).sqrt() * inv_spacing;
            mass[j] += w;
            disp_sum[j] += w * dist;
            disp_sq[j] += w * dist * dist;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formulation these kernels replace: broadcast to `[m, n, d]`,
    /// square, sum. Kept as the oracle rather than as a comment, because "bit
    /// identical to the broadcast form" is a claim that needs a broadcast form
    /// to be checked against.
    fn oracle_sqdist(a: &[f32], m: usize, b: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for c in 0..d {
                    let t = a[i * d + c] - b[j * d + c];
                    acc += t * t;
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn oracle_resp(a: &[f32], m: usize, b: &[f32], n: usize, d: usize, temp: f64) -> Vec<f32> {
        let mut q = oracle_sqdist(a, m, b, n, d);
        let scale = (-1.0 / temp.max(1e-12)) as f32;
        for i in 0..m {
            let row = &mut q[i * n..(i + 1) * n];
            let mut mx = f32::NEG_INFINITY;
            for v in row.iter_mut() {
                *v *= scale;
                if *v > mx {
                    mx = *v;
                }
            }
            let mut s = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                s += *v;
            }
            let z = s.max(1e-30);
            for v in row.iter_mut() {
                *v /= z;
            }
        }
        q
    }

    fn pts(n: usize, d: usize, seed: f32) -> Vec<f32> {
        (0..n * d).map(|i| ((i as f32 + seed) * 0.6180339).sin() * 3.0).collect()
    }

    /// The governing property. Not "close": identical, because these kernels
    /// land under caches that promise a cached value equals an uncached one.
    #[test]
    fn the_exact_kernel_is_bit_identical_to_the_broadcast_form() {
        for &(m, n, d) in &[
            (1usize, 1usize, 1usize),
            (4, 8, 16),
            (7, 30, 16),      // the production codebook shape
            (33, 5, 3),       // ragged on both axes
            (128, 64, 32),
            (2, 3, 129),      // a feature width past any tile
        ] {
            let a = pts(m, d, 0.0);
            let b = pts(n, d, 11.0);
            let got = sqdist_dense(&a, m, &b, n, d, Accum::Exact).unwrap();
            let want = oracle_sqdist(&a, m, &b, n, d);
            assert_eq!(got, want, "mismatch at {m}x{n}x{d}");
        }
    }

    /// The fused form is faster and is NOT the same number. Asserting the
    /// difference is small AND that the exact form is the one that matches
    /// keeps the trade-off documented in something that runs.
    #[test]
    fn the_fused_kernel_is_close_but_deliberately_not_identical() {
        let (m, n, d) = (32usize, 24usize, 48usize);
        let a = pts(m, d, 0.0);
        let b = pts(n, d, 7.0);
        let exact = sqdist_dense(&a, m, &b, n, d, Accum::Exact).unwrap();
        let fused = sqdist_dense(&a, m, &b, n, d, Accum::Fused).unwrap();
        let mut max_rel = 0.0f32;
        for (x, y) in exact.iter().zip(&fused) {
            let denom = x.abs().max(1e-6);
            max_rel = max_rel.max((x - y).abs() / denom);
        }
        assert!(max_rel < 1e-5, "the fused form drifted by {max_rel}");
        assert_eq!(exact, oracle_sqdist(&a, m, &b, n, d));
    }

    /// Padding columns must not be readable as answers, and the boundary
    /// between valid and padded has to fall where the caller thinks it does.
    #[test]
    fn padded_columns_carry_poison_and_valid_ones_do_not() {
        let (m, n, d) = (4usize, 30usize, 16usize);
        let a = pts(m, d, 0.0);
        let b = pts(n, d, 3.0);
        let pa = Panel::from_rows(&a, m, d).unwrap();
        let pb = PackedSupport::pack(&b, n, d).unwrap();
        let stride = pb.stride();
        assert_eq!(stride, 32);
        let mut out = vec![0.0f32; m * stride];
        sqdist_into(&pa, &pb, &mut out, stride, Accum::Exact).unwrap();
        for i in 0..m {
            for j in 0..n {
                assert!(out[i * stride + j] < 1e6, "a valid entry looks like poison");
            }
            for j in n..stride {
                assert!(
                    out[i * stride + j] > 1e30,
                    "padding column {j} was not poisoned: {}",
                    out[i * stride + j]
                );
                assert!(out[i * stride + j].is_finite(), "padding overflowed to inf");
            }
        }
    }

    #[test]
    fn responsibilities_match_the_tensor_formulation_bit_for_bit() {
        for &(m, n, d, t) in &[
            (5usize, 30usize, 16usize, 4.0f64),
            (64, 8, 4, 0.25),
            (1, 1, 1, 1.0),
            (17, 13, 7, 100.0),
        ] {
            let a = pts(m, d, 0.0);
            let b = pts(n, d, 5.0);
            let pa = Panel::from_rows(&a, m, d).unwrap();
            let pb = PackedSupport::pack(&b, n, d).unwrap();
            let stride = pb.stride();
            let mut out = vec![0.0f32; m * stride];
            responsibilities_into(&pa, &pb, t, &mut out, stride, Accum::Exact, ExpMode::Libm)
                .unwrap();
            let want = oracle_resp(&a, m, &b, n, d, t);
            for i in 0..m {
                for j in 0..n {
                    assert_eq!(
                        out[i * stride + j],
                        want[i * n + j],
                        "row {i} col {j} at {m}x{n}x{d}, T={t}"
                    );
                }
                for j in n..stride {
                    assert_eq!(out[i * stride + j], 0.0, "a padding column survived the mask");
                }
                let s: f32 = out[i * stride..i * stride + n].iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
            }
        }
    }

    /// A point far from every support column drives the whole row to
    /// underflow. Without the partition-function clamp that is 0/0.
    #[test]
    fn a_point_far_from_everything_yields_a_finite_row() {
        let (n, d) = (4usize, 2usize);
        let b = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let a = vec![1.0e6f32, 1.0e6];
        let pa = Panel::from_rows(&a, 1, d).unwrap();
        let pb = PackedSupport::pack(&b, n, d).unwrap();
        let stride = pb.stride();
        let mut out = vec![0.0f32; stride];
        responsibilities_into(&pa, &pb, 1e-3, &mut out, stride, Accum::Exact, ExpMode::Libm).unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        // The nearest column still wins outright rather than the row being NaN.
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn nn_spacing_reads_the_valid_block_and_skips_the_diagonal() {
        // Atoms on a line at unit spacing: every nearest neighbour is at 1.
        let k = 6usize;
        let atoms: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let c = sqdist_dense(&atoms, k, &atoms, k, 1, Accum::Exact).unwrap();
        assert!((nn_spacing_from_sqdist(&c, k, k).unwrap() - 1.0).abs() < 1e-9);

        // The same block inside a padded buffer must give the same answer —
        // reading the poison would give a spacing of ~1e17.
        let pb = PackedSupport::pack(&atoms, k, 1).unwrap();
        let pa = Panel::from_rows(&atoms, k, 1).unwrap();
        let stride = pb.stride();
        let mut padded = vec![0.0f32; k * stride];
        sqdist_into(&pa, &pb, &mut padded, stride, Accum::Exact).unwrap();
        assert!((nn_spacing_from_sqdist(&padded, k, stride).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_one_atom_support_degrades_rather_than_reporting_a_spacing() {
        assert_eq!(nn_spacing_from_sqdist(&[0.0], 1, 1).unwrap(), 1.0);
        assert!(nn_spacing_from_sqdist(&[], 0, 0).is_err());
    }

    #[test]
    fn displacement_moments_match_a_scalar_accumulation() {
        let (rows, valid, stride) = (5usize, 3usize, 8usize);
        let resp: Vec<f32> =
            (0..rows * stride).map(|i| if i % stride < valid { 0.1 * (i % 7) as f32 } else { 0.0 }).collect();
        let sq: Vec<f32> = (0..rows * stride).map(|i| (i as f32 * 0.31).sin().abs() * 4.0).collect();
        let inv = 0.5;

        let (mut m, mut s, mut q) = (vec![0.0; valid], vec![0.0; valid], vec![0.0; valid]);
        weighted_displacement_moments(&resp, &sq, rows, valid, stride, inv, &mut m, &mut s, &mut q)
            .unwrap();

        let (mut m2, mut s2, mut q2) = (vec![0.0f64; valid], vec![0.0f64; valid], vec![0.0f64; valid]);
        for i in 0..rows {
            for j in 0..valid {
                let w = resp[i * stride + j] as f64;
                if w <= 0.0 {
                    continue;
                }
                let dist = (sq[i * stride + j].max(0.0) as f64).sqrt() * inv;
                m2[j] += w;
                s2[j] += w * dist;
                q2[j] += w * dist * dist;
            }
        }
        assert_eq!(m, m2);
        assert_eq!(s, s2);
        assert_eq!(q, q2);
    }

    fn oracle_gmm(
        z: &[f32],
        m: usize,
        mu: &[f32],
        var: &[f32],
        k: usize,
        d: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * k];
        for i in 0..m {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..k {
                let mut q = 0.0f32;
                let mut hl = 0.0f32;
                for c in 0..d {
                    let t = z[i * d + c] - mu[j * d + c];
                    q += (t * t) / var[j * d + c];
                    hl += var[j * d + c].ln();
                }
                let w = 1.0f32 / k as f32;
                let v = q * -0.5 - 0.5 * hl + w.ln();
                out[i * k + j] = v;
                if v > mx {
                    mx = v;
                }
            }
            let mut s = 0.0f32;
            for j in 0..k {
                out[i * k + j] = (out[i * k + j] - mx).exp();
                s += out[i * k + j];
            }
            for j in 0..k {
                out[i * k + j] /= s.max(1e-30);
            }
        }
        out
    }

    #[test]
    fn mixture_responsibilities_match_the_scalar_form_bit_for_bit() {
        for &(m, k, d) in &[(5usize, 4usize, 3usize), (33, 30, 16), (1, 1, 1)] {
            let z = pts(m, d, 0.0);
            let mu = pts(k, d, 4.0);
            let var: Vec<f32> = (0..k * d).map(|i| 0.5 + ((i * 3 % 5) as f32) * 0.25).collect();

            let pz = Panel::from_rows(&z, m, d).unwrap();
            let pmu = PackedSupport::pack(&mu, k, d).unwrap();
            let pvar = PackedSupport::pack_with_pad(&var, k, d, 1.0).unwrap();
            let stride = pmu.stride();
            let half_logdet: Vec<f32> = (0..k)
                .map(|j| 0.5 * (0..d).map(|c| var[j * d + c].ln()).sum::<f32>())
                .collect();
            let log_w = vec![(1.0f32 / k as f32).ln(); k];

            let mut out = vec![0.0f32; m * stride];
            gmm_responsibilities_into(&pz, &pmu, &pvar, &half_logdet, &log_w, &mut out, stride)
                .unwrap();
            let want = oracle_gmm(&z, m, &mu, &var, k, d);
            for i in 0..m {
                for j in 0..k {
                    assert_eq!(out[i * stride + j], want[i * k + j], "({i},{j}) at {m}x{k}x{d}");
                }
                assert!(out[i * stride + k..(i + 1) * stride].iter().all(|v| *v == 0.0));
            }
        }
    }

    /// A variance panel padded with the position poison would divide the
    /// poisoned distance back into range. This is the test that catches it.
    #[test]
    fn a_padding_component_cannot_win_the_mixture() {
        let (m, k, d) = (2usize, 3usize, 2usize);
        let z = vec![0.0f32; m * d];
        let mu = vec![0.0f32; k * d]; // every real component AT the query
        let var = vec![1.0f32; k * d];
        let pz = Panel::from_rows(&z, m, d).unwrap();
        let pmu = PackedSupport::pack(&mu, k, d).unwrap();
        let pvar = PackedSupport::pack_with_pad(&var, k, d, 1.0).unwrap();
        let stride = pmu.stride();
        let mut out = vec![0.0f32; m * stride];
        gmm_responsibilities_into(&pz, &pmu, &pvar, &vec![0.0; k], &vec![0.0; k], &mut out, stride)
            .unwrap();
        for i in 0..m {
            let s: f32 = out[i * stride..i * stride + k].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {i} sums to {s}");
            for j in 0..k {
                assert!((out[i * stride + j] - 1.0 / k as f32).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_mismatched_variance_panel_is_refused() {
        let z = Panel::from_rows(&pts(4, 3, 0.0), 4, 3).unwrap();
        let mu = PackedSupport::pack(&pts(5, 3, 1.0), 5, 3).unwrap();
        let var = PackedSupport::pack_with_pad(&pts(6, 3, 2.0), 6, 3, 1.0).unwrap();
        let mut out = vec![0.0f32; 4 * 8];
        assert!(mahalanobis_diag_into(&z, &mu, &var, &mut out, 8).is_err());
    }

    #[test]
    fn weighted_centroids_match_a_per_column_accumulation() {
        let (rows, valid, d) = (9usize, 5usize, 4usize);
        let rs = 8usize;
        let zrows = pts(rows, d, 0.0);
        let z = Panel::from_rows(&zrows, rows, d).unwrap();
        let resp: Vec<f32> = (0..rows * rs)
            .map(|i| if i % rs < valid { ((i * 3 % 7) as f32) * 0.1 } else { 0.0 })
            .collect();

        let (mut acc, mut w) = (vec![0.0f64; valid * d], vec![0.0f64; valid]);
        weighted_centroid_accumulate(&resp, rs, z.view(), valid, &mut acc, &mut w).unwrap();

        // The formulation it replaces: a loop per column over every point.
        let (mut acc2, mut w2) = (vec![0.0f64; valid * d], vec![0.0f64; valid]);
        for j in 0..valid {
            for i in 0..rows {
                let wt = resp[i * rs + j] as f64;
                if wt <= 0.0 {
                    continue;
                }
                w2[j] += wt;
                for c in 0..d {
                    acc2[j * d + c] += wt * zrows[i * d + c] as f64;
                }
            }
        }
        assert_eq!(acc, acc2);
        assert_eq!(w, w2);
    }

    #[test]
    fn weighted_centroids_refuse_undersized_accumulators() {
        let z = Panel::from_rows(&pts(4, 3, 0.0), 4, 3).unwrap();
        let resp = vec![0.0f32; 4 * 8];
        let mut acc = vec![0.0f64; 2];
        let mut w = vec![0.0f64; 5];
        assert!(weighted_centroid_accumulate(&resp, 8, z.view(), 5, &mut acc, &mut w).is_err());
    }

    #[test]
    fn shape_mismatches_are_refused_rather_than_read_out_of_bounds() {
        let a = Panel::from_rows(&pts(4, 8, 0.0), 4, 8).unwrap();
        let b = PackedSupport::pack(&pts(3, 5, 1.0), 3, 5).unwrap();
        let mut out = vec![0.0f32; 4 * 8];
        // Feature widths disagree.
        assert!(sqdist_into(&a, &b, &mut out, 8, Accum::Exact).is_err());

        let b2 = PackedSupport::pack(&pts(3, 8, 1.0), 3, 8).unwrap();
        // out_stride narrower than the padded support.
        assert!(sqdist_into(&a, &b2, &mut out, 4, Accum::Exact).is_err());
        // Output too small.
        let mut tiny = vec![0.0f32; 3];
        assert!(sqdist_into(&a, &b2, &mut tiny, 8, Accum::Exact).is_err());
        assert!(softmax_neg_scaled_rows(&mut out, 4, 0, 8, 1.0, ExpMode::Libm).is_err());
        // The vector path needs a whole number of vectors per row.
        assert!(softmax_neg_scaled_rows(&mut out, 4, 3, 5, 1.0, ExpMode::Vector).is_err());
    }
}
