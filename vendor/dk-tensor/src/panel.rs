//! Cache-line-aligned, vector-padded buffers — the layout a cached artefact
//! should already be in.
//!
//! # The problem this exists to remove
//!
//! Every cache on the transport path stores a framework tensor and every
//! consumer of one begins by copying it out (`flatten_all()?.to_vec1()?`),
//! usually widening `f32` to `f64` on the way. So a "cache hit" is a hit on
//! the *arithmetic* and a miss on the *layout*: the bytes still have to be
//! walked, re-allocated and re-laid-out before a single FLOP happens, once per
//! consumer, every time.
//!
//! A [`Panel`] is the other half of that hit. It is what the kernel wants —
//! 64-byte-aligned base, rows padded to a whole number of vector registers, no
//! strides to interpret — so a cached one is loadable as-is and the copy
//! disappears rather than being made faster.
//!
//! # Two layouts, because a contraction has two operands
//!
//! | | shape | indexed by | who reads it |
//! |---|---|---|---|
//! | [`Panel`] | `[rows, stride]` row-major | one row at a time | the **streaming** operand |
//! | [`PackedSupport`] | `[d, n_padded]` dim-major | one dim at a time | the **resident** operand |
//!
//! The transpose in [`PackedSupport`] is the entire reason the micro-kernel
//! can be written at all: at a fixed feature dimension, `LANES` consecutive
//! support columns are contiguous, so they are one vector load. Row-major, the
//! same eight values are `d` floats apart and the load becomes a gather.
//!
//! # Padding columns are poisoned, not zeroed
//!
//! `PackedSupport` rounds `n` up to a whole vector, and those padding columns
//! produce output the caller must not read. Zero-filling them would make a
//! padding column look like an atom sitting at the origin — which, for a point
//! near the origin, *wins the softmax*. So they are filled with
//! [`PAD_COORD`] instead: the squared distance to one is enormous but finite,
//! every softmax weight it could earn underflows to exactly zero, and the
//! failure mode of forgetting to mask is a harmless zero rather than a
//! confident wrong assignment.
//!
//! That is belt and braces on purpose. The stride and the valid count are also
//! carried separately in every signature, so a caller *can* mask; the poison
//! is what happens when one does not.

use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::cache_model::{round_up, CACHE_LINE, LANES};
use crate::error::{Error, Result};

/// Coordinate written into a packed support's padding columns.
///
/// Chosen so `d * PAD_COORD^2` stays finite in `f32` for every feature width
/// this tree uses (`1e34` per dimension against an `f32` ceiling of `3.4e38`
/// leaves room for `d` up to ~4000), because an `inf` here would propagate
/// into a max-shift as `inf - inf = NaN`. Large enough that any temperature a
/// caller could reasonably pass sends `exp(-d^2/T)` to exactly zero.
pub const PAD_COORD: f32 = 1.0e17;

// ---------------------------------------------------------------------------
// AlignedF32
// ---------------------------------------------------------------------------

/// A `f32` buffer whose base address is a multiple of [`CACHE_LINE`].
///
/// `Vec<f32>` guarantees only 4-byte alignment, which is enough for
/// correctness (every vector load in [`crate::pairwise`] is an unaligned one)
/// and not enough for speed: an unaligned base makes every tile straddle one
/// more cache line than it needs to, and the extra line is charged on every
/// pass over the panel, not once.
pub struct AlignedF32 {
    ptr: NonNull<f32>,
    len: usize,
}

// Safety: this is a plain owned allocation of `f32`, with no interior
// mutability and no thread affinity.
unsafe impl Send for AlignedF32 {}
unsafe impl Sync for AlignedF32 {}

impl AlignedF32 {
    fn layout(len: usize) -> Result<Layout> {
        // A zero-sized allocation is UB, so the empty buffer is one element
        // wide and reports a length of zero.
        Layout::from_size_align(len.max(1) * std::mem::size_of::<f32>(), CACHE_LINE)
            .map_err(|e| Error::Msg(format!("aligned buffer layout for {len} f32: {e}")))
    }

    /// A zero-filled buffer of `len` floats.
    pub fn zeros(len: usize) -> Result<Self> {
        Self::filled(len, 0.0)
    }

    /// A buffer of `len` floats, every one set to `v`.
    pub fn filled(len: usize, v: f32) -> Result<Self> {
        let layout = Self::layout(len)?;
        // Safety: layout has a non-zero size by construction.
        let raw = unsafe { alloc(layout) } as *mut f32;
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error::Msg(format!("out of memory allocating {len} aligned f32")))?;
        // Safety: `raw` owns `len.max(1)` f32 slots, and f32 has no drop glue,
        // so writing over uninitialised memory is sound.
        unsafe {
            for i in 0..len {
                ptr.as_ptr().add(i).write(v);
            }
        }
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The base address, for asserting the alignment this type exists for.
    pub fn addr(&self) -> usize {
        self.ptr.as_ptr() as usize
    }
}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        if let Ok(layout) = Self::layout(self.len) {
            // Safety: allocated by `alloc` with this exact layout.
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) }
        }
    }
}

impl Deref for AlignedF32 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // Safety: `len` elements were initialised at construction.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedF32 {
    fn deref_mut(&mut self) -> &mut [f32] {
        // Safety: as above, and `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Clone for AlignedF32 {
    fn clone(&self) -> Self {
        let mut out = Self::zeros(self.len).expect("cloning an existing allocation cannot overflow");
        out.copy_from_slice(self);
        out
    }
}

impl std::fmt::Debug for AlignedF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedF32").field("len", &self.len).finish()
    }
}

// ---------------------------------------------------------------------------
// Panel — the streaming operand
// ---------------------------------------------------------------------------

/// A row-major `[rows, cols]` matrix stored with a vector-padded stride.
///
/// The padding is what lets a kernel read `stride` floats from any row without
/// a masked tail; the values in `cols..stride` are zero and no kernel here
/// contracts over them.
#[derive(Clone, Debug)]
pub struct Panel {
    buf: AlignedF32,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl Panel {
    /// An all-zero `[rows, cols]` panel.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        let stride = round_up(cols, LANES).max(LANES);
        Ok(Self { buf: AlignedF32::zeros(rows * stride)?, rows, cols, stride })
    }

    /// Copy a row-major `[rows, cols]` slice into a padded panel.
    pub fn from_rows(data: &[f32], rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Msg(format!(
                "panel: expected {rows}x{cols} = {} values, got {}",
                rows * cols,
                data.len()
            )));
        }
        let mut p = Self::zeros(rows, cols)?;
        for i in 0..rows {
            let dst = &mut p.buf[i * p.stride..i * p.stride + cols];
            dst.copy_from_slice(&data[i * cols..(i + 1) * cols]);
        }
        Ok(p)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Floats between the starts of consecutive rows. A multiple of [`LANES`].
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The whole backing store, `rows * stride` floats.
    pub fn as_slice(&self) -> &[f32] {
        &self.buf
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buf
    }

    /// One row's valid `cols` values.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.buf[i * self.stride..i * self.stride + self.cols]
    }

    /// One row including its zero padding — `stride` floats.
    pub fn row_padded(&self, i: usize) -> &[f32] {
        &self.buf[i * self.stride..(i + 1) * self.stride]
    }

    /// Copy back into an unpadded row-major `[rows, cols]` vector.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            out[i * self.cols..(i + 1) * self.cols].copy_from_slice(self.row(i));
        }
        out
    }

    /// Bytes the panel occupies, for comparing against a
    /// [`CacheBudget`](crate::cache_model::CacheBudget) level.
    pub fn bytes(&self) -> usize {
        self.buf.len() * std::mem::size_of::<f32>()
    }

    pub fn base_addr(&self) -> usize {
        self.buf.addr()
    }

    /// A borrowed view of the whole panel.
    pub fn view(&self) -> PanelRef<'_> {
        PanelRef { data: self.as_slice(), rows: self.rows, cols: self.cols, stride: self.stride }
    }

    /// A borrowed view of `count` rows starting at `start`.
    ///
    /// The reason views exist at all: a fused kernel walks its input in
    /// L1-sized row blocks, and copying each block out would reintroduce
    /// exactly the pass over memory the fusion is there to remove.
    pub fn rows_view(&self, start: usize, count: usize) -> Result<PanelRef<'_>> {
        if start + count > self.rows {
            return Err(Error::Msg(format!(
                "panel view: rows {start}..{} of a {}-row panel",
                start + count,
                self.rows
            )));
        }
        Ok(PanelRef {
            data: &self.as_slice()[start * self.stride..(start + count) * self.stride],
            rows: count,
            cols: self.cols,
            stride: self.stride,
        })
    }
}

/// A borrowed `[rows, stride]` row-major block. What the kernels actually
/// take, so a caller may hand them a slice of a larger panel.
#[derive(Clone, Copy, Debug)]
pub struct PanelRef<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
}

impl<'a> From<&'a Panel> for PanelRef<'a> {
    fn from(p: &'a Panel) -> Self {
        p.view()
    }
}

// ---------------------------------------------------------------------------
// PackedSupport — the resident operand
// ---------------------------------------------------------------------------

/// A `[n, d]` support set stored **dim-major** as `[d, n_padded]`.
///
/// This is the layout a codebook's atoms, a GP's inducing points or a
/// mixture's means should be *cached in*: packing is a pure function of the
/// fitted artefact, it happens once per fit, and every message thereafter
/// streams past it without touching a transpose.
#[derive(Clone, Debug)]
pub struct PackedSupport {
    buf: AlignedF32,
    n: usize,
    d: usize,
    stride: usize,
    sqnorm: AlignedF32,
}

impl PackedSupport {
    /// Pack a row-major `[n, d]` support set, poisoning the padding columns
    /// with [`PAD_COORD`].
    pub fn pack(data: &[f32], n: usize, d: usize) -> Result<Self> {
        Self::pack_with_pad(data, n, d, PAD_COORD)
    }

    /// Pack with a chosen padding value.
    ///
    /// The default poison is right for a set of POSITIONS, where a padding
    /// column must sit impossibly far from every query. It is wrong for a set
    /// of SCALES: a diagonal-Mahalanobis kernel divides by the variance, and a
    /// padding variance of `1e17` would divide the poisoned distance back down
    /// to something a query could plausibly be assigned to. Pack a variance
    /// panel with `1.0` so the position's poison survives the division.
    pub fn pack_with_pad(data: &[f32], n: usize, d: usize, pad: f32) -> Result<Self> {
        if data.len() != n * d {
            return Err(Error::Msg(format!(
                "packed support: expected {n}x{d} = {} values, got {}",
                n * d,
                data.len()
            )));
        }
        if n == 0 || d == 0 {
            return Err(Error::Msg("packed support: n and d must both be non-zero".into()));
        }
        let stride = round_up(n, LANES);
        // Poisoned, not zeroed — see the module docs.
        let mut buf = AlignedF32::filled(stride * d, pad)?;
        for j in 0..n {
            for c in 0..d {
                buf[c * stride + j] = data[j * d + c];
            }
        }
        let mut sqnorm = AlignedF32::zeros(stride)?;
        for j in 0..n {
            sqnorm[j] = data[j * d..(j + 1) * d].iter().map(|v| v * v).sum();
        }
        Ok(Self { buf, n, d, stride, sqnorm })
    }

    /// Valid support columns.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Feature width.
    pub fn d(&self) -> usize {
        self.d
    }
    /// Padded column count — `n` rounded up to [`LANES`]. Output tiles are
    /// this wide.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The `stride` values at feature dimension `c`, contiguous. This
    /// contiguity is the whole point of the layout.
    pub fn dim(&self, c: usize) -> &[f32] {
        &self.buf[c * self.stride..(c + 1) * self.stride]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.buf
    }

    /// `||b_j||^2` for each column, padding entries zero. Not used by the
    /// direct kernel — which subtracts before squaring, and is more accurate
    /// for that reason — but it is the quantity a caller needs to reach for
    /// the expanded `||a||^2 + ||b||^2 - 2a·b` form, so it is computed once
    /// here rather than by whoever wants it.
    pub fn sqnorms(&self) -> &[f32] {
        &self.sqnorm
    }

    pub fn bytes(&self) -> usize {
        (self.buf.len() + self.sqnorm.len()) * std::mem::size_of::<f32>()
    }

    pub fn base_addr(&self) -> usize {
        self.buf.addr()
    }

    /// Reconstruct the row-major `[n, d]` form. Round-trip check, and the
    /// escape hatch for a consumer that needs the original layout.
    pub fn to_row_major(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.n * self.d];
        for j in 0..self.n {
            for c in 0..self.d {
                out[j * self.d + c] = self.buf[c * self.stride + j];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    #[test]
    fn an_aligned_buffer_starts_on_a_cache_line() {
        for len in [1usize, 7, 64, 1000] {
            let b = AlignedF32::zeros(len).unwrap();
            assert_eq!(b.addr() % CACHE_LINE, 0, "len {len} allocated at {:x}", b.addr());
            assert_eq!(b.len(), len);
            assert!(b.iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn a_filled_buffer_holds_the_fill_and_clones_it() {
        let b = AlignedF32::filled(9, 2.5).unwrap();
        assert!(b.iter().all(|v| *v == 2.5));
        let c = b.clone();
        assert_eq!(&*c, &*b);
        assert_eq!(c.addr() % CACHE_LINE, 0);
    }

    #[test]
    fn a_panel_round_trips_and_pads_to_a_vector_multiple() {
        let data = seq(5 * 3);
        let p = Panel::from_rows(&data, 5, 3).unwrap();
        assert_eq!(p.stride() % LANES, 0);
        assert!(p.stride() >= p.cols());
        assert_eq!(p.base_addr() % CACHE_LINE, 0);
        assert_eq!(p.to_dense(), data);
        for i in 0..5 {
            assert_eq!(p.row(i), &data[i * 3..(i + 1) * 3]);
            // The padding is zero and is not part of the row.
            assert!(p.row_padded(i)[3..].iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn a_panel_rejects_a_length_that_does_not_match_its_shape() {
        assert!(Panel::from_rows(&seq(10), 3, 4).is_err());
    }

    #[test]
    fn packing_transposes_and_round_trips() {
        let (n, d) = (5usize, 3usize);
        let data = seq(n * d);
        let s = PackedSupport::pack(&data, n, d).unwrap();
        assert_eq!(s.stride(), 8, "5 columns must pad to one whole vector");
        assert_eq!(s.base_addr() % CACHE_LINE, 0);
        assert_eq!(s.to_row_major(), data);
        // The defining property: at a fixed dim, columns are contiguous.
        for c in 0..d {
            for j in 0..n {
                assert_eq!(s.dim(c)[j], data[j * d + c]);
            }
        }
    }

    /// The hazard the poison exists for. A zero-filled padding column is an
    /// atom at the origin, and a point near the origin would be assigned to
    /// it — a confident wrong answer with no error anywhere.
    #[test]
    fn padding_columns_are_poisoned_so_a_forgotten_mask_yields_zero_not_a_win() {
        let (n, d) = (3usize, 4usize);
        let atoms = vec![0.0f32; n * d]; // every real atom AT the origin
        let s = PackedSupport::pack(&atoms, n, d).unwrap();
        for c in 0..d {
            for j in n..s.stride() {
                assert_eq!(s.dim(c)[j], PAD_COORD, "padding column {j} was not poisoned");
            }
        }
        // A query at the origin: real columns score 0, padding scores enormous.
        let q = vec![0.0f32; d];
        for j in n..s.stride() {
            let d2: f32 = (0..d).map(|c| (q[c] - s.dim(c)[j]).powi(2)).sum();
            assert!(d2.is_finite(), "padding produced a non-finite distance");
            assert!(d2 > 1e30, "padding distance {d2} is not large enough to underflow");
            assert_eq!((-d2).exp(), 0.0, "a padding column could still win a softmax");
        }
    }

    #[test]
    fn a_variance_panel_pads_with_one_so_the_position_poison_survives_division() {
        let (n, d) = (3usize, 4usize);
        let var = vec![0.25f32; n * d];
        let v = PackedSupport::pack_with_pad(&var, n, d, 1.0).unwrap();
        let mu = PackedSupport::pack(&vec![0.0f32; n * d], n, d).unwrap();
        let q = vec![0.0f32; d];
        for j in n..v.stride() {
            let quad: f32 =
                (0..d).map(|c| (q[c] - mu.dim(c)[j]).powi(2) / v.dim(c)[j]).sum();
            assert!(quad > 1e30, "padding quad {quad} is small enough to win");
            assert!(quad.is_finite());
        }
    }

    #[test]
    fn sqnorms_are_precomputed_for_the_valid_columns_only() {
        let (n, d) = (3usize, 2usize);
        let data = vec![3.0f32, 4.0, 1.0, 0.0, 0.0, 2.0];
        let s = PackedSupport::pack(&data, n, d).unwrap();
        assert_eq!(&s.sqnorms()[..n], &[25.0, 1.0, 4.0]);
        assert!(s.sqnorms()[n..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn packing_rejects_empty_or_mismatched_input() {
        assert!(PackedSupport::pack(&[], 0, 3).is_err());
        assert!(PackedSupport::pack(&seq(6), 3, 3).is_err());
    }
}
