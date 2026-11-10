//! The Tensor value type.
//!
//! Design (from the port plan): PyTorch's `Variable` is `at::Tensor` with
//! `AutogradMeta` welded inside `c10::TensorImpl`, which makes the autograd
//! layer unliftable. Here the split is inverted — `Tensor` is autograd-free
//! and `dk_autograd::Variable` composes over it. The one autograd-facing
//! concession is the version counter: an `Arc<AtomicU64>` shared by every
//! view of the same data, bumped on in-place mutation, checked by
//! `SavedVariable::unpack` (PyTorch's VariableVersion, exactly).
//!
//! Layout is (shape, strides-in-elements, offset-in-elements) over untyped
//! byte storage; `expand` uses stride-0 broadcasting like PyTorch.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::storage::{Device, Storage};
use crate::DType;

pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for (i, &d) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= d.max(1);
    }
    strides
}

#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    dtype: DType,
    shape: Vec<usize>,
    /// In elements, not bytes. May contain 0 (broadcast dims from `expand`).
    strides: Vec<usize>,
    /// In elements from the start of storage.
    offset: usize,
    /// Shared across every view of the same data; bumped on in-place writes.
    version: Arc<AtomicU64>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor[{:?}; {}; {:?}]",
            self.shape,
            self.dtype,
            self.device()
        )
    }
}

// ---------------------------------------------------------------------------
// element codecs (CPU)
// ---------------------------------------------------------------------------

fn read_elem(bytes: &[u8], dtype: DType, idx: usize) -> f64 {
    let s = dtype.size_in_bytes();
    let b = &bytes[idx * s..(idx + 1) * s];
    match dtype {
        DType::F32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
        DType::F64 => f64::from_le_bytes(b.try_into().unwrap()),
        DType::F16 => half::f16::from_le_bytes(b.try_into().unwrap()).to_f64(),
        DType::BF16 => half::bf16::from_le_bytes(b.try_into().unwrap()).to_f64(),
    }
}

fn write_elem(bytes: &mut [u8], dtype: DType, idx: usize, v: f64) {
    let s = dtype.size_in_bytes();
    let b = &mut bytes[idx * s..(idx + 1) * s];
    match dtype {
        DType::F32 => b.copy_from_slice(&(v as f32).to_le_bytes()),
        DType::F64 => b.copy_from_slice(&v.to_le_bytes()),
        DType::F16 => b.copy_from_slice(&half::f16::from_f64(v).to_le_bytes()),
        DType::BF16 => b.copy_from_slice(&half::bf16::from_f64(v).to_le_bytes()),
    }
}

/// Iterates storage offsets (in elements) of a strided view in logical
/// (row-major) order. Handles stride-0 broadcast dims.
pub struct StridedIndex<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    idx: Vec<usize>,
    offset: usize,
    remaining: usize,
}

impl<'a> StridedIndex<'a> {
    pub fn new(shape: &'a [usize], strides: &'a [usize], offset: usize) -> Self {
        let n: usize = shape.iter().product();
        StridedIndex {
            shape,
            strides,
            idx: vec![0; shape.len()],
            offset,
            remaining: n,
        }
    }
}

impl Iterator for StridedIndex<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let cur = self.offset;
        self.remaining -= 1;
        // advance the multi-index, updating offset incrementally
        for d in (0..self.shape.len()).rev() {
            self.idx[d] += 1;
            self.offset += self.strides[d];
            if self.idx[d] < self.shape[d] {
                break;
            }
            self.offset -= self.strides[d] * self.shape[d];
            self.idx[d] = 0;
        }
        Some(cur)
    }
}

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

impl Tensor {
    fn from_cpu_bytes(bytes: Vec<u8>, dtype: DType, shape: Vec<usize>) -> Tensor {
        let strides = contiguous_strides(&shape);
        Tensor {
            storage: Storage::Cpu(Arc::new(bytes)),
            dtype,
            shape,
            strides,
            offset: 0,
            version: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_cpu_bytes(vec![0u8; n * dtype.size_in_bytes()], dtype, shape.to_vec())
    }

    pub fn full(shape: &[usize], dtype: DType, v: f64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut bytes = vec![0u8; n * dtype.size_in_bytes()];
        for i in 0..n {
            write_elem(&mut bytes, dtype, i, v);
        }
        Tensor::from_cpu_bytes(bytes, dtype, shape.to_vec())
    }

    pub fn ones(shape: &[usize], dtype: DType) -> Tensor {
        Tensor::full(shape, dtype, 1.0)
    }

    pub fn from_f64_slice(data: &[f64], shape: &[usize], dtype: DType) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(Error::ShapeMismatch {
                op: "from_f64_slice",
                lhs: shape.to_vec(),
                rhs: vec![data.len()],
            });
        }
        let mut bytes = vec![0u8; n * dtype.size_in_bytes()];
        for (i, &v) in data.iter().enumerate() {
            write_elem(&mut bytes, dtype, i, v);
        }
        Ok(Tensor::from_cpu_bytes(bytes, dtype, shape.to_vec()))
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(Error::ShapeMismatch {
                op: "from_f32_slice",
                lhs: shape.to_vec(),
                rhs: vec![data.len()],
            });
        }
        let mut bytes = Vec::with_capacity(n * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Tensor::from_cpu_bytes(bytes, DType::F32, shape.to_vec()))
    }

    pub fn scalar(v: f64, dtype: DType) -> Tensor {
        Tensor::full(&[], dtype, v)
    }

    /// Standard-normal init from a caller-supplied RNG (keeps this crate
    /// deterministic under test seeds; there is no global RNG here).
    pub fn randn<R: rand::Rng>(rng: &mut R, shape: &[usize], dtype: DType) -> Tensor {
        use rand::prelude::Distribution;
        let normal = rand_normal();
        let n: usize = shape.iter().product();
        let mut bytes = vec![0u8; n * dtype.size_in_bytes()];
        for i in 0..n {
            write_elem(&mut bytes, dtype, i, normal.sample(rng));
        }
        Tensor::from_cpu_bytes(bytes, dtype, shape.to_vec())
    }
}

/// Box–Muller via rand's Standard uniform — avoids a rand_distr dependency.
fn rand_normal() -> impl rand::distributions::Distribution<f64> {
    struct BoxMuller;
    impl rand::distributions::Distribution<f64> for BoxMuller {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
            let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    BoxMuller
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> Device {
        self.storage.device()
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// True if `other` aliases the same allocation.
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        self.storage.ptr_eq(&other.storage)
    }

    // -- version counter (the SavedVariable contract) -----------------------

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Record an in-place mutation. Every view sharing this data observes it.
    pub fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    pub fn version_counter(&self) -> &Arc<AtomicU64> {
        &self.version
    }
}

// ---------------------------------------------------------------------------
// views (metadata-only, storage shared, version counter shared)
// ---------------------------------------------------------------------------

impl Tensor {
    fn view_with(&self, shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Tensor {
        Tensor {
            storage: self.storage.clone(),
            dtype: self.dtype,
            shape,
            strides,
            offset,
            version: self.version.clone(),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if n != self.elem_count() {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                lhs: self.shape.clone(),
                rhs: shape.to_vec(),
            });
        }
        if self.is_contiguous() {
            Ok(self.view_with(shape.to_vec(), contiguous_strides(shape), self.offset))
        } else {
            self.contiguous()?.reshape(shape)
        }
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor> {
        if d0 >= self.rank() || d1 >= self.rank() {
            return Err(Error::Msg(format!(
                "transpose: dims ({d0},{d1}) out of range for rank {}",
                self.rank()
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(d0, d1);
        strides.swap(d0, d1);
        Ok(self.view_with(shape, strides, self.offset))
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        if dims.len() != self.rank() {
            return Err(Error::Msg(format!(
                "permute: got {} dims for rank {}",
                dims.len(),
                self.rank()
            )));
        }
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            if d >= dims.len() || seen[d] {
                return Err(Error::Msg(format!("permute: invalid dims {dims:?}")));
            }
            seen[d] = true;
        }
        let shape = dims.iter().map(|&d| self.shape[d]).collect();
        let strides = dims.iter().map(|&d| self.strides[d]).collect();
        Ok(self.view_with(shape, strides, self.offset))
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        if dim >= self.rank() || start + len > self.shape[dim] {
            return Err(Error::Msg(format!(
                "narrow: dim {dim} start {start} len {len} on shape {:?}",
                self.shape
            )));
        }
        let mut shape = self.shape.clone();
        shape[dim] = len;
        let offset = self.offset + start * self.strides[dim];
        Ok(self.view_with(shape, self.strides.clone(), offset))
    }

    pub fn squeeze(&self, dim: usize) -> Result<Tensor> {
        if dim >= self.rank() || self.shape[dim] != 1 {
            return Err(Error::Msg(format!(
                "squeeze: dim {dim} of shape {:?} is not 1",
                self.shape
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.remove(dim);
        strides.remove(dim);
        Ok(self.view_with(shape, strides, self.offset))
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        if dim > self.rank() {
            return Err(Error::Msg(format!(
                "unsqueeze: dim {dim} > rank {}",
                self.rank()
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        // stride value is immaterial for a size-1 dim; use the contiguous one
        let s = if dim == strides.len() {
            1
        } else {
            strides[dim] * shape[dim].max(1)
        };
        shape.insert(dim, 1);
        strides.insert(dim, s);
        Ok(self.view_with(shape, strides, self.offset))
    }

    /// Broadcast size-1 dims to `shape` with stride 0 (PyTorch expand).
    pub fn expand(&self, shape: &[usize]) -> Result<Tensor> {
        if shape.len() < self.rank() {
            return Err(Error::ShapeMismatch {
                op: "expand",
                lhs: self.shape.clone(),
                rhs: shape.to_vec(),
            });
        }
        let lead = shape.len() - self.rank();
        let mut strides = vec![0usize; shape.len()];
        for i in 0..self.rank() {
            let (own, want) = (self.shape[i], shape[lead + i]);
            if own == want {
                strides[lead + i] = self.strides[i];
            } else if own == 1 {
                strides[lead + i] = 0;
            } else {
                return Err(Error::ShapeMismatch {
                    op: "expand",
                    lhs: self.shape.clone(),
                    rhs: shape.to_vec(),
                });
            }
        }
        Ok(self.view_with(shape.to_vec(), strides, self.offset))
    }
}

// ---------------------------------------------------------------------------
// CPU data access
// ---------------------------------------------------------------------------

impl Tensor {
    fn cpu_bytes(&self, op: &'static str) -> Result<&[u8]> {
        self.storage
            .cpu_bytes()
            .ok_or(Error::DeviceMismatch { op })
    }

    /// Logical-order copy-out as f64 (any dtype, any strides).
    pub fn to_vec_f64(&self) -> Result<Vec<f64>> {
        let bytes = self.cpu_bytes("to_vec_f64")?;
        let mut out = Vec::with_capacity(self.elem_count());
        for off in StridedIndex::new(&self.shape, &self.strides, self.offset) {
            out.push(read_elem(bytes, self.dtype, off));
        }
        Ok(out)
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.to_vec_f64()?.into_iter().map(|v| v as f32).collect())
    }

    /// Single element of a scalar tensor.
    pub fn to_scalar_f64(&self) -> Result<f64> {
        if self.elem_count() != 1 {
            return Err(Error::Msg(format!(
                "to_scalar_f64 on shape {:?}",
                self.shape
            )));
        }
        let bytes = self.cpu_bytes("to_scalar_f64")?;
        Ok(read_elem(bytes, self.dtype, self.offset))
    }

    /// Materialize into fresh contiguous storage (same dtype). A copy — the
    /// result has a fresh version counter, as it aliases nothing.
    pub fn contiguous(&self) -> Result<Tensor> {
        let bytes = self.cpu_bytes("contiguous")?;
        let n = self.elem_count();
        let s = self.dtype.size_in_bytes();
        let mut out = vec![0u8; n * s];
        for (i, off) in StridedIndex::new(&self.shape, &self.strides, self.offset).enumerate() {
            out[i * s..(i + 1) * s].copy_from_slice(&bytes[off * s..(off + 1) * s]);
        }
        Ok(Tensor::from_cpu_bytes(out, self.dtype, self.shape.clone()))
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype {
            return self.contiguous();
        }
        Tensor::from_f64_slice(&self.to_vec_f64()?, &self.shape.clone(), dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_views_read_correctly() {
        let t = Tensor::from_f64_slice(&[1., 2., 3., 4., 5., 6.], &[2, 3], DType::F32).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.to_vec_f64().unwrap(), vec![1., 4., 2., 5., 3., 6.]);
        let n = t.narrow(1, 1, 2).unwrap();
        assert_eq!(n.to_vec_f64().unwrap(), vec![2., 3., 5., 6.]);
        assert!(n.shares_storage(&t));
        assert!(!n.contiguous().unwrap().shares_storage(&t));
    }

    #[test]
    fn expand_broadcasts_with_zero_stride() {
        let t = Tensor::from_f64_slice(&[1., 2.], &[2, 1], DType::F64).unwrap();
        let e = t.expand(&[2, 3]).unwrap();
        assert_eq!(e.to_vec_f64().unwrap(), vec![1., 1., 1., 2., 2., 2.]);
        assert_eq!(e.strides(), &[1, 0]);
    }

    #[test]
    fn version_counter_is_shared_across_views() {
        let t = Tensor::zeros(&[4], DType::F32);
        let v = t.narrow(0, 1, 2).unwrap();
        assert_eq!(v.version(), 0);
        t.bump_version();
        assert_eq!(v.version(), 1);
        // a contiguous copy detaches from the counter
        let c = t.contiguous().unwrap();
        t.bump_version();
        assert_eq!(c.version(), 0);
    }

    #[test]
    fn half_dtypes_round_trip() {
        let t = Tensor::from_f64_slice(&[0.5, -1.25], &[2], DType::F16).unwrap();
        assert_eq!(t.to_vec_f64().unwrap(), vec![0.5, -1.25]);
        let b = Tensor::from_f64_slice(&[1.0, -2.0], &[2], DType::BF16).unwrap();
        assert_eq!(b.to_vec_f64().unwrap(), vec![1.0, -2.0]);
    }
}
