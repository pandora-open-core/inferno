//! Inferno — Reusable caches for caller-owned model and tensor computations.

use crate::{matrix, Result, Tensor};
use std::collections::VecDeque;
pub trait Cache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V>;
    fn insert(&mut self, key: K, value: V);
    fn clear(&mut self);
}
/// Entry-bounded LRU for small developer caches; O(capacity) lookup.
/// Capacity zero disables storage. Values and keys are owned, never global.
#[derive(Debug)]
pub struct LruCache<K, V> {
    capacity: usize,
    entries: VecDeque<(K, V)>,
}
impl<K: Eq, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
impl<K: Eq, V> Cache<K, V> for LruCache<K, V> {
    fn get(&mut self, key: &K) -> Option<&V> {
        let i = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(i)?;
        self.entries.push_back(entry);
        self.entries.back().map(|(_, v)| v)
    }
    fn insert(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }
        if let Some(i) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(i);
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((key, value));
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
}
/// Exact input bits, dtypes and shapes avoid hash-only aliasing. Revision identifies
/// kernel parameters, dtype policy and any other external dependencies.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CovarianceKey {
    revision: u64,
    x_dtype: crate::DType,
    y_dtype: crate::DType,
    x_shape: (usize, usize),
    y_shape: (usize, usize),
    x: Vec<u64>,
    y: Vec<u64>,
}
impl CovarianceKey {
    pub fn new(revision: u64, x: &Tensor, y: &Tensor) -> Result<Self> {
        let x_dtype = x.dtype();
        let y_dtype = y.dtype();
        let (n, d, x) = matrix(x)?;
        let (m, e, y) = matrix(y)?;
        Ok(Self {
            revision,
            x_dtype,
            y_dtype,
            x_shape: (n, d),
            y_shape: (m, e),
            x: x.into_iter().map(f64::to_bits).collect(),
            y: y.into_iter().map(f64::to_bits).collect(),
        })
    }
}
/// Compute only on a miss. Errors are never inserted. The returned tensor is a
/// clone of the cached value; use only with detached CPU reference tensors.
pub fn covariance<C: Cache<CovarianceKey, Tensor>, K: crate::covariance::Covariance>(
    cache: &mut C,
    kernel: &K,
    revision: u64,
    x: &Tensor,
    y: &Tensor,
) -> Result<Tensor> {
    let key = CovarianceKey::new(revision, x, y)?;
    if let Some(value) = cache.get(&key) {
        return Ok(value.clone());
    }
    let value = kernel.covariance(x, y)?;
    let (n, m, _) = matrix(&value)?;
    crate::ensure(
        (n, m) == (x.shape()[0], y.shape()[0]),
        "cached kernel shape mismatch",
    )?;
    cache.insert(key, value.clone());
    Ok(value)
}

/// Plug any developer cache directly into `ExactGp` as its covariance component.
/// The mutex permits shared posterior prediction when `K` and `C` are thread-safe.
/// Misses compute while holding the lock; kernels must not reenter this adapter.
pub struct CachedCovariance<K, C = LruCache<CovarianceKey, Tensor>> {
    kernel: K,
    cache: std::sync::Mutex<C>,
    revision: u64,
}
impl<K, C: Cache<CovarianceKey, Tensor>> CachedCovariance<K, C> {
    pub fn new(kernel: K, cache: C, revision: u64) -> Self {
        Self {
            kernel,
            cache: std::sync::Mutex::new(cache),
            revision,
        }
    }
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
    /// Clear cached matrices before allowing a parameter update.
    pub fn kernel_mut(&mut self) -> Result<&mut K> {
        self.invalidate()?;
        Ok(&mut self.kernel)
    }
    pub fn invalidate(&mut self) -> Result<()> {
        self.cache
            .get_mut()
            .map_err(|_| crate::Error("covariance cache poisoned".into()))?
            .clear();
        Ok(())
    }
    /// Change the external dependency revision and discard old entries.
    pub fn set_revision(&mut self, revision: u64) -> Result<()> {
        self.invalidate()?;
        self.revision = revision;
        Ok(())
    }
}
impl<K> CachedCovariance<K> {
    pub fn lru(kernel: K, capacity: usize, revision: u64) -> Self {
        Self::new(kernel, LruCache::new(capacity), revision)
    }
}
impl<K: crate::covariance::Covariance, C: Cache<CovarianceKey, Tensor>>
    crate::covariance::Covariance for CachedCovariance<K, C>
{
    fn covariance(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| crate::Error("covariance cache poisoned".into()))?;
        covariance(&mut *cache, &self.kernel, self.revision, x, y)
    }
}
