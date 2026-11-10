//! Inferno — Cached Krylov information for repeated conjugate-gradient solves.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use super::ciq::LanczosProbe;
use super::precond::{CiqPreconditioner, PrecondKind};

/// Reused VERBATIM from the recycling path — one key type, not two that can
/// disagree about what "the same operator" means.
pub use super::recycle::RecycleKey;

/// Default work budget, in matvec-columns, before an entry is refreshed.
pub const DEFAULT_REFRESH_EVERY: u64 = 120;

/// One operator's cached spectral furniture.
#[derive(Clone)]
pub struct CgCacheEntry {
    probe: Option<LanczosProbe>,
    /// The preconditioner built from that probe.
    pub precond: Option<Arc<CiqPreconditioner>>,
    /// Memo of `precond.log_det()` — closed form, but it is a host reduction
    /// over the rank-p coefficients and the SLQ path asks for it per probe.
    pub log_det_m: Option<f64>,
    /// Which construction `precond` is, so a bake-off or a telemetry line can
    /// report it without unwrapping the payload.
    pub kind: Option<PrecondKind>,
    /// Odometer reading when this entry was built.
    stamp: u64,
}

impl CgCacheEntry {
    pub fn new(
        probe: Option<LanczosProbe>,
        precond: Option<Arc<CiqPreconditioner>>,
        stamp: u64,
    ) -> Self {
        let kind = precond.as_ref().map(|p| p.kind());
        Self {
            probe,
            precond,
            log_det_m: None,
            kind,
            stamp,
        }
    }

    /// The cached probe with its `bounds` REMOVED.
    ///
    /// See the module docs: the basis, Ritz values and Gauss weights are
    /// accelerator inputs and survive staleness gracefully; the bounds are
    /// quadrature nodes and do not. Handing back a probe whose `bounds` are
    /// empty is what makes "do not use these as nodes" a type-level fact
    /// rather than a comment somebody skips.
    pub fn probe_for_preconditioning(&self) -> Option<LanczosProbe> {
        self.probe.as_ref().map(|p| LanczosProbe {
            q: p.q.clone(),
            ritz: p.ritz.clone(),
            weights: p.weights.clone(),
            kept: p.kept.clone(),
            bounds: Vec::new(),
        })
    }

    pub fn stamp(&self) -> u64 {
        self.stamp
    }
}

/// A keyed, work-aged store of [`CgCacheEntry`].
///
/// `RefCell` rather than a global `Mutex`: this matches `settings`'s
/// scoped-guard discipline, avoids a lock candle may hit from a rayon worker,
/// and — because the natural home is a `thread_local!` — stops a `formula`
/// hash collision from crossing models undetectably.
pub struct CgCache {
    inner: RefCell<HashMap<RecycleKey, CgCacheEntry>>,
    odometer: Cell<u64>,
    refresh_every: Cell<u64>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl Default for CgCache {
    fn default() -> Self {
        Self::with_refresh(DEFAULT_REFRESH_EVERY)
    }
}

impl CgCache {
    pub fn with_refresh(refresh_every: u64) -> Self {
        Self {
            inner: RefCell::new(HashMap::new()),
            odometer: Cell::new(0),
            refresh_every: Cell::new(refresh_every.max(1)),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    /// Charge `columns` of matvec work against every entry's age.
    ///
    /// Call ONCE per solve, at exit — `k_done * n_cols`. Inside the loop it
    /// would be both slower and, worse, would age entries by an amount that
    /// depends on where the loop happened to break.
    pub fn tick(&self, columns: u64) {
        self.odometer
            .set(self.odometer.get().saturating_add(columns));
    }

    pub fn odometer(&self) -> u64 {
        self.odometer.get()
    }

    /// `(hits, misses)` since construction — the number a bake-off must
    /// report, because a cache with a 0% hit rate measures the same latency as
    /// no cache and looks like a win that isn't.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }

    /// A live entry for `key`, or `None`.
    ///
    /// Misses on: absent, a different generation (unconditional — the
    /// odometer does not get a say across a generation boundary), or an
    /// odometer delta past `refresh_every`.
    pub fn get(&self, key: &RecycleKey) -> Option<CgCacheEntry> {
        let map = self.inner.borrow();
        let hit = map
            .get(key)
            .filter(|e| self.odometer.get().saturating_sub(e.stamp) < self.refresh_every.get());
        match hit {
            Some(e) => {
                let e = e.clone();
                drop(map);
                self.hits.set(self.hits.get() + 1);
                Some(e)
            }
            None => {
                drop(map);
                self.misses.set(self.misses.get() + 1);
                None
            }
        }
    }

    /// Store an entry stamped at the CURRENT odometer reading.
    ///
    /// Any entry under a different generation for the same `(formula, n)` is
    /// dropped: keeping it would let a `generation` that later cycles back
    /// resurrect a subspace built against parameters two generations old.
    pub fn put(
        &self,
        key: RecycleKey,
        probe: Option<LanczosProbe>,
        precond: Option<Arc<CiqPreconditioner>>,
    ) {
        let entry = CgCacheEntry::new(probe, precond, self.odometer.get());
        let mut map = self.inner.borrow_mut();
        map.retain(|k, _| !(k.formula == key.formula && k.n == key.n));
        map.insert(key, entry);
    }

    /// Memoise `precond.log_det()` on an existing entry.
    pub fn set_log_det(&self, key: &RecycleKey, value: f64) {
        if let Some(e) = self.inner.borrow_mut().get_mut(key) {
            e.log_det_m = Some(value);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.inner.borrow_mut().clear();
        self.odometer.set(0);
        self.hits.set(0);
        self.misses.set(0);
    }
}

thread_local! {
    /// The process-wide (per-thread) cache. Scoped like `settings`.
    pub static CG_CACHE: CgCache = CgCache::default();
}

/// Charge work against the thread's cache. `cg::run` calls this once at exit.
pub fn tick(columns: u64) {
    CG_CACHE.with(|c| c.tick(columns));
}

/// Run `f` against the thread's cache.
pub fn with_cache<R>(f: impl FnOnce(&CgCache) -> R) -> R {
    CG_CACHE.with(f)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Verification of cached krylov information for repeated conjugate-gradient solves.

    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn key(generation: u64) -> RecycleKey {
        RecycleKey {
            formula: 7,
            generation,
            n: 16,
        }
    }

    fn probe(dev: &Device) -> LanczosProbe {
        LanczosProbe {
            q: Tensor::zeros((16, 3), DType::F64, dev).unwrap(),
            ritz: Tensor::new(&[1.0f64, 2.0, 4.0], dev).unwrap(),
            weights: Tensor::new(&[0.5f64, 0.3, 0.2], dev).unwrap(),
            kept: vec![3],
            bounds: vec![(1.0, 4.0)],
        }
    }

    #[test]
    fn a_fresh_entry_hits_and_an_aged_one_misses() {
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        c.put(key(0), Some(probe(&dev)), None);
        assert!(c.get(&key(0)).is_some(), "a just-stored entry must hit");

        c.tick(119);
        assert!(c.get(&key(0)).is_some(), "119 < 120 columns is still fresh");
        c.tick(1);
        assert!(c.get(&key(0)).is_none(), "120 columns must expire it");

        let (hits, misses) = c.stats();
        assert_eq!((hits, misses), (2, 1));
    }

    /// The odometer must not get a say across a generation boundary: the
    /// parameters the subspace was built against have been replaced, and no
    /// amount of "not much work has happened since" makes that stale entry
    /// describe the new operator.
    ///
    /// **Vacuous version to refuse:** bumping the generation AND the odometer
    /// together, which cannot distinguish the two rules.
    #[test]
    fn a_generation_bump_misses_even_at_zero_work() {
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        c.put(key(0), Some(probe(&dev)), None);
        assert_eq!(c.odometer(), 0, "the fixture must not have aged");
        assert!(c.get(&key(0)).is_some());
        assert!(
            c.get(&key(1)).is_none(),
            "a new generation hit an entry built under the old one"
        );
    }

    /// A new generation evicts the old entry for the same operator rather than
    /// accumulating one per generation — otherwise a generation counter that
    /// cycles could resurrect a subspace built two generations ago.
    #[test]
    fn a_new_generation_evicts_the_old_entry() {
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        c.put(key(0), Some(probe(&dev)), None);
        c.put(key(1), Some(probe(&dev)), None);
        assert_eq!(c.len(), 1, "generations accumulated instead of replacing");
        assert!(c.get(&key(0)).is_none());
        assert!(c.get(&key(1)).is_some());
    }

    /// The bounds must never leave the cache. They are quadrature nodes, and a
    /// stale `λmin` that is too large breaks a contour's error bound silently
    /// — the one failure mode in this file that is not merely slow.
    #[test]
    fn the_cached_probe_refuses_to_hand_back_its_bounds() {
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        c.put(key(0), Some(probe(&dev)), None);
        let e = c.get(&key(0)).unwrap();
        let p = e.probe_for_preconditioning().unwrap();
        assert!(p.bounds.is_empty(), "bounds escaped the cache");
        // …while everything a preconditioner legitimately needs survives.
        assert_eq!(p.kept, vec![3]);
        assert_eq!(p.ritz.dims(), &[3]);
        assert_eq!(p.weights.dims(), &[3]);
        assert_eq!(p.q.dims(), &[16, 3]);
    }

    #[test]
    fn the_kind_and_logdet_memo_round_trip() {
        use crate::operators::precond::CiqPreconditioner;
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        let diag = Tensor::full(3.0f64, 16, &dev).unwrap();
        let p = Arc::new(CiqPreconditioner::jacobi(&diag, 1e-8).unwrap());
        let want = p.log_det().unwrap().to_scalar::<f64>().unwrap();
        c.put(key(0), None, Some(p));
        c.set_log_det(&key(0), want);
        let e = c.get(&key(0)).unwrap();
        assert_eq!(e.kind, Some(PrecondKind::Jacobi));
        assert_eq!(e.log_det_m, Some(want));
        assert_eq!(e.precond.unwrap().rank(), 0);
    }

    /// A shape change is a miss, not a panic.
    #[test]
    fn a_size_change_misses() {
        let dev = Device::Cpu;
        let c = CgCache::with_refresh(120);
        c.put(key(0), Some(probe(&dev)), None);
        let other = RecycleKey {
            formula: 7,
            generation: 0,
            n: 32,
        };
        assert!(c.get(&other).is_none());
    }

    #[test]
    fn the_thread_local_odometer_accumulates() {
        with_cache(|c| c.clear());
        tick(40);
        tick(2);
        assert_eq!(with_cache(|c| c.odometer()), 42);
        with_cache(|c| c.clear());
    }
}
