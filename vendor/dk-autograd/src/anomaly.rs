//! Port of torch/csrc/autograd/anomaly_mode.{h,cpp} — a global flag; when
//! on, every Node records its creation backtrace, and the engine attaches
//! it to backward errors so the failure points at the *forward* op.

use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::Relaxed);
}

/// RAII guard, upstream `DetectAnomalyGuard`.
pub struct DetectAnomalyGuard {
    prev: bool,
}

impl DetectAnomalyGuard {
    pub fn new() -> DetectAnomalyGuard {
        let prev = is_enabled();
        set_enabled(true);
        DetectAnomalyGuard { prev }
    }
}

impl Default for DetectAnomalyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DetectAnomalyGuard {
    fn drop(&mut self) {
        set_enabled(self.prev);
    }
}

/// Backtrace at node creation, when anomaly mode is on.
pub fn capture_trace() -> Option<String> {
    if is_enabled() {
        Some(std::backtrace::Backtrace::force_capture().to_string())
    } else {
        None
    }
}
