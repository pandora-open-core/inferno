//! The memory hierarchy, as a number the kernels can actually divide by.
//!
//! Every blocked kernel in [`crate::pairwise`] asks the same question — "how
//! many rows may I hold resident before I start evicting the panel I am
//! streaming against" — and that question has exactly one honest answer per
//! machine. This module is where the answer lives, so no kernel carries a
//! hard-coded `32` whose provenance nobody remembers.
//!
//! # Three numbers, three different jobs
//!
//! * **L1d** is the only level a micro-kernel may plan against. A tile that
//!   fits L1 runs at load-to-use latency; one that spills runs at L2 latency
//!   and the register blocking stops paying for itself.
//! * **L2** is where the *packed* panel lives when the support is too wide
//!   for L1 — it is a staging area, not a working set.
//! * **L3** bounds the whole problem. Past it the kernel is memory-bound and
//!   the tiling is irrelevant; what matters then is the streaming order, and
//!   that is a property of the caller's loop, not of a tile size.
//!
//! # Why the numbers are not simply read off the machine
//!
//! Three corrections, each of which changes the tile size materially:
//!
//! 1. **SMT splits L1.** Two hardware threads on one core share one L1d. A
//!    tiling computed against the full L1 on an SMT machine is computed
//!    against twice the cache the thread will actually get, and it thrashes
//!    exactly when the pool is saturated — i.e. under load. [`threads_per_core`]
//!    divides it out.
//! 2. **Nothing gets 100% of a cache.** The stack, the loop's own scalars and
//!    whatever the allocator touched last are all resident too. [`OCCUPANCY`]
//!    is the fraction a kernel may plan against; the remainder is not slack,
//!    it is the rest of the program.
//! 3. **The build target is not the dev box.** These caches are sized for a
//!    deployment profile ([`CacheBudget::TARGET`]); a developer's laptop or a
//!    shared CI runner will report something smaller. Detection is therefore
//!    the default and the target is the fallback, never the other way round —
//!    a tiling computed for a cache larger than the one present is the failure
//!    mode that has no symptom other than being slow.
//!
//! Resolution order is env → sysfs → [`CacheBudget::TARGET`], and
//! [`CacheBudget::global`] resolves once per process.

use std::sync::OnceLock;

/// Bytes in a coherency line on every CPU this runs on. Not detected as a
/// *constant* — [`CacheBudget::line_bytes`] carries the detected value — but
/// alignment has to be a compile-time constant to be useful to `repr(align)`,
/// and 64 is correct on every x86-64 and every aarch64 in the fleet.
pub const CACHE_LINE: usize = 64;

/// f32 lanes in one 256-bit vector — the width [`crate::pairwise`]'s
/// micro-kernel is written against, and therefore the granularity every packed
/// panel is padded to.
pub const LANES: usize = 8;

/// Fraction of a cache level a kernel may plan its working set against,
/// as `(numerator, denominator)`.
///
/// 3/4 rather than a rounder number: at 7/8 the measured tail-latency on the
/// pairwise kernel became bimodal (the packed panel and the streaming block
/// alias into the same sets often enough to matter on an 8-way L1), and at
/// 1/2 the tile is small enough that loop overhead shows up in the inner
/// dimension. This is the value that leaves a full associativity way spare.
pub const OCCUPANCY: (usize, usize) = (3, 4);

/// A machine's data-cache hierarchy in bytes.
///
/// All figures are **per core** except [`l3_bytes`](Self::l3_bytes), which is
/// shared and is only ever used as a problem-size ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBudget {
    /// Per-core L1 data cache.
    pub l1_bytes: usize,
    /// Per-core L2.
    pub l2_bytes: usize,
    /// Shared last level. `0` when the machine has none to report.
    pub l3_bytes: usize,
    /// Coherency line size.
    pub line_bytes: usize,
    /// Hardware threads sharing one L1d. `1` when SMT is off.
    pub threads_per_core: usize,
}

impl CacheBudget {
    /// The deployment profile: 128 KiB L1d, 512 KiB L2, 32 MiB L3, no SMT.
    ///
    /// Used when nothing better can be discovered. Deliberately the *fallback*
    /// and not the default — see the module docs for why a tiling computed
    /// against a cache the machine does not have fails silently.
    pub const TARGET: Self = Self {
        l1_bytes: 128 << 10,
        l2_bytes: 512 << 10,
        l3_bytes: 32 << 20,
        line_bytes: CACHE_LINE,
        threads_per_core: 1,
    };

    /// Resolve once per process: env, then sysfs, then [`Self::TARGET`].
    pub fn global() -> &'static Self {
        static G: OnceLock<CacheBudget> = OnceLock::new();
        G.get_or_init(|| {
            let mut b = Self::detect();
            b.apply_env_overrides();
            b.sanitise();
            b
        })
    }

    /// Read the hierarchy from the operating system, falling back per-field.
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            if let Some(b) = sysfs::read() {
                return b;
            }
        }
        Self::TARGET
    }

    /// Bytes of L1d one thread may plan against: the level, scaled by
    /// [`OCCUPANCY`] and divided by the threads sharing it.
    pub fn usable_l1(&self) -> usize {
        Self::usable(self.l1_bytes, self.threads_per_core)
    }

    /// Bytes of L2 one thread may plan against.
    ///
    /// L2 is per-core on every machine in the fleet, so it is divided by
    /// `threads_per_core` for the same reason L1 is. If that ever stops being
    /// true the division is the thing to revisit, not the occupancy.
    pub fn usable_l2(&self) -> usize {
        Self::usable(self.l2_bytes, self.threads_per_core)
    }

    /// Bytes of L3 the whole process may plan against. NOT divided by
    /// `threads_per_core` — L3 is shared by the socket, and a kernel that
    /// divided it by the SMT factor would be reserving cache for a sibling
    /// that may not exist.
    pub fn usable_l3(&self) -> usize {
        Self::usable(self.l3_bytes, 1)
    }

    fn usable(level: usize, share: usize) -> usize {
        let (num, den) = OCCUPANCY;
        (level / share.max(1)) * num / den
    }

    fn apply_env_overrides(&mut self) {
        if let Some(v) = env_bytes("DK_CACHE_L1") {
            self.l1_bytes = v;
        }
        if let Some(v) = env_bytes("DK_CACHE_L2") {
            self.l2_bytes = v;
        }
        if let Some(v) = env_bytes("DK_CACHE_L3") {
            self.l3_bytes = v;
        }
        if let Some(v) = env_bytes("DK_CACHE_LINE") {
            self.line_bytes = v;
        }
        if let Ok(s) = std::env::var("DK_CACHE_SMT") {
            if let Ok(v) = s.trim().parse::<usize>() {
                self.threads_per_core = v;
            }
        }
    }

    /// Clamp anything a bad detection or a typo'd env var could have produced.
    ///
    /// A zero L1 would make every tiling degenerate to one row and a
    /// nonsensically large one would make it degenerate to no blocking at all;
    /// both are slow rather than wrong, which is precisely why they need a
    /// guard here instead of an assertion in a kernel.
    fn sanitise(&mut self) {
        // Zero means "nothing reported", not "a 16-byte line". Clamping it
        // like a value would silently give every tile a wrong straddle model.
        if !self.line_bytes.is_power_of_two() || !(16..=256).contains(&self.line_bytes) {
            self.line_bytes = CACHE_LINE;
        }
        self.threads_per_core = self.threads_per_core.clamp(1, 8);
        self.l1_bytes = self.l1_bytes.clamp(8 << 10, 4 << 20);
        self.l2_bytes = self.l2_bytes.clamp(self.l1_bytes, 256 << 20);
        if self.l3_bytes != 0 {
            self.l3_bytes = self.l3_bytes.clamp(self.l2_bytes, 4 << 30);
        }
    }
}

impl Default for CacheBudget {
    fn default() -> Self {
        *Self::global()
    }
}

fn env_bytes(key: &str) -> Option<usize> {
    let raw = std::env::var(key).ok()?;
    parse_size(raw.trim())
}

/// `"512K"`, `"2M"`, `"1048576"` — the forms sysfs and a human both write.
fn parse_size(s: &str) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    let (digits, mult) = match s.as_bytes()[s.len() - 1] {
        b'K' | b'k' => (&s[..s.len() - 1], 1 << 10),
        b'M' | b'm' => (&s[..s.len() - 1], 1 << 20),
        b'G' | b'g' => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    digits.trim().parse::<usize>().ok().map(|n| n * mult)
}

#[cfg(target_os = "linux")]
mod sysfs {
    use super::{parse_size, CacheBudget, CACHE_LINE};

    const ROOT: &str = "/sys/devices/system/cpu/cpu0/cache";

    /// `None` when the directory is absent (containers with a masked sysfs,
    /// some hypervisors) — the caller then uses the target profile rather than
    /// a half-populated budget, because a hierarchy with a real L1 and a
    /// defaulted L2 is worse than a consistent guess.
    pub fn read() -> Option<CacheBudget> {
        let mut b = CacheBudget { l1_bytes: 0, l2_bytes: 0, l3_bytes: 0, line_bytes: 0, threads_per_core: 1 };
        for p in index_dirs()? {
            let field = |n: &str| std::fs::read_to_string(p.join(n)).ok();
            let level: usize = match field("level").and_then(|s| s.trim().parse().ok()) {
                Some(l) => l,
                None => continue,
            };
            let kind = field("type").unwrap_or_default();
            let kind = kind.trim().to_string();
            let size = match field("size").and_then(|s| parse_size(s.trim())) {
                Some(s) => s,
                None => continue,
            };
            if let Some(l) = field("coherency_line_size").and_then(|s| s.trim().parse().ok()) {
                b.line_bytes = l;
            }
            match (level, kind.as_str()) {
                // Instruction cache is not a place data tiles live.
                (1, "Data") | (1, "Unified") => b.l1_bytes = b.l1_bytes.max(size),
                (2, _) => b.l2_bytes = b.l2_bytes.max(size),
                (3, _) | (4, _) => b.l3_bytes = b.l3_bytes.max(size),
                _ => {}
            }
        }
        if b.l1_bytes == 0 || b.l2_bytes == 0 {
            return None;
        }
        if b.line_bytes == 0 {
            b.line_bytes = CACHE_LINE;
        }
        b.threads_per_core = threads_per_core().unwrap_or(1);
        Some(b)
    }

    /// Siblings sharing cpu0's L1d, from the cache's own `shared_cpu_list`.
    ///
    /// Read from the CACHE rather than from `topology/thread_siblings`: what a
    /// tiling needs to know is how many threads contend for this cache, and on
    /// a machine where those two ever disagree the cache's own view is the one
    /// that is true by construction.
    fn threads_per_core() -> Option<usize> {
        for p in index_dirs()? {
            // `continue`, never `?`: the directory also contains `uevent`, and
            // an early return on it made this scan give up before reaching the
            // L1 entry — reporting no SMT on a machine that has it, which is
            // the silent-slowness failure this module's docs warn about.
            let level: usize = match std::fs::read_to_string(p.join("level"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
            {
                Some(l) => l,
                None => continue,
            };
            let kind = std::fs::read_to_string(p.join("type")).unwrap_or_default();
            if level != 1 || kind.trim() == "Instruction" {
                continue;
            }
            if let Some(list) = std::fs::read_to_string(p.join("shared_cpu_list")).ok() {
                return Some(count_cpu_list(list.trim()).max(1));
            }
        }
        None
    }

    /// The `index*` subdirectories, and only those. `read_dir` also yields
    /// `uevent`, and every field read against it fails.
    fn index_dirs() -> Option<Vec<std::path::PathBuf>> {
        let mut out: Vec<_> = std::fs::read_dir(ROOT)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("index"))
            })
            .collect();
        // `read_dir` order is unspecified; sorting makes detection produce the
        // same budget on every run of the same machine.
        out.sort();
        Some(out)
    }

    /// `"0,4"` -> 2, `"0-1"` -> 2, `"0-1,8-9"` -> 4.
    fn count_cpu_list(s: &str) -> usize {
        s.split(',')
            .filter(|p| !p.is_empty())
            .map(|part| match part.split_once('-') {
                Some((a, b)) => {
                    let (a, b): (usize, usize) = (a.trim().parse().unwrap_or(0), b.trim().parse().unwrap_or(0));
                    b.saturating_sub(a) + 1
                }
                None => 1,
            })
            .sum()
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn cpu_lists_parse_in_every_form_sysfs_writes() {
            assert_eq!(super::count_cpu_list("0"), 1);
            assert_eq!(super::count_cpu_list("0,4"), 2);
            assert_eq!(super::count_cpu_list("0-1"), 2);
            assert_eq!(super::count_cpu_list("0-1,8-9"), 4);
        }
    }
}

// ---------------------------------------------------------------------------
// Tilings
// ---------------------------------------------------------------------------

/// Which level the resident (packed) panel was sized to fit.
///
/// Reported rather than inferred because it changes what the figure MEANS: an
/// `L1` tiling is a promise about latency, an `L2` one is a promise only about
/// bandwidth, and a `Streaming` one is neither — it is the kernel saying the
/// problem does not fit and the loop order is now the only thing that matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    L1,
    L2,
    Streaming,
}

/// How to walk an `[M, D] x [N, D] -> [M, N]` contraction.
///
/// `N` is the **support** axis (atoms, inducing points, mixture components) —
/// small, reused by every row, and therefore the thing to hold resident. `M`
/// is the **streaming** axis (corpus points, batch rows) — large, touched
/// once. That asymmetry is the whole design: pack the support once, stream the
/// corpus past it, and never materialise the `[M, N, D]` intermediate that a
/// broadcast formulation of the same contraction builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairTiling {
    /// Streaming rows per pass.
    pub mc: usize,
    /// Support columns held resident per pass.
    pub nc: usize,
    /// Padded support stride — `nc` rounded up to [`LANES`]. Output tiles are
    /// this wide, so the micro-kernel never needs a masked tail.
    pub nc_padded: usize,
    /// Where the packed support panel was sized to live.
    pub residency: Residency,
}

impl PairTiling {
    /// Bytes the working set occupies: packed support + streaming block +
    /// output tile.
    pub fn working_bytes(&self, d: usize, elem: usize) -> usize {
        self.nc_padded * d * elem + self.mc * d * elem + self.mc * self.nc_padded * elem
    }
}

/// Micro-kernel row count — four independent accumulator chains, which is what
/// hides a 4-cycle FMA latency without spilling the 8 vector accumulators a
/// 4x8 tile needs.
pub const MR: usize = 4;

impl CacheBudget {
    /// Tile an `[m, d] x [n, d] -> [m, n]` contraction over `elem`-byte
    /// elements.
    ///
    /// The support panel is placed first, because it is the operand every row
    /// re-reads; whatever L1 is left after it goes to the streaming block and
    /// its output tile. When the support alone will not fit L1 the support is
    /// itself split into `nc` chunks that do, and the caller loops — which is
    /// why `nc` may come back smaller than `n`.
    pub fn pair_tiling(&self, m: usize, n: usize, d: usize, elem: usize) -> PairTiling {
        let l1 = self.usable_l1();
        let row = d * elem; // one packed support column, or one streamed row

        // Half of L1 is the ceiling for the resident panel: the streaming
        // block and its output tile need somewhere to live too, and a support
        // panel allowed to fill L1 leaves the streaming side with nothing but
        // evictions.
        let panel_ceiling = (l1 / 2).max(row);
        let mut nc = round_up(n.max(1), LANES);
        let mut residency = Residency::L1;
        if nc * row > panel_ceiling {
            let fit = round_down(panel_ceiling / row.max(1), LANES).max(LANES);
            nc = fit.min(round_up(n.max(1), LANES));
            residency = if nc * row <= self.usable_l2() { Residency::L2 } else { Residency::Streaming };
        }
        let nc_padded = nc;
        let nc = nc.min(n.max(1));

        // Whatever L1 the panel did not take, split between the streamed rows
        // and the output tile they produce.
        let panel_bytes = nc_padded * row;
        let spare = l1.saturating_sub(panel_bytes).max(row + nc_padded * elem);
        let per_row = row + nc_padded * elem;
        // Rounded DOWN, then floored at one micro-kernel tile. Rounding up is
        // the obvious thing and it is wrong: it overshoots the budget the
        // whole calculation exists to respect, by up to `MR` rows, which on a
        // narrow feature width is several kilobytes past L1.
        let mut mc = round_down(spare / per_row, MR).max(MR);
        mc = mc.min(round_up(m.max(1), MR));

        PairTiling { mc, nc, nc_padded, residency }
    }

    /// Rows of a `[rows, cols]` row-major buffer that fit L1 together — the
    /// tiling for a pure row-wise pass (a softmax, a normalisation, a
    /// reduction) where there is no second operand to hold resident.
    pub fn row_block(&self, rows: usize, cols: usize, elem: usize) -> usize {
        let per_row = (cols * elem).max(1);
        round_up((self.usable_l1() / per_row).max(1), MR).min(round_up(rows.max(1), MR))
    }
}

/// Round `v` up to the next multiple of `q`. `q` is a power of two at every
/// call site here, but the general form costs nothing and does not silently
/// misbehave if that stops being true.
pub const fn round_up(v: usize, q: usize) -> usize {
    if q == 0 {
        return v;
    }
    v.div_ceil(q) * q
}

/// Round `v` down to a multiple of `q`.
pub const fn round_down(v: usize, q: usize) -> usize {
    if q == 0 {
        return v;
    }
    (v / q) * q
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(l1: usize, l2: usize, smt: usize) -> CacheBudget {
        CacheBudget { l1_bytes: l1, l2_bytes: l2, l3_bytes: 32 << 20, line_bytes: 64, threads_per_core: smt }
    }

    #[test]
    fn sizes_parse_in_the_forms_sysfs_and_humans_write() {
        assert_eq!(parse_size("512K"), Some(512 << 10));
        assert_eq!(parse_size("2M"), Some(2 << 20));
        assert_eq!(parse_size("1048576"), Some(1048576));
        assert_eq!(parse_size("1G"), Some(1 << 30));
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("banana"), None);
    }

    /// The correction that has no symptom other than being slow: two threads
    /// on one core get half the cache each, so the tile must be smaller.
    #[test]
    fn smt_halves_the_budget_a_thread_may_plan_against() {
        let solo = budget(128 << 10, 512 << 10, 1);
        let smt = budget(128 << 10, 512 << 10, 2);
        assert_eq!(smt.usable_l1() * 2, solo.usable_l1());

        let a = solo.pair_tiling(4096, 64, 16, 4);
        let b = smt.pair_tiling(4096, 64, 16, 4);
        assert!(b.mc < a.mc, "an SMT tiling was not smaller: {} vs {}", b.mc, a.mc);
    }

    /// L3 is shared by the socket, so reserving half of it for a sibling
    /// thread would be reserving cache for a contender that may not exist.
    #[test]
    fn l3_is_not_divided_by_the_smt_factor() {
        let smt = budget(128 << 10, 512 << 10, 2);
        assert_eq!(smt.usable_l3(), CacheBudget::usable(32 << 20, 1));
    }

    /// The governing invariant of the whole module. Everything else here is
    /// in service of this one property.
    #[test]
    fn every_tiling_working_set_fits_the_level_it_claims() {
        let b = budget(128 << 10, 512 << 10, 1);
        for &(m, n, d) in &[
            (1usize, 1usize, 1usize),
            (4096, 30, 16),
            (100_000, 30, 16),
            (512, 512, 16),
            (64, 64, 4096),
            (7, 13, 3),
        ] {
            let t = b.pair_tiling(m, n, d, 4);
            assert!(t.mc >= 1 && t.nc >= 1, "degenerate tiling for {m}x{n}x{d}: {t:?}");
            assert!(t.nc <= n.max(1), "nc exceeded the support for {m}x{n}x{d}");
            assert!(t.nc_padded % LANES == 0, "padded stride is not a vector multiple");
            assert!(t.nc_padded >= t.nc);
            assert!(t.mc % MR == 0, "mc is not a micro-kernel multiple: {}", t.mc);

            let ws = t.working_bytes(d, 4);
            let ceiling = match t.residency {
                Residency::L1 => b.usable_l1(),
                Residency::L2 => b.usable_l2(),
                // A streaming tiling has already declared it does not fit.
                Residency::Streaming => usize::MAX,
            };
            // The one exception: a single micro-kernel row against a single
            // padded support tile is the floor, and no cache is smaller.
            let floor = MR * d * 4 + LANES * d * 4 + MR * LANES * 4;
            assert!(
                ws <= ceiling || ws <= floor,
                "{m}x{n}x{d} claimed {:?} but its working set is {ws} against {ceiling}",
                t.residency
            );
        }
    }

    /// A support panel that already fits stays whole — the common case in this
    /// tree, where K is 30 atoms and D is 16.
    #[test]
    fn a_small_support_is_held_entire_in_l1() {
        let b = budget(128 << 10, 512 << 10, 1);
        let t = b.pair_tiling(100_000, 30, 16, 4);
        assert_eq!(t.residency, Residency::L1);
        assert_eq!(t.nc, 30);
        assert_eq!(t.nc_padded, 32, "30 atoms must pad to a whole number of vectors");
        assert!(t.mc >= 64, "the streaming block collapsed to {} rows", t.mc);
    }

    /// A support too wide for L1 is split rather than allowed to evict the
    /// stream it is being read against.
    #[test]
    fn a_wide_support_is_split_and_says_so() {
        let b = budget(32 << 10, 512 << 10, 1);
        let t = b.pair_tiling(1024, 4096, 64, 4);
        assert!(t.nc < 4096, "the support was not split: nc = {}", t.nc);
        assert_ne!(t.residency, Residency::L1);
    }

    #[test]
    fn a_bad_budget_is_clamped_rather_than_producing_a_degenerate_tiling() {
        let mut b = CacheBudget { l1_bytes: 0, l2_bytes: 0, l3_bytes: 0, line_bytes: 0, threads_per_core: 0 };
        b.sanitise();
        assert!(b.l1_bytes >= 8 << 10);
        assert!(b.l2_bytes >= b.l1_bytes);
        assert_eq!(b.threads_per_core, 1);
        assert_eq!(b.line_bytes, 64);
        let t = b.pair_tiling(1024, 32, 16, 4);
        assert!(t.mc >= 1 && t.nc >= 1);
    }

    #[test]
    fn rounding_helpers_agree_with_their_names() {
        assert_eq!(round_up(30, 8), 32);
        assert_eq!(round_up(32, 8), 32);
        assert_eq!(round_up(0, 8), 0);
        assert_eq!(round_down(30, 8), 24);
        assert_eq!(round_down(32, 8), 32);
    }

    /// Detection must produce something a kernel can divide by on whatever
    /// machine the suite runs on, sysfs present or not.
    #[test]
    fn the_global_budget_is_usable_wherever_the_suite_runs() {
        let b = CacheBudget::global();
        assert!(b.usable_l1() >= 4 << 10, "usable L1 came back at {}", b.usable_l1());
        assert!(b.usable_l2() >= b.usable_l1());
        assert!(b.line_bytes.is_power_of_two());
    }
}
