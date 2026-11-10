//! Measure the blocked pairwise kernels against the broadcast formulation they
//! replace, on the shapes this tree actually runs.
//!
//! `cargo run --release -p dk-tensor --example pairwise_bench`
//!
//! Debug builds measure nothing useful here — the whole claim is about what
//! LLVM does with the micro-kernel — so the example says so and carries on.

use std::time::Instant;

use dk_tensor::cache_model::CacheBudget;
use dk_tensor::pairwise::{responsibilities_into, sqdist_dense, sqdist_into, Accum, ExpMode};
use dk_tensor::panel::{PackedSupport, Panel};

/// The formulation in the tree today: broadcast to `[m, n, d]`, square, sum.
fn broadcast_sqdist(a: &[f32], m: usize, b: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut wide = vec![0.0f32; m * n * d];
    for i in 0..m {
        for j in 0..n {
            for c in 0..d {
                wide[(i * n + j) * d + c] = a[i * d + c] - b[j * d + c];
            }
        }
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m * n {
        let mut acc = 0.0f32;
        for c in 0..d {
            let t = wide[i * d + c];
            acc += t * t;
        }
        out[i] = acc;
    }
    out
}

fn pts(n: usize, d: usize, seed: f32) -> Vec<f32> {
    (0..n * d).map(|i| ((i as f32 + seed) * 0.6180339).sin() * 3.0).collect()
}

fn time<T>(iters: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f(); // warm the caches and the branch predictors
    let t0 = Instant::now();
    for _ in 0..iters {
        last = f();
    }
    (t0.elapsed().as_secs_f64() * 1e3 / iters as f64, last)
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("warning: debug build — these figures measure the optimiser being off\n");
    }
    let b = CacheBudget::global();
    println!(
        "cache: L1 {} KiB  L2 {} KiB  L3 {} MiB  line {}B  smt {}",
        b.l1_bytes >> 10,
        b.l2_bytes >> 10,
        b.l3_bytes >> 20,
        b.line_bytes,
        b.threads_per_core
    );
    println!("usable per thread: L1 {} KiB  L2 {} KiB\n", b.usable_l1() >> 10, b.usable_l2() >> 10);

    println!("{:<26} {:>10} {:>10} {:>10} {:>8} {:>8}", "shape (m x n x d)", "bcast ms", "exact ms", "fused ms", "x", "x(f)");
    for &(m, n, d, iters) in &[
        (4096usize, 30usize, 16usize, 20usize), // corpus -> production codebook
        (16384, 30, 16, 8),                     // a long conversation
        (30, 30, 16, 400),                      // a codebook's own ground cost
        (2048, 512, 64, 4),                     // a GP inducing-point block
        (512, 512, 16, 20),                     // square, small feature width
    ] {
        let a = pts(m, d, 0.0);
        let s = pts(n, d, 11.0);

        let (t_b, want) = time(iters, || broadcast_sqdist(&a, m, &s, n, d));
        let (t_e, got) = time(iters, || sqdist_dense(&a, m, &s, n, d, Accum::Exact).unwrap());
        let (t_f, _) = time(iters, || sqdist_dense(&a, m, &s, n, d, Accum::Fused).unwrap());

        assert_eq!(got, want, "the exact kernel disagreed with the broadcast form at {m}x{n}x{d}");
        println!(
            "{:<26} {:>10.3} {:>10.3} {:>10.3} {:>7.1}x {:>7.1}x",
            format!("{m} x {n} x {d}"),
            t_b,
            t_e,
            t_f,
            t_b / t_e,
            t_b / t_f
        );
    }

    // The full responsibilities path. Reported separately from the distance
    // because the transcendental, not the contraction, turns out to be where
    // the time goes — which is the whole reason `ExpMode` exists.
    println!("\nresponsibilities (support packed once — the cached path):");
    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "shape (m x n x d)", "dist ms", "libm ms", "vec ms", "x", "max rel"
    );
    for &(m, n, d, iters) in &[(4096usize, 30usize, 16usize, 30usize), (16384, 30, 16, 10)] {
        let a = pts(m, d, 0.0);
        let s = pts(n, d, 11.0);
        let panel = Panel::from_rows(&a, m, d).unwrap();
        let packed = PackedSupport::pack(&s, n, d).unwrap();
        let stride = packed.stride();
        let mut out = vec![0.0f32; m * stride];

        let (t_dist, _) = time(iters, || {
            sqdist_into(&panel, &packed, &mut out, stride, Accum::Exact).unwrap()
        });
        let (t_libm, _) = time(iters, || {
            responsibilities_into(&panel, &packed, 4.0, &mut out, stride, Accum::Exact, ExpMode::Libm)
                .unwrap()
        });
        let exact = out.clone();
        let (t_vec, _) = time(iters, || {
            responsibilities_into(&panel, &packed, 4.0, &mut out, stride, Accum::Fused, ExpMode::Vector)
                .unwrap()
        });

        let mut max_rel = 0.0f32;
        for (x, y) in exact.iter().zip(&out) {
            if *x > 1e-6 {
                max_rel = max_rel.max((x - y).abs() / x);
            }
        }
        println!(
            "{:<26} {:>10.3} {:>10.3} {:>10.3} {:>7.1}x {:>10.2e}",
            format!("{m} x {n} x {d}"),
            t_dist,
            t_libm,
            t_vec,
            t_libm / t_vec,
            max_rel
        );
    }

    // Packing is a fit-time cost the cache is supposed to make disappear.
    // Reported so the claim is a number rather than an assertion.
    let (t_pack, _) = time(2000, || PackedSupport::pack(&pts(30, 16, 11.0), 30, 16).unwrap());
    let (t_pack_big, _) = time(50, || PackedSupport::pack(&pts(4096, 64, 11.0), 4096, 64).unwrap());
    println!("\npacking the support: 30x16 {t_pack:.4} ms   4096x64 {t_pack_big:.4} ms (once per fit)");
}
