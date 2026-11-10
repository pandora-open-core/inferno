//! Inferno — Temporal convolution layers for sequence feature extraction.

use candle_core::{Result, Tensor, D};
use candle_nn::ConvTranspose1dConfig;

/// `x: [B, C_in, L]`, `w: [C_in, C_out, K]` → `[B, C_out, L_out]`.
///
/// Implements transposed convolution with differentiable core operations.
/// Checked against an independent scatter reference, rather than Candle's
/// native CPU transposed-convolution fast path, which can disagree numerically.
pub fn conv_transpose1d_diff(
    x: &Tensor,
    w: &Tensor,
    cfg: ConvTranspose1dConfig,
) -> Result<Tensor> {
    let (_b, c_in, l_in) = x.dims3()?;
    let (c_in_k, _c_out, k) = w.dims3()?;
    let ConvTranspose1dConfig { padding: p, output_padding: op, stride: s, dilation: d, groups } = cfg;

    if c_in != c_in_k {
        candle_core::bail!(
            "conv_transpose1d_diff: in-channel mismatch, input has {c_in} but kernel has {c_in_k}"
        );
    }
    // Grouped transposed convolution needs a per-group slice-and-concat. The
    // model uses groups=1 everywhere; refusing is better than quietly
    // computing the ungrouped result under a grouped config.
    if groups != 1 {
        candle_core::bail!(
            "conv_transpose1d_diff: groups={groups} is unsupported (only 1); \
             grouped transposed convolution needs a per-group loop"
        );
    }
    // p' = d(k−1) − p must be non-negative, or the identity does not hold and
    // conv1d would reject the padding anyway — but several frames later, with
    // a message that says nothing about why.
    let dk = d * k.saturating_sub(1);
    if dk < p {
        candle_core::bail!(
            "conv_transpose1d_diff: padding {p} exceeds dilation·(k−1) = {dk}; \
             the strided-up-convolution identity requires d·(k−1) ≥ padding"
        );
    }
    let p_eff = dk - p;

    // -- 1. Zero-insert `s−1` zeros between input samples ------------------
    //
    // [B, C, L] → [B, C, (L−1)·s + 1]. Built with cat/reshape/narrow so the
    // whole thing stays on the tape.
    let up = if s <= 1 {
        x.clone()
    } else {
        let zeros = Tensor::zeros((_b, c_in, l_in, s - 1), x.dtype(), x.device())?;
        Tensor::cat(&[&x.unsqueeze(D::Minus1)?, &zeros], D::Minus1)?
            .reshape((_b, c_in, l_in * s))?
            // The trailing s−1 zeros after the LAST real sample are not part
            // of the upsampled signal; keeping them would lengthen the output.
            .narrow(D::Minus1, 0, (l_in - 1) * s + 1)?
    };

    // -- 2. output_padding: extra zeros on the right of the upsampled input -
    let up = if op == 0 {
        up
    } else {
        let tail = Tensor::zeros((_b, c_in, op), up.dtype(), up.device())?;
        Tensor::cat(&[&up, &tail], D::Minus1)?
    };

    // -- 3. Flip the taps and swap the channel axes ------------------------
    let rev: Vec<u32> = (0..k as u32).rev().collect();
    let rev = Tensor::from_vec(rev, k, w.device())?;
    let w_flipped = w.index_select(&rev, D::Minus1)?;
    let kernel = w_flipped.transpose(0, 1)?.contiguous()?; // [C_out, C_in, K]

    // -- 4. The convolution ------------------------------------------------
    up.contiguous()?.conv1d(&kernel, p_eff, 1, d, 1)
}

/// Drop-in replacement for `<ConvTranspose1d as Module>::forward`.
///
/// Takes the layer rather than raw tensors so the module keeps owning its
/// parameters — the checkpoint names (`deconv.0.weight`, …) are unchanged, and
/// switching a call site over is a one-word edit.
pub fn forward(layer: &candle_nn::ConvTranspose1d, x: &Tensor) -> Result<Tensor> {
    let y = conv_transpose1d_diff(x, layer.weight(), *layer.config())?;
    match layer.bias() {
        None => Ok(y),
        Some(bias) => {
            let b = bias.dims1()?;
            y.broadcast_add(&bias.reshape((1, b, 1))?)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Verification of temporal convolution layers for sequence feature extraction.

    use super::*;
    use candle_core::{DType, Device};

    fn cfg(p: usize, op: usize, s: usize, d: usize) -> ConvTranspose1dConfig {
        ConvTranspose1dConfig { padding: p, output_padding: op, stride: s, dilation: d, groups: 1 }
    }

    /// The whole correctness argument in one test: the composite must agree
    /// with candle's own forward everywhere the identity claims to hold.
    ///
    /// A grid, not one case — `p'`, the flip and the `output_padding`
    /// placement are each derived from `(p, s, d, op)`, so a single
    /// configuration could pass with two of the three wrong.
    #[test]
    fn it_matches_direct_scatter_across_the_config_grid() {
        let dev = Device::Cpu;
        for (b, c_in, c_out, l, k) in [(2, 3, 4, 7, 4), (1, 1, 1, 5, 2), (3, 2, 5, 9, 3)] {
            let x = Tensor::randn(0f32, 1f32, (b, c_in, l), &dev).unwrap();
            let w = Tensor::randn(0f32, 1f32, (c_in, c_out, k), &dev).unwrap();
            for s in [1usize, 2, 3] {
                for d in [1usize, 2] {
                    for p in 0..=(d * (k - 1)) {
                        // Only positive-length output configurations are valid.
                        if (l - 1) * s + d * (k - 1) + 1 <= 2 * p {
                            continue;
                        }
                        // output_padding must be < max(stride, dilation) —
                        // candle's own constraint on the op.
                        for op in 0..s.max(d).min(3) {
                            let c = cfg(p, op, s, d);
                            // Independent definition: each input sample scatters
                            // its weighted kernel taps into the output. Candle's
                            // native unpadded CPU fast path is not a reliable
                            // numerical oracle at the pinned PR revision.
                            let out_len = (l - 1) * s + d * (k - 1) + op + 1 - 2 * p;
                            let xv = x.to_vec3::<f32>().unwrap();
                            let wv = w.to_vec3::<f32>().unwrap();
                            let mut reference = vec![0f32; b * c_out * out_len];
                            for bi in 0..b {
                                for ci in 0..c_in {
                                    for li in 0..l {
                                        for co in 0..c_out {
                                            for tap in 0..k {
                                                let position = li * s + tap * d;
                                                if position >= p && position - p < out_len {
                                                    reference[(bi * c_out + co) * out_len + position - p] += xv[bi][ci][li] * wv[ci][co][tap];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let want = Tensor::from_vec(reference, (b, c_out, out_len), &dev).unwrap();
                            let got = conv_transpose1d_diff(&x, &w, c).unwrap();
                            assert_eq!(
                                got.dims(),
                                want.dims(),
                                "shape mismatch at p={p} op={op} s={s} d={d} k={k}"
                            );
                            let err = got
                                .sub(&want)
                                .unwrap()
                                .abs()
                                .unwrap()
                                .max_all()
                                .unwrap()
                                .to_scalar::<f32>()
                                .unwrap();
                            assert!(
                                err < 1e-4,
                                "value mismatch {err} at p={p} op={op} s={s} d={d} k={k}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The point of the exercise. candle's own op errors here.
    #[test]
    fn it_differentiates_where_candles_own_op_refuses() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let w = vb
            .get_with_hints((3, 4, 4), "w", candle_nn::Init::Randn { mean: 0.0, stdev: 0.5 })
            .unwrap();
        let x = Tensor::randn(0f32, 1f32, (2, 3, 6), &dev).unwrap();

        // Baseline: candle's op refuses, which is the bug this file exists for.
        let native = x.conv_transpose1d(&w, 1, 0, 2, 1, 1).unwrap();
        let e = native.sqr().unwrap().mean_all().unwrap().backward().unwrap_err();
        assert!(
            format!("{e}").contains("conv-transpose1d"),
            "expected the BackwardNotSupported this file works around, got: {e}"
        );

        // The composite differentiates, and the gradient is real.
        let out = conv_transpose1d_diff(&x, &w, cfg(1, 0, 2, 1)).unwrap();
        let grads = out.sqr().unwrap().mean_all().unwrap().backward().unwrap();
        let g = grads.get(&w).expect("no gradient reached the kernel");
        assert_eq!(g.dims(), w.dims());
        let n = g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(n.is_finite() && n > 0.0, "kernel gradient is {n}");
    }

    /// The gradient is candle's, so it should be right — but "should" is not a
    /// test. Central finite differences in F64; f32 central FD is hopeless at
    /// this step size.
    ///
    /// This also guards the PR #4044 dependency pin: convolution backward
    /// creates noncontiguous kernel views, which must be materialised correctly.
    #[test]
    fn the_kernel_gradient_matches_central_finite_differences() {
        let dev = Device::Cpu;
        let (b, c_in, c_out, l, k) = (2usize, 2, 3, 5, 3);
        let x = Tensor::randn(0f64, 1f64, (b, c_in, l), &dev)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let w0 = Tensor::randn(0f64, 1f64, (c_in, c_out, k), &dev)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let c = cfg(1, 0, 2, 1);

        let loss_of = |w: &Tensor| -> f64 {
            conv_transpose1d_diff(&x, w, c)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };

        let wv = candle_core::Var::from_tensor(&w0).unwrap();
        let analytic = conv_transpose1d_diff(&x, &wv, c)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .backward()
            .unwrap();
        let g = analytic.get(&wv).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let h = 1e-4;
        let flat = w0.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        // A handful of coordinates: each FD probe is two full forwards.
        for &i in &[0usize, 5, 11, flat.len() - 1] {
            let bump = |delta: f64| {
                let mut v = flat.clone();
                v[i] += delta;
                Tensor::from_vec(v, (c_in, c_out, k), &dev).unwrap()
            };
            let fd = (loss_of(&bump(h)) - loss_of(&bump(-h))) / (2.0 * h);
            let rel = (fd - g[i]).abs() / fd.abs().max(1e-8);
            assert!(rel < 1e-5, "coord {i}: analytic {} vs fd {fd} (rel {rel})", g[i]);
        }
    }

    /// Guards the vendor patch directly, on plain `conv1d` — no code of ours
    /// involved.
    ///
    /// Two independent statements, because the second follows from the first
    /// and naming both is what makes a future failure diagnosable:
    ///
    /// 1. `conv1d` agrees with itself whether or not the KERNEL is contiguous.
    ///    Stock candle fails this by O(1) — a transposed view and its
    ///    contiguous twin give different answers.
    /// 2. Therefore the weight gradient is right, at the shape
    ///    `ConvolutionalLoopEncoder`'s stem actually uses.
    #[test]
    fn conv1d_is_correct_for_a_non_contiguous_kernel() {
        let dev = Device::Cpu;

        // (1) the forward, which is the actual bug.
        let x = Tensor::randn(0f64, 1f64, (2, 3, 8), &dev).unwrap();
        let kk = Tensor::randn(0f64, 1f64, (3, 4, 3), &dev).unwrap();
        let kt = kk.transpose(0, 1).unwrap();
        assert!(!kt.is_contiguous(), "the fixture must actually be a view");
        let diff = x
            .conv1d(&kt, 1, 1, 1, 1)
            .unwrap()
            .sub(&x.conv1d(&kt.contiguous().unwrap(), 1, 1, 1, 1).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        assert!(
            diff < 1e-12,
            "conv1d disagrees with itself by {diff} depending on kernel contiguity — \
             the vendor/candle/candle-core patch is not applied"
        );

        // (2) the consequence: a correct weight gradient at the encoder's
        // stem shape, Conv1d(in→32, k=7, s=2, p=3).
        let (b, ci, co, l, k, p, s) = (2usize, 6, 8, 16, 7, 3, 2);
        let x = Tensor::randn(0f64, 1f64, (b, ci, l), &dev).unwrap();
        let w0 = Tensor::randn(0f64, 1f64, (co, ci, k), &dev).unwrap();
        let loss_of = |w: &Tensor| -> f64 {
            x.conv1d(w, p, s, 1, 1)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };
        let v = candle_core::Var::from_tensor(&w0).unwrap();
        let g = x
            .conv1d(v.as_tensor(), p, s, 1, 1)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .backward()
            .unwrap();
        let g = g.get(&v).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();

        let flat = w0.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let h = 1e-6;
        for &i in &[0usize, 17, 100, flat.len() - 1] {
            let bump = |d: f64| {
                let mut vv = flat.clone();
                vv[i] += d;
                Tensor::from_vec(vv, (co, ci, k), &dev).unwrap()
            };
            let fd = (loss_of(&bump(h)) - loss_of(&bump(-h))) / (2.0 * h);
            let rel = (fd - g[i]).abs() / fd.abs().max(1e-8);
            assert!(rel < 1e-5, "conv1d weight grad coord {i}: {} vs fd {fd}", g[i]);
        }
    }

    #[test]
    fn unsupported_configs_are_refused_rather_than_miscomputed() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 2, 4), &dev).unwrap();
        let w = Tensor::randn(0f32, 1f32, (2, 2, 3), &dev).unwrap();

        let grouped = conv_transpose1d_diff(
            &x,
            &w,
            ConvTranspose1dConfig { padding: 0, output_padding: 0, stride: 1, dilation: 1, groups: 2 },
        );
        assert!(format!("{:?}", grouped.unwrap_err()).contains("groups"));

        // d(k−1) = 2, so padding 3 breaks the identity.
        let over_padded = conv_transpose1d_diff(&x, &w, cfg(3, 0, 1, 1));
        assert!(format!("{:?}", over_padded.unwrap_err()).contains("padding"));

        // Mismatched channels must not silently broadcast.
        let bad = Tensor::randn(0f32, 1f32, (5, 2, 3), &dev).unwrap();
        assert!(conv_transpose1d_diff(&x, &bad, cfg(0, 0, 1, 1)).is_err());
    }

    // =====================================================================
    // Conv1d backward parity — guards the SECOND vendor patch
    // =====================================================================

    /// One `Conv1d` geometry the repo actually instantiates.
    ///
    /// Channel counts are shrunk for test speed; `(l_in, k, stride, padding,
    /// groups)` is exact, because those five are the only things the backward
    /// arithmetic depends on. `l_in` is the length the layer really sees,
    /// traced forward from `seq_len = 32` through the encoder's strides.
    struct ConvSite {
        name: &'static str,
        c_in: usize,
        c_out: usize,
        l_in: usize,
        k: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    }

    /// Every first-party `Conv1d` config, in the order the data meets them.
    ///
    /// `stage3.conv2` is the one that panicked; `stage3.conv1` sits exactly on
    /// the boundary — `(l_out − 1)·s == 2p` — so one more strided stage, or a
    /// shorter input, and it lands on the same defect.
    const CONV1D_SITES: &[ConvSite] = &[
        // pandorapurple::model::encoder — stem + three residual stages.
        ConvSite { name: "encoder.stem",          c_in: 6,  c_out: 8, l_in: 32, k: 7, stride: 2, padding: 3, groups: 1 },
        ConvSite { name: "encoder.stage1.conv1",  c_in: 8,  c_out: 8, l_in: 16, k: 5, stride: 2, padding: 2, groups: 1 },
        ConvSite { name: "encoder.stage1.conv2",  c_in: 8,  c_out: 8, l_in: 8,  k: 5, stride: 1, padding: 2, groups: 1 },
        ConvSite { name: "encoder.stage1.short",  c_in: 8,  c_out: 8, l_in: 16, k: 1, stride: 2, padding: 0, groups: 1 },
        ConvSite { name: "encoder.stage2.conv1",  c_in: 8,  c_out: 8, l_in: 8,  k: 3, stride: 2, padding: 1, groups: 1 },
        ConvSite { name: "encoder.stage2.conv2",  c_in: 8,  c_out: 8, l_in: 4,  k: 3, stride: 1, padding: 1, groups: 1 },
        ConvSite { name: "encoder.stage3.conv1",  c_in: 8,  c_out: 8, l_in: 4,  k: 3, stride: 2, padding: 1, groups: 1 },
        // ↓ the panic: (2 − 1)·1 = 1 < 2·1.
        ConvSite { name: "encoder.stage3.conv2",  c_in: 8,  c_out: 8, l_in: 2,  k: 3, stride: 1, padding: 1, groups: 1 },
        // pandorapurple::model::mamba — depthwise, one group per channel.
        ConvSite { name: "mamba.conv1d",          c_in: 8,  c_out: 8, l_in: 16, k: 4, stride: 1, padding: 0, groups: 8 },
        // pandorapurple::model::row_head — depthwise state kernel.
        ConvSite { name: "row_head.row_conv",     c_in: 8,  c_out: 8, l_in: 12, k: 4, stride: 1, padding: 0, groups: 8 },
        // crates/dk-claude-suggests::encoder — every stage is stride 1, so the
        // length never shrinks and `Dims::t` (default 12) IS the l_in at all
        // four convs. That also makes it the one site whose safety is a
        // caller's choice: `t` of 1 or 2 puts it on the underflow below.
        ConvSite { name: "suggests.encoder.conv", c_in: 8,  c_out: 8, l_in: 12, k: 3, stride: 1, padding: 1, groups: 1 },
    ];

    /// Central-difference parity for BOTH gradients of one site.
    ///
    /// Both are checked because they fail independently and for different
    /// reasons: `dL/dinput` runs through `conv_transpose1d` (where the
    /// `l_out` underflow lives), `dL/dkernel` through a transposed — i.e.
    /// non-contiguous — `conv1d` (where the contiguity bug lives). A test of
    /// only one would have passed throughout both defects.
    fn assert_conv1d_grad_parity(site: &ConvSite) {
        let dev = Device::Cpu;
        let (b, ci, co, l, k, s, p, g) =
            (2usize, site.c_in, site.c_out, site.l_in, site.k, site.stride, site.padding, site.groups);
        let name = site.name;

        let x0 = Tensor::randn(0f64, 1f64, (b, ci, l), &dev).unwrap();
        let w0 = Tensor::randn(0f64, 1f64, (co, ci / g, k), &dev).unwrap();

        let loss = |x: &Tensor, w: &Tensor| -> f64 {
            x.conv1d(w, p, s, 1, g)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
        };

        let vx = candle_core::Var::from_tensor(&x0).unwrap();
        let vw = candle_core::Var::from_tensor(&w0).unwrap();
        let grads = vx
            .as_tensor()
            .conv1d(vw.as_tensor(), p, s, 1, g)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .backward()
            .unwrap();

        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let gx = flat(&grads.get(&vx).unwrap_or_else(|| panic!("{name}: no input gradient")));
        let gw = flat(&grads.get(&vw).unwrap_or_else(|| panic!("{name}: no kernel gradient")));

        // A gradient of exactly zero everywhere would satisfy any relative
        // test vacuously — the failure mode of a silently-skipped arm.
        assert!(gx.iter().any(|v| v.abs() > 1e-9), "{name}: input gradient is all zero");
        assert!(gw.iter().any(|v| v.abs() > 1e-9), "{name}: kernel gradient is all zero");

        // `mean(conv(x, w)^2)` is EXACTLY quadratic in x for fixed w, and in w
        // for fixed x — a convolution is linear in each argument separately.
        // So the central difference carries no truncation error at all (the
        // leading term is f''' = 0) and h is free to be large, which is what
        // suppresses the cancellation roundoff that otherwise dominates at
        // small-magnitude coordinates. At h = 1e-6 this test failed on a
        // coordinate whose true gradient was 2e-5, purely on f64 subtraction
        // noise; at 1e-3 the same coordinate agrees to ~1e-8.
        let h = 1e-3;
        let mut check = |what: &str, analytic: &[f64], base: &[f64], dims: (usize, usize, usize)| {
            let n = base.len();
            for &i in &[0usize, n / 3, 2 * n / 3, n - 1] {
                let bump = |d: f64| {
                    let mut v = base.to_vec();
                    v[i] += d;
                    Tensor::from_vec(v, dims, &dev).unwrap()
                };
                let fd = if what == "d/dx" {
                    (loss(&bump(h), &w0) - loss(&bump(-h), &w0)) / (2.0 * h)
                } else {
                    (loss(&x0, &bump(h)) - loss(&x0, &bump(-h))) / (2.0 * h)
                };
                let rel = (fd - analytic[i]).abs() / fd.abs().max(1e-8);
                assert!(
                    rel < 1e-7,
                    "{name} {what} coord {i}: analytic {} vs finite-difference {fd} (rel {rel})",
                    analytic[i],
                );
            }
        };

        check("d/dx", &gx, &flat(&x0), (b, ci, l));
        check("d/dw", &gw, &flat(&w0), (co, ci / g, k));
    }

    /// Gradient parity at every `Conv1d` geometry the repo instantiates.
    ///
    /// Table-driven rather than one test per site so that adding a layer means
    /// adding a row — a new config that trips either vendor bug then fails
    /// here instead of at training time, which is where `stage3.conv2` was
    /// found.
    #[test]
    fn every_conv1d_site_has_a_correct_gradient() {
        for site in CONV1D_SITES {
            assert_conv1d_grad_parity(site);
        }
    }

    /// Guards the second vendor patch directly, and names its trigger.
    ///
    /// `Op::Conv1D`'s backward arm computes `dL/dinput` as a transposed
    /// convolution over the upstream gradient, so a forward `Conv1d(k, s, p)`
    /// backprops as `conv_transpose1d(l_in = l_out, stride = s, padding = p)`.
    /// Upstream's `ParamsConvTranspose1D::l_out` evaluates
    /// `(l_in − 1)·stride − 2·padding + …` left to right in `usize`, so it
    /// underflows whenever `(l_in − 1)·stride < 2·padding` — despite the total
    /// being non-negative and correct.
    ///
    /// `ConvolutionalLoopEncoder`'s `stage3.conv2` is exactly that: `k=3, s=1,
    /// p=1` on a sequence already strided down to length 2, so `1 < 2` and the
    /// whole backward pass panicked rather than returning a wrong number.
    #[test]
    fn conv1d_backward_survives_padding_wider_than_the_strided_span() {
        let dev = Device::Cpu;
        let (b, c, l, k, s, p) = (2usize, 4, 2, 3, 1, 1);

        let l_out = (l + 2 * p - (k - 1) - 1) / s + 1;
        assert!(
            (l_out - 1) * s < 2 * p,
            "fixture no longer exercises the underflow: ({l_out} − 1)·{s} ≥ 2·{p}"
        );

        let x = candle_core::Var::from_tensor(&Tensor::randn(0f64, 1f64, (b, c, l), &dev).unwrap()).unwrap();
        let w = candle_core::Var::from_tensor(&Tensor::randn(0f64, 1f64, (c, c, k), &dev).unwrap()).unwrap();

        // Panicked at vendor/candle/candle-core/src/conv.rs before the patch.
        let grads = x
            .as_tensor()
            .conv1d(w.as_tensor(), p, s, 1, 1)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .backward()
            .unwrap();

        let gx = grads.get(&x).expect("no input gradient");
        assert_eq!(gx.dims(), &[b, c, l], "input gradient must match the input");
        assert!(
            gx.abs().unwrap().max_all().unwrap().to_scalar::<f64>().unwrap() > 1e-9,
            "input gradient is identically zero"
        );
    }
}
