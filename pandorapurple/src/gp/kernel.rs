//! Inferno — Legacy packed-feature neural kernel and SIMD sparse variational GP prediction.
//!
//! Inputs use the legacy 198-wide, five-primitive packed feature schema.
//! Hyperparameters and posterior tensors are supplied by the caller.

use anyhow::{bail, Context};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::f32::consts::PI;
use wide::f32x8;

// ---------------------------------------------------------------------------
// Feature layout & constraint bounds
// ---------------------------------------------------------------------------

/// Three-primitive packed VAE output layout; distinct from the legacy GP input.
pub mod layout {
    //! Packed VAE feature block offsets.
    pub const D: usize = 134;
    pub const BLOCK: usize = 32;
    pub const GATES_LO: usize = 0;
    pub const GATES_HI: usize = 8;
    pub const MAT52_LO: usize = 8;
    pub const MAT52_HI: usize = 40;
    pub const RBF_LO: usize = 40;
    pub const RBF_HI: usize = 72;
    pub const RQ_LO: usize = 72;
    pub const RQ_HI: usize = 104;
    pub const PI_LO: usize = 104;
    pub const PI_HI: usize = 134;
}

/// Five-primitive, 198-wide feature layout consumed by the legacy GP.
pub mod legacy_layout {
    //! Legacy neural-kernel input block offsets.
    pub const D: usize = 198;
    pub const GATES_LO: usize = 0;
    pub const GATES_HI: usize = 8;
    pub const LIN_LO: usize = 8;
    pub const LIN_HI: usize = 40;
    pub const PER_LO: usize = 40;
    pub const PER_HI: usize = 72;
    pub const RAT_LO: usize = 72;
    pub const RAT_HI: usize = 104;
    pub const POLY_LO: usize = 104;
    pub const POLY_HI: usize = 136;
    pub const MAT_LO: usize = 136;
    pub const MAT_HI: usize = 168;
    pub const PI_LO: usize = 168;
    pub const PI_HI: usize = 198;
}

pub mod bounds {
    //! Constraint intervals for legacy kernel hyperparameters.
    pub const S12: (f32, f32) = (0.05, 5.0);
    pub const S32: (f32, f32) = (0.05, 5.0);
    pub const S52: (f32, f32) = (0.05, 5.0);
    pub const RQ_ALPHA: (f32, f32) = (0.5, 3.0);
    pub const POLY_OFFSET: (f32, f32) = (0.05, 1.25);
    pub const LIN_SCALE: (f32, f32) = (0.05, 1.5);
    pub const OUTSCALE: (f32, f32) = (0.05, 3.5);
    pub const LATENT_AMP: (f32, f32) = (0.05, 3.5);
    pub const INV_BW: (f32, f32) = (0.05, 1.15);
    pub const NKN_W: (f32, f32) = (0.1, 1.1);
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline(always)]
fn interval_xform(raw: f32, lo_hi: (f32, f32)) -> f32 {
    lo_hi.0 + (lo_hi.1 - lo_hi.0) * sigmoid(raw)
}

// ---------------------------------------------------------------------------
// Hyperparameter structs (unchanged)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RawHyper {
    pub raw_scale_12: f32,
    pub raw_scale_32: f32,
    pub raw_scale_52: f32,
    pub raw_rq_alpha: f32,
    pub raw_poly_offset: f32,
    pub raw_linear_scale: f32,
    pub raw_outputscale: [f32; 8],
    pub raw_latent_amplitude: [f32; 8],
    pub raw_inv_bandwidth: [f32; 32],
    pub raw_nkn_weights: [[f32; 8]; 4],
}

#[derive(Clone)]
pub struct KernelHyper {
    pub scale_12: f32,
    pub scale_32: f32,
    pub scale_52: f32,
    pub rq_alpha: f32,
    pub poly_offset: f32,
    pub linear_scale_over_32: f32,
    pub outputscale: [f32; 8],
    pub latent_amplitude: [f32; 8],
    pub per_latent_scale: [f32; 8],
    pub inv_bw_raw: [f32; 32],
    pub inv_bw_simd: [f32x8; 4],
    pub nkn_weights: [[f32; 8]; 4],
}

impl KernelHyper {
    pub fn from_raw(raw: &RawHyper) -> Self {
        use bounds::*;
        let mut outputscale = [0.0_f32; 8];
        let mut latent_amplitude = [0.0_f32; 8];
        let mut per_latent_scale = [0.0_f32; 8];
        for e in 0..8 {
            outputscale[e] = interval_xform(raw.raw_outputscale[e], OUTSCALE);
            latent_amplitude[e] = interval_xform(raw.raw_latent_amplitude[e], LATENT_AMP);
            per_latent_scale[e] = outputscale[e] * latent_amplitude[e];
        }

        let mut inv_bw_raw = [0.0_f32; 32];
        for d in 0..32 {
            inv_bw_raw[d] = interval_xform(raw.raw_inv_bandwidth[d], INV_BW);
        }
        let inv_bw_simd = [
            f32x8::from([
                inv_bw_raw[0], inv_bw_raw[1], inv_bw_raw[2], inv_bw_raw[3],
                inv_bw_raw[4], inv_bw_raw[5], inv_bw_raw[6], inv_bw_raw[7],
            ]),
            f32x8::from([
                inv_bw_raw[8],  inv_bw_raw[9],  inv_bw_raw[10], inv_bw_raw[11],
                inv_bw_raw[12], inv_bw_raw[13], inv_bw_raw[14], inv_bw_raw[15],
            ]),
            f32x8::from([
                inv_bw_raw[16], inv_bw_raw[17], inv_bw_raw[18], inv_bw_raw[19],
                inv_bw_raw[20], inv_bw_raw[21], inv_bw_raw[22], inv_bw_raw[23],
            ]),
            f32x8::from([
                inv_bw_raw[24], inv_bw_raw[25], inv_bw_raw[26], inv_bw_raw[27],
                inv_bw_raw[28], inv_bw_raw[29], inv_bw_raw[30], inv_bw_raw[31],
            ]),
        ];

        let mut nkn_weights = [[0.0_f32; 8]; 4];
        for n in 0..4 {
            for p in 0..8 {
                let c = interval_xform(raw.raw_nkn_weights[n][p], NKN_W);
                nkn_weights[n][p] = softplus(c) / 8.0;
            }
        }

        Self {
            scale_12: interval_xform(raw.raw_scale_12, S12),
            scale_32: interval_xform(raw.raw_scale_32, S32),
            scale_52: interval_xform(raw.raw_scale_52, S52),
            rq_alpha: interval_xform(raw.raw_rq_alpha, RQ_ALPHA),
            poly_offset: interval_xform(raw.raw_poly_offset, POLY_OFFSET),
            linear_scale_over_32: interval_xform(raw.raw_linear_scale, LIN_SCALE) / 32.0,
            outputscale,
            latent_amplitude,
            per_latent_scale,
            inv_bw_raw,
            inv_bw_simd,
            nkn_weights,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LmcHyper {
    pub raw_fsa_scale: f32,
    pub raw_noise_floor: f32,
    pub lmc_weights: [f32; 8],
}

impl LmcHyper {
    #[inline(always)]
    pub fn fsa_weight(&self) -> f32 { softplus(self.raw_fsa_scale) }
    #[inline(always)]
    pub fn noise_floor(&self) -> f32 { softplus(self.raw_noise_floor) + 1e-4 }
}

// ---------------------------------------------------------------------------
// SIMD helpers — 32-dim weighted reductions
// ---------------------------------------------------------------------------

#[inline(always)]
fn dot_scaled_32(a: &[f32], b: &[f32], inv_bw: &[f32x8; 4]) -> f32 {
    debug_assert!(a.len() >= 32 && b.len() >= 32);
    let mut acc = f32x8::splat(0.0);
    for k in 0..4 {
        let av = load_f32x8(&a[k * 8..(k + 1) * 8]);
        let bv = load_f32x8(&b[k * 8..(k + 1) * 8]);
        let w = inv_bw[k];
        acc += av * bv * w * w;
    }
    horiz_sum(acc)
}

#[inline(always)]
fn dist_sq_32(a: &[f32], b: &[f32], inv_bw: &[f32x8; 4]) -> (f32, [f32x8; 4]) {
    debug_assert!(a.len() >= 32 && b.len() >= 32);
    let mut acc = f32x8::splat(0.0);
    let mut diffs = [f32x8::splat(0.0); 4];
    for k in 0..4 {
        let av = load_f32x8(&a[k * 8..(k + 1) * 8]);
        let bv = load_f32x8(&b[k * 8..(k + 1) * 8]);
        let dw = (av - bv) * inv_bw[k];
        diffs[k] = dw;
        acc += dw * dw;
    }
    (horiz_sum(acc), diffs)
}

#[inline(always)]
fn load_f32x8(slice: &[f32]) -> f32x8 {
    f32x8::from([slice[0], slice[1], slice[2], slice[3],
                 slice[4], slice[5], slice[6], slice[7]])
}

#[inline(always)]
fn horiz_sum(v: f32x8) -> f32 {
    let a: [f32; 8] = v.into();
    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}

// ---------------------------------------------------------------------------
// Forward-only kernel: K_base(x1, x2)
// ---------------------------------------------------------------------------

pub fn kernel_base(x1: &[f32], x2: &[f32], hp: &KernelHyper) -> f32 {
    debug_assert!(x1.len() >= legacy_layout::D && x2.len() >= legacy_layout::D);

    let inner_lin = dot_scaled_32(&x1[legacy_layout::LIN_LO..legacy_layout::LIN_HI],
                                  &x2[legacy_layout::LIN_LO..legacy_layout::LIN_HI],
                                  &hp.inv_bw_simd);
    let inner_poly = dot_scaled_32(&x1[legacy_layout::POLY_LO..legacy_layout::POLY_HI],
                                   &x2[legacy_layout::POLY_LO..legacy_layout::POLY_HI],
                                   &hp.inv_bw_simd);
    let (d2_mat, _)  = dist_sq_32(&x1[legacy_layout::MAT_LO..legacy_layout::MAT_HI],
                                  &x2[legacy_layout::MAT_LO..legacy_layout::MAT_HI],
                                  &hp.inv_bw_simd);
    let (d2_rat, _)  = dist_sq_32(&x1[legacy_layout::RAT_LO..legacy_layout::RAT_HI],
                                  &x2[legacy_layout::RAT_LO..legacy_layout::RAT_HI],
                                  &hp.inv_bw_simd);
    let (d2_per, _)  = dist_sq_32(&x1[legacy_layout::PER_LO..legacy_layout::PER_HI],
                                  &x2[legacy_layout::PER_LO..legacy_layout::PER_HI],
                                  &hp.inv_bw_simd);

    let d_mat = (d2_mat + 1e-12).sqrt();
    let d_per = (d2_per + 1e-11).sqrt();

    let k_lin = inner_lin * hp.linear_scale_over_32;
    let k_poly_inner = hp.poly_offset + inner_poly / 32.0;
    let k_poly = k_poly_inner * k_poly_inner;
    let sinp = (PI * d_per).sin();
    let k_per = (-2.0 * sinp * sinp).exp();
    let k_m12 = (-d_mat * hp.scale_12).exp();
    let sqrt3d = 1.7320508_f32 * d_mat * hp.scale_32;
    let k_m32 = (1.0 + sqrt3d) * (-sqrt3d).exp();
    let sqrt5d = 2.2360680_f32 * d_mat * hp.scale_52;
    let s52sq = hp.scale_52 * hp.scale_52;
    let k_m52 = (1.0 + sqrt5d + (5.0 / 3.0) * d2_mat * s52sq) * (-sqrt5d).exp();
    let w_rq = 1.0 + d2_rat / (2.0 * hp.rq_alpha);
    let k_rq = w_rq.powf(-hp.rq_alpha);
    let k_rbf = (-d2_rat).exp();

    let prims = [k_lin, k_poly, k_per, k_m12, k_m32, k_m52, k_rq, k_rbf];

    let g_i = &x1[legacy_layout::GATES_LO..legacy_layout::GATES_HI];
    let g_j = &x2[legacy_layout::GATES_LO..legacy_layout::GATES_HI];
    let mut gg = [0.0_f32; 8];
    for p in 0..8 { gg[p] = g_i[p] * g_j[p]; }

    const NODE_JITTER: f32 = 1e-3;
    let mut n_val = [NODE_JITTER; 4];
    for n in 0..4 {
        for p in 0..8 {
            n_val[n] += gg[p] * prims[p] * hp.nkn_weights[n][p];
        }
    }
    const SCALE: f32 = 0.025;
    (n_val[0] * n_val[1] + SCALE) + (n_val[2] * n_val[3] + SCALE)
}

// ---------------------------------------------------------------------------
// Analytic-gradient sibling: K_base + ∂K_base/∂θ
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct KernelGrad {
    pub d_scale_12: f32,
    pub d_scale_32: f32,
    pub d_scale_52: f32,
    pub d_rq_alpha: f32,
    pub d_poly_offset: f32,
    pub d_linear_scale_over_32: f32,
    pub d_inv_bw: [f32; 32],
    pub d_nkn_weights: [[f32; 8]; 4],
    pub d_per_latent_scale: [f32; 8],
}

pub fn kernel_base_with_grad(
    x1: &[f32], x2: &[f32], hp: &KernelHyper, coeff: f32, grad: &mut KernelGrad,
) -> f32 {
    let inner_lin = dot_scaled_32(&x1[legacy_layout::LIN_LO..legacy_layout::LIN_HI],
                                  &x2[legacy_layout::LIN_LO..legacy_layout::LIN_HI],
                                  &hp.inv_bw_simd);
    let inner_poly = dot_scaled_32(&x1[legacy_layout::POLY_LO..legacy_layout::POLY_HI],
                                   &x2[legacy_layout::POLY_LO..legacy_layout::POLY_HI],
                                   &hp.inv_bw_simd);
    let (d2_mat, _diff_mat) = dist_sq_32(&x1[legacy_layout::MAT_LO..legacy_layout::MAT_HI],
                                         &x2[legacy_layout::MAT_LO..legacy_layout::MAT_HI],
                                         &hp.inv_bw_simd);
    let (d2_rat, _diff_rat) = dist_sq_32(&x1[legacy_layout::RAT_LO..legacy_layout::RAT_HI],
                                         &x2[legacy_layout::RAT_LO..legacy_layout::RAT_HI],
                                         &hp.inv_bw_simd);
    let (d2_per, _diff_per) = dist_sq_32(&x1[legacy_layout::PER_LO..legacy_layout::PER_HI],
                                         &x2[legacy_layout::PER_LO..legacy_layout::PER_HI],
                                         &hp.inv_bw_simd);

    let d_mat = (d2_mat + 1e-12).sqrt();
    let d_per = (d2_per + 1e-11).sqrt();

    let k_lin = inner_lin * hp.linear_scale_over_32;
    let k_poly_inner = hp.poly_offset + inner_poly / 32.0;
    let k_poly = k_poly_inner * k_poly_inner;
    let two_pi_d = PI * d_per;
    let sinp = two_pi_d.sin();
    let _cosp = two_pi_d.cos();
    let k_per = (-2.0 * sinp * sinp).exp();
    let k_m12 = (-d_mat * hp.scale_12).exp();
    let sqrt3d = 1.7320508_f32 * d_mat * hp.scale_32;
    let e_m32 = (-sqrt3d).exp();
    let k_m32 = (1.0 + sqrt3d) * e_m32;
    let sqrt5d = 2.2360680_f32 * d_mat * hp.scale_52;
    let s52sq = hp.scale_52 * hp.scale_52;
    let e_m52 = (-sqrt5d).exp();
    let v_m52 = 1.0 + sqrt5d + (5.0 / 3.0) * d2_mat * s52sq;
    let k_m52 = v_m52 * e_m52;
    let w_rq = 1.0 + d2_rat / (2.0 * hp.rq_alpha);
    let k_rq = w_rq.powf(-hp.rq_alpha);
    let k_rbf = (-d2_rat).exp();

    let prims = [k_lin, k_poly, k_per, k_m12, k_m32, k_m52, k_rq, k_rbf];

    let g_i = &x1[legacy_layout::GATES_LO..legacy_layout::GATES_HI];
    let g_j = &x2[legacy_layout::GATES_LO..legacy_layout::GATES_HI];
    let mut gg = [0.0_f32; 8];
    for p in 0..8 { gg[p] = g_i[p] * g_j[p]; }

    const NODE_JITTER: f32 = 1e-3;
    let mut n_val = [NODE_JITTER; 4];
    for n in 0..4 {
        for p in 0..8 {
            n_val[n] += gg[p] * prims[p] * hp.nkn_weights[n][p];
        }
    }
    const SCALE: f32 = 0.025;
    let k_value = (n_val[0] * n_val[1] + SCALE) + (n_val[2] * n_val[3] + SCALE);

    let dk_dn = [n_val[1], n_val[0], n_val[3], n_val[2]];

    let mut dk_dprim = [0.0_f32; 8];
    for p in 0..8 {
        let mut acc = 0.0_f32;
        for n in 0..4 {
            acc += dk_dn[n] * gg[p] * hp.nkn_weights[n][p];
        }
        dk_dprim[p] = acc;
    }

    grad.d_linear_scale_over_32 += coeff * dk_dprim[0] * inner_lin;
    let dkpoly_dpoff = 2.0 * k_poly_inner;
    grad.d_poly_offset += coeff * dk_dprim[1] * dkpoly_dpoff;
    grad.d_scale_12 += coeff * dk_dprim[3] * (-d_mat * k_m12);
    let du32_ds32 = 1.7320508_f32 * d_mat;
    grad.d_scale_32 += coeff * dk_dprim[4] * (-sqrt3d * e_m32 * du32_ds32);
    let du52_ds52 = 2.2360680_f32 * d_mat;
    let dv52_ds52 = (10.0 / 3.0) * d2_mat * hp.scale_52;
    grad.d_scale_52 += coeff * dk_dprim[5] * e_m52
                     * (-v_m52 * du52_ds52 + dv52_ds52);
    let ln_w = w_rq.ln();
    let dkrq_dalpha = -k_rq * ln_w + k_rq * d2_rat / (2.0 * hp.rq_alpha * w_rq);
    grad.d_rq_alpha += coeff * dk_dprim[6] * dkrq_dalpha;

    for n in 0..4 {
        for p in 0..8 {
            grad.d_nkn_weights[n][p] += coeff * dk_dn[n] * gg[p] * prims[p];
        }
    }

    let _ = k_poly; let _ = k_per; let _ = k_rbf; // silence unused under no-grad-extension
    k_value
}

// ---------------------------------------------------------------------------
// Vector forms: K(x*, Z) over M inducing points
// ---------------------------------------------------------------------------

pub fn kernel_vector(
    x_star: &[f32], z: &ArrayView2<f32>, hp: &KernelHyper, out: &mut ArrayViewMut1<f32>,
) {
    debug_assert_eq!(z.ncols(), legacy_layout::D);
    debug_assert_eq!(out.len(), z.nrows());
    for (m, row) in z.axis_iter(ndarray::Axis(0)).enumerate() {
        let slice = row.as_slice().expect("inducing points must be contiguous");
        out[m] = kernel_base(x_star, slice, hp);
    }
}

pub fn kernel_vector_contracted_grad(
    x_star: &[f32], z: &ArrayView2<f32>, coeffs: &ArrayView1<f32>,
    hp: &KernelHyper, grad: &mut KernelGrad,
) -> f32 {
    debug_assert_eq!(z.ncols(), legacy_layout::D);
    debug_assert_eq!(coeffs.len(), z.nrows());
    let mut acc = 0.0_f32;
    for (m, row) in z.axis_iter(ndarray::Axis(0)).enumerate() {
        let slice = row.as_slice().expect("contiguous");
        let c = coeffs[m];
        let kv = kernel_base_with_grad(x_star, slice, hp, c, grad);
        acc += c * kv;
    }
    acc
}

// ---------------------------------------------------------------------------
// Mean module — ProbabilisticMixtureMean mirror
// ---------------------------------------------------------------------------

pub struct ProbabilisticMixtureMean {
    pub cluster_constants: Array2<f32>,
}

impl ProbabilisticMixtureMean {
    pub fn forward_one(&self, x_star: &[f32]) -> [f32; 8] {
        debug_assert_eq!(self.cluster_constants.nrows(), 8);
        let k_atoms = self.cluster_constants.ncols();
        debug_assert_eq!(k_atoms, legacy_layout::PI_HI - legacy_layout::PI_LO);
        let pi = &x_star[legacy_layout::PI_LO..legacy_layout::PI_HI];
        let mut m = [0.0_f32; 8];
        for e in 0..8 {
            let row = self.cluster_constants.row(e);
            let mut s = 0.0_f32;
            for a in 0..k_atoms { s += pi[a] * row[a]; }
            m[e] = s;
        }
        m
    }
}

// ---------------------------------------------------------------------------
// SvgpPredictor — the covar_module replacement at serving
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Prediction {
    pub mean: f32,
    pub variance: f32,
    pub per_latent_mean: [f32; 8],
    pub per_latent_variance: [f32; 8],
}

pub struct SvgpPredictor {
    pub hp: KernelHyper,
    pub lmc: LmcHyper,
    pub z: Array2<f32>,
    pub alpha: Array2<f32>,
    pub l_chol: Array3<f32>,
    pub p_cache: Array3<f32>,
    pub mean_module: ProbabilisticMixtureMean,
    k_star: Array1<f32>,
    v_buf: Array1<f32>,
    pt_v_buf: Array1<f32>,
}

impl SvgpPredictor {
    /// Load a precomputed GP artefact from a safetensors file.
    ///
    /// Expected schema (all tensors **f32**, little-endian, contiguous):
    ///
    /// | tensor name           | shape       | source                                        |
    /// |-----------------------|-------------|-----------------------------------------------|
    /// | `raw_scale_12`        | `[]` or `[1]` | `gp.covar_module.base_kernel.raw_scale_12`   |
    /// | `raw_scale_32`        | scalar      | …`.raw_scale_32`                              |
    /// | `raw_scale_52`        | scalar      | …`.raw_scale_52`                              |
    /// | `raw_rq_alpha`        | scalar      | …`.raw_rq_alpha`                              |
    /// | `raw_poly_offset`     | scalar      | …`.raw_poly_offset`                           |
    /// | `raw_linear_scale`    | scalar      | …`.raw_linear_scale`                          |
    /// | `raw_outputscale`     | `[8]`       | `gp.covar_module.raw_outputscale`             |
    /// | `raw_latent_amplitude`| `[8]`       | `gp.covar_module.raw_latent_amplitude`        |
    /// | `raw_inv_bandwidth`   | `[32]`      | `gp.covar_module.base_kernel.raw_inv_bandwidth` |
    /// | `raw_nkn_weights`     | `[4, 8]`    | `gp.covar_module.base_kernel.raw_nkn_weights` |
    /// | `raw_fsa_scale`       | scalar      | `gp.variational_strategy.raw_fsa_scale`       |
    /// | `raw_noise_floor`     | scalar      | `gp.variational_strategy.raw_noise_floor`     |
    /// | `z`                   | `[M, 198]`  | inducing points, collapsed across latents     |
    /// | `alpha`               | `[8, M]`    | `K_ZZ⁻¹ m_q` (precomputed per latent)         |
    /// | `l_chol`              | `[8, M, M]` | `chol(K_ZZ_unscaled)` lower triangular        |
    /// | `p_cache`             | `[8, M, M]` | `L_e⁻¹ L_Sq`                                  |
    /// | `cluster_constants`   | `[8, 30]`   | `gp.mean_module.cluster_constants`            |
    ///
    /// **Important convention**: `l_chol[e]` must be the Cholesky of the
    /// *unscaled* K_ZZ (i.e. K_ZZ computed via `kernel_base` without the
    /// `outputscale[e] * latent_amplitude[e]` multiplier). The
    /// per-latent scale is reapplied inside `predict()`. If your Python
    /// export uses GPyTorch's default scaled `K_ZZ`, divide each
    /// `K_ZZ[e]` by `per_latent_scale[e]` before the Cholesky.
    ///
    /// `lmc_weights` is initialised to a uniform placeholder of `1/8`
    /// per latent — callers MUST overwrite per request via
    /// `set_lmc_weights(...)` before calling `predict`.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read GP artefact {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parse safetensors {}", path.display()))?;

        // ----- raw hyperparameters → KernelHyper ---------------------------
        let raw = RawHyper {
            raw_scale_12:        read_scalar(&st, "raw_scale_12")?,
            raw_scale_32:        read_scalar(&st, "raw_scale_32")?,
            raw_scale_52:        read_scalar(&st, "raw_scale_52")?,
            raw_rq_alpha:        read_scalar(&st, "raw_rq_alpha")?,
            raw_poly_offset:     read_scalar(&st, "raw_poly_offset")?,
            raw_linear_scale:    read_scalar(&st, "raw_linear_scale")?,
            raw_outputscale:     read_fixed_n(&st, "raw_outputscale", 8)?,
            raw_latent_amplitude: read_fixed_n(&st, "raw_latent_amplitude", 8)?,
            raw_inv_bandwidth:   read_fixed_n_32(&st, "raw_inv_bandwidth")?,
            raw_nkn_weights:     read_nkn(&st, "raw_nkn_weights")?,
        };
        let hp = KernelHyper::from_raw(&raw);

        // ----- LMC scalars; per-request weights initialised to uniform ----
        let lmc = LmcHyper {
            raw_fsa_scale:   read_scalar(&st, "raw_fsa_scale")?,
            raw_noise_floor: read_scalar(&st, "raw_noise_floor")?,
            lmc_weights: [1.0 / 8.0; 8],
        };

        // ----- posterior tensors ------------------------------------------
        let z       = read_array2(&st, "z")?;
        let alpha   = read_array2(&st, "alpha")?;
        let l_chol  = read_array3(&st, "l_chol")?;
        let p_cache = read_array3(&st, "p_cache")?;
        let cluster_constants = read_array2(&st, "cluster_constants")?;

        // ----- shape validation (friendly errors > new()'s asserts) -------
        let m = z.nrows();
        if z.ncols() != legacy_layout::D {
            bail!("z must be [M, {}], got {:?}", legacy_layout::D, z.shape());
        }
        if alpha.shape() != [8, m] {
            bail!("alpha must be [8, M={m}], got {:?}", alpha.shape());
        }
        if l_chol.shape() != [8, m, m] {
            bail!("l_chol must be [8, {m}, {m}], got {:?}", l_chol.shape());
        }
        if p_cache.shape() != [8, m, m] {
            bail!("p_cache must be [8, {m}, {m}], got {:?}", p_cache.shape());
        }
        let k_atoms = legacy_layout::PI_HI - legacy_layout::PI_LO;
        if cluster_constants.shape() != [8, k_atoms] {
            bail!("cluster_constants must be [8, {k_atoms}], got {:?}",
                cluster_constants.shape());
        }

        Ok(Self::new(
            hp, lmc, z, alpha, l_chol, p_cache,
            ProbabilisticMixtureMean { cluster_constants },
        ))
    }

    pub fn new(
        hp: KernelHyper, lmc: LmcHyper, z: Array2<f32>, alpha: Array2<f32>,
        l_chol: Array3<f32>, p_cache: Array3<f32>, mean_module: ProbabilisticMixtureMean,
    ) -> Self {
        let m = z.nrows();
        assert_eq!(z.ncols(), legacy_layout::D);
        assert_eq!(alpha.shape(), &[8, m]);
        assert_eq!(l_chol.shape(), &[8, m, m]);
        assert_eq!(p_cache.shape(), &[8, m, m]);
        Self {
            hp, lmc, z, alpha, l_chol, p_cache, mean_module,
            k_star: Array1::zeros(m),
            v_buf:  Array1::zeros(m),
            pt_v_buf: Array1::zeros(m),
        }
    }

    #[inline(always)]
    pub fn set_lmc_weights(&mut self, w: [f32; 8]) {
        self.lmc.lmc_weights = w;
    }

    pub fn predict(&mut self, x_star: &[f32]) -> Prediction {
        debug_assert!(x_star.len() >= legacy_layout::D);
        let m = self.z.nrows();

        {
            let mut view = self.k_star.view_mut();
            kernel_vector(x_star, &self.z.view(), &self.hp, &mut view);
        }
        let prior_means = self.mean_module.forward_one(x_star);
        let k_ss_base = kernel_base(x_star, x_star, &self.hp);

        let mut per_latent_mean = [0.0_f32; 8];
        let mut per_latent_var = [0.0_f32; 8];

        for e in 0..8 {
            let scale_e = self.hp.per_latent_scale[e];

            let alpha_e = self.alpha.row(e);
            let mut dot = 0.0_f32;
            for i in 0..m { dot += alpha_e[i] * self.k_star[i]; }
            per_latent_mean[e] = prior_means[e] + scale_e * dot;

            let l_e = self.l_chol.index_axis(ndarray::Axis(0), e);
            tri_solve_lower(&l_e, &self.k_star.view(), &mut self.v_buf.view_mut());
            let mut v_norm_sq = 0.0_f32;
            for i in 0..m { v_norm_sq += self.v_buf[i] * self.v_buf[i]; }

            let p_e = self.p_cache.index_axis(ndarray::Axis(0), e);
            let mut u_norm_sq = 0.0_f32;
            for j in 0..m {
                let col = p_e.column(j);
                let mut s = 0.0_f32;
                for i in 0..m { s += col[i] * self.v_buf[i]; }
                u_norm_sq += s * s;
            }

            let prior_reduced = k_ss_base - v_norm_sq;
            per_latent_var[e] = scale_e * (prior_reduced + u_norm_sq).max(0.0);
        }

        let w = &self.lmc.lmc_weights;
        let mut mixed_mean = 0.0_f32;
        let mut mixed_var = 0.0_f32;
        let mut full_prior_mix = 0.0_f32;
        for e in 0..8 {
            mixed_mean += w[e] * per_latent_mean[e];
            mixed_var += w[e] * w[e] * per_latent_var[e];
            full_prior_mix += w[e] * (k_ss_base * self.hp.per_latent_scale[e]);
        }

        let residual = (full_prior_mix - mixed_var).max(1e-4);
        let total_var = mixed_var
                      + residual * self.lmc.fsa_weight()
                      + self.lmc.noise_floor();

        Prediction {
            mean: mixed_mean,
            variance: total_var,
            per_latent_mean,
            per_latent_variance: per_latent_var,
        }
    }
}

// ---------------------------------------------------------------------------
// safetensors readers (private — only used by SvgpPredictor::load)
// ---------------------------------------------------------------------------

fn check_f32(name: &str, t: &TensorView) -> anyhow::Result<()> {
    if t.dtype() != Dtype::F32 {
        bail!("tensor `{name}` must be f32, got {:?}", t.dtype());
    }
    Ok(())
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0,
        "safetensors f32 payload length {} not divisible by 4", bytes.len());
    bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_scalar(st: &SafeTensors, name: &str) -> anyhow::Result<f32> {
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let data = bytes_to_f32(t.data());
    if data.len() != 1 {
        bail!("`{name}` must be a scalar (1 element), got shape {:?}", t.shape());
    }
    Ok(data[0])
}

/// Read a fixed-length f32 vector into a `[f32; 8]`. Generic over the length
/// via `expected_len` so the same helper works for `raw_outputscale` and
/// `raw_latent_amplitude`.
fn read_fixed_n(st: &SafeTensors, name: &str, expected_len: usize) -> anyhow::Result<[f32; 8]> {
    if expected_len != 8 {
        bail!("read_fixed_n only supports length 8, got {expected_len}");
    }
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let data = bytes_to_f32(t.data());
    if data.len() != 8 {
        bail!("`{name}` must have 8 elements, got {} (shape {:?})",
            data.len(), t.shape());
    }
    let mut out = [0.0_f32; 8];
    out.copy_from_slice(&data);
    Ok(out)
}

fn read_fixed_n_32(st: &SafeTensors, name: &str) -> anyhow::Result<[f32; 32]> {
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let data = bytes_to_f32(t.data());
    if data.len() != 32 {
        bail!("`{name}` must have 32 elements, got {} (shape {:?})",
            data.len(), t.shape());
    }
    let mut out = [0.0_f32; 32];
    out.copy_from_slice(&data);
    Ok(out)
}

fn read_nkn(st: &SafeTensors, name: &str) -> anyhow::Result<[[f32; 8]; 4]> {
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let shape = t.shape();
    if !(shape == [4, 8] || shape == [32]) {
        bail!("`{name}` must be [4, 8] or flattened [32], got {:?}", shape);
    }
    let data = bytes_to_f32(t.data());
    if data.len() != 32 {
        bail!("`{name}` must have 32 elements, got {}", data.len());
    }
    let mut out = [[0.0_f32; 8]; 4];
    // Row-major: data[n*8 + p] = nkn[n][p]
    for n in 0..4 {
        for p in 0..8 {
            out[n][p] = data[n * 8 + p];
        }
    }
    Ok(out)
}

fn read_array2(st: &SafeTensors, name: &str) -> anyhow::Result<Array2<f32>> {
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("`{name}` must be 2-D, got shape {:?}", shape);
    }
    let data = bytes_to_f32(t.data());
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .with_context(|| format!("reshape `{name}` to [{}, {}]", shape[0], shape[1]))
}

fn read_array3(st: &SafeTensors, name: &str) -> anyhow::Result<Array3<f32>> {
    let t = st.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    check_f32(name, &t)?;
    let shape = t.shape();
    if shape.len() != 3 {
        bail!("`{name}` must be 3-D, got shape {:?}", shape);
    }
    let data = bytes_to_f32(t.data());
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
        .with_context(|| format!("reshape `{name}` to [{}, {}, {}]",
            shape[0], shape[1], shape[2]))
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

fn tri_solve_lower(l: &ArrayView2<f32>, b: &ArrayView1<f32>, x: &mut ArrayViewMut1<f32>) {
    let m = l.nrows();
    debug_assert_eq!(l.ncols(), m);
    debug_assert_eq!(b.len(), m);
    debug_assert_eq!(x.len(), m);
    for i in 0..m {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }
}

// ===========================================================================
// Python export script (reference)
// ===========================================================================
//
// Drop this on the Python side next to your training script. It produces a
// `gp_artefact.safetensors` file that `SvgpPredictor::load` consumes.
//
// ```python
// import torch
// from safetensors.torch import save_file
//
// def export_gp(gp, out_path: str):
//     """
//     gp: trained AcceleratedKernelGP with .covar_module (GenerativeKernel
//         wrapped in ScaleKernel + per-latent amplitude), .mean_module
//         (ProbabilisticMixtureMean), and .variational_strategy
//         (LMCVariationalStrategy).
//     """
//     ck = gp.covar_module                 # ScaleKernel(GenerativeKernel)
//     bk = ck.base_kernel                  # GenerativeKernel (NKN composition)
//     vs = gp.variational_strategy
//     bvs = vs.base_variational_strategy   # the per-latent VariationalStrategy
//
//     # ----- inducing points: collapse the per-latent copies -----------
//     # `inducing_points` is [8, M, 198]; we average to a single shared [M, 198]
//     # because the features feeding the kernel are identical across latents.
//     Z = bvs.inducing_points.detach().mean(dim=0).contiguous()
//     M = Z.shape[0]
//
//     # ----- K_ZZ per latent (unscaled), Cholesky, P = L⁻¹ L_Sq --------
//     with torch.no_grad():
//         # GPyTorch returns the SCALED K_ZZ; divide out per-latent scale.
//         outscale = torch.nn.functional.softplus(ck.raw_outputscale) * 0  # placeholder
//         # The real per-latent scale is the Interval-transformed product:
//         lo, hi = 0.05, 3.5
//         out_s = lo + (hi - lo) * torch.sigmoid(ck.raw_outputscale)     # [8]
//         lat_s = lo + (hi - lo) * torch.sigmoid(ck.raw_latent_amplitude) # [8]
//         per_latent_scale = (out_s * lat_s).view(8, 1, 1)               # [8,1,1]
//
//         # K_ZZ_scaled : [8, M, M]
//         K_ZZ_scaled = ck(Z.unsqueeze(0).expand(8, -1, -1)).evaluate()
//         K_ZZ_unscaled = K_ZZ_scaled / per_latent_scale
//         L = torch.linalg.cholesky(K_ZZ_unscaled)                       # [8, M, M]
//
//         # alpha = K_ZZ_unscaled⁻¹ @ m_q  (per latent)
//         m_q = bvs.variational_distribution.variational_mean             # [8, M]
//         alpha = torch.cholesky_solve(m_q.unsqueeze(-1), L).squeeze(-1)  # [8, M]
//
//         # L_Sq = chol(S_q)  (CholeskyVariationalDistribution stores it directly)
//         L_Sq = bvs.variational_distribution.chol_variational_covar      # [8, M, M]
//
//         # P = L⁻¹ L_Sq   (lower-triangular forward solve)
//         P = torch.linalg.solve_triangular(L, L_Sq, upper=False)         # [8, M, M]
//
//     tensors = {
//         "raw_scale_12":         bk.raw_scale_12.detach(),
//         "raw_scale_32":         bk.raw_scale_32.detach(),
//         "raw_scale_52":         bk.raw_scale_52.detach(),
//         "raw_rq_alpha":         bk.raw_rq_alpha.detach(),
//         "raw_poly_offset":      bk.raw_poly_offset.detach(),
//         "raw_linear_scale":     bk.raw_linear_scale.detach(),
//         "raw_outputscale":      ck.raw_outputscale.detach(),       # [8]
//         "raw_latent_amplitude": ck.raw_latent_amplitude.detach(),  # [8]
//         "raw_inv_bandwidth":    bk.raw_inv_bandwidth.detach(),     # [32]
//         "raw_nkn_weights":      bk.raw_nkn_weights.detach(),       # [4, 8]
//         "raw_fsa_scale":        vs.raw_fsa_scale.detach(),
//         "raw_noise_floor":      vs.raw_noise_floor.detach(),
//         "z":                    Z,                                  # [M, 198]
//         "alpha":                alpha,                              # [8, M]
//         "l_chol":               L,                                  # [8, M, M]
//         "p_cache":              P,                                  # [8, M, M]
//         "cluster_constants":    gp.mean_module.cluster_constants.detach(),
//     }
//     # Force f32 (avoid surprises if anything trained in bfloat16).
//     tensors = {k: v.to(torch.float32).contiguous() for k, v in tensors.items()}
//     save_file(tensors, out_path)
// ```
//
// Validate the round-trip by predicting a known applicant in Python and
// asserting Rust matches to within `~1e-4` absolute. The most common drift
// sources are (a) forgetting to divide K_ZZ by per_latent_scale before chol
// and (b) saving the scaled S_q where you meant the unscaled — both produce
// a uniform variance scaling that's easy to spot against a calibration set.

#[cfg(test)]
mod layout_tests {
    //! Verification of legacy packed-feature neural kernel and simd sparse variational gp prediction.

    use super::{layout, legacy_layout};

    /// The offsets are written out by hand, so the arithmetic gets checked
    /// rather than trusted: each range must start where the previous one
    /// ended, and the last must land exactly on `D`. A gap or an overlap
    /// here does not error — it silently feeds the kernel the wrong 32
    /// columns for a primitive.
    #[test]
    fn layout_tiles_without_gaps_or_overlap() {
        for (name, ranges, d) in [
            (
                "layout",
                vec![
                    ("gates", layout::GATES_LO, layout::GATES_HI),
                    ("mat52", layout::MAT52_LO, layout::MAT52_HI),
                    ("rbf", layout::RBF_LO, layout::RBF_HI),
                    ("rq", layout::RQ_LO, layout::RQ_HI),
                    ("pi", layout::PI_LO, layout::PI_HI),
                ],
                layout::D,
            ),
            (
                "legacy_layout",
                vec![
                    ("gates", legacy_layout::GATES_LO, legacy_layout::GATES_HI),
                    ("lin", legacy_layout::LIN_LO, legacy_layout::LIN_HI),
                    ("per", legacy_layout::PER_LO, legacy_layout::PER_HI),
                    ("rat", legacy_layout::RAT_LO, legacy_layout::RAT_HI),
                    ("poly", legacy_layout::POLY_LO, legacy_layout::POLY_HI),
                    ("mat", legacy_layout::MAT_LO, legacy_layout::MAT_HI),
                    ("pi", legacy_layout::PI_LO, legacy_layout::PI_HI),
                ],
                legacy_layout::D,
            ),
        ] {
            let mut cursor = 0usize;
            for (field, lo, hi) in &ranges {
                assert_eq!(*lo, cursor, "{name}: {field} starts at {lo}, expected {cursor}");
                assert!(hi > lo, "{name}: {field} is empty");
                cursor = *hi;
            }
            assert_eq!(cursor, d, "{name}: ranges cover {cursor}, but D is {d}");
        }
    }

    /// Three primitives of one BLOCK each, plus the gates and the simplex.
    /// If someone widens a primitive without widening `D`, the tiling test
    /// above catches it; this one says what the numbers MEAN.
    #[test]
    fn the_live_layout_is_three_primitives_of_one_block() {
        assert_eq!(layout::D, 134);
        assert_eq!(layout::BLOCK, 32);
        for (lo, hi) in [
            (layout::MAT52_LO, layout::MAT52_HI),
            (layout::RBF_LO, layout::RBF_HI),
            (layout::RQ_LO, layout::RQ_HI),
        ] {
            assert_eq!(hi - lo, layout::BLOCK, "every primitive is one block wide");
        }
        assert_eq!(layout::GATES_HI - layout::GATES_LO, 8, "one gate per LMC latent");
        assert_eq!(layout::PI_HI - layout::PI_LO, 30, "k_atoms slots for the simplex");
        // The two schemas must stay distinct — the whole point of splitting
        // them is that a checkpoint written for one is not readable as the
        // other.
        assert_ne!(layout::D, legacy_layout::D);
    }
}
