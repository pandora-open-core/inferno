//! Inferno — Differentiable Candle implementation of the legacy neural kernel and multitask GP posterior.
//!
//! Inputs use the legacy 198-wide, five-primitive packed feature schema.
//! Hyperparameters and posterior tensors are supplied by the caller.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use std::sync::Mutex;
use crate::model::{RefinementCertainty, StateSpaceOutput, vae_forward, forward_substitute, softplus, Vae};
use crate::gp::{KernelHyper};
use super::DifferentiablePrediction;

// KernelHyperTensor
// ---------------------------------------------------------------------------

/// Hyperparameters carried as Candle tensors. Any leaf may be wrapped in
/// `Var` upstream to enable autograd; this struct is agnostic.
pub struct KernelHyperTensor {
    pub scale_12: Tensor,
    pub scale_32: Tensor,
    pub scale_52: Tensor,
    pub rq_alpha: Tensor,
    pub poly_offset: Tensor,
    pub linear_scale_over_32: Tensor,
    pub outputscale: Tensor,      // [8]
    pub latent_amplitude: Tensor, // [8]
    pub inv_bw: Tensor,           // [32]
    pub nkn_weights: Tensor,      // [4, 8]
}

impl KernelHyperTensor {
    /// Lift a pure-Rust `KernelHyper` onto the device. For inference-only
    /// use; wrap any leaf you want to fine-tune in `candle_core::Var`
    /// before calling `.backward()` downstream.
    pub fn from_kernel_hyper(
        hp: &KernelHyper,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let scalar = |v: f32| -> Result<Tensor> {
            Ok(Tensor::new(&[v], device)?.to_dtype(dtype)?)
        };
        let vec = |v: &[f32]| -> Result<Tensor> {
            Ok(Tensor::from_slice(v, v.len(), device)?.to_dtype(dtype)?)
        };

        let mut nkn_flat = Vec::with_capacity(4 * 8);
        for row in &hp.nkn_weights {
            nkn_flat.extend_from_slice(row);
        }
        let nkn_weights = Tensor::from_slice(&nkn_flat, (4, 8), device)?.to_dtype(dtype)?;

        Ok(Self {
            scale_12: scalar(hp.scale_12)?,
            scale_32: scalar(hp.scale_32)?,
            scale_52: scalar(hp.scale_52)?,
            rq_alpha: scalar(hp.rq_alpha)?,
            poly_offset: scalar(hp.poly_offset)?,
            linear_scale_over_32: scalar(hp.linear_scale_over_32)?,
            outputscale: vec(&hp.outputscale)?,
            latent_amplitude: vec(&hp.latent_amplitude)?,
            inv_bw: vec(&hp.inv_bw_raw)?,
            nkn_weights,
        })
    }
}

// ---------------------------------------------------------------------------
// Candle NKN kernel
// ---------------------------------------------------------------------------

/// Batched NKN kernel matrix, Candle version. Returns `[B, M]`. Gradients
/// flow through `x1`, `x2`, and every field of `hp` when those leaves are
/// `Var`-wrapped upstream.
///
/// Formula is bit-for-bit identical to `kernel_module::kernel_base` —
/// any divergence here is a bug, fix it before training. The 198-d layout:
///   `gates(8) | lin(32) | per(32) | rat(32) | poly(32) | mat(32) | pi(30)`
pub fn kernel_base_candle(
    x1: &Tensor,
    x2: &Tensor,
    hp: &KernelHyperTensor,
) -> Result<Tensor> {
    use std::f64::consts::PI;

    let slc1 = |off: usize, len: usize| x1.narrow(1, off, len);
    let slc2 = |off: usize, len: usize| x2.narrow(1, off, len);

    let gates_i = slc1(0, 8)?;
    let gates_j = slc2(0, 8)?;
    let lin_i = slc1(8, 32)?;
    let lin_j = slc2(8, 32)?;
    let per_i = slc1(40, 32)?;
    let per_j = slc2(40, 32)?;
    let rat_i = slc1(72, 32)?;
    let rat_j = slc2(72, 32)?;
    let poly_i = slc1(104, 32)?;
    let poly_j = slc2(104, 32)?;
    let mat_i = slc1(136, 32)?;
    let mat_j = slc2(136, 32)?;

    let inv_bw_row = hp.inv_bw.unsqueeze(0)?; // [1, 32]

    // Σ a·b·inv_bw²  = (a ⊙ inv_bw)(b ⊙ inv_bw)ᵀ
    let dot_w2 = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
        let aw = a.broadcast_mul(&inv_bw_row)?;
        let bw = b.broadcast_mul(&inv_bw_row)?;
        Ok(aw.matmul(&bw.t()?)?)
    };
    let inner_lin = dot_w2(&lin_i, &lin_j)?;
    let inner_poly = dot_w2(&poly_i, &poly_j)?;

    // Σ ((a - b)·inv_bw)²   → [B, M]
    let dist_sq = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
        let a_exp = a.unsqueeze(1)?; // [B, 1, 32]
        let b_exp = b.unsqueeze(0)?; // [1, M, 32]
        let diff = a_exp.broadcast_sub(&b_exp)?;
        let w = hp.inv_bw.reshape((1, 1, 32))?;
        let weighted = diff.broadcast_mul(&w)?;
        Ok(weighted.sqr()?.sum(D::Minus1)?)
    };
    let d2_mat = dist_sq(&mat_i, &mat_j)?;
    let d2_rat = dist_sq(&rat_i, &rat_j)?;
    let d2_per = dist_sq(&per_i, &per_j)?;

    let d_mat = d2_mat.affine(1.0, 1e-12)?.sqrt()?;
    let d_per = d2_per.affine(1.0, 1e-11)?.sqrt()?;

    let k_lin = inner_lin.broadcast_mul(&hp.linear_scale_over_32)?;

    let k_poly_inner = inner_poly
        .affine(1.0 / 32.0, 0.0)?
        .broadcast_add(&hp.poly_offset)?;
    let k_poly = k_poly_inner.sqr()?;

    let sinp = d_per.affine(PI, 0.0)?.sin()?;
    let k_per = sinp.sqr()?.affine(-2.0, 0.0)?.exp()?;

    let k_m12 = d_mat.broadcast_mul(&hp.scale_12)?.neg()?.exp()?;

    let sqrt3d = d_mat
        .broadcast_mul(&hp.scale_32)?
        .affine(1.732_050_8, 0.0)?;
    let k_m32 = sqrt3d.affine(1.0, 1.0)?.mul(&sqrt3d.neg()?.exp()?)?;

    let sqrt5d = d_mat
        .broadcast_mul(&hp.scale_52)?
        .affine(2.236_068, 0.0)?;
    let s52sq = hp.scale_52.sqr()?;
    let v_m52_term3 = d2_mat.broadcast_mul(&s52sq)?.affine(5.0 / 3.0, 0.0)?;
    let v_m52 = sqrt5d.affine(1.0, 1.0)?.add(&v_m52_term3)?;
    let k_m52 = v_m52.mul(&sqrt5d.neg()?.exp()?)?;

    let w_rq = d2_rat
        .broadcast_div(&hp.rq_alpha.affine(2.0, 0.0)?)?
        .affine(1.0, 1.0)?;
    // x^(-α) = exp(-α · log x) — keep in-graph.
    let k_rq = hp
        .rq_alpha
        .neg()?
        .broadcast_mul(&w_rq.log()?)?
        .exp()?;

    let k_rbf = d2_rat.neg()?.exp()?;

    // Gate outer product [B, M, 8].
    let gg = gates_i.unsqueeze(1)?.broadcast_mul(&gates_j.unsqueeze(0)?)?;

    // Stack primitives along last axis — canonical NKN order:
    // [lin, poly, per, m12, m32, m52, rq, rbf].
    let prims = Tensor::stack(
        &[&k_lin, &k_poly, &k_per, &k_m12, &k_m32, &k_m52, &k_rq, &k_rbf],
        D::Minus1,
    )?;

    let gg_prims = gg.broadcast_mul(&prims)?;

    // nodes[b, m, n] = Σ_p gg_prims[b, m, p] · w[n, p]   →   gg_prims @ wᵀ
    let nkn_wt = hp.nkn_weights.t()?; // [8, 4]
    let nodes = gg_prims.matmul(&nkn_wt)?.affine(1.0, 1.0e-3)?; // [B, M, 4]

    let n0 = nodes.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let n1 = nodes.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let n2 = nodes.narrow(D::Minus1, 2, 1)?.squeeze(D::Minus1)?;
    let n3 = nodes.narrow(D::Minus1, 3, 1)?.squeeze(D::Minus1)?;

    let first = n0.mul(&n1)?.affine(1.0, 0.025)?;
    let second = n2.mul(&n3)?.affine(1.0, 0.025)?;
    Ok(first.add(&second)?)
}

// ---------------------------------------------------------------------------
// DifferentiableGp
// ---------------------------------------------------------------------------

/// Candle-native differentiable GP. Mirrors `SvgpPredictor` topology but
/// every op is a tensor op.
pub struct DifferentiableGp {
    pub hp: KernelHyperTensor,
    pub z: Tensor,                 // [M, 198]
    pub alpha: Tensor,             // [8, M]
    pub l_chol: Tensor,            // [8, M, M] lower triangular
    pub p_cache: Tensor,           // [8, M, M]
    pub cluster_constants: Tensor, // [8, 30]
    pub raw_fsa_scale: Tensor,
    pub raw_noise_floor: Tensor,
    pub lmc_weights: Mutex<Tensor>,
    pub device: Device,
    pub dtype: DType,
}

impl DifferentiableGp {
    pub fn predict(&self, x_star: &Tensor) -> Result<DifferentiablePrediction> {
        let k_star = kernel_base_candle(x_star, &self.z, &self.hp)?; // [B, M]
        let k_ss = kernel_base_candle(x_star, x_star, &self.hp)?;    // [B, B]
        let b = k_ss.dim(0)?;
        let eye = Tensor::eye(b, k_ss.dtype(), k_ss.device())?;
        let k_ss_diag = k_ss.broadcast_mul(&eye)?.sum(0)?;

        // Prior means from ProbabilisticMixtureMean: π @ cᵀ  →  [B, 8].
        let pi = x_star.narrow(1, 168, 30)?;
        let prior_means = pi.matmul(&self.cluster_constants.t()?)?;

        let per_latent_scale = self.hp.outputscale.mul(&self.hp.latent_amplitude)?; // [8]

        // Per-latent mean: m(x*) + scale · k*ᵀ α
        let dot_alpha = k_star.matmul(&self.alpha.t()?)?; // [B, 8]
        let per_latent_mean = prior_means.add(
            &dot_alpha.broadcast_mul(&per_latent_scale.unsqueeze(0)?)?,
        )?;

        // Per-latent variance via batched forward substitution.
        let b = x_star.dim(0)?;
        let m = self.z.dim(0)?;
        let k_star_t = k_star.t()?.unsqueeze(0)?.broadcast_as((8, m, b))?;
        let v = forward_substitute(&self.l_chol, &k_star_t)?; // [8, M, B]
        let v_norm_sq = v.sqr()?.sum(1)?; // [8, B]

        let u = self.p_cache.t()?.matmul(&v)?; // [8, M, B]
        let u_norm_sq = u.sqr()?.sum(1)?; // [8, B]

        let prior_reduced = k_ss_diag.unsqueeze(0)?.broadcast_sub(&v_norm_sq)?;
        let per_latent_var_t = per_latent_scale.unsqueeze(D::Minus1)?.broadcast_mul(
            &prior_reduced
                .add(&u_norm_sq)?
                .clamp(0.0_f32, f32::INFINITY)?,
        )?;
        let per_latent_variance = per_latent_var_t.t()?; // [B, 8]

        // LMC mixing.
        let w = self
            .lmc_weights
            .lock()
            .map_err(|_| anyhow!("lmc_weights mutex poisoned"))?
            .clone();
        let mixed_mean = per_latent_mean
            .broadcast_mul(&w.unsqueeze(0)?)?
            .sum(D::Minus1)?;
        let w_sq = w.sqr()?;
        let mixed_var = per_latent_variance
            .broadcast_mul(&w_sq.unsqueeze(0)?)?
            .sum(D::Minus1)?;

        // FSA residual + noise floor.
        let full_prior_per_latent = k_ss_diag
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&per_latent_scale.unsqueeze(0)?)?;
        let full_prior_mix = full_prior_per_latent
            .broadcast_mul(&w.unsqueeze(0)?)?
            .sum(D::Minus1)?;

        let fsa_weight = softplus(&self.raw_fsa_scale)?;
        let noise_floor = softplus(&self.raw_noise_floor)?.affine(1.0, 1e-4)?;

        let residual = full_prior_mix
            .sub(&mixed_var)?
            .clamp(1e-4_f32, f32::INFINITY)?;
        let variance = mixed_var
            .add(&residual.broadcast_mul(&fsa_weight)?)?
            .broadcast_add(&noise_floor)?;

        Ok(DifferentiablePrediction {
            mean: mixed_mean,
            variance,
            per_latent_mean,
            per_latent_variance,
        })
    }

    pub fn set_lmc_weights(&self, w: &[f32; 8]) -> Result<()> {
        let tensor_w = Tensor::new(w.as_slice(), &self.device)?.to_dtype(self.dtype)?;
        let mut guard = self
            .lmc_weights
            .lock()
            .map_err(|_| anyhow!("lmc_weights mutex poisoned"))?;
        *guard = tensor_w;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

/// Run the differentiable GP serving path. Caller must wrap any tensor
/// they want to differentiate through (e.g. `x`, hp leaves) in
/// `candle_core::Var` BEFORE calling — this function is autograd-agnostic.
pub fn serve(
    vae: &Vae,
    gp: &DifferentiableGp,
    certainty: RefinementCertainty,
    x: &Tensor,
    state: Option<&StateSpaceOutput>,
) -> Result<(DifferentiablePrediction, StateSpaceOutput)> {
    let (state_out, zz, lmc_w) = vae_forward(vae, certainty, x, state)?;
    gp.set_lmc_weights(&lmc_w)?;
    let zz_t = Tensor::from_vec(zz.clone(), (1, zz.len()), x.device())?;
    let pred = gp.predict(&zz_t)?;
    Ok((pred, state_out))
}
