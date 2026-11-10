//! Inferno — Sparsemax and entmax probability transforms with a counterfactual prediction head.

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// entmax-1.5 along the last dim. `z: [.., E] → p: [.., E]` (sparse simplex).
pub fn entmax15(z: &Tensor, n_iter: usize) -> Result<Tensor> {
    let last = z.rank() - 1;
    let h = z.affine(0.5, 0.0)?; // ½z
    let h_max = h.max_keepdim(last)?; // [.., 1]
    // Bracket: sum(τ_hi)=0 < 1 ≤ sum(τ_lo).
    let mut tau_hi = h_max.clone();
    let mut tau_lo = h_max.affine(1.0, -1.0)?; // h_max − 1
    let one = h_max.affine(0.0, 1.0)?; // all-ones, [.., 1]

    for _ in 0..n_iter {
        let tau = ((&tau_lo + &tau_hi)? * 0.5)?;
        let p = h.broadcast_sub(&tau)?.relu()?.sqr()?; // [.., E]
        let s = p.sum_keepdim(last)?; // [.., 1]
        // sum decreases in τ: if s > 1 root is at larger τ ⇒ raise the floor.
        let gt = s.gt(&one)?; // u8 mask [.., 1]
        tau_lo = gt.where_cond(&tau, &tau_lo)?;
        tau_hi = gt.where_cond(&tau_hi, &tau)?;
    }

    let tau = ((&tau_lo + &tau_hi)? * 0.5)?;
    let p = h.broadcast_sub(&tau)?.relu()?.sqr()?;
    // renormalise to kill residual bisection error.
    p.broadcast_div(&p.sum_keepdim(last)?)
}

/// **sparsemax** (α-entmax at α = 2) along the last dim. `z: [.., E] → p`.
///
/// Closed form `p_i = [z_i − τ]_+` with `τ` fixed by `Σ_i p_i = 1`, solved by
/// the same bisection as [`entmax15`] — monotone decreasing in `τ`, bracketed by
/// `[max(z) − 1, max(z)]` because the top coordinate alone already contributes 1
/// at the lower end and everything is clipped away at the upper.
///
/// The difference from [`entmax15`] is the whole Tsallis ladder in two ops:
/// α = 1.5 halves and SQUARES the hinge, α = 2 does neither. Sparsemax
/// therefore produces MORE exact zeros than entmax-1.5 on the same logits,
/// which is why `flows::state::SimplexMap` offers both rather than picking one.
///
/// Same caution as entmax: outside the support the Jacobian is zero, so a
/// coordinate driven to 0 receives no gradient of its own. It is not strictly
/// absorbing — the threshold moves as the survivors update — but the dynamics
/// are slow there. Watch the support SIZE, not just the entropy.
pub fn sparsemax(z: &Tensor, n_iter: usize) -> Result<Tensor> {
    let last = z.rank() - 1;
    let z_max = z.max_keepdim(last)?; // [.., 1]
    let mut tau_hi = z_max.clone();
    let mut tau_lo = z_max.affine(1.0, -1.0)?; // z_max − 1
    let one = z_max.affine(0.0, 1.0)?;

    for _ in 0..n_iter {
        let tau = ((&tau_lo + &tau_hi)? * 0.5)?;
        let p = z.broadcast_sub(&tau)?.relu()?; // [.., E]
        let s = p.sum_keepdim(last)?;
        let gt = s.gt(&one)?;
        tau_lo = gt.where_cond(&tau, &tau_lo)?;
        tau_hi = gt.where_cond(&tau_hi, &tau)?;
    }

    let tau = ((&tau_lo + &tau_hi)? * 0.5)?;
    let p = z.broadcast_sub(&tau)?.relu()?;
    p.broadcast_div(&p.sum_keepdim(last)?)
}

/// Sparse counterfactual head: routes `[B, E, D]` expert outputs into
/// `num_sparse` counterfactual bottlenecks via per-counterfactual entmax-1.5
/// distributions over the experts.
pub struct EntmaxCounterfactualHead {
    router: Linear,     // routing_dim → num_sparse·num_experts
    projection: Linear, // latent_dim → bottleneck_dim
    num_experts: usize,
    num_sparse: usize,
    entmax_iters: usize,
}

impl EntmaxCounterfactualHead {
    pub fn new(
        vb: VarBuilder,
        num_experts: usize,
        num_sparse: usize,
        latent_dim: usize,
        bottleneck_dim: usize,
        routing_dim: Option<usize>,
    ) -> Result<Self> {
        let routing_dim = routing_dim.unwrap_or(num_experts);
        Ok(Self {
            router: crate::model::parametrize::sobol_linear(&vb.pp("router"), routing_dim, num_sparse * num_experts)?,
            projection: crate::model::parametrize::sobol_linear(&vb.pp("projection"), latent_dim, bottleneck_dim)?,
            num_experts,
            num_sparse,
            entmax_iters: 50,
        })
    }

    /// `expert_outputs: [B, E, D]`, `routing_signal: [B, routing_dim]`.
    /// Returns `(counterfactual_bottlenecks [B, K, bottleneck],
    /// counterfactual_latents [B, K, D], routing_weights [B, K, E])` — the
    /// middle element is the PRE-projection latent-space counterfactuals
    /// (what the SDE bridge / LatentSSM scores).
    /// Returns `(bottlenecks, latents, routing_weights, routing_logits)` —
    /// the RAW pre-entmax logits ride along for the router-K KL (each
    /// entmax row is a simplex, so any expert-sum of the WEIGHTS is
    /// identically 1 with a zero Jacobian; the K-distribution must come
    /// from the logits).
    pub fn forward(
        &self,
        expert_outputs: &Tensor,
        routing_signal: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let b = expert_outputs.dim(0)?;
        let (k, e) = (self.num_sparse, self.num_experts);

        let routing_logits = self.router.forward(routing_signal)?.reshape((b, k, e))?;
        let routing_weights = entmax15(&routing_logits, self.entmax_iters)?; // [B, K, E]

        // [B, K, E] · [B, E, D] → [B, K, D]
        let per_cf = routing_weights.matmul(expert_outputs)?;
        let bottlenecks = self.projection.forward(&per_cf)?; // [B, K, bottleneck]
        Ok((bottlenecks, per_cf, routing_weights, routing_logits))
    }
}

#[cfg(test)]
mod tests {
    //! Verification of sparsemax and entmax probability transforms with a counterfactual prediction head.

    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn entmax15_is_a_sparse_simplex() {
        let dev = Device::Cpu;
        // A peaked logit row should route (near) all mass to one expert and
        // produce exact zeros elsewhere.
        let z = Tensor::new(&[[4.0f32, 0.0, -3.0, -3.0, -3.0]], &dev).unwrap();
        let p = entmax15(&z, 60).unwrap();
        let s = p.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((s - 1.0).abs() < 1e-4, "entmax sum = {s}");
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().any(|x| *x == 0.0), "entmax should be sparse: {v:?}");
        assert!(v.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn head_shapes() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (b, e, d, k, bn) = (3usize, 6usize, 10usize, 4usize, 8usize);
        let head = EntmaxCounterfactualHead::new(vb, e, k, d, bn, Some(e)).unwrap();
        let experts = Tensor::randn(0f32, 1f32, (b, e, d), &dev).unwrap();
        let routing = Tensor::randn(0f32, 1f32, (b, e), &dev).unwrap();
        let (out, latents, w, _lg) = head.forward(&experts, &routing).unwrap();
        assert_eq!(out.dims(), &[b, k, bn]);
        assert_eq!(latents.dims(), &[b, k, d]);
        assert_eq!(w.dims(), &[b, k, e]);
    }
}
