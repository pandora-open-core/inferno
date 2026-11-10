//! Inferno — Neural feature heads producing Matérn, RBF and rational-quadratic feature blocks and gates.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use crate::model::norm::{layer_norm, LayerNorm, LayerNormConfig};


use crate::model::{GpParams, KernelNetworkModule, softsign};
use crate::config::{KernelNetworkConfig};




// ============================================================================
// KernelNetwork
// ============================================================================

pub struct KernelNetwork {
    /// FiLM time conditioning (`film_projector`, zero-init in Python ⇒
    /// identity at start): `t_emb [B, time_dim] → (scale, bias)` on the
    /// bottleneck. `None` on checkpoints from before the Sinkhorn migration
    /// — identity conditioning, matching the zero-init.
    film_projector: Option<Linear>,

    // Three GEOC-tractable primitive heads — weight-normed on the trainable
    // path (`parametrizations.weight.original{0,1}`), folded on cleaned
    // serving checkpoints.
    mat52_k: crate::model::parametrize::WeightNormLinear,
    rq_k: crate::model::parametrize::WeightNormLinear,
    rbf_k: crate::model::parametrize::WeightNormLinear,

    // Spectral feedback loop: three Linears + three LayerNorms; the Linears
    // are spectral-normed on the trainable path (`weight_orig`), folded on
    // cleaned serving checkpoints. ELU is applied between layers 0→1 and
    // 1→2 (no params). The final LayerNorm is applied without a trailing ELU.
    sfl_lin0: crate::model::parametrize::SpectralNormLinear,
    sfl_ln0: LayerNorm,
    sfl_lin1: crate::model::parametrize::SpectralNormLinear,
    sfl_ln1: LayerNorm,
    sfl_lin2: crate::model::parametrize::SpectralNormLinear,
    sfl_ln2: LayerNorm,

    // Gate head — unadorned Linear → Sigmoid.
    gate_head_lin: Linear,

    config: KernelNetworkConfig,
}

impl KernelNetwork {
    /// Load weights from a [`VarBuilder`] rooted at the Python
    /// `kernel_network` submodule.
    ///
    /// Expected layout after both parametrization strips:
    
    ///   mat52.weight                     [32, 64]
    ///   mat52.bias                       [32]
    ///   rq.weight                   [32, 64]
    ///   rq.bias                     [32]
    ///   rbf.weight [32, 64]
    ///   rbf.bias [32]
    
    ///
    ///   spectral_feedback_loop.0.weight   [512, 96]
    ///   spectral_feedback_loop.0.bias     [512]
    ///   spectral_feedback_loop.1.weight   [512]     (LayerNorm gamma)
    ///   spectral_feedback_loop.1.bias     [512]     (LayerNorm beta)
    ///   spectral_feedback_loop.3.weight   [1024, 512]
    ///   spectral_feedback_loop.3.bias     [1024]
    ///   spectral_feedback_loop.4.weight   [1024]
    ///   spectral_feedback_loop.4.bias     [1024]
    ///   spectral_feedback_loop.6.weight   [spectral_emb_dim, 1024]
    ///   spectral_feedback_loop.6.bias     [spectral_emb_dim]
    ///   spectral_feedback_loop.7.weight   [spectral_emb_dim]  (LayerNorm gamma)
    ///   spectral_feedback_loop.7.bias     [spectral_emb_dim]  (LayerNorm beta)
    ///
    ///   gate_head.0.weight                [8, 96]
    ///   gate_head.0.bias                  [8]
    
    ///
    /// Note that indices 2 and 5 in `spectral_feedback_loop` are `nn.ELU`
    /// layers with no params, which is why the Linear/LayerNorm indices
    /// jump from 1 → 3 → 4 → 6 → 7.
    pub fn load(config: KernelNetworkConfig, vb: VarBuilder) -> Result<Self> {
        let bottleneck = config.bottleneck_dim;
        let k_out = config.individual_kernel_dim_out;
        let primitives_total = config.primitives_total_dim();

        // FiLM head — zero-init in Python ⇒ identity conditioning at start.
        // ALWAYS built on a trainable map; absent only on pre-Sinkhorn
        // tensor-backed checkpoints.
        let film_projector = crate::model::parametrize::build_or_absent(
            &vb,
            "film_projector.weight",
            || {
                crate::model::parametrize::zero_linear(
                    vb.pp("film_projector"),
                    config.time_dim,
                    bottleneck * 2,
                )
            },
        )?;

        // The three GEOC primitives are weight-normed in Python
        // (`P.weight_norm` after an orthogonal init with gain 0.8 ⇒ row norms
        // ≈ 0.8 = `original0` seed). Trainable path registers
        // `{mat52,rbf,rq}.parametrizations.weight.original{0,1}`.
        let prim_init = candle_nn::init::Init::Randn {
            mean: 0.0,
            stdev: 0.8 / (bottleneck as f64).sqrt(),
        };
        let mat52_k =
            crate::model::parametrize::WeightNormLinear::new(vb.pp("mat52"), bottleneck, k_out, prim_init, 0.8)?;
        let rbf_k =
            crate::model::parametrize::WeightNormLinear::new(vb.pp("rbf"), bottleneck, k_out, prim_init, 0.8)?;
        let rq_k =
            crate::model::parametrize::WeightNormLinear::new(vb.pp("rq"), bottleneck, k_out, prim_init, 0.8)?;

        let sfl = vb.pp("spectral_feedback_loop");
        let ln_cfg = LayerNormConfig {
            eps: config.norm_eps,
            remove_mean: true,
            affine: true,
        };

        // Sequential Linears at 0/3/6 are spectral-normed in Python
        // (`weight_orig` on the trainable path).
        let sn = |vb: VarBuilder, i: usize, o: usize| {
            crate::model::parametrize::SpectralNormLinear::new(
                vb,
                i,
                o,
                candle_nn::init::Init::Randn { mean: 0.0, stdev: 1.0 / (i as f64).sqrt() },
            )
        };
        let sfl_lin0 = sn(sfl.pp("0"), primitives_total, 512)?;
        let sfl_ln0 = layer_norm(512, ln_cfg, sfl.pp("1"))?;

        // index 2 — ELU (no params)
        let sfl_lin1 = sn(sfl.pp("3"), 512, 1024)?;
        let sfl_ln1 = layer_norm(1024, ln_cfg, sfl.pp("4"))?;

        // index 5 — ELU (no params)
        let sfl_lin2 = sn(sfl.pp("6"), 1024, config.spectral_emb_dim)?;
        let sfl_ln2 = layer_norm(config.spectral_emb_dim, ln_cfg, sfl.pp("7"))?;

        let gate_head_lin = crate::model::parametrize::sobol_linear(&vb.pp("gate_head").pp("0"), primitives_total, config.num_gates)?;

        Ok(Self {
            film_projector,
            mat52_k,
            rq_k,
            rbf_k,
            sfl_lin0,
            sfl_ln0,
            sfl_lin1,
            sfl_ln1,
            sfl_lin2,
            sfl_ln2,
            gate_head_lin,
            config,
        })
    }

    pub fn config(&self) -> &KernelNetworkConfig {
        &self.config
    }
}



// ============================================================================
// Trait impl
// ============================================================================

impl KernelNetworkModule for KernelNetwork {
    fn set_power_iteration(&self, on: bool) {
        self.sfl_lin0.set_power_iteration(on);
        self.sfl_lin1.set_power_iteration(on);
        self.sfl_lin2.set_power_iteration(on);
    }

    fn forward(&self, bottleneck: &Tensor, t_emb: Option<&Tensor>) -> Result<(GpParams, Tensor)> {
        // FiLM on the flow-matching time: x_t = x·(1 + scale) + bias.
        // Zero-init in Python ⇒ identity at start; absent head or absent
        // conditioning ⇒ identity here.
        let x_t = match (&self.film_projector, t_emb) {
            (Some(film), Some(t)) => {
                let fb = film.forward(t)?; // [B, 2·bottleneck]
                let bn = self.config.bottleneck_dim;
                let scale = fb.narrow(D::Minus1, 0, bn)?;
                let bias = fb.narrow(D::Minus1, bn, bn)?;
                bottleneck
                    .broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(&bias)?
            }
            _ => bottleneck.clone(),
        };
        let bottleneck = &x_t;

        // Three parallel primitive heads, each followed by softsign.
        let mat52 = softsign(&self.mat52_k.forward(bottleneck)?)?;
        let rbf = softsign(&self.rbf_k.forward(bottleneck)?)?;
        let rq = softsign(&self.rq_k.forward(bottleneck)?)?;


        // Concat order is load-bearing — matches the Python reference and
        // the downstream `spectral_feedback_loop` expects the primitives in
        // this exact layout (mat52 | rbf | rq )

        let kernel_features = Tensor::cat(
            &[&mat52, &rbf, &rq],
            D::Minus1,
        )?;

        // Spectral feedback loop: Linear → LayerNorm → ELU, twice, then
        // Linear → LayerNorm (no trailing ELU on the last stage).
        let h = self.sfl_lin0.forward(&kernel_features)?;
        let h = self.sfl_ln0.forward(&h)?;
        let h = h.elu(1.0)?;

        let h = self.sfl_lin1.forward(&h)?;
        let h = self.sfl_ln1.forward(&h)?;
        let h = h.elu(1.0)?;

        let h = self.sfl_lin2.forward(&h)?;
        let features_large = self.sfl_ln2.forward(&h)?;

        // Gate head: Linear → Sigmoid on the *primitive concat* (not on
        // `features_large`).
        let gates_logits = self.gate_head_lin.forward(&kernel_features)?;
        let gates = candle_nn::ops::sigmoid(&gates_logits)?;

        let gp = GpParams {
            gates,
            gates_logits,
            mat52,
            rbf,
            rq
        };

        Ok((gp, features_large))
    }
}

// ============================================================================
// Tests (shape / helper only)
// ============================================================================

#[cfg(test)]
mod tests {
    //! Verification of neural feature heads producing matérn, rbf and rational-quadratic feature blocks and gates.

    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn config_default_matches_python() {
        let c = KernelNetworkConfig::default();
        assert_eq!(c.bottleneck_dim, 64);
        assert_eq!(c.spectral_emb_dim, 2048);
        assert_eq!(c.individual_kernel_dim_out, 32);
        assert_eq!(c.num_primitives, 3);
        assert_eq!(c.num_gates, 8);
        assert_eq!(c.primitives_total_dim(), 96);
    }

    #[test]
    fn softsign_zero() {
        let d = Device::Cpu;
        let x = Tensor::zeros((4,), DType::F32, &d).unwrap();
        let y = softsign(&x).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        for vi in v {
            assert!(vi.abs() < 1e-6);
        }
    }

    #[test]
    fn softsign_large_positive_saturates_near_one() {
        let d = Device::Cpu;
        let x = Tensor::new(&[1e6_f32], &d).unwrap();
        let y = softsign(&x).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softsign_large_negative_saturates_near_neg_one() {
        let d = Device::Cpu;
        let x = Tensor::new(&[-1e6_f32], &d).unwrap();
        let y = softsign(&x).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        assert!((v[0] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn softsign_matches_formula_at_one() {
        let d = Device::Cpu;
        let x = Tensor::new(&[1.0_f32, -1.0, 2.0, -2.0], &d).unwrap();
        let y = softsign(&x).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        // 1/(1+1) = 0.5; -1/(1+1) = -0.5; 2/(1+2) = 0.666…; -2/(1+2) = -0.666…
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] + 0.5).abs() < 1e-6);
        assert!((v[2] - 2.0 / 3.0).abs() < 1e-6);
        assert!((v[3] + 2.0 / 3.0).abs() < 1e-6);
    }
}
