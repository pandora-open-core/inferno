//! Inferno — Spectral variational autoencoders with configurable encoders, latent distributions and decoders.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use crate::model::{Decoder, DecoderMode, SpectralDecoder};
use crate::model::{Encoder, ConvolutionalLoopEncoder};
use crate::model::{Dirichlet, DirichletMode, AmortisedDirichlet};
use crate::model::KernelNetwork;
use crate::config::{SpectralVaeConfig, RootConfig};

use crate::model::ewma::EwmaEncoder;
use crate::model::mamba::{LatentSSM, SsmState};
use crate::model::time::FourierTimeEmbedding;
use crate::model::noise::{BayesianPosteriorSampler, NoiseBundle, SamplerConfig};

use crate::gp::{layout};

/// Rolling Itô window the GP sees (Python `SpectralVAE.T_HISTORY`).
pub const T_HISTORY: usize = 12;

// ---------------------------------------------------------------------------
// Caller-facing refinement certainty knob
// ---------------------------------------------------------------------------

/// The number of refinement iterations the VAE performs before handing its
/// feature block to the Rust GP. Higher = tighter posterior at roughly
/// linear wall-clock cost.
///
///   * `Full`     — `T_HISTORY` steps: the TRAINING default, and the only
///     setting that yields a real Itô window (see the variant docs)
///   * `Quick`    — 1 step  (~1 VAE forward, fastest; SERVING only)
///   * `Standard` — 2 steps (matches the Python default)
///   * `High`     — 3 steps (tight posterior; use for high-stakes decisions
///                          or when upstream drift detection fired)
///
/// Map directly to the `steps` argument of the Python `refinement_loop`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RefinementCertainty {
    Quick,
    Standard,
    High,
    /// `T_HISTORY` refinements — the PRODUCTION default (Python
    /// `steps=None → T_HISTORY`, 2026-07-30).
    ///
    /// # Why anything less is broken, not merely cheaper
    ///
    /// `refinement_loop` collects one latent per step and LEFT-PADS the stack
    /// to `T_HISTORY` by repeating its first entry. So `Quick` (1 step) yields
    /// a `z_history` that is the current latent copied 12 times — a window
    /// with no time variation at all. Every consumer built on that window is
    /// then reading a constant: the spatiotemporal GP's `T` axis carries no
    /// signal, the Itô Savitzky–Golay drift target is identically zero, and
    /// `forward_gp`'s window rebuild substitutes `x_t` into 11 copies of
    /// itself.
    ///
    /// None of that errors. It trains, reports finite losses, and is blind to
    /// time — which is why the default moved rather than being left to the
    /// caller.
    Full,
}

impl RefinementCertainty {
    pub fn steps(self) -> usize {
        match self {
            RefinementCertainty::Quick => 1,
            RefinementCertainty::Standard => 2,
            RefinementCertainty::High => 3,
            RefinementCertainty::Full => T_HISTORY,
        }
    }
}

impl Default for RefinementCertainty {
    /// `Full` — the TRAINING default (Python `steps=None → T_HISTORY`,
    /// 2026-07-30). See the variant's docs: anything less produces a
    /// degenerate Itô window and the spatiotemporal GP cannot see time.
    fn default() -> Self {
        RefinementCertainty::Full
    }
}

// ---------------------------------------------------------------------------
// State carried across refinement steps (mirror of the Python NamedTuple)
// ---------------------------------------------------------------------------

/// Full state carried between refinement steps. Tensor shapes are
/// annotated inline. All tensors live on the configured device; host-side
/// copies happen only at the GP handoff.
#[derive(Clone)]
pub struct StateSpaceOutput {
    // ── Kernel feature blocks consumed by the GP ────────────────────────
    pub gates: Tensor,        // [B, 8]
    pub mat52: Tensor,        // [B, 32]
    pub rbf: Tensor,          // [B, 32]
    pub rq: Tensor,           // [B, 32]
    pub pi: Tensor,           // [B, k_atoms]  (k_atoms=30 typically)

    // ── Bottleneck / latent plumbing ────────────────────────────────────
    pub bottleneck: Tensor,               // [B, bottleneck_dim]
    pub alpha: Tensor,                    // [B, evidence_dim]
    pub alpha_mu: Tensor,                 // [B, evidence_dim]
    pub alpha_factor: Tensor,             // [B, evidence_dim, rank]
    pub alpha_diag: Tensor,               // [B, evidence_dim]
    pub gp_features: Tensor,              // [B, num_latents, latent_dim]
    pub parameters_per_expert: Tensor,    // [B, num_latents, latent_dim]
    pub recon: Tensor,                    // [B, 1, input_dim]
    /// Composite predictive variance paired with `recon` — the decoder's
    /// heteroscedastic `1/P`. `None` only on zero-states; a real forward
    /// always carries it. The likelihood network's outcome predictive is
    /// `N(recon, recon_var)`.
    pub recon_var: Option<Tensor>,        // [B, 1, input_dim]
    pub bandwidth_mod: Tensor,            // [B, num_latents]
    pub amp: Tensor,                      // [B, 1, input_dim]
    pub trend: Tensor,                    // [B, 1, input_dim]
    pub res: Tensor,                      // [B, input_dim]
    pub ls: Tensor,                       // [B, k_atoms]
    pub ls_logvar: Option<Tensor>,        // [B, k] lengthscale posterior logvar
    pub bw_learned: Option<Tensor>,       // [B, k, latent_dim] learned bandwidths
    pub mu_z: Tensor,                     // [B, latent_dim]
    pub logvar_z: Tensor,                 // [B, latent_dim]
    pub lmc_matrices: Tensor,             // [B, k_atoms, k_atoms]
    pub lmc_consensus: Tensor,            // [B, k_atoms, num_latents]
    pub real_x: Tensor,                   // [B, 1, input_dim]   (anchor)

    // ── New optional state (models/state.py additions) ──────────────────
    /// Pre-sigmoid NKN gate logits `[B, 8]` (routing signal for the
    /// counterfactual head / capped-simplex weighting).
    pub gates_logits: Option<Tensor>,
    /// Sampled latent `z [B, latent_dim]` of the LAST refinement step.
    pub z: Option<Tensor>,
    /// Rolling Itô window `[B, T_HISTORY, latent_dim]` of per-step latents.
    pub z_history: Option<Tensor>,
    /// Rolling Itô window `[B, ≤T_HISTORY, 134]` of per-step packed kernel
    /// features (`gates|mat52|rbf|rq|π`) — the windowed input of the
    /// SB-GP's per-position binary/probit head and `call_windowed`.
    pub feature_history: Option<Tensor>,
    /// Pre-SSM posterior-mean window `[B, T_HISTORY, latent_dim]` — the SSM
    /// InfoNCE's KEYS (Python `z_mu_history`; on the DAG path this is also
    /// what Python republishes as `_last_z_pre_history`).
    pub z_mu_history: Option<Tensor>,
    /// EWMA bank features per step `[B, T_HISTORY, D·(K+1)]` (80 at the
    /// production geometry) — detached BY CONSTRUCTION (the bank never
    /// participates in the gradient); the latent policy's substrate when
    /// `ewma_to_policy`.
    pub z_ewma_history: Option<Tensor>,
    /// EWMA bank carry `[B, K, D]`. NEVER folded or collapsed — Python's
    /// `_NEVER_FOLDED`: its leading dim is `B`, not `B·T`, and at `T == 1`
    /// a shape signature cannot tell the two apart; a name can.
    pub ewma_state: Option<Tensor>,
    /// Mamba scan carry `(h [B, d_inner, N], conv_cache [B, d_inner, k−1])`.
    /// NEVER folded or collapsed — same discipline as `ewma_state`. Python
    /// writes it unconditionally at the end of every rollout; before this
    /// field existed the Rust loop dropped the recurrence at return.
    pub ssm_state: Option<crate::model::mamba::SsmState>,
    /// VP diffusion time of THIS forward `[B, 1]` (stamped by time_norm).
    pub diffusion_t: Option<Tensor>,
    /// Sparse counterfactual bottleneck lifts `[B, K, bottleneck_dim]`.
    pub counterfactual_bottlenecks: Option<Tensor>,
    /// Pre-projection LATENT-space counterfactuals `[B, K, latent_dim]` —
    /// what the SDE bridge / LatentSSM scores (from the entmax head).
    pub counterfactual_latents: Option<Tensor>,
    /// Entmax-sparse routing `[B, K, num_experts]`.
    pub counterfactual_routing: Option<Tensor>,
    /// `[B, K, num_experts]` raw router logits (pre-entmax).
    pub counterfactual_routing_logits: Option<Tensor>,
    /// `[B, num_experts]` dense MoE gate weights (softmax of gates_logits) —
    /// the gate-alignment target of the two-piece irl_routing.
    pub moe_weights: Option<Tensor>,
    /// Routing moments accumulated across the refinement steps
    /// (count-weighted running mean).
    pub moments: Option<crate::model::moments::GaussianMoments>,

    // ── Loss stashes (the Python state's `-loss terms-` block): computed by
    // the KL-enabled dirichlet/decoder forwards (OFF the serving path; the
    // SGLD potential + fetcher LossTerms read them) ──────────────────────
    pub global_divergence: Option<Tensor>,
    pub local_divergence: Option<Tensor>,
    pub inverse_wishart: Option<Tensor>,
    pub lengthscale_kl: Option<Tensor>,
    pub recon_kl: Option<Tensor>,
    /// Heteroscedastic 3-expert reconstruction NLL stash (decoder-computed).
    pub recon_likelihood: Option<Tensor>,
    /// `[B]` per-sample reconstruction NLL — Σ over features AND the heads'
    /// temporal axis (same map as `recon_likelihood`, which mean-reduces it:
    /// `recon_likelihood == recon_nll_per_sample.mean() / T`). The IWAE
    /// evidence bound's required input: `log p(x|z) = −recon_nll_per_sample`
    /// up to the dropped `0.5·D·log2π` constant, which is constant in z and
    /// cancels in self-normalised importance weights.
    pub recon_nll_per_sample: Option<Tensor>,
    /// Dense-MoE mixture reconstruction NLL stash (decoder-computed).
    pub moe_recon_nll: Option<Tensor>,
    /// `(log bw̄ − log inv_bw)²`-style consistency stash (else the term
    /// derives it live from `bw_learned` + the kernel's `inv_bandwidth`).
    pub bandwidth_loss_term: Option<Tensor>,
    /// Global Kumaraswamy stick state (`KumaraswamyGlobal`) — the score
    /// terms / SGLD variational potential read q_a/q_b/log(1−v)/β off it.
    pub kumaraswamy_global: Option<crate::model::concentration::KumaraswamyConcentration>,
}

impl StateSpaceOutput {
    /// Build the zero/neutral state. Port of Python `get_zero_state`.
    pub fn stochastic_zero_state(
        x: &Tensor,
        config: &SpectralVaeConfig,
        device: &Device,
        batch_size: usize,
    ) -> Result<Self> {
        let b = batch_size;
        let k = config.k_atoms;
        let e = config.num_latents;
        let f = config.latent_dim;
        let r = config.rank;
        let d_in = config.input_dim;
        let bn = config.bottleneck_dim;
        let ev = config.evidence_dim;
        let eps = config.eps as f32;

        let init_pi = uniform_pi(b, k, device)?;

        // SEEDED, like every other init in this tree: `Tensor::randn` reaches
        // candle's device RNG, and the vendored CPU backend refuses
        // `Device::set_seed` — an unseeded zero state makes every forward that
        // brings one up (the deterministic pass brings one up per call)
        // irreproducible. Python's torch.randn here is covered by the global
        // manual_seed; this is the Rust equivalent. Distinct salt per field.
        let zr = |shape: &[usize], scale: f32, salt: u64| -> Result<Tensor> {
            Ok(crate::model::noise::sobol_randn_like_per_dim(
                shape,
                device,
                0x7a65_726f_5f73 ^ salt,
            )?
            .reshape(shape)?
            .affine(scale as f64, 0.0)?)
        };

        Ok(Self {
            recon: zr(&[b, 1, d_in], eps, 1)?,
            recon_var: None,
            amp: Tensor::ones((b, 1, d_in), DType::F32, device)?,
            trend: Tensor::zeros((b, 1, d_in), DType::F32, device)?,
            gp_features: zr(&[b, e, f], eps, 2)?,
            pi: init_pi,
            alpha: Tensor::ones((b, ev), DType::F32, device)?,

            // FIX: Cast config values to f32 so Tensor::full infers DType::F32
            alpha_mu: Tensor::full(config.neutral_logit as f32, (b, ev), device)?,
            alpha_factor: Tensor::zeros((b, ev, r), DType::F32, device)?,
            
            alpha_diag: Tensor::full(config.alpha_diag_init as f32, (b, ev), device)?,

            bottleneck: zr(&[b, bn], eps, 3)?,
            parameters_per_expert: zr(&[b, e, f], eps, 4)?,
            bandwidth_mod: Tensor::ones((b, e), DType::F32, device)?,
            res: Tensor::zeros((b, d_in), DType::F32, device)?,
            ls: Tensor::ones((b, k), DType::F32, device)?,
            mu_z: zr(&[b, f], eps, 5)?,
            
            // .affine(multiplier, addition)
            logvar_z: Tensor::ones((b, f), DType::F32, device)?.affine(0.05, 0.0)?,
            
            lmc_matrices: zr(&[b, k, k], 1e-4, 6)?,
            real_x: x.clone(),
            
            gates: Tensor::ones((b, 8), DType::F32, device)?.affine(0.125, 0.0)?,
            mat52: zr(&[b, 32], 0.01, 7)?,
            rbf: zr(&[b, 32], 0.01, 8)?,
            rq: zr(&[b, 32], 0.01, 9)?,
            lmc_consensus: zr(&[b, k, e], 1e-4, 10)?,

            gates_logits: None,
            z: None,
            z_history: None,
            feature_history: None,
            z_mu_history: None,
            z_ewma_history: None,
            ewma_state: None,
            ssm_state: None,
            diffusion_t: None,
            counterfactual_bottlenecks: None,
            counterfactual_latents: None,
            counterfactual_routing: None,
            counterfactual_routing_logits: None,
            moe_weights: None,
            moments: None,
            ls_logvar: None,
            bw_learned: None,
            global_divergence: None,
            local_divergence: None,
            inverse_wishart: None,
            lengthscale_kl: None,
            recon_kl: None,
            recon_likelihood: None,
            recon_nll_per_sample: None,
            moe_recon_nll: None,
            bandwidth_loss_term: None,
            kumaraswamy_global: None,
        })
    }

    /// Every tensor field `.detach()`ed — the step-boundary carry (Python
    /// `detach_carried_state`). A warm start for the recurrent carries, not
    /// BPTT: nothing survives with a graph.
    pub fn detach_all(&self) -> Result<Self> {
        let d = |t: &Tensor| t.detach();
        let dopt = |t: &Option<Tensor>| t.as_ref().map(|x| x.detach());
        Ok(Self {
            gates: d(&self.gates),
            mat52: d(&self.mat52),
            rbf: d(&self.rbf),
            rq: d(&self.rq),
            pi: d(&self.pi),
            bottleneck: d(&self.bottleneck),
            alpha: d(&self.alpha),
            alpha_mu: d(&self.alpha_mu),
            alpha_factor: d(&self.alpha_factor),
            alpha_diag: d(&self.alpha_diag),
            gp_features: d(&self.gp_features),
            parameters_per_expert: d(&self.parameters_per_expert),
            recon: d(&self.recon),
            recon_var: dopt(&self.recon_var),
            bandwidth_mod: d(&self.bandwidth_mod),
            amp: d(&self.amp),
            trend: d(&self.trend),
            res: d(&self.res),
            ls: d(&self.ls),
            ls_logvar: dopt(&self.ls_logvar),
            bw_learned: dopt(&self.bw_learned),
            mu_z: d(&self.mu_z),
            logvar_z: d(&self.logvar_z),
            lmc_matrices: d(&self.lmc_matrices),
            lmc_consensus: d(&self.lmc_consensus),
            real_x: d(&self.real_x),
            gates_logits: dopt(&self.gates_logits),
            z: dopt(&self.z),
            z_history: dopt(&self.z_history),
            feature_history: dopt(&self.feature_history),
            z_mu_history: dopt(&self.z_mu_history),
            z_ewma_history: dopt(&self.z_ewma_history),
            ewma_state: dopt(&self.ewma_state),
            ssm_state: self
                .ssm_state
                .as_ref()
                .map(|(h, c)| (h.detach(), c.detach())),
            diffusion_t: dopt(&self.diffusion_t),
            counterfactual_bottlenecks: dopt(&self.counterfactual_bottlenecks),
            counterfactual_latents: dopt(&self.counterfactual_latents),
            counterfactual_routing: dopt(&self.counterfactual_routing),
            counterfactual_routing_logits: dopt(&self.counterfactual_routing_logits),
            moe_weights: dopt(&self.moe_weights),
            moments: self.moments.as_ref().map(|m| crate::model::moments::GaussianMoments {
                mean: m.mean.detach(),
                variance: m.variance.detach(),
                skew: m.skew.detach(),
                excess_kurtosis: m.excess_kurtosis.detach(),
                entropy: m.entropy.detach(),
                count: m.count,
            }),
            global_divergence: dopt(&self.global_divergence),
            local_divergence: dopt(&self.local_divergence),
            inverse_wishart: dopt(&self.inverse_wishart),
            lengthscale_kl: dopt(&self.lengthscale_kl),
            recon_kl: dopt(&self.recon_kl),
            recon_likelihood: dopt(&self.recon_likelihood),
            recon_nll_per_sample: dopt(&self.recon_nll_per_sample),
            moe_recon_nll: dopt(&self.moe_recon_nll),
            bandwidth_loss_term: dopt(&self.bandwidth_loss_term),
            kumaraswamy_global: self.kumaraswamy_global.as_ref().map(|k| {
                crate::model::concentration::KumaraswamyConcentration {
                    q_a: dopt(&k.q_a),
                    q_b: dopt(&k.q_b),
                    log_one_minus_v: dopt(&k.log_one_minus_v),
                    v: dopt(&k.v),
                    u: dopt(&k.u),
                    log_pi: dopt(&k.log_pi),
                    pi_posterior: dopt(&k.pi_posterior),
                    beta: dopt(&k.beta),
                    log_beta: dopt(&k.log_beta),
                    gamma: dopt(&k.gamma),
                }
            }),
        })
    }

    /// Push one refinement step's latent into the rolling Itô window
    /// (`[B, T_HISTORY, latent_dim]`, oldest dropped once full). 
    // this is z_history
    pub fn push_z_history(prev: Option<&Tensor>, z: &Tensor, t_history: usize) -> Result<Tensor> {
        let step = z.unsqueeze(1)?; // [B, 1, D]
        match prev {
            None => Ok(step),
            Some(h) => {
                let l = h.dim(1)?;
                let kept = if l >= t_history { h.narrow(1, l + 1 - t_history, t_history - 1)? } else { h.clone() };
                Ok(Tensor::cat(&[&kept, &step], 1)?)
            }
        }
    }
}

/// Left-pad a per-step history to a fixed `t`-wide window by repeating the
/// earliest step: `[B, D]×n → [B, t, D]`. The repeat is inert for the EWMA
/// history (levels seeded AT μ₀ make a constant prefix exact) and a measured
/// signal loss for `z_history` — see the warning at the call site.
fn pad_window(tail: &[Tensor], t: usize) -> Result<Tensor> {
    let mut window: Vec<&Tensor> = Vec::with_capacity(t);
    for _ in tail.len()..t {
        window.push(&tail[0]);
    }
    window.extend(tail.iter());
    Ok(Tensor::stack(&window, 1)?)
}

/// Unfold a `[B·T, …]`-assembled state back to `B` and collapse it: window
/// fields keep the full `[B, T, …]` axis, the `_NEVER_FOLDED` carries pass
/// through untouched, `moments` takes the window MEAN, and everything else
/// takes the clean END step (`[:, T−1]`) — Python's `deterministic_pass`
/// sweep (vae.py:834-877).
///
/// EXHAUSTIVE DESTRUCTURING on purpose: adding a field to `StateSpaceOutput`
/// without deciding its collapse rule is a COMPILE ERROR here, which is the
/// Rust twin of Python's "the rule is the whole state" discipline (that side
/// paid two incidents for a subset sweep).
fn unfold_and_collapse(state: StateSpaceOutput, b: usize, t: usize) -> Result<StateSpaceOutput> {
    let bt = b * t;
    // `[B·T, rest…] → [B, rest…]` at the clean end; leading-dim mismatch
    // (batchless globals, already-[B] tensors) passes through untouched.
    let clean_end = |v: &Tensor| -> Result<Tensor> {
        if v.rank() >= 1 && v.dim(0)? == bt {
            let mut dims = vec![b, t];
            dims.extend(v.dims()[1..].iter().copied());
            Ok(v.reshape(dims)?.narrow(1, t - 1, 1)?.squeeze(1)?)
        } else {
            Ok(v.clone())
        }
    };
    let clean_opt = |v: &Option<Tensor>| -> Result<Option<Tensor>> {
        Ok(match v {
            Some(x) => Some(clean_end(x)?),
            None => None,
        })
    };
    // Window MEAN over the unfolded T axis — for the routing moments only.
    let window_mean = |v: &Tensor| -> Result<Tensor> {
        if v.rank() >= 1 && v.dim(0)? == bt {
            let mut dims = vec![b, t];
            dims.extend(v.dims()[1..].iter().copied());
            Ok(v.reshape(dims)?.mean(1)?)
        } else {
            Ok(v.clone())
        }
    };
    // Window fields: keep the full [B, T, …] axis.
    let window = |v: &Tensor| -> Result<Tensor> {
        if v.rank() >= 1 && v.dim(0)? == bt {
            let mut dims = vec![b, t];
            dims.extend(v.dims()[1..].iter().copied());
            Ok(v.reshape(dims)?)
        } else {
            Ok(v.clone())
        }
    };

    let StateSpaceOutput {
        gates,
        mat52,
        rbf,
        rq,
        pi,
        bottleneck,
        alpha,
        alpha_mu,
        alpha_factor,
        alpha_diag,
        gp_features,
        parameters_per_expert,
        recon,
        recon_var,
        bandwidth_mod,
        amp,
        trend,
        res,
        ls,
        ls_logvar,
        bw_learned,
        mu_z,
        logvar_z,
        lmc_matrices,
        lmc_consensus,
        real_x,
        gates_logits,
        z,
        z_history,
        feature_history,
        z_mu_history,
        z_ewma_history,
        ewma_state,
        ssm_state,
        diffusion_t,
        counterfactual_bottlenecks,
        counterfactual_latents,
        counterfactual_routing,
        counterfactual_routing_logits,
        moe_weights,
        moments,
        global_divergence,
        local_divergence,
        inverse_wishart,
        lengthscale_kl,
        recon_kl,
        recon_likelihood,
        recon_nll_per_sample,
        moe_recon_nll,
        bandwidth_loss_term,
        kumaraswamy_global,
    } = state;

    // The GP kernel blocks stay windowed [B, T, …] — and their packed concat
    // IS the binary head's window on this path (replaces the rolling push).
    let gates_w = window(&gates)?;
    let mat52_w = window(&mat52)?;
    let rbf_w = window(&rbf)?;
    let rq_w = window(&rq)?;
    let pi_w = window(&pi)?;
    let feature_history = Some(Tensor::cat(
        &[&gates_w, &mat52_w, &rbf_w, &rq_w, &pi_w],
        candle_core::D::Minus1,
    )?);

    Ok(StateSpaceOutput {
        // WINDOW fields — the GP consumes the packed [B, T, 134] above; the
        // per-field state keeps the clean end so every [B, …]-shaped consumer
        // (forward_gp's pack134, the drift terms) reads one position.
        gates: clean_end(&gates)?,
        mat52: clean_end(&mat52)?,
        rbf: clean_end(&rbf)?,
        rq: clean_end(&rq)?,
        pi: clean_end(&pi)?,
        feature_history,
        // Histories/carries: stamped by the caller's overrides.
        z_history,
        z_mu_history,
        z_ewma_history,
        ewma_state,
        ssm_state,
        diffusion_t,
        // Clean-end everything else.
        bottleneck: clean_end(&bottleneck)?,
        alpha: clean_end(&alpha)?,
        alpha_mu: clean_end(&alpha_mu)?,
        alpha_factor: clean_end(&alpha_factor)?,
        alpha_diag: clean_end(&alpha_diag)?,
        gp_features: clean_end(&gp_features)?,
        parameters_per_expert: clean_end(&parameters_per_expert)?,
        recon: clean_end(&recon)?,
        recon_var: clean_opt(&recon_var)?,
        bandwidth_mod: clean_end(&bandwidth_mod)?,
        amp: clean_end(&amp)?,
        trend: clean_end(&trend)?,
        res: clean_end(&res)?,
        ls: clean_end(&ls)?,
        ls_logvar: clean_opt(&ls_logvar)?,
        bw_learned: clean_opt(&bw_learned)?,
        mu_z: clean_end(&mu_z)?,
        logvar_z: clean_end(&logvar_z)?,
        lmc_matrices: clean_end(&lmc_matrices)?,
        lmc_consensus: clean_end(&lmc_consensus)?,
        real_x: clean_end(&real_x)?,
        gates_logits: clean_opt(&gates_logits)?,
        z: clean_opt(&z)?,
        counterfactual_bottlenecks: clean_opt(&counterfactual_bottlenecks)?,
        counterfactual_latents: clean_opt(&counterfactual_latents)?,
        counterfactual_routing: clean_opt(&counterfactual_routing)?,
        counterfactual_routing_logits: clean_opt(&counterfactual_routing_logits)?,
        moe_weights: clean_opt(&moe_weights)?,
        // Window MEAN — a routing estimate over the whole window, not the
        // last position's (Python `_window_mean`).
        moments: match moments {
            Some(m) => Some(crate::model::moments::GaussianMoments {
                mean: window_mean(&m.mean)?,
                variance: window_mean(&m.variance)?,
                skew: window_mean(&m.skew)?,
                excess_kurtosis: window_mean(&m.excess_kurtosis)?,
                entropy: window_mean(&m.entropy)?,
                count: m.count,
            }),
            None => None,
        },
        global_divergence: clean_opt(&global_divergence)?,
        local_divergence: clean_opt(&local_divergence)?,
        inverse_wishart: clean_opt(&inverse_wishart)?,
        lengthscale_kl: clean_opt(&lengthscale_kl)?,
        recon_kl: clean_opt(&recon_kl)?,
        recon_likelihood: clean_opt(&recon_likelihood)?,
        recon_nll_per_sample: clean_opt(&recon_nll_per_sample)?,
        moe_recon_nll: clean_opt(&moe_recon_nll)?,
        bandwidth_loss_term: clean_opt(&bandwidth_loss_term)?,
        // Global stick container: batchless raws, recomputed statelessly per
        // fold — its tensors never carry a B·T leading dim, so clean-end
        // through the same helper is the identity while staying under the
        // exhaustive-sweep discipline.
        kumaraswamy_global: match kumaraswamy_global {
            Some(k) => Some(crate::model::concentration::KumaraswamyConcentration {
                q_a: clean_opt(&k.q_a)?,
                q_b: clean_opt(&k.q_b)?,
                log_one_minus_v: clean_opt(&k.log_one_minus_v)?,
                v: clean_opt(&k.v)?,
                u: clean_opt(&k.u)?,
                log_pi: clean_opt(&k.log_pi)?,
                pi_posterior: clean_opt(&k.pi_posterior)?,
                beta: clean_opt(&k.beta)?,
                log_beta: clean_opt(&k.log_beta)?,
                gamma: clean_opt(&k.gamma)?,
            }),
            None => None,
        },
    })
}

/// Uniform simplex initialisation: `1/k` everywhere.
fn uniform_pi(b: usize, k: usize, device: &Device) -> Result<Tensor> {
    let pi = Tensor::ones((b, k), DType::F32, device)?;
    Ok((pi * (1.0 / k as f64))?)
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Candle port of `SpectralVAE`. Generic over the three heavy submodules so
/// you can swap real ports for mocks in unit tests; the SSM / sampler / time
/// embedding are concrete orchestrator plumbing (like the Python model).
pub struct SpectralVae<E: Encoder, Dm: Dirichlet, Dc: Decoder> {
    pub encoder: E,
    /// Mamba latent SSM on the `encoder_z → Dirichlet` boundary — present on
    /// checkpoints trained with `model.vae.latent_ssm`; identity at init.
    pub latent_ssm: Option<LatentSSM>,
    pub dirichlet: Dm,
    pub decoder: Dc,
    /// Master Fourier time embedding (`self.time_emb` in Python): `[B, 1]`
    /// diffusion time → `[B, 256]` conditioning. `None` on checkpoints from
    /// before the master-timer migration (the old z-additive `time_net` MLP,
    /// which the current flow no longer applies).
    pub time_emb: Option<FourierTimeEmbedding>,
    /// Deterministic per-Itô-step reparam noise (`self._noise_sampler`):
    /// every site seeded via `site_seed`, distinct per step via
    /// `salt = step_idx` — common random numbers across forwards.
    pub sampler: BayesianPosteriorSampler,
    /// The multi-scale EWMA bank (`models/ewma.py`). Constructed
    /// UNCONDITIONALLY, like Python (vae.py:193): it is parameter-free, so
    /// the module tree — and hence the checkpoint — does not depend on the
    /// `ewma_enabled` flag.
    pub ewma_encoder: EwmaEncoder,
    /// `config.ewma.ewma` — gates the wiring, not the construction.
    pub ewma_enabled: bool,
    pub config: SpectralVaeConfig,
    pub device: Device,
}


impl SpectralVae<ConvolutionalLoopEncoder, AmortisedDirichlet<KernelNetwork>, SpectralDecoder> {

    pub fn new(
        encoder: ConvolutionalLoopEncoder,
        latent_ssm: Option<LatentSSM>,
        dirichlet: AmortisedDirichlet<KernelNetwork>,
        decoder: SpectralDecoder,
        time_emb: Option<FourierTimeEmbedding>,
        config: SpectralVaeConfig,        // ← narrow
        device: Device,
    ) -> Self {
        let sampler = BayesianPosteriorSampler::new(SamplerConfig::default());
        // Default bank; `build`/`load` override from `root.ewma` via
        // `with_ewma`. `default_bank` cannot fail on its static half-life
        // list (non-empty, ascending).
        let ewma_encoder = EwmaEncoder::default_bank(config.latent_dim, DType::F32, &device)
            .expect("default EWMA bank is statically valid");
        Self {
            encoder,
            latent_ssm,
            dirichlet,
            decoder,
            time_emb,
            sampler,
            ewma_encoder,
            ewma_enabled: false,
            config,
            device,
        }
    }

    /// Install the configured bank. The first reader `EWMAConfig` has had.
    pub fn with_ewma(mut self, cfg: &crate::config::EWMAConfig) -> Result<Self> {
        self.ewma_encoder = EwmaEncoder::new(
            self.config.latent_dim,
            &cfg.ewma_half_lives,
            cfg.ewma_ctx_gain,
            DType::F32,
            &self.device,
        )?;
        self.ewma_enabled = cfg.ewma;
        Ok(self)
    }

    /// Port of Python `deterministic_pass` (variationalautoencoder.py:641-878).
    ///
    /// One `[B·T, 1, D]` fold over the pooled window: every position seeded
    /// from a zero state (no state feedback), submodules EVAL-FORCED via the
    /// mode-parameter seams, ONE salt-0 noise bundle. The stochastic 1-step
    /// companion (`refinement_loop`) can be called alongside it to pair T
    /// deterministic positions with one stochastic step.
    ///
    /// The eval forcing closes exactly the reparameterisation branches
    /// (`conv_bottleneck = mu_data`, `z = mu_z`, Kumaraswamy at the median);
    /// `logvar_z` is still emitted, so the analytic spread consumers keep
    /// their gradient path through THIS state.
    pub fn deterministic_pass(
        &self,
        x_pooled: &Tensor,
        t: f64,
        indices: Option<&Tensor>,
        current_state: Option<&StateSpaceOutput>,
    ) -> Result<StateSpaceOutput> {
        let (b, t_win, d_in) = x_pooled.dims3().map_err(|_| {
            anyhow!(
                "deterministic_pass expects the pooled window [B, T, D]; got {:?}",
                x_pooled.dims()
            )
        })?;
        let bt = b * t_win;

        // 1. Fold the time axis into the batch.
        let xf = x_pooled.reshape((bt, 1, d_in))?;

        // 2. Scalar time conditioning at batch B·T (the Rust call chain
        //    reduces the clock to a scalar before the VAE — the per-step
        //    [B, T] clock of vae.py:905-914 is not reachable from here).
        let dt = Tensor::full(t as f32, (bt, 1), &self.device)?.to_dtype(xf.dtype())?;
        let t_cond = match &self.time_emb {
            Some(f) => Some(f.forward(&dt)?),
            None => None,
        };

        // 3. Zero seed at B·T — no state feedback between positions. Only the
        //    two recurrent carries survive from `current_state` (the Python
        //    `_NEVER_FOLDED` pair); `prev_bw` deliberately does NOT carry.
        let seed = StateSpaceOutput::stochastic_zero_state(&xf, &self.config, &self.device, bt)?;

        // 4. ONE noise bundle, salt 0 — identical draws across the fold, so
        //    any bundle-reading site is pinned (the Dirichlet's Deterministic
        //    mode uses the uniform median, which is stronger still).
        let bundle =
            self.sampler
                .draw_single(&noise_site_shapes(&self.config, bt), &self.device, 0)?;

        // 5. Encoder ONCE, eval-forced: z = mu_z, logvar_z still emitted.
        let enc_out = self.encoder.forward_with_training(
            false,
            &xf,
            t_cond.as_ref(),
            &seed,
            indices,
            Some(&bundle),
        )?;

        // 6. Restore the time axis on the posterior means.
        let latent_dim = self.config.latent_dim;
        let mu_win = enc_out.mu_z.reshape((b, t_win, latent_dim))?;

        // 7. EWMA over the window: momentum context ADDED to the SSM input
        //    (never substituted — grad flows through mu_win; the bank is
        //    detached by construction and exactly zero on a constant path).
        let (ssm_in, ewma_win) = if self.ewma_enabled {
            let ctx = self
                .ewma_encoder
                .window_momentum(&mu_win)?
                .affine(self.ewma_encoder.ctx_gain(), 0.0)?;
            (
                mu_win.add(&ctx.to_dtype(mu_win.dtype())?)?,
                Some(self.ewma_encoder.encode_window(&mu_win)?),
            )
        } else {
            (mu_win.clone(), None)
        };

        // 8. ONE L=T SSM scan with the carried state. Equivalent to T
        //    stateful L=1 calls: the conv cache carries the k−1 history and
        //    the h-scan is sequential inside the block.
        let carried_ssm = current_state.and_then(|s| s.ssm_state.clone());
        let (z_win, ssm_out) = match &self.latent_ssm {
            Some(ssm) => {
                let (out, st) = ssm.forward(&ssm_in, carried_ssm)?;
                (out, Some(st))
            }
            None => (ssm_in, current_state.and_then(|s| s.ssm_state.clone())),
        };

        // 9. Dirichlet ONCE, eval-forced, on the folded post-SSM latents.
        let mut enc_for_dir = enc_out;
        let z_flat = z_win.reshape((bt, latent_dim))?;
        enc_for_dir.z = z_flat.clone();
        let dir_out = self.dirichlet.forward_with_mode(
            DirichletMode::Deterministic,
            &z_flat,
            t_cond.as_ref(),
            &enc_for_dir,
            indices,
            1,
            0,
            false,
            Some(&bundle),
            None, // prev_bw: no carry across the fold — only ssm/ewma carry
        )?;

        // 10. Decoder ONCE, eval-forced.
        let step_indices = match indices {
            Some(ix) => ix.clone(),
            None => dir_out.pi.argmax(candle_core::D::Minus1)?,
        };
        let dec_out = self.decoder.forward_with_mode(
            DecoderMode::Deterministic,
            &dir_out.features,
            t_cond.as_ref(),
            &dir_out,
            Some(&step_indices),
            1,
            0,
            false,
            Some(&bundle),
        )?;

        // 11. Per-position routing moments at B·T (window MEAN applied in the
        //     collapse below — Python `_window_mean`, not `[:, -1]`).
        let moments = match &dec_out.counterfactual_routing {
            Some(r) => Some(crate::model::moments::GaussianMoments::from_routing(r)?),
            None => None,
        };

        // Assemble at B·T with the SAME literal both paths share, then
        // unfold + collapse.
        let flat = StateSpaceOutput {
            recon: dec_out.recon,
            recon_var: Some(dec_out.recon_var.clone()),
            gp_features: dec_out.gp_features,
            pi: dec_out.pi,
            alpha: dec_out.alpha,
            alpha_mu: dec_out.alpha_mu,
            alpha_factor: dec_out.alpha_factor,
            alpha_diag: dec_out.alpha_diag,
            bottleneck: dec_out.bottleneck,
            parameters_per_expert: dec_out.parameters_per_expert,
            bandwidth_mod: dec_out.bandwidth_mod,
            amp: dec_out.amp,
            trend: dec_out.trend,
            res: dec_out.res,
            ls: dec_out.ls,
            mu_z: dec_out.mu_z,
            logvar_z: dec_out.logvar_z,
            lmc_matrices: dec_out.lmc_matrices,
            real_x: enc_for_dir.real_x,
            gates: dec_out.gates,
            mat52: dec_out.mat52,
            rbf: dec_out.rbf,
            rq: dec_out.rq,
            lmc_consensus: dec_out.lmc_consensus,
            gates_logits: dec_out.gates_logits.clone(),
            z: Some(z_flat),
            z_history: None,
            feature_history: None,
            z_mu_history: None,
            z_ewma_history: None,
            ewma_state: None,
            ssm_state: None,
            diffusion_t: None,
            counterfactual_bottlenecks: dec_out.counterfactual_bottlenecks,
            counterfactual_latents: dec_out.counterfactual_latents,
            counterfactual_routing: dec_out.counterfactual_routing,
            counterfactual_routing_logits: dec_out.counterfactual_routing_logits,
            moe_weights: dec_out.moe_weights.clone(),
            moments,
            ls_logvar: Some(dec_out.ls_logvar),
            bw_learned: Some(dec_out.bw_learned),
            global_divergence: None,
            local_divergence: None,
            inverse_wishart: None,
            lengthscale_kl: None,
            recon_kl: None,
            recon_likelihood: dec_out.recon_likelihood.clone(),
            recon_nll_per_sample: dec_out.recon_nll_per_sample.clone(),
            moe_recon_nll: dec_out.moe_recon_nll.clone(),
            bandwidth_loss_term: dir_out.bandwidth_loss_term.clone(),
            kumaraswamy_global: Dirichlet::kumaraswamy_global(&self.dirichlet, bundle.get("global_stick"))?,
        };

        let mut state = unfold_and_collapse(flat, b, t_win)?;

        // Explicit overrides AFTER the sweep (vae.py:862-877):
        state.z = Some(z_win.narrow(1, t_win - 1, 1)?.squeeze(1)?);
        state.z_history = Some(z_win);
        state.z_mu_history = Some(mu_win);
        state.z_ewma_history = ewma_win;
        // ssm_state IS advanced; ewma_state is passed through NOT advanced —
        // the bank is stateless across windows on the DAG path
        // (`window_momentum`/`encode_window` re-seed at μ₀ every call).
        state.ssm_state = ssm_out;
        state.ewma_state = current_state.and_then(|s| s.ewma_state.clone());
        state.diffusion_t = current_state.and_then(|s| s.diffusion_t.clone());

        debug_assert!(
            state.z_history.as_ref().map(|z| z.dims()[..2] == [b, t_win]).unwrap_or(false),
            "window contract: z_history must be [B, T, ..]"
        );
        Ok(state)
    }

    /// Python `model.train()` / `model.eval()` twin: flips the encoder's
    /// training flag (reparameterised bundle-seeded draws vs the
    /// deterministic `z = mu_z` eval path). The Dirichlet's equivalent knob
    /// is its `DirichletMode`, set at construction.
    pub fn set_training(&mut self, training: bool) {
        self.encoder.training = training;
    }

    pub fn load(
        path: &std::path::Path,
        root: RootConfig,                 // ← broad; load needs to read sub-configs
        dirichlet_mode: DirichletMode,
        decoder_mode: DecoderMode,
        device: Device,
    ) -> anyhow::Result<Self> {
        
        
        // Checkpoint-tolerant: strips a `vae.` prefix and FOLDS weight/
        // spectral-norm parametrizations at load time, so both the cleaned
        // export and the raw training checkpoint serve.

        let cleaned = crate::utils::load_weights::load_cleaned_map(path, &device, Some("vae."))?;
        Self::load_from_cleaned_map(cleaned, root, dirichlet_mode, decoder_mode, device)
    }

    /// [`load`](Self::load) over in-memory safetensors bytes — the queue /
    /// blob-store path (dk-inference), where weights arrive as `Vec<u8>`
    /// rather than a file. Identical fold/strip/reconcile behaviour.
    pub fn load_from_bytes(
        bytes: &[u8],
        root: RootConfig,
        dirichlet_mode: DirichletMode,
        decoder_mode: DecoderMode,
        device: Device,
    ) -> anyhow::Result<Self> {
        let cleaned =
            crate::utils::load_weights::load_cleaned_map_from_bytes(bytes, &device, Some("vae."))?;
        Self::load_from_cleaned_map(cleaned, root, dirichlet_mode, decoder_mode, device)
    }

    /// The shared load core: reconcile config dims from the (already folded)
    /// checkpoint map — the checkpoint is the source of truth for weight
    /// SHAPES — then build over a tensor-backed VarBuilder.
    fn load_from_cleaned_map(
        cleaned: std::collections::HashMap<String, Tensor>,
        root: RootConfig,
        dirichlet_mode: DirichletMode,
        decoder_mode: DecoderMode,
        device: Device,
    ) -> anyhow::Result<Self> {
        // Reconcile `input_dim` with the checkpoint. The checkpoint is the
        // source of truth for weight SHAPES — the config's job is the
        // non-structural knobs (modes, thresholds). Serving without a
        // config.yaml falls back to `INPUT_DIM` defaults, so a model trained
        // on a different feature count would otherwise blow up at
        // `encoder.stem.0.weight` ("shape mismatch … expected [32,128,7],
        // got [32,30,7]"). The stem conv weight is `[32, input_dim, 7]`, so
        // its dim-1 IS the trained input_dim; override all four sub-configs
        // (encoder stem + decoder recon heads both key off it) and warn.
        let mut root = root;
        if let Some(stem) = cleaned.get("encoder.stem.0.weight") {
            if let Ok(ckpt_input_dim) = stem.dim(1) {
                if ckpt_input_dim != root.vae.input_dim {
                    tracing::warn!(
                        config = root.vae.input_dim,
                        checkpoint = ckpt_input_dim,
                        "input_dim mismatch — overriding config from the checkpoint stem \
                         (serve with a matching config.yaml to silence this)"
                    );
                    root.vae.input_dim = ckpt_input_dim;
                    root.encoder.input_dim = ckpt_input_dim;
                    root.decoder.input_dim = ckpt_input_dim;
                    root.dirichlet.input_dim = ckpt_input_dim;
                }
            }
        }

        // Reconcile the Dirichlet latent window `T` the same way: the
        // bottleneck mixer's weight is `[bottleneck_dim, latent_dim·T]`, so
        // its dim-1 divided by latent_dim IS the trained window (1 = legacy
        // single-step, 12 = the Itô-window mixer). The two copies of the
        // knob — the `vae` block gates the refinement-loop windowing, the
        // `dirichlet` block sizes the mixer — MUST agree, so set both.
        if let Some(mix) = cleaned.get("dirichlet.bottleneck_mixer.0.weight") {
            let lat = root.dirichlet.latent_dim.max(1);
            if let Ok(mixer_in) = mix.dim(1) {
                if mixer_in % lat == 0 {
                    let t_win = mixer_in / lat;
                    if t_win != root.dirichlet.dirichlet_t_window {
                        tracing::warn!(
                            config = root.dirichlet.dirichlet_t_window,
                            checkpoint = t_win,
                            "dirichlet_t_window mismatch — overriding config from the bottleneck mixer"
                        );
                        root.vae.dirichlet_t_window = t_win;
                        root.dirichlet.dirichlet_t_window = t_win;
                    }
                }
            }
        }

        // Reconcile the time-conditioning widths. The whole trained stack is
        // constructed at ONE `time_dim` (Python `config.model.time_dim`,
        // default 64 — NOT the 256 the AmortisedDirichlet signature default
        // suggests): the FourierTimeEmbedding's embed_dim and every FiLM
        // head's input width. `time_emb.mlp.0.weight` is
        // `[time_dim, 2·n_features]`; the film weights are
        // `[2·bottleneck, time_dim]` — the checkpoint is the source of truth
        // (a hardcoded 256 died with "shape mismatch for
        // dirichlet.kernel_network.film_projector.weight,
        // expected [128, 256], got [128, 64]").
        if let Some(mlp0) = cleaned.get("time_emb.mlp.0.weight") {
            if let (Ok(embed), Ok(two_n)) = (mlp0.dim(0), mlp0.dim(1)) {
                if embed != root.vae.time_dim || two_n / 2 != root.vae.time_rff_dim {
                    tracing::warn!(
                        config_embed = root.vae.time_dim,
                        config_rff = root.vae.time_rff_dim,
                        checkpoint_embed = embed,
                        checkpoint_rff = two_n / 2,
                        "time embedding dims mismatch — overriding config from time_emb.mlp.0"
                    );
                    root.vae.time_dim = embed;
                    root.vae.time_rff_dim = two_n / 2;
                }
            }
        }
        for (key, slot) in [
            ("encoder.film_projector.weight", &mut root.encoder.time_dim),
            ("decoder.film_projector.weight", &mut root.decoder.time_dim),
            ("dirichlet.film_projector.weight", &mut root.dirichlet.time_dim),
            (
                "dirichlet.kernel_network.film_projector.weight",
                &mut root.kernel_network.time_dim,
            ),
        ] {
            if let Some(w) = cleaned.get(key) {
                if let Ok(t_in) = w.dim(1) {
                    if t_in != *slot {
                        tracing::warn!(key, config = *slot, checkpoint = t_in,
                            "film_projector time_dim mismatch — overriding config from the checkpoint");
                        *slot = t_in;
                    }
                }
            }
        }

        let vb = VarBuilder::from_tensors(cleaned, DType::F32, &device);
        Self::build(vb, root, dirichlet_mode, decoder_mode, device)
    }

    /// Construct from ANY [`VarBuilder`] — the trainable path: hand in a
    /// `VarBuilder::from_varmap` builder and every weight registers as a
    /// `Var` the optimizer islands can step (warm-start by `Var::set` from a
    /// cleaned checkpoint map afterwards). [`load`](Self::load) is this over
    /// a folded tensor-backed builder.
    pub fn build(
        vb: VarBuilder,
        root: RootConfig,
        dirichlet_mode: DirichletMode,
        decoder_mode: DecoderMode,
        device: Device,
    ) -> anyhow::Result<Self> {
        let kernel_network = KernelNetwork::load(
            root.kernel_network.clone(),
            vb.pp("dirichlet").pp("kernel_network"),
        )?;
        let encoder   = ConvolutionalLoopEncoder::load(vb.pp("encoder"), root.encoder.clone())?;
        let dirichlet = AmortisedDirichlet::<KernelNetwork>::load(
            root.dirichlet.clone(), dirichlet_mode, vb.pp("dirichlet"), kernel_network,
        )?;
        let decoder   = SpectralDecoder::load(root.decoder.clone(), decoder_mode, vb.pp("decoder"))?;

        // Master Fourier time embedding (`time_emb.mlp.{0,2}` + `omega_t`
        // buffer) — the Python `self.time_emb` producing the `time_dim`-wide
        // conditioning (dims reconciled from the checkpoint in `load`).
        // Absent on pre-migration checkpoints (whose z-additive `time_net`
        // MLP the current refinement flow no longer applies).
        // ALWAYS built on a trainable map (Python registers it
        // unconditionally); `None` only on pre-migration tensor-backed
        // checkpoints (whose z-additive `time_net` MLP the current
        // refinement flow no longer applies).
        let time_emb = crate::model::parametrize::build_or_absent(
            &vb,
            "time_emb.mlp.0.weight",
            || {
                FourierTimeEmbedding::new(
                    vb.pp("time_emb"),
                    root.vae.time_dim,
                    root.vae.time_rff_dim,
                    10.0,
                )
            },
        )?;

        // Mamba latent SSM (encoder_z → Dirichlet boundary) — identity at
        // init (zero out_proj). ALWAYS built on a trainable map; absent only
        // on tensor-backed checkpoints from before it existed.
        let latent_ssm = crate::model::parametrize::build_or_absent(
            &vb,
            "latent_ssm.norm.weight",
            || {
                LatentSSM::with_config(
                    vb.pp("latent_ssm"),
                    root.vae.latent_dim,
                    crate::model::mamba::MambaConfig::default(),
                    (root.vae.ssm_predict_dim > 0).then(|| {
                        crate::model::mamba::PredictorConfig {
                            predict_dim: root.vae.ssm_predict_dim,
                            temp: root.vae.ssm_predict_temp,
                            horizons: root.vae.ssm_predict_horizons.clone(),
                            ..Default::default()
                        }
                    }),
                )
            },
        )?;

        let ewma = root.ewma.clone();
        Ok(Self::new(encoder, latent_ssm, dirichlet, decoder, time_emb, root.vae, device)
            .with_ewma(&ewma)?)
    }
}

impl<E: Encoder, Dm: Dirichlet, Dc: Decoder> SpectralVae<E, Dm, Dc> {
    /// Port of the Python `refinement_loop` — the data flow matches step for
    /// step:
    ///
    /// 1. per-Itô-step deterministic noise bundle (`draw_single`, salt =
    ///    step index; a caller-supplied `noise_bundle` — e.g. an external
    ///    IWAE draw — overrides every step),
    /// 2. dream gating on the refinement-pass index / TRAILING-aligned input
    ///    read (consume the LAST `steps` rows so the window ENDS at the
    ///    current/labeled step),
    /// 3. encoder → Mamba latent SSM (state carried across steps) →
    ///    Dirichlet boundary → decoder,
    /// 4. routing-moment accumulation (count-weighted running mean, fresh
    ///    per call) + per-step latent history,
    /// 5. final fixed-`T_HISTORY` `z_history` window, left-padded by
    ///    repeating the earliest step.
    ///
    /// `t` is the diffusion/horizon time (the Python `t_scalar`) — constant
    /// across the refinement steps, embedded once by the master Fourier
    /// timer.
    pub fn refinement_loop(
        &self,
        x: &Tensor,
        t: f64,
        certainty: RefinementCertainty,
        indices: Option<&Tensor>,
        current_state: Option<StateSpaceOutput>,
        generative_mode: bool,
        noise_bundle: Option<&NoiseBundle>,
    ) -> Result<StateSpaceOutput> {
        let steps = certainty.steps();
        let batch_size = x.dim(0)?;

        // Bring up initial state if the caller didn't supply one.
        let mut state = match current_state {
            Some(s) => s,
            None => StateSpaceOutput::stochastic_zero_state(x, &self.config, &self.device, batch_size)?,
        };

        // Python: dream_steps = max(1, int(0.75 * steps))
        let dream_steps = ((0.75_f32 * steps as f32) as usize).max(1);

        // Python: x_seq = x if x.dim() == 3 else x.unsqueeze(1)
        let x_seq = if x.dims().len() == 3 {
            x.clone()
        } else {
            x.unsqueeze(1)?
        };
        let seq_len = x_seq.dim(1)?;

        // Master time conditioning: dt = [B, 1] filled with t_scalar →
        // FourierTimeEmbedding → [B, 256]. This is the `dt` the Python
        // passes into enc/dir/dec — the Dirichlet FiLMs on it (its own
        // projector + the kernel network's); the encoder/decoder ports have
        // no time input yet.
        let dt = Tensor::full(t as f32, (batch_size, 1), &self.device)?
            .to_dtype(x_seq.dtype())?;
        let t_cond = match &self.time_emb {
            Some(f) => Some(f.forward(&dt)?),
            None => None,
        };

        // Carried across refinement ticks: the Mamba SSM recurrence, the
        // per-step latents (→ the fixed-T z_history window after the loop),
        // and the count-weighted routing moments (fresh accumulator each
        // call, like the Python `acc_moments = None`).
        // Both recurrent carries are seeded from the INCOMING state (Python's
        // _NEVER_FOLDED pair): a caller that threads state across calls keeps
        // the recurrence; one that does not gets a cold start rather than
        // another call's leftovers.
        let mut ssm_state: Option<SsmState> = state.ssm_state.clone();
        let mut ewma_state: Option<Tensor> = state.ewma_state.clone();
        let mut z_mu_hist: Vec<Tensor> = Vec::with_capacity(steps);
        let mut ewma_hist: Vec<Tensor> = Vec::with_capacity(steps);
        // The bandwidth carry, threaded across steps exactly like `ssm_state`
        // (Python's `_NEVER_FOLDED = {ssm_state, ewma_state}` discipline). It
        // used to live in a `Mutex` on the Dirichlet, which made every forward
        // a function of call history — see `bw_with_previous`. Seeded from the
        // incoming state so a caller that threads state across `refinement_loop`
        // invocations keeps the carry, and one that does not gets a cold start
        // rather than another call's leftovers.
        let mut prev_bw: Option<Tensor> = state.bw_learned.clone();
        let mut z_hist: Vec<Tensor> = Vec::with_capacity(steps);
        let mut acc_moments: Option<crate::model::moments::GaussianMoments> = None;
        let mut feat_history = state.feature_history.clone();

        for step_idx in 0..steps {
            // --- deterministic reparam noise for THIS Itô step -----------
            // Every site seeded via `site_seed` (reproducible;
            // common-random-numbers across forwards → low gradient
            // variance), distinct per step via salt = step_idx. A
            // caller-supplied bundle (e.g. an external IWAE draw) overrides.
            let step_bundle = match noise_bundle {
                Some(nb) => nb.clone(),
                None => self.sampler.draw_single(
                    &noise_site_shapes(&self.config, batch_size),
                    &self.device,
                    step_idx as u64,
                )?,
            };

            // --- pick next input ----------------------------------------
            // Dream gating uses the refinement-pass index, not the horizon
            // time. Otherwise: TRAILING-aligned read — consume the LAST
            // `steps` rows of the input window so z_history ENDS at the
            // current/labeled step (a head-aligned read would leave the
            // label row outside the receptive field).
            let x_t = if generative_mode && step_idx > 0 && step_idx % (dream_steps + 1) != 0 {
                state.recon.clone()
            } else {
                let row = seq_len as isize - steps as isize + step_idx as isize;
                if row >= 0 && (row as usize) < seq_len {
                    x_seq.narrow(1, row as usize, 1)? // [B, 1, input_dim]
                } else {
                    state.recon.clone()
                }
            };

            // --- encoder ------------------------------------------------
            let mut encoder_out = self
                .encoder
                .forward(&x_t, t_cond.as_ref(), &state, indices, Some(&step_bundle))?;

            // --- Mamba latent SSM (encoder_z → Dirichlet boundary) -------
            // Streaming mode: one [B, D] step per refinement tick, carrying
            // the (ssm_state, conv_cache) across ticks so the SSM models the
            // z_0 → z_1 → … trajectory. Identity at init. The SSM runs on
            // the RAW encoder latent (no z-additive time MLP — the master
            // timer conditions the submodules instead).
            // --- EWMA bank: streaming step on the PRE-SSM posterior mean.
            // Python vae.py:567-576. The context is ADDED to the SSM input,
            // never substituted for it: `encoder_out.z` carries the gradient;
            // `ctx` is a detached momentum bias beside it (the bank detaches
            // internally), exactly zero on a constant path.
            z_mu_hist.push(encoder_out.mu_z.clone());
            let ssm_in = if self.ewma_enabled {
                let (feats, ctx, new_ewma) =
                    self.ewma_encoder.step(&encoder_out.mu_z, ewma_state.as_ref())?;
                ewma_hist.push(feats);
                ewma_state = Some(new_ewma);
                encoder_out.z.add(&ctx.to_dtype(encoder_out.z.dtype())?)?
            } else {
                encoder_out.z.clone()
            };

            let z_ssm = match &self.latent_ssm {
                Some(ssm) => {
                    let (out, s) = ssm.forward(&ssm_in, ssm_state.take())?;
                    ssm_state = Some(s);
                    out
                }
                None => ssm_in,
            };
            // Python: encoder_out = encoder_out._replace(z=z_ssm)
            encoder_out.z = z_ssm.clone();
            z_hist.push(z_ssm.clone()); // [B, latent_dim] this Itô step

            // --- dirichlet ----------------------------------------------
            // PER-STEP boundary input (the Python flow): the Dirichlet now
            // operates in the T=12 Itô time space itself — the windowed GP
            // path fans [B, T, D] → [B·T, D] through it, and this rollout
            // feeds one step at a time. (Legacy windowed-mixer checkpoints
            // are tiled up to `latent_dim·T` inside the module.)
            let dirichlet_out = self.dirichlet.forward(
                &z_ssm,
                t_cond.as_ref(),
                &encoder_out,
                indices,
                1,
                step_idx,
                generative_mode,
                Some(&step_bundle),
                prev_bw.as_ref(),
            )?;
            prev_bw = Some(dirichlet_out.bw_learned.clone());

            // --- decoder indices:  argmax(pi) unless caller forced them -
            // Python: dream_indices = argmax(pi); step_indices = indices or dream_indices
            let step_indices = match indices {
                Some(ix) => ix.clone(),
                None => dirichlet_out.pi.argmax(candle_core::D::Minus1)?,
            };

            // --- decoder ------------------------------------------------
            let decoder_out = self.decoder.forward(
                &dirichlet_out.features,
                t_cond.as_ref(),
                &dirichlet_out,
                Some(&step_indices),
                1,
                step_idx,
                generative_mode,
                Some(&step_bundle),
            )?;

            // --- routing-moment accumulation -----------------------------
            // Python: moments = routing_moments(decoder_out);
            //         acc_moments = accumulate_moments(acc_moments, moments)
            // A routing-less step contributes nothing; the first real step
            // becomes the estimate.
            let step_moments = match &decoder_out.counterfactual_routing {
                Some(r) => Some(crate::model::moments::GaussianMoments::from_routing(r)?),
                None => None,
            };
            acc_moments = crate::model::moments::GaussianMoments::accumulate(acc_moments, step_moments)?;

            // --- per-step 134-wide packed kernel features → rolling window
            // (the probit/binary head's input; see `pack_features` for the
            // layout). Rust-side extra: the Python assembles this window in
            // `_assemble_state`.
            let x134 = Tensor::cat(
                &[
                    &decoder_out.gates,
                    &decoder_out.mat52,
                    &decoder_out.rbf,
                    &decoder_out.rq,
                    &decoder_out.pi,
                ],
                candle_core::D::Minus1,
            )?;
            feat_history = Some(StateSpaceOutput::push_z_history(
                feat_history.as_ref(),
                &x134,
                T_HISTORY,
            )?);

            // --- unified assembly: pull each field from its true source --
            state = StateSpaceOutput {
                recon: decoder_out.recon,
                recon_var: Some(decoder_out.recon_var.clone()),
                gp_features: decoder_out.gp_features,
                pi: decoder_out.pi,
                alpha: decoder_out.alpha,
                alpha_mu: decoder_out.alpha_mu,
                alpha_factor: decoder_out.alpha_factor,
                alpha_diag: decoder_out.alpha_diag,
                bottleneck: decoder_out.bottleneck,
                parameters_per_expert: decoder_out.parameters_per_expert,
                bandwidth_mod: decoder_out.bandwidth_mod,
                amp: decoder_out.amp,
                trend: decoder_out.trend,
                res: decoder_out.res,
                ls: decoder_out.ls,
                mu_z: decoder_out.mu_z,
                logvar_z: decoder_out.logvar_z,
                lmc_matrices: decoder_out.lmc_matrices,
                real_x: encoder_out.real_x,
                gates: decoder_out.gates,
                mat52: decoder_out.mat52,
                rbf: decoder_out.rbf,
                rq: decoder_out.rq,
                lmc_consensus: decoder_out.lmc_consensus,

                gates_logits: decoder_out.gates_logits.clone(),
                // The SSM'd latent — the Python state's z after the
                // `encoder_out._replace(z=z_ssm)`.
                z: Some(z_ssm.clone()),
                // Stamped ONCE after the loop (Python's final `_replace`) —
                // the fixed-T left-padded stack of this call's z_hist.
                z_history: None,
                feature_history: feat_history.clone(),
                // Histories + carries are stamped ONCE after the loop, like
                // z_history (Python writes them in the final _replace).
                z_mu_history: None,
                z_ewma_history: None,
                ewma_state: None,
                ssm_state: None,
                diffusion_t: state.diffusion_t.clone(),
                counterfactual_bottlenecks: decoder_out.counterfactual_bottlenecks,
                counterfactual_latents: decoder_out.counterfactual_latents,
                counterfactual_routing: decoder_out.counterfactual_routing,
                counterfactual_routing_logits: decoder_out.counterfactual_routing_logits,
                // Dense MoE responsibilities — now DECODER-computed (softmax
                // over the first num_experts gate logits, renormalised) so it
                // matches the mixture the moe_recon NLL actually used. The old
                // fold recomputed softmax over ALL gates, which silently
                // diverged from the recon mixture whenever num_gates≠num_experts.
                moe_weights: decoder_out.moe_weights.clone(),
                moments: acc_moments.clone(),
                ls_logvar: Some(decoder_out.ls_logvar),
                bw_learned: Some(decoder_out.bw_learned),
                // KL stashes stay None on the serving path (the KL-enabled
                // forwards fill them); the global stick container is cheap —
                // recomputed statelessly from the raws each fold.
                global_divergence: None,
                local_divergence: None,
                inverse_wishart: None,
                lengthscale_kl: None,
                recon_kl: None,
                // Heteroscedastic recon NLL + dense-MoE mixture NLL now
                // computed by the decoder forward (were dead None stubs).
                recon_likelihood: decoder_out.recon_likelihood.clone(),
                recon_nll_per_sample: decoder_out.recon_nll_per_sample.clone(),
                moe_recon_nll: decoder_out.moe_recon_nll.clone(),
                // The Dirichlet's [current ‖ previous] bandwidth stash (bw_32).
                bandwidth_loss_term: dirichlet_out.bandwidth_loss_term.clone(),
                kumaraswamy_global: Dirichlet::kumaraswamy_global(&self.dirichlet, step_bundle.get("global_stick"))?,
            };
        }

        // Rolling window of the last T_HISTORY per-step latents →
        // [B, T_HISTORY, latent_dim] — the explicit Itô-time axis the GP
        // consumes. Left-pad by repeating the earliest step when fewer than
        // T_HISTORY steps ran, so the window shape is always fixed (avoids
        // downstream shape confusion).
        if !z_hist.is_empty() {
            let start = z_hist.len().saturating_sub(T_HISTORY);
            let tail = &z_hist[start..];
            if tail.len() < T_HISTORY {
                // The padding below repeats `tail[0]`, so the window carries
                // fewer than T_HISTORY DISTINCT positions. That is not a
                // cosmetic shape fix: every consumer reads the duplicates as
                // real time steps, so the GP's T axis loses signal and the Itô
                // Savitzky-Golay drift target collapses toward zero — silently,
                // with finite losses throughout. Warn ONCE per run (the
                // one-shot latch idiom used by `DynamicalSystem`), because a
                // per-forward warning on this path is noise that gets filtered,
                // which is exactly how this went unnoticed.
                //
                // Under `deterministic_dag` this call is the 1-step COMPANION,
                // whose window is never published — the warning then describes
                // a state nothing consumes.
                static PAD_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !PAD_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(
                        distinct_steps = tail.len(),
                        t_history = T_HISTORY,
                        "z_history left-padded: only {} of {} window positions are \
                         distinct. Use RefinementCertainty::Full on any training or \
                         expert-pool path; anything less trains against a window with \
                         no time variation.",
                        tail.len(),
                        T_HISTORY,
                    );
                }
            }
            state.z_history = Some(pad_window(tail, T_HISTORY)?);
        }
        // The sibling histories, same left-pad discipline (Python
        // `_pad_window` on all three). The EWMA history only exists when the
        // bank is enabled.
        if !z_mu_hist.is_empty() {
            let start = z_mu_hist.len().saturating_sub(T_HISTORY);
            state.z_mu_history = Some(pad_window(&z_mu_hist[start..], T_HISTORY)?);
        }
        if !ewma_hist.is_empty() {
            let start = ewma_hist.len().saturating_sub(T_HISTORY);
            state.z_ewma_history = Some(pad_window(&ewma_hist[start..], T_HISTORY)?);
        }
        // Recurrent carries written UNCONDITIONALLY (Python vae.py:634-635) —
        // before these fields existed the Rust loop dropped the SSM recurrence
        // at every return.
        state.ssm_state = ssm_state;
        state.ewma_state = ewma_state;

        Ok(state)
    }

    /// Default entry point matching the Python `forward` — equivalent to
    /// calling `refinement_loop` at diffusion time 0 with the supplied
    /// `certainty`, no generative fallback, and sampler-drawn noise.
    pub fn forward(
        &self,
        x: &Tensor,
        vae_out: Option<StateSpaceOutput>,
        indices: Option<&Tensor>,
        certainty: RefinementCertainty,
    ) -> Result<StateSpaceOutput> {
        self.refinement_loop(x, 0.0, certainty, indices, vae_out, false, None)
    }

    /// Port of the Python `sample_variational_posterior`: run the encoder to
    /// get `q(z | x)`, then optionally refine `z` with `n_langevin_steps` of
    /// unadjusted Langevin against the CLOSED-FORM variational score
    ///
    /// ```text
    ///   ∇_z log q(z | x) = −(z − μ_z) / σ_z²
    /// ```
    ///
    /// No autodiff: the score of a diagonal Gaussian is two ops. At `B = 1`,
    /// `D = 16` a step is ~1.4 µs and 16 steps measure ~24 µs off a pre-drawn
    /// noise pool, so Langevin refinement of ONE latent is genuinely cheap.
    ///
    /// # `certainty` is `Full` for a reason
    ///
    /// The Python passes `steps=None`, which means `T_HISTORY` refinements —
    /// NOT one. A single step stamps a DEGENERATE `z_history` (the current `z`
    /// repeated `T_HISTORY` times) onto `last_state`, which blinds every
    /// subsequent `forward_gp`: the GP's `T` axis carries no signal and the
    /// Itô Savitzky–Golay drift target is identically zero. It does not error.
    /// Callers that genuinely only read `z`/`mu_z` may pass `Quick`; anything
    /// touching the window must not.
    ///
    /// Noise is Sobol (`sobol_randn_like`), not `Tensor::randn`: the vendored
    /// candle CPU backend ignores `Device::set_seed`, so the QMC path is the
    /// only reproducible one.
    pub fn sample_variational_posterior(
        &self,
        x: &Tensor,
        t: f64,
        n_langevin_steps: usize,
        step_size: f64,
        certainty: RefinementCertainty,
        noise_bundle: Option<&NoiseBundle>,
        seed: u64,
    ) -> Result<StateSpaceOutput> {
        let state = self.refinement_loop(x, t, certainty, None, None, false, noise_bundle)?;
        if n_langevin_steps == 0 {
            return Ok(state);
        }
        let mu_z = state.mu_z.clone();
        let var_z = state.logvar_z.exp()?;
        let mut z = state
            .z
            .clone()
            .ok_or_else(|| anyhow!("sample_variational_posterior: state.z is None"))?;
        let dims = z.dims().to_vec();
        let sq = (2.0 * step_size).sqrt();
        for k in 0..n_langevin_steps {
            // score = −(z − μ_z)/σ_z², closed form.
            let score = z.sub(&mu_z)?.div(&var_z)?.affine(-1.0, 0.0)?;
            let eta = crate::model::noise::sobol_randn_like(
                &dims,
                z.device(),
                seed.wrapping_add(k as u64),
            )?;
            z = z
                .add(&score.affine(step_size, 0.0)?)?
                .add(&eta.to_dtype(z.dtype())?.affine(sq, 0.0)?)?;
        }
        Ok(StateSpaceOutput { z: Some(z), ..state })
    }

    /// Port of the Python `langevin_joint_sample`: joint unadjusted Langevin on
    /// the latent `z` AND the Dirichlet evidence `α`, both against closed-form
    /// scores.
    ///
    /// `z` uses the diagonal variational score as above. `α` is low-rank
    /// Gaussian — `Σ = diag(alpha_diag) + alpha_factor·alpha_factorᵀ` — so its
    /// score `−Σ⁻¹(α − μ_α)` comes from
    /// [`lowrank_score`](crate::model::gaussian::lowrank_score), the Woodbury
    /// identity, rather than forming or inverting `Σ`. That is the Rust
    /// equivalent of the Python `_lowrank_inverse`, and it already exists.
    ///
    /// Both updates share one Sobol stream with distinct site salts so the two
    /// noise sequences are independent but reproducible.
    pub fn langevin_joint_sample(
        &self,
        state: &StateSpaceOutput,
        n_steps: usize,
        eps: f64,
        seed: u64,
    ) -> Result<StateSpaceOutput> {
        if n_steps == 0 {
            return Ok(state.clone());
        }
        let mu_z = state.mu_z.clone();
        let var_z = state.logvar_z.exp()?;
        let mut z = state
            .z
            .clone()
            .ok_or_else(|| anyhow!("langevin_joint_sample: state.z is None"))?;
        let mut alpha = state.alpha.clone();
        let mu_alpha = state.alpha_mu.clone();
        let (a_diag, a_fac) = (state.alpha_diag.clone(), state.alpha_factor.clone());

        let z_dims = z.dims().to_vec();
        let a_dims = alpha.dims().to_vec();
        let sq = (2.0 * eps).sqrt();
        let z_salt = crate::model::noise::site_seed("latent_z", seed);
        let a_salt = crate::model::noise::site_seed("alpha_joint", seed);

        for k in 0..n_steps {
            let s_z = z.sub(&mu_z)?.div(&var_z)?.affine(-1.0, 0.0)?;
            let e_z = crate::model::noise::sobol_randn_like(
                &z_dims,
                z.device(),
                z_salt.wrapping_add(k as u64),
            )?;
            z = z
                .add(&s_z.affine(eps, 0.0)?)?
                .add(&e_z.to_dtype(z.dtype())?.affine(sq, 0.0)?)?;

            // −Σ⁻¹(α − μ_α) via Woodbury; never forms Σ.
            let s_a = crate::model::gaussian::lowrank_score(&alpha, &mu_alpha, &a_diag, &a_fac)?;
            let e_a = crate::model::noise::sobol_randn_like(
                &a_dims,
                alpha.device(),
                a_salt.wrapping_add(k as u64),
            )?;
            alpha = alpha
                .add(&s_a.affine(eps, 0.0)?)?
                .add(&e_a.to_dtype(alpha.dtype())?.affine(sq, 0.0)?)?;
        }
        Ok(StateSpaceOutput { z: Some(z), alpha, ..state.clone() })
    }

    /// Lite-bridge avenue — port of the Python `encode_latent`: re-encode a
    /// LATENT `z` through the SSM + Dirichlet boundary WITHOUT the conv
    /// encoder. This is what the SDE bridge calls between Itô steps to read
    /// the zt-conditioned posterior π off an evolving latent.
    ///
    /// `z`: `[B, latent_dim]` or `[B, L, latent_dim]` (the last step is
    /// taken — strictly one step feeds the streaming SSM). `vae_out` carries
    /// the encoder evidence when the caller has it (bridge states usually
    /// do); `None` substitutes empty tensors, for which the Dirichlet's
    /// documented safe defaults kick in.
    ///
    /// Returns the `DirichletOutput` with `z_in` replaced by the SSM'd
    /// latent and `real_x` by the input latent (`[B, 1, latent_dim]`) — the
    /// Python `dirichlet_out._replace(z=z_ssm, real_x=z)`.
    /// Hold (`false`) or advance (`true`) the power iteration of every
    /// `SpectralNormLinear` in the model.
    ///
    /// `false` is PyTorch's `eval()` semantics and is what a serving, decode or
    /// parity path wants: without it the σ estimate advances one iteration per
    /// forward, so repeated evaluation of the SAME input drifts and then
    /// settles rather than agreeing. Not the default, because the training
    /// path must keep advancing.
    pub fn set_power_iteration(&self, on: bool) {
        self.encoder.set_power_iteration(on);
        self.dirichlet.set_power_iteration(on);
        self.decoder.set_power_iteration(on);
    }

    /// What a real layer says, not what was last requested — see
    /// [`Dirichlet::power_iteration`](crate::model::Dirichlet::power_iteration).
    pub fn power_iteration(&self) -> Option<bool> {
        self.dirichlet.power_iteration()
    }

    pub fn encode_latent(
        &self,
        z: &Tensor,
        t: f64,
        vae_out: Option<&crate::model::EncoderOutput>,
        noise_bundle: Option<&NoiseBundle>,
    ) -> Result<crate::model::DirichletOutput> {
        // 1. Strictly one step for the streaming SSM.
        let z2 = match z.rank() {
            3 => {
                let l = z.dim(1)?;
                z.narrow(1, l - 1, 1)?.squeeze(1)? // last step → [B, D]
            }
            2 => z.clone(),
            r => return Err(anyhow!("encode_latent expects [B, D] or [B, L, D], got rank {r}")),
        };
        let b = z2.dim(0)?;

        // 2. Time conditioning (the Python format_conditioning + time_emb).
        let dt = Tensor::full(t as f32, (b, 1), &self.device)?.to_dtype(z2.dtype())?;
        let t_cond = match &self.time_emb {
            Some(f) => Some(f.forward(&dt)?),
            None => None,
        };

        // 3. Latent SSM — fresh recurrence, single step.
        let z_ssm = match &self.latent_ssm {
            Some(ssm) => ssm.forward(&z2, None)?.0,
            None => z2.clone(),
        };

        // 4. Dirichlet boundary — per-step input (the module tiles up to a
        // legacy windowed mixer itself when needed).
        let enc;
        let enc_ref = match vae_out {
            Some(e) => e,
            None => {
                enc = empty_encoder_output(&z_ssm)?;
                &enc
            }
        };
        let mut out = self
            .dirichlet
            .forward(&z_ssm, t_cond.as_ref(), enc_ref, None, 1, 0, false, noise_bundle, None)?;

        // 6. Python: dirichlet_out._replace(z=z_ssm, real_x=z)
        out.z_in = z_ssm;
        out.real_x = z2.unsqueeze(1)?;
        Ok(out)
    }
}

/// Evidence-less [`EncoderOutput`] for [`SpectralVae::encode_latent`] when
/// the bridge caller carries no encoder state: the alpha/ls carriers are
/// EMPTY tensors — the Dirichlet forward's documented fallback substitutes
/// its safe defaults — and the latent slots echo `z` (`logvar_z = 0` ⇒ unit
/// variance).
pub(crate) fn empty_encoder_output(z: &Tensor) -> Result<crate::model::EncoderOutput> {
    let empty = Tensor::zeros(0, z.dtype(), z.device())?;
    Ok(crate::model::EncoderOutput {
        alpha_mu: empty.clone(),
        alpha_factor: empty.clone(),
        alpha_diag: empty.clone(),
        alpha: empty.clone(),
        log_pi: empty.clone(),
        pi: empty.clone(),
        z: z.clone(),
        mu_z: z.clone(),
        logvar_z: z.zeros_like()?,
        ls: empty,
        real_x: z.unsqueeze(1)?,
    })
}

// ---------------------------------------------------------------------------
// GP handoff — extract the 198-dim feature vector and LMC weights
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// spectral_mixture — the amortised-Dirichlet mixture over the atom support
// ---------------------------------------------------------------------------

/// `p(z | x) = Σ_k π_k · N(atom_loc_k, diag(atom_scale_k²))` — the Bridge's
/// policy-mixture extractor (Python `Bridge.spectral_mixture`). The batch
/// dependence lives ENTIRELY in the posterior simplex π: pass the live
/// refinement `state` (its `pi` is the zt-conditioned posterior — the
/// encode_latent output in Python); with no state the Dirichlet PRIOR π
/// (deterministic median sticks in Deterministic mode) is the cold-start.
/// The atoms are the shared support; aggregation over the batch is the
/// diagonal `from_batched` (`first` | `average` | `concat`).
pub fn spectral_mixture<KN: crate::model::KernelNetworkModule>(
    dirichlet: &crate::model::AmortisedDirichlet<KN>,
    state: Option<&StateSpaceOutput>,
    batch: usize,
    device: &Device,
    aggregate: &str,
    obs_jitter: f64,
) -> Result<crate::stochastics::GaussianMixture> {
    let pi = match state {
        Some(s) => s.pi.clone(),
        None => dirichlet.sample_prior(batch, device)?,
    };
    let pi = pi.to_dtype(DType::F32)?;
    let (b, k) = pi.dims2()?;
    let norm = pi.sum_keepdim(1)?.maximum(1e-10)?;
    let weights_b = pi.broadcast_div(&norm)?; // [B, K]

    let (loc, var) = dirichlet.atoms()?; // [K, D] each
    let d = loc.dim(1)?;
    let means_b = loc.unsqueeze(0)?.broadcast_as((b, k, d))?.contiguous()?;
    let vars_b = var
        .affine(1.0, obs_jitter)?
        .unsqueeze(0)?
        .broadcast_as((b, k, d))?
        .contiguous()?;
    Ok(crate::stochastics::GaussianMixture::from_batched(
        &weights_b, &means_b, &vars_b, aggregate,
    )?)
}

// ---------------------------------------------------------------------------
// Per-forward-pass posterior sampling — the BayesianPosteriorSampler wiring
// ---------------------------------------------------------------------------

/// The VAE's reparameterisation-site manifest (the Python
/// `model.noise_site_shapes`): per-sample shapes WITHOUT the S axis, keyed by
/// the canonical site names the sampler's strata/antithetic config uses.
pub fn noise_site_shapes(
    cfg: &SpectralVaeConfig,
    batch: usize,
) -> std::collections::HashMap<String, Vec<usize>> {
    std::collections::HashMap::from([
        // Dirichlet manifest (Python `AmortisedDirichlet.noise_site_shapes`):
        // global_stick is BATCHLESS (the global q_a is [K−1]); the evidence
        // reparam is two split Gaussian sites (η_d, η_f).
        ("global_stick".to_string(), vec![cfg.k_atoms - 1]),
        ("local_stick".to_string(), vec![batch, cfg.k_atoms - 1]),
        ("local_evidence_d".to_string(), vec![batch, cfg.evidence_dim]),
        ("local_evidence_f".to_string(), vec![batch, cfg.rank]),
        ("lengthscale".to_string(), vec![batch, cfg.k_atoms]),
        // Encoder manifest (Python `ConvolutionalLoopEncoder.noise_site_shapes`
        // — the keys its forward actually READS; the old "encoder_z"
        // placeholder matched nothing and left the reparam sites on ad-hoc
        // noise).
        ("conv_bottleneck".to_string(), vec![batch, cfg.bottleneck_dim]),
        ("latent_z".to_string(), vec![batch, cfg.latent_dim]),
        ("default_pi".to_string(), vec![batch, cfg.k_atoms]),
    ])
}

/// One deterministic noise bundle PER FORWARD PASS of the refinement loop:
/// `salt = step` gives each Itô step distinct-but-reproducible noise while
/// the per-site seeds keep every site's draw stable across runs (the GEOC
/// determinism boundary). Call at the top of each `refinement_loop` pass;
/// the bundle's sites feed the encoder-z reparam (`encoder.rs` `eps`), the
/// Dirichlet's stochastic-mode sticks, and the alpha evidence draw — those
/// three internal `randn`/`rand` sites are the remaining injection seams
/// (each is a one-argument threading once its forward takes `Option<&Tensor>`
/// noise).
pub fn draw_step_bundle(
    cfg: &SpectralVaeConfig,
    sampler: &crate::model::noise::BayesianPosteriorSampler,
    batch: usize,
    step: usize,
    device: &Device,
) -> Result<crate::model::noise::NoiseBundle> {
    Ok(sampler.draw_bundle(&noise_site_shapes(cfg, batch), device, step as u64)?)
}

/// Pack VAE features as row-major `[B, 134]` values:
/// `gates(8) | mat52(32) | rbf(32) | rq(32) | pi(30)`.
///
/// This three-primitive schema is distinct from the 198-wide, five-primitive
/// inputs required by the legacy GP. It cannot be passed directly to
/// `SvgpPredictor::predict`. The width check rejects configuration drift,
/// such as changing the number of simplex atoms from 30.
pub fn pack_features(state: &StateSpaceOutput) -> Result<Vec<f32>> {
    let cat = Tensor::cat(
        &[
            &state.gates,
            &state.mat52,
            &state.rbf,
            &state.rq,
            // pi is [B, k_atoms]; kernel layout reserves 30 slots. If
            // k_atoms != 30 you need a projection — handled here with a
            // strict check to catch config drift at the type boundary.
            &state.pi,
        ],
        candle_core::D::Minus1,
    )?;
    let (b, d) = cat.dims2()?;
    if d != layout::D {
        return Err(anyhow!(
            "packed feature width {} != expected {} (check k_atoms == 30)",
            d,
            layout::D
        ));
    }
    // Force host + contiguous, then flatten.
    let host = cat.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
    let flat = host.reshape((b * d,))?;
    Ok(flat.to_vec1::<f32>()?)
}

/// Project the `lmc_consensus` tensor to an `[f32; 8]` LMC weight vector
/// for the GP predictor.
///
/// Faithful port of the Python chain:
///     lmc_raw  = lmc_consensus.mean(dim=0)
///     top4, _  = topk(lmc_raw, k=4, dim=0)
///     min_val  = top4.min(dim=0, keepdim=True)
///     sparse   = where(lmc_raw >= min_val, lmc_raw, 0)
///     squashed = sparse.sum(dim=0)
///     final    = squashed / (squashed.sum() + 1e-7)
///
/// We do the dominance-and-normalise step on the host because the topk
/// + `where_cond` kernel trip is cheap on 30x8 and simpler to audit.
pub fn lmc_project(lmc_consensus: &Tensor) -> Result<[f32; 8]> {
    // NEW consensus shape: [B, num_latents] (per-sample latent simplex from
    // softmax(π@Λ/τ)) — batch-mean is already a normalised weight vector,
    // no atom-axis dominance step needed.
    if lmc_consensus.rank() == 2 && lmc_consensus.dim(1)? == 8 {
        let w = lmc_consensus
            .mean(0)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let sum: f32 = w.iter().sum::<f32>() + 1e-7;
        let mut out = [0.0_f32; 8];
        for (o, v) in out.iter_mut().zip(w) {
            *o = v / sum;
        }
        return Ok(out);
    }

    // LEGACY shape: [B, K, E]. Python `.mean(dim=0)` reduces the
    // first dim of whatever shape comes in — at serving B=1 this is the
    // batch reduction. The output is [K, E].
    let mean_over_batch = lmc_consensus.mean(0)?;
    let (k, e) = mean_over_batch.dims2()?;
    if e != 8 {
        return Err(anyhow!(
            "LMC consensus has {} latents, expected 8 (num_latents mismatch)",
            e
        ));
    }

    // Pull to host — 30*8 f32 is 960 bytes, nothing.
    let host = mean_over_batch
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let flat: Vec<f32> = host.reshape((k * e,))?.to_vec1()?;

    // Row-major [K, E] — compute per-atom dominance, then per-latent sum.
    // The Python semantic: for each position in the [K, E] grid, keep it
    // only if it's in the top-4 of its ATOM (i.e., top-4 across K).
    // That matches `topk(lmc_raw, k=4, dim=0)` on [K, E] — top across K.
    let mut column_topk_mins = [f32::NEG_INFINITY; 8];
    for ei in 0..8 {
        // Collect column e's K values, sort desc, take 4th.
        let mut col: Vec<f32> = (0..k).map(|ki| flat[ki * 8 + ei]).collect();
        col.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let topk = col.get(3).copied().unwrap_or(f32::NEG_INFINITY);
        column_topk_mins[ei] = topk;
    }

    let mut squashed = [0.0_f32; 8];
    for ki in 0..k {
        for ei in 0..8 {
            let v = flat[ki * 8 + ei];
            if v >= column_topk_mins[ei] {
                squashed[ei] += v;
            }
        }
    }

    let sum: f32 = squashed.iter().sum::<f32>() + 1e-7;
    for w in &mut squashed { *w /= sum; }
    Ok(squashed)
}




// ---------------------------------------------------------------------------
// Notes on the port
// ---------------------------------------------------------------------------
//
// * No autograd: Candle has an autograd engine via `candle_core::Var` + the
//   `.backward()` API, but it's slower than libtorch and the design here
//   is inference-only. If you later want gradient-based explanations, add
//   a parallel `DifferentiableSpectralVae` that wraps the same submodules
//   with `Var` leaves instead of pre-loaded Tensors.
//
// * Generative vs inference mode: `generative_mode=true` is only useful
//   for trajectory rollouts where you want the VAE to hallucinate future
//   inputs. For a single-applicant risk score, leave it false — the
//   refinement loop just re-processes `real_x` each step and the three
//   certainty levels reflect real posterior tightening.
//
// * Memory: refinement_loop clones tensors into the state struct each
//   step. Candle tensors are Arc-backed so this is cheap (refcount bump),
//   but if you're serving at sustained high QPS on CPU, a scratch-buffer
//   pool per engine would reduce allocator churn. Measure before
//   optimising — at B=1 the allocations are probably noise against the
//   Dirichlet forward.
//
// * LMC projection: pure-Rust path on the host. The tensor is only 30×8
//   f32 = 960 bytes, so the device→host copy and the O(K·E) sort are
//   noise against the kernel evaluations.
//
// * Replacing sskp_engine.rs: this file is a complete substitute for the
//   tch-rs `SskpEngine` on the Fast path. Drop the tch-rs dependency from
//   Cargo.toml, delete `DifferentiableGp` (it was tch-specific), and
//   re-wire the API pool in `server.rs` to hold `CandleSskpEngine<E, D, Dc>`
//   instead. The handler contract stays the same: `serve_fast(&x, &state)`
//   becomes `serve(&x, Some(state), certainty)`.

#[cfg(test)]
mod certainty_tests {
    //! Verification of spectral variational autoencoders with configurable encoders, latent distributions and decoders.

    use super::{RefinementCertainty, T_HISTORY};

    /// `refinement_loop` reads ONE input row per step and left-pads
    /// `z_history` to `T_HISTORY` by repeating its first entry. So a certainty
    /// with `steps() < T_HISTORY` cannot produce a window whose positions are
    /// all distinct — the shortfall is exactly the number of duplicated rows.
    ///
    /// This pins the arithmetic rather than the behaviour because the failure
    /// is silent: `Standard` trained for months reporting finite losses on a
    /// window that was 10 copies plus 2 real steps.
    #[test]
    fn only_full_certainty_fills_the_ito_window() {
        assert_eq!(
            RefinementCertainty::Full.steps(),
            T_HISTORY,
            "Full must fill the window exactly, or the padding branch fires on \
             the training path",
        );
        assert_eq!(
            RefinementCertainty::default(),
            RefinementCertainty::Full,
            "the default must be the non-degenerate setting",
        );

        for c in [
            RefinementCertainty::Quick,
            RefinementCertainty::Standard,
            RefinementCertainty::High,
        ] {
            let dupes = T_HISTORY - c.steps();
            assert!(
                dupes > 0,
                "{c:?} is a serving-only setting; if it now fills the window, \
                 move it into the assertion above",
            );
        }

        // The specific numbers the production regression turned on.
        assert_eq!(T_HISTORY - RefinementCertainty::Standard.steps(), 10);
        assert_eq!(T_HISTORY - RefinementCertainty::Quick.steps(), 11);
    }
}
