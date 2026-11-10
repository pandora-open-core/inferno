//! Inferno — Amortised Dirichlet latent distributions and configurable kernel-network features.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::init::Init;
use candle_nn::{linear, Linear, VarBuilder};
use crate::kernels::scalar_f64;
use crate::model::norm::{layer_norm, LayerNorm, LayerNormConfig};

use crate::model::{
    EncoderOutput, softplus,
    interval, greater_than, positive,
    GpParams, DirichletMode
};
use crate::config::DirichletConfig;

/// Sobol salt for `uniform_like`'s stochastic fallback (no bundle site).
/// Distinct from every other salt in the tree so two sites never share a draw.
const UNIFORM_FALLBACK_SALT: u64 = 0x756e_6966_6f72_6d;


/// Raw value at which the stick's variational Beta is UNIFORM.
///
/// `softplus(0.5413) = 1.0`, so `q(v) = Beta(1, 1)`. Python's `neutral_stick`.
const NEUTRAL_STICK: f64 = 0.5413;

/// Centre of the `raw_lmc_matrix` init — Python's `initial_lmc_val`, its
/// legacy raw literal under the `(−5, 5)` logistic box.
const LMC_INIT_MEAN: f64 = -2.25;

/// Spread of the `raw_lmc_matrix` symmetry break — Python's
/// `LMC_RESEED_SIGMA`. Unit std in RAW space; see the init site for why a ramp
/// at the `atom_scale` amplitudes is not enough.
const LMC_INIT_SIGMA: f64 = 1.0;

/// What the latent contributed to the concentration logits on one call.
///
/// The share is the number to watch: `|dz| / (|dz| + |evidence|)` is exactly 0
/// at the zero init and says what fraction of the concentration logits the
/// latent is responsible for once it trains. A share that never leaves ~0
/// means the coupling was wired but the landscape gave it no reason to move —
/// which is a statement about the loss terms, not about the head.
#[derive(Clone, Copy, Debug)]
pub struct ZEvidence {
    pub z_absmean: f64,
    pub evidence_absmean: f64,
    pub share: f64,
}

#[derive(Debug, Clone)]
pub struct DirichletOutput {
    // Fourier / kernel outputs
    pub features: Tensor,     // [B, k * 2M]
    pub frequencies: Tensor,  // [B, k, M, latent_dim]
    pub gated_weights: Tensor,// [B, spectral_emb_dim]
    pub ls_pred: Tensor,      // [B, k] or [k, 1]
    pub bw_learned: Tensor,   // [B, k, latent_dim] or [1, k, latent_dim]
    pub ls_logvar: Tensor,    // [B, k] or [1, k]

    // Stick-breaking
    pub beta: Tensor,         // [k_atoms]
    pub pi: Tensor,           // [B, k_atoms]
    pub local_conc: Tensor,   // [B, evidence_dim]

    // Bottleneck / carriers
    pub z_in: Tensor,         // [B, latent_dim]  (echo of input)
    pub bottleneck: Tensor,   // [B, bottleneck_dim]
    /// The latent → simplex coupling's measurement for this call. `None` when
    /// the head is absent or the flag is off.
    pub z_evidence: Option<ZEvidence>,

    // Carry-over from the VAE (passed through unchanged)
    pub mu_z: Tensor,
    pub logvar_z: Tensor,
    pub real_x: Tensor,

    // Coregionalisation
    pub lmc_matrices: Tensor, // [B, k, k]
    /// Per-sample latent simplex `softmax(smooth_clamp(π@Λ)/τ)` — `[B,
    /// num_latents]` (the old port batch-averaged an atom-axis softmax to
    /// `[k, num_latents]`).
    pub lmc_consensus: Tensor,

    /// `[current ‖ previous-step]` bandwidth `[B, k, 2·latent_dim]` — the
    /// bandwidth-consistency loss stash (Python `bw_32`).
    pub bandwidth_loss_term: Option<Tensor>,

    // GP primitives (flattened in from the NKN)
    pub gates: Tensor,
    pub gates_logits: Tensor,
    pub mat52: Tensor,
    pub rbf: Tensor,
    pub rq: Tensor,
}

// ============================================================================
// Traits
// ============================================================================

/// The neural kernel network (NKN) used to produce the GP primitive
/// parameters and the spectral-feature gate. Implementers must accept the
/// bottleneck tensor `[B, bottleneck_dim]` plus the master time conditioning
/// `t_emb [B, time_dim]` (the SinkhornKernelNetwork FiLMs on it; `None` or a
/// missing FiLM head is identity conditioning) and return a `GpParams`
/// struct plus the `[B, spectral_emb_dim]` gate tensor.
pub trait KernelNetworkModule {
    fn forward(&self, bottleneck: &Tensor, t_emb: Option<&Tensor>) -> Result<(GpParams, Tensor)>;
    /// Hold or advance any `SpectralNormLinear` power iteration. Default no-op;
    /// see [`Dirichlet::set_power_iteration`].
    fn set_power_iteration(&self, _on: bool) {}
}


pub trait Dirichlet {
    /// Hold or advance any `SpectralNormLinear` power iteration this module
    /// owns. `false` is PyTorch's `eval()` semantics and is what makes a
    /// serving or parity path reproducible — see
    /// [`SpectralNormLinear::set_power_iteration`](crate::model::SpectralNormLinear::set_power_iteration).
    ///
    /// Default is a NO-OP, so a module with no spectral-norm layer need not
    /// implement it. A module that HAS one and does not override this stays
    /// silently call-history dependent — `the_vae_can_freeze_its_power_iteration`
    /// is the gate that catches that for the production tree.
    fn set_power_iteration(&self, _on: bool) {}

    /// Ask a real layer whether its power iteration is advancing.
    ///
    /// The mirror of [`set_power_iteration`](Self::set_power_iteration), and
    /// not redundant with it: a parity artefact must record what the run DID,
    /// not what the caller requested, and those differ the moment two DAGs
    /// share one model and the second one's pins win. `None` from an
    /// implementation that owns no spectral-norm layer.
    fn power_iteration(&self) -> Option<bool> {
        None
    }

    /// Typed global Kumaraswamy stick state (None for implementations that
    /// don't expose it — the default).
    ///
    /// `u_noise` is the SAME `global_stick` site the forward's own stick
    /// break reads. Threading it is not only a determinism fix: this used to
    /// hard-code `None`, so in `DirichletMode::Stochastic` it drew a fresh
    /// independent uniform and the `KumaraswamyGlobal` published on the state
    /// described a stick the forward never took. Every consumer of it — the
    /// SGLD potential, the global-divergence score terms — was reading a
    /// different draw from the one that produced `pi`.
    fn kumaraswamy_global(
        &self,
        _u_noise: Option<&Tensor>,
    ) -> Result<Option<crate::model::concentration::KumaraswamyConcentration>> {
        Ok(None)
    }

    /// `t_emb` is the master Fourier time conditioning `[B, 256]` (the
    /// Python `dt`): the FiLM projectors (dirichlet's own + the kernel
    /// network's) modulate the bottleneck on it; `None` (or checkpoints
    /// without the FiLM heads) is identity conditioning.
    ///
    /// `noise` is the per-Itô-step deterministic reparam bundle. Sites:
    /// `global_stick [K−1]` / `local_stick [B, K−1]` (Gaussians, mapped to
    /// the sticks' uniforms via Φ), `local_evidence_d [B, evidence_dim]` +
    /// `local_evidence_f [B, rank]` (the low-rank MVN evidence reparam).
    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        vae_out: &EncoderOutput,
        indices: Option<&Tensor>,
        steps: usize,
        t: usize,
        generative_mode: bool,
        noise: Option<&crate::model::noise::NoiseBundle>,
        // The PREVIOUS step's `bw_learned`, threaded by the caller. Explicit
        // rather than remembered, for the same reason `ssm_state` and
        // `ewma_state` are: hidden carried state makes the forward a function
        // of call history, and a serving path that drifts for a dozen calls
        // and then settles looks correct to any test that calls it once.
        // `None` is a cold start and self-bootstraps.
        prev_bw: Option<&Tensor>,
    ) -> Result<DirichletOutput>;
}

// ============================================================================
// AmortisedDirichlet
// ============================================================================

pub struct AmortisedDirichlet<K: KernelNetworkModule> {
    config: DirichletConfig,
    mode: DirichletMode,
    raw_q_a_global: Tensor,              // [k_atoms - 1]
    raw_q_b_global: Tensor,              // [k_atoms - 1]
    raw_gamma: Tensor,                   // [1]
    raw_h_mu: Tensor,                    // [1, 1, latent_dim]
    raw_h_sigma: Tensor,                 // [1, 1, latent_dim] or [1]
    raw_atom_loc: Tensor,                // [k_atoms, 1, latent_dim]
    raw_atom_scale: Tensor,              // [k_atoms, 1, latent_dim]
    /// The SPREAD of the RFF frequency draw, per atom per dim:
    /// `omega = base + (bw · omega_scale) · eps`, so this is the standard
    /// deviation of the spectral density each atom samples from.
    ///
    /// Bounded by construction, which is the point — the only thing holding
    /// frequencies in before it was `smooth_clamp(omega, -30, 20)` applied
    /// AFTER the fact, which bounds the result without bounding the thing that
    /// generates it. A clamp saturates and stops passing gradient; a
    /// constrained parameter stays live at its limit.
    ///
    /// It is also the knob `raw_atom_scale` was supposed to be and is not:
    /// `_refit_atoms` writes the within-cluster k-means std (~0.003) into that
    /// one, which sits ON the box floor, so the frequency scale has been pinned
    /// and unlearnable. This one is never written by the refit.
    raw_omega_scale: Tensor,             // [k_atoms, 1, latent_dim]
    raw_lengthscale_uncertainty: Tensor, // [1, k_atoms]
    raw_lmc_var: Tensor,                 // [k_atoms]
    raw_lmc_matrix: Tensor,              // [k_atoms, num_latents]
    noise_weights: Tensor, // [k_atoms, M, latent_dim]
    noise_bias: Tensor,    // [k_atoms, M]

    // Hypernets — spectral-normed on the trainable path (`weight_orig`
    // names), folded plain weights on cleaned serving checkpoints.
    compress_spectral_features_head: crate::model::parametrize::SpectralNormLinear,
    bottleneck_linear: crate::model::parametrize::SpectralNormLinear,
    bottleneck_norm: LayerNorm,
    /// Declared input width of the bottleneck mixer — `latent_dim` on the
    /// per-step Itô checkpoints, `latent_dim·T` on the legacy windowed-mixer
    /// ones (the forward tiles a narrower input up to this width).
    mixer_in: usize,
    /// Dirichlet-side FiLM head (`film_projector`, zero-init ⇒ identity):
    /// modulates the mixed bottleneck on the master time embedding BEFORE
    /// the kernel network (which FiLMs again internally). `None` on
    /// pre-migration checkpoints.
    film_projector: Option<Linear>,
    /// The latent's own contribution to the concentration logits
    /// (`latent_evidence_head`, zero-init ⇒ adds exactly 0 at load). Reads the
    /// FiLM'd bottleneck, so the correction is time-conditioned by
    /// construction and shares the representation the module already pays for.
    /// `None` on a checkpoint that predates it.
    latent_evidence_head: Option<Linear>,


    // Neural kernel network (trait object — supplied by caller)
    kernel_network: K,

    device: Device,
    dtype: DType,
}

impl<K: KernelNetworkModule> AmortisedDirichlet<K> {
    /// Build the module by reading raw parameters and buffers from a
    /// [`VarBuilder`] rooted at the Python `dirichlet` submodule.
    ///
    /// Expected safetensors layout (after spectral-norm strip):
    ///
    /// ```text
    ///   raw_q_a_global                            [k - 1]
    ///   raw_q_b_global                            [k - 1]
    ///   raw_gamma                                 [1]
    ///   raw_h_mu                                  [1, 1, latent]
    ///   raw_h_sigma                               [1, 1, latent]
    ///   raw_atom_loc                              [k, 1, latent]
    ///   raw_atom_scale                            [k, 1, latent]
    ///   raw_lengthscale_uncertainty               [1, k]
    ///   raw_lmc_var                               [k]
    ///   raw_lmc_matrix                            [k, num_latents]
    ///   noise_weights                             [k, M, latent]
    ///   noise_bias                                [k, M]
    ///   compress_spectral_features_head.weight    [spec_emb, k * M * 2]
    ///   compress_spectral_features_head.bias      [spec_emb]
    ///   bottleneck_mixer.0.weight                 [bot, latent]
    ///   bottleneck_mixer.0.bias                   [bot]
    ///   bottleneck_mixer.1.weight                 [bot]
    ///   bottleneck_mixer.1.bias                   [bot]
    /// ```
    pub fn load(
        config: DirichletConfig,
        mode: DirichletMode,
        vb: VarBuilder,
        kernel_network: K,
    ) -> Result<Self> {
        let k = config.k_atoms;
        let d = config.latent_dim;
        let m = config.num_fourier_features;

        // THE VARIATIONAL BETA STARTS UNIFORM, which is what `0.5413` is for:
        // both concentrations pass through `positive` (softplus), and
        // `softplus(0.5413) = 1.0` exactly, so `q(v) = Beta(1, 1)` — the
        // uniform distribution on the stick. Python calls this the
        // `neutral_stick` and inits both to it.
        //
        // The default raw ZERO is not neutral and is not harmless:
        // `softplus(0) = ln 2 = 0.693`, i.e. `Beta(0.693, 0.693)`, which is
        // U-SHAPED — it puts its mass at the two rails and starts the
        // stick-breaking already committed to breaking early or late rather
        // than uncommitted. These are the two parameters in this module that
        // must NOT be given a draw: a symmetry break here is a break of the
        // uniform prior itself, which is the one thing the init is supposed to
        // express.
        let neutral_stick = candle_nn::init::Init::Const(NEUTRAL_STICK);
        let raw_q_a_global = vb.get_with_hints(k - 1, "raw_q_a_global", neutral_stick)?;
        let raw_q_b_global = vb.get_with_hints(k - 1, "raw_q_b_global", neutral_stick)?;
        let raw_gamma = vb.get(1, "raw_gamma")?;
        let raw_h_mu = vb.get((1, 1, d), "raw_h_mu")?;
        let raw_h_sigma = vb.get((1, 1, d), "raw_h_sigma")?;
        let raw_atom_loc = vb.get((k, 1, d), "raw_atom_loc")?;
        let raw_atom_scale = vb.get((k, 1, d), "raw_atom_scale")?;
        // INITIALISED AT EXACTLY 1.0, not at the raw zero.
        //
        // Raw zero maps to the box MIDPOINT — 2.025 on the default (0.05, 4.0)
        // box — and the formula before this parameter existed had no scale
        // factor at all, i.e. an implicit 1.0. Seeding at the midpoint would
        // therefore DOUBLE the frequency spread of every existing checkpoint
        // the moment the key appeared, silently: a missing key loads as "fresh
        // init", the shapes all match, and nothing reports that the spectral
        // mixture just moved. Seeded through the constraint's own inverse, so
        // the whole feature is inert until someone trains or configures it.
        let (om_lo, om_hi) = config.omega_scale_bounds();
        let raw_omega_scale = vb.get_with_hints(
            (k, 1, d),
            "raw_omega_scale",
            candle_nn::init::Init::Const(crate::model::interval_ns_inv(1.0, om_lo, om_hi)),
        )?;
        let raw_lengthscale_uncertainty = vb.get((1, k), "raw_lengthscale_uncertainty")?;
        let raw_lmc_var = vb.get(k, "raw_lmc_var")?;
        // SYMMETRY-BROKEN, and it has to be — this is the ONLY source of latent
        // structure in the model.
        //
        // `coreg_matrix` reaches L only through the Gram `L Lᵀ`, so its VJP is
        // `dL = (G + Gᵀ + ε·I)·L` — a LEFT action. Every column of `dL` is the
        // same operator applied to the corresponding column of `L`, which makes
        // the latent-column subspace a CONSERVED QUANTITY of the gradient flow:
        // if the columns start identical they stay identical, exactly, forever.
        // Python measured the per-row spread of `dL` across latent columns at
        // 0.0 — not small, zero — with 200 steps leaving the pairwise column
        // cosines at 1.0. This is not a defect in the analytic backward;
        // ordinary autograd through `(LLᵀ)⊙(ππᵀ)` does the identical thing. It
        // is a property of the Gram parameterisation, and its consequence is
        // that latent asymmetry can only be INJECTED, never learned.
        //
        // A DEFAULT-ZERO INIT IS THEREFORE FATAL, in two compounding ways.
        // Every column is identical, so `π @ L` collapses to the same value in
        // every column for every sample and the consensus softmax is exactly
        // uniform — Python measured `lmc_consensus` carrying zero per-window
        // information for an entire run. And the `W_base` tie projects the RAW
        // matrix (`atoms_to_z.project_matrix(raw_lmc_matrix)`), so at raw zero
        // the latent×task cross-covariance is identically 0 and the GP's task
        // posterior mean is exactly zero with it.
        //
        // Ramps are NOT enough here (they are for `raw_atom_scale`): the
        // π-weighted sum over 30 atoms averages a monotone ramp down to ~1e-5
        // of consensus std. A unit-std raw draw is what measures right.
        //
        // REPRODUCIBLE, like Python's. Python seeds this from a generator
        // pinned at 240806; `Init::Randn` cannot match that, because it reaches
        // candle's device RNG and the vendored CPU backend refuses
        // `Device::set_seed`. The seeded draw is applied by
        // `Sota::SYMMETRY_BREAKS` from the Sobol/QMC engine — see that table,
        // which is where every symmetry break in this tree is declared.
        //
        // The `Init` here is the FALLBACK, for a Dirichlet built standalone
        // without the model around it: it must still be non-degenerate, since
        // the conservation argument above does not care who constructed the
        // module. A checkpoint overrides both.
        let raw_lmc_matrix = crate::model::parametrize::sobol_randn(
            &vb,
            "raw_lmc_matrix",
            (k, config.num_latents),
            LMC_INIT_MEAN,
            LMC_INIT_SIGMA,
        )?;

        // BUFFERS in Python (`register_buffer` — dirichlet.py:244-246), not
        // parameters: loaded from a tensor-backed checkpoint, plain random
        // draws on a fresh trainable map. Registering them as Vars put two
        // never-trainable names into every optimizer-island diff.
        let (noise_weights, noise_bias) = if vb.contains_tensor("noise_weights") {
            (vb.get((k, m, d), "noise_weights")?.detach(), vb.get((k, m), "noise_bias")?.detach())
        } else {
            (
                Tensor::randn(0f32, 1f32, (k, m, d), vb.device())?.to_dtype(vb.dtype())?,
                Tensor::rand(0f32, 1f32, (k, m), vb.device())?.to_dtype(vb.dtype())?,
            )
        };

        let compress_spectral_features_head = crate::model::parametrize::SpectralNormLinear::new(
            vb.pp("compress_spectral_features_head"),
            k * m * 2,
            config.spectral_emb_dim,
            candle_nn::init::Init::Randn {
                mean: 0.0,
                stdev: 1.0 / ((k * m * 2) as f64).sqrt(),
            },
        )?;

        let bmix = vb.pp("bottleneck_mixer");
        // Mixer input width. The CURRENT Python model is per-step
        // (`Linear(latent_dim, bottleneck)` — the T=12 Itô axis rides the
        // BATCH dim, `[B·T, latent_dim]`); legacy windowed-mixer checkpoints
        // trained `Linear(latent_dim·T, bottleneck)`. The loader reconciles
        // `dirichlet_t_window` off the checkpoint weight, so this expression
        // sizes correctly for both; the forward tiles a narrower input up to
        // `mixer_in` (degenerate no-history window) for the legacy case.
        let mixer_in = config.latent_dim * config.dirichlet_t_window.max(1);
        let bottleneck_linear = crate::model::parametrize::SpectralNormLinear::new(
            bmix.pp("0"),
            mixer_in,
            config.bottleneck_dim,
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 1.0 / (mixer_in as f64).sqrt() },
        )?;
        let bottleneck_norm = layer_norm(
            config.bottleneck_dim,
            LayerNormConfig {
                eps: config.norm_eps,
                remove_mean: true,
                affine: true,
            },
            bmix.pp("1"),
        )?;

        // Dirichlet-side FiLM head — zero-init in Python (identity at
        // start); absent on pre-migration checkpoints (identity here too).
        // Input width = the reconciled `time_dim` (Python config.model
        // .time_dim, default 64).
        let film_projector = crate::model::parametrize::build_or_absent(
            &vb,
            "film_projector.weight",
            || {
                crate::model::parametrize::zero_linear(
                    vb.pp("film_projector"),
                    config.time_dim,
                    config.bottleneck_dim * 2,
                )
            },
        )?;

        // The latent → simplex coupling. Zero-init like `film_projector`, and
        // for the same reason: an architectural path that changed the simplex
        // at step 0 would be indistinguishable from a regression in every
        // downstream transport reading.
        let latent_evidence_head = crate::model::parametrize::build_or_absent(
            &vb,
            "latent_evidence_head.weight",
            || {
                crate::model::parametrize::zero_linear(
                    vb.pp("latent_evidence_head"),
                    config.bottleneck_dim,
                    config.evidence_dim,
                )
            },
        )?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            config,
            mode,
            raw_q_a_global,
            raw_q_b_global,
            raw_gamma,
            raw_h_mu,
            raw_h_sigma,
            raw_atom_loc,
            raw_atom_scale,
            raw_omega_scale,
            raw_lengthscale_uncertainty,
            raw_lmc_var,
            raw_lmc_matrix,
            noise_weights,
            noise_bias,
            compress_spectral_features_head,
            bottleneck_linear,
            bottleneck_norm,
            mixer_in,
            film_projector,
            latent_evidence_head,
            kernel_network,
            device,
            dtype,
        })
    }

    pub fn mode(&self) -> DirichletMode {
        self.mode
    }

    /// The GLOBAL Kumaraswamy stick state as a typed container (the Python
    /// `KumaraswamyGlobal`) — recomputed statelessly from the raws (u = ½ in
    /// deterministic mode), so the score terms / SGLD potential can read
    /// q_a, q_b, log(1−v), β, log β without re-deriving the stick math.
    pub fn kumaraswamy_global(
        &self,
        u_noise: Option<&Tensor>,
    ) -> Result<crate::model::concentration::KumaraswamyGlobal> {
        let q_a = (softplus(&self.q_a_global()?)? + 1.035)?;
        let q_b = (softplus(&self.q_b_global()?)? + 1.035)?;
        let u = self.uniform_like(&q_a, u_noise)?;
        let v = crate::model::kumaraswamy_icdf(&u, &q_a, &q_b, 1e-12)?;
        let one_minus_v = (v.ones_like()? - &v)?.clamp(1e-12, 1.0)?;
        let log_1mv = one_minus_v.log()?;
        let beta = crate::model::stick_break(&v, 1e-12)?;
        let log_beta = beta.clamp(1e-30, f64::INFINITY)?.log()?;
        Ok(crate::model::concentration::KumaraswamyGlobal {
            q_a: Some(q_a),
            q_b: Some(q_b),
            log_one_minus_v: Some(log_1mv),
            v: Some(v),
            u: Some(u),
            beta: Some(beta),
            log_beta: Some(log_beta),
            // gamma_conc = interval-constrained γ + stick_breaking_epsilon
            // (the Python `self.gamma + self.stick_breaking_epsilon`).
            gamma: Some(self.gamma()?.affine(1.0, self.config.stick_breaking_epsilon)?),
            log_pi: None,
            pi_posterior: None,
        })
    }

    /// Analytic backward for DETERMINISTIC mode — the `DirichletBackward`
    /// assembler: given the upstream sensitivities the loss stack produces
    /// (`dpi = ∂L/∂π`, optionally `db = ∂L/∂B` with the `pi` the forward
    /// used), return closed-form gradients w.r.t. the raw parameters with NO
    /// autograd graph through this module.
    ///
    /// # How the "autograd function" is composed, in full
    ///
    /// There is no `torch.autograd.Function` in candle, and we deliberately
    /// do not want one: the whole point of this path is that the Dirichlet's
    /// forward is a chain of SMOOTH CLOSED FORMS, so its backward can be
    /// written as an explicit chain of vector-Jacobian products (VJPs) that
    /// we compose BY HAND, in reverse order of the forward. A VJP is the
    /// only primitive reverse-mode autograd is made of: for a forward step
    /// `y = f(x)`, the VJP maps an upstream cotangent `∂L/∂y` to the
    /// downstream cotangent `∂L/∂x = (∂y/∂x)ᵀ · ∂L/∂y` — WITHOUT ever
    /// materialising the Jacobian `∂y/∂x`. Composing VJPs back-to-front IS
    /// backpropagation; an autograd engine merely automates the bookkeeping
    /// we do explicitly here.
    ///
    /// The forward chain (deterministic mode) is:
    ///
    /// ```text
    ///   raw_q_a ─softplus+1.035→ q_a ┐
    ///   raw_q_b ─softplus+1.035→ q_b ┼─icdf(u=½)→ v ─stick_break→ π ──→ L
    ///   raw_lmc_matrix ─interval──→ Λ ┐                    (⊙ ππᵀ)  ↑
    ///   raw_lmc_var ─greater_than→ τ ┴─── coreg ────────→ B ────────┘
    /// ```
    ///
    /// and the backward composes the five VJPs in exactly the reverse order:
    ///
    /// 1. **Coregionalisation** (only when `db` is given). `coreg_vjp` maps
    ///    `∂L/∂B` to three cotangents at once — `(∂L/∂Λ, ∂L/∂τ, ∂L/∂π|_B)` —
    ///    because `B = (ΛΛᵀ)⊙(ππᵀ) + diag(τ+jitter)` has three inputs. The
    ///    π-cotangent is ADDED to the direct `dpi`: when one quantity (π)
    ///    feeds two consumers (the loss directly, and B), reverse mode SUMS
    ///    the cotangents arriving along each path. That sum,
    ///    `dpi_total = dpi + dpi_B`, is the single cotangent the rest of the
    ///    chain sees — this is the multivariate chain rule made explicit.
    /// 2. **Stick-breaking.** `stick_break_vjp` pulls `∂L/∂π` back through
    ///    `π_k = v_k·∏_{j<k}(1−v_j)` to `∂L/∂v`. Internally it never builds
    ///    the `[K, K−1]` Jacobian — the suffix sums collapse it to O(K) ops,
    ///    which is exactly why hand-composing beats generic autograd here.
    /// 3. **Kumaraswamy inverse-CDF.** `kumaraswamy_icdf_vjp` pulls `∂L/∂v`
    ///    back through `v = (1−(1−u)^{1/q_b})^{1/q_a}` to `(∂L/∂q_a,
    ///    ∂L/∂q_b)`. `u` is the reparameterisation NOISE input, not a
    ///    parameter: its cotangent is discarded (torch would `return None`
    ///    for it). In deterministic mode `u ≡ ½` — a constant the forward
    ///    can re-derive — which is what makes this whole backward STATELESS:
    ///    no saved activations, everything recomputed from raws + the mode.
    /// 4. **Constraint transforms** — the LAST link, mapping constrained
    ///    cotangents onto the raw leaves: `softplus_vjp` (`∂q/∂raw = σ(raw)`,
    ///    same for `greater_than` since the additive lower bound has zero
    ///    derivative) and `interval_vjp` (`(hi−lo)·σ(1−σ)`). These MUST
    ///    mirror the forward's transforms exactly — they are the Rust twins
    ///    of the gpytorch constraint Jacobians the Python resolves by
    ///    autograd on the transform alone.
    ///
    /// Batch reduction: parameters are shared across the batch while the
    /// cotangents carry batch axes, so each VJP SUMS over broadcast axes
    /// when it lands on a parameter (`reduce_to` inside the pure functions)
    /// — reverse mode's rule that a broadcast in the forward becomes a
    /// reduction in the backward.
    ///
    /// Every pure VJP used here is finite-difference-verified in
    /// `dirichlet_backward.rs`, including an end-to-end test of exactly this
    /// composition order.
    pub fn analytic_backward(
        &self,
        dpi: &Tensor,
        db: Option<&Tensor>,
        pi: Option<&Tensor>,
    ) -> Result<crate::model::DirichletGrads> {
        use crate::model::{
            coreg_vjp, interval_vjp, kumaraswamy_icdf, kumaraswamy_icdf_vjp, softplus_vjp,
            stick_break_vjp,
        };
        let eps = 1e-12;
        let mut dpi_total = dpi.clone();
        let mut grads = crate::model::DirichletGrads {
            raw_q_a: None,
            raw_q_b: None,
            raw_lmc_matrix: None,
            raw_lmc_var: None,
            dpi_total: None,
        };

        // ---- (1) coregionalisation branch: ∂L/∂B → (∂L/∂Λ, ∂L/∂τ, ∂L/∂π) --
        if let (Some(db), Some(pi)) = (db, pi) {
            let lmc = self.lmc_matrix()?; // Λ = interval(raw)
            let lmc_var = self.lmc_var()?; // τ = greater_than(raw)
            let (dl, dvar, dpi_b) = coreg_vjp(&lmc, &lmc_var, pi, db)?;
            let (lo, hi) = self.config.lmc_constraint;
            grads.raw_lmc_matrix = Some(interval_vjp(&self.raw_lmc_matrix, &dl, lo, hi)?);
            grads.raw_lmc_var = Some(softplus_vjp(&self.raw_lmc_var, &dvar)?);
            dpi_total = dpi_total.add(&dpi_b)?; // fan-in: sum the two π paths
        }
        grads.dpi_total = Some(dpi_total.clone());

        // ---- (2)+(3)+(4) stick chain: dπ → dv → (dq_a, dq_b) → raws -------
        let qa = (softplus(&self.q_a_global()?)? + 1.035)?;
        let qb = (softplus(&self.q_b_global()?)? + 1.035)?;
        // deterministic mode: the forward's u is the constant median ½,
        // so v (and hence the whole tape) is recomputed, not stored.
        let u = qa.zeros_like()?.affine(0.0, 0.5)?;
        let v = kumaraswamy_icdf(&u, &qa, &qb, eps)?;
        let dv = stick_break_vjp(&v, &dpi_total, eps)?;
        let (gqa, gqb) = kumaraswamy_icdf_vjp(&u, &qa, &qb, &dv, eps)?;
        grads.raw_q_a = Some(softplus_vjp(&self.q_a_global()?, &gqa)?);
        grads.raw_q_b = Some(softplus_vjp(&self.q_b_global()?, &gqb)?);
        Ok(grads)
    }

    pub fn set_mode(&mut self, mode: DirichletMode) {
        self.mode = mode;
    }

    // -----------------------------------------------------------------------
    // Property accessors — apply the GPyTorch constraint transforms on read.
    // These are cheap (one softplus or sigmoid) so we do not cache.
    // -----------------------------------------------------------------------

    fn q_a_global(&self) -> Result<Tensor> {
        positive(&self.raw_q_a_global)
    }

    fn q_b_global(&self) -> Result<Tensor> {
        positive(&self.raw_q_b_global)
    }

    /// `0.35 + (6.0 - 0.35) * sigmoid(raw_gamma)` — hand-coded interval.
    fn gamma(&self) -> Result<Tensor> {
        interval(&self.raw_gamma, 0.35, 6.0)
    }

    fn h_mu(&self) -> Result<Tensor> {
        interval(
            &self.raw_h_mu,
            self.config.freq_bounds.0,
            self.config.freq_bounds.1,
        )
    }

    fn h_sigma(&self) -> Result<Tensor> {
        interval(
            &self.raw_h_sigma,
            self.config.broad_var.0,
            self.config.broad_var.1,
        )
    }

    /// Public atom support for the spectral mixture: `(loc [K, D], var [K, D])`
    /// — the Dirichlet's constrained atom locations and squared scales.
    pub fn atoms(&self) -> Result<(Tensor, Tensor)> {
        let loc = self.atom_loc()?.squeeze(1)?; // [K, 1, D] → [K, D]
        let var = self.atom_scale()?.squeeze(1)?.sqr()?;
        Ok((loc, var))
    }

    fn atom_loc(&self) -> Result<Tensor> {
        interval(
            &self.raw_atom_loc,
            self.config.freq_bounds.0,
            self.config.freq_bounds.1,
        )
    }

    /// `raw_omega_scale` through its box. NON-SATURATING, unlike every other
    /// interval in this module — see [`interval_ns`](crate::model::interval_ns)
    /// for why the two are not interchangeable.
    fn omega_scale(&self) -> Result<Tensor> {
        crate::model::interval_ns(
            &self.raw_omega_scale,
            self.config.omega_scale_bounds().0,
            self.config.omega_scale_bounds().1,
        )
    }

    fn atom_scale(&self) -> Result<Tensor> {
        interval(
            &self.raw_atom_scale,
            self.config.tight_var.0,
            self.config.tight_var.1,
        )
    }

    fn lengthscale_uncertainty(&self) -> Result<Tensor> {
        interval(
            &self.raw_lengthscale_uncertainty,
            self.config.min_ls,
            self.config.max_ls,
        )
    }

    fn lmc_var(&self) -> Result<Tensor> {
        greater_than(&self.raw_lmc_var, self.config.scale_constraint_lower)
    }

    /// Public constrained LMC matrix `Λ [k_atoms, num_latents]` — consumed
    /// by per-latent projections in downstream models.
    pub fn lmc_matrix_constrained(&self) -> Result<Tensor> {
        self.lmc_matrix()
    }

    /// The RAW atom loadings `[k_atoms, num_latents]`, untransformed.
    ///
    /// This — not [`lmc_matrix_constrained`](Self::lmc_matrix_constrained) —
    /// is what the `W_base` tie projects
    /// (`atoms_to_z.project_matrix(raw_lmc_matrix)`). The interval transform
    /// exists to bound `Λ` for the `(ΛΛᵀ) ⊙ ππᵀ` coregionalisation, where a
    /// runaway loading would break PSD-ness; the cross-covariance `W_base`
    /// has no such constraint and squashing it through `interval` would cap
    /// the task↔latent coupling for no reason.
    pub fn raw_lmc_matrix(&self) -> &Tensor {
        &self.raw_lmc_matrix
    }

    fn lmc_matrix(&self) -> Result<Tensor> {
        interval(
            &self.raw_lmc_matrix,
            self.config.lmc_constraint.0,
            self.config.lmc_constraint.1,
        )
    }

    // -----------------------------------------------------------------------
    // Core forward helpers
    // -----------------------------------------------------------------------

    /// Uniform draw for a Kumaraswamy stick, in priority order:
    ///
    /// 1. bundle noise (a GAUSSIAN site — mapped to Uniform(0,1) via Φ,
    ///    which exactly recovers the sampler's low-discrepancy uniform);
    /// 2. mode fallback — the median ½ in [`DirichletMode::Deterministic`],
    ///    a fresh `rand` in [`DirichletMode::Stochastic`].
    ///
    /// Always finished with the smooth `affine_unit` squash into
    /// `(uniform_dist_clamp, 1 − uniform_dist_clamp)` — the Python replaced
    /// the hard clamp (dead gradient at the rails) with this.
    fn uniform_like(&self, template: &Tensor, u_noise: Option<&Tensor>) -> Result<Tensor> {
        self.uniform_like_with(self.mode, template, u_noise)
    }

    /// [`uniform_like`](Self::uniform_like) with the mode as a PARAMETER —
    /// the seam `deterministic_pass` uses to run one eval-forced fold from a
    /// `&self` call chain without ambient mode flips (a leaked flip would
    /// silently flip the whole model; a parameter cannot leak).
    fn uniform_like_with(
        &self,
        mode: DirichletMode,
        template: &Tensor,
        u_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let u = match u_noise {
            Some(eta) => {
                let eta = eta.to_dtype(template.dtype())?;
                // Broadcast a batchless site (e.g. global_stick [K−1]) up to
                // the template's shape.
                let eta = if eta.shape() != template.shape() {
                    eta.broadcast_as(template.shape())?
                } else {
                    eta
                };
                crate::model::normal_cdf(&eta)?
            }
            None => match mode {
                DirichletMode::Deterministic => {
                    // Median — not a draw.
                    Tensor::full(0.5f32, template.shape(), template.device())?
                        .to_dtype(template.dtype())?
                }
                DirichletMode::Stochastic => {
                    // Sobol at a FIXED salt, not `Tensor::rand`: candle's CPU
                    // backend ignores `Device::set_seed`, so a plain draw here
                    // makes any caller that passes no site irreproducible —
                    // which is exactly how `kumaraswamy_global` leaked. Mapped
                    // through Φ so the fallback has the same law as the bundle
                    // path above. CONSTANT across calls by construction: a
                    // caller wanting independent draws supplies the site.
                    let g = crate::model::noise::sobol_randn_like(
                        template.dims(),
                        template.device(),
                        UNIFORM_FALLBACK_SALT,
                    )?
                    .reshape(template.dims())?
                    .to_dtype(template.dtype())?;
                    crate::model::normal_cdf(&g)?
                }
            },
        };
        crate::model::affine_unit(&u, self.config.uniform_dist_clamp)
    }

    //gamma_conc = self.gamma + self.stick_breaking_epsilon
    /// Global Kumaraswamy stick-breaking → `beta` of length `k_atoms`.
    /// `u_noise` is the bundle's `global_stick` site (`[K−1]`, Gaussian).
    ///
    /// PUBLIC because `beta` is BATCHLESS — a model constant per step, not a
    /// per-request quantity — so an online path that enters at the fused
    /// bottleneck and calls [`restick_from_alpha`](Self::restick_from_alpha)
    /// per request wants to compute it ONCE at load and reuse it. Passing
    /// `None` for the noise is the deterministic serving draw, exactly as
    /// the internal `dirichlet_from_alpha_heads` path does.
    pub fn global_stick_breaking_kumaraswamy(&self, u_noise: Option<&Tensor>) -> Result<Tensor> {
        let q_a = (softplus(&self.q_a_global()?)? + 1.035)?;
        let q_b = (softplus(&self.q_b_global()?)? + 1.035)?;
        let u = self.uniform_like(&q_a, u_noise)?;

        // v = (1 - (1 - u)^(1/b))^(1/a)
        let one_minus_u = (u.ones_like()? - &u)?;
        let inv_b = q_b.recip()?;
        let inv_a = q_a.recip()?;

        let pow_inner = pow_tensor(&one_minus_u, &inv_b)?;
        let base = (pow_inner.ones_like()? - &pow_inner)?;
        let qv = pow_tensor(&base, &inv_a)?;

        // Smooth boundary squash (Python: affine_unit, was a hard clamp).
        let qv = crate::model::affine_unit(&qv, self.config.uniform_dist_clamp)?;

        let log_v = qv.log()?;
        let one_minus_v = (qv.ones_like()? - &qv)?;
        let log_1mv = one_minus_v.log()?;

        // F.pad(log_1_minus_v, (1, 0), value=0.0)  → prepend zero along last dim.
        let padded = prepend_zero_last_dim(&log_1mv)?;
        let cum = padded.cumsum(D::Minus1)?;

        // log_pi_k = log_v + cum[..., :-1]
        // log_pi_last = cum[..., -1:]
        let n = log_v.dim(D::Minus1)?;
        let cum_head = cum.narrow(D::Minus1, 0, n)?; // length = k - 1
        let cum_tail = cum.narrow(D::Minus1, n, 1)?; // length = 1

        let log_pi_k = (log_v + cum_head)?;
        let log_beta = Tensor::cat(&[&log_pi_k, &cum_tail], D::Minus1)?;
        let beta = log_beta.exp()?;
        let sum = beta.sum_keepdim(D::Minus1)?;
        beta.broadcast_div(&sum)
    }

    /// Local stick-breaking conditioned on the global `beta`.
    /// Returns `pi` of shape `[B, k_atoms]`. `u_noise` is the bundle's
    /// `local_stick` site (`[B, K−1]`, Gaussian).
    fn local_stick_breaking(
        &self,
        qa: &Tensor,
        qb: &Tensor,
        global_beta: &Tensor,
        u_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.local_stick_breaking_with(self.mode, qa, qb, global_beta, u_noise)
    }

    /// Mode-parameterised twin — see [`Self::uniform_like_with`].
    fn local_stick_breaking_with(
        &self,
        mode: DirichletMode,
        qa: &Tensor,
        qb: &Tensor,
        global_beta: &Tensor,
        u_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let prior_a = 1.0_f64;
        // prior_b = global_beta[..., :-1] * concentration_strength + 1.0
        // (Python 77.0 — the global-stick prior strength; was 37 in
        // pretraining).
        let k = global_beta.dim(D::Minus1)?;
        let head = global_beta.narrow(D::Minus1, 0, k - 1)?;
        let prior_b = head.affine(self.config.concentration_strength as f64, 1.0)?;

        // Hard clamp → smooth clamp on the concentration params.
        let clamped_qa = crate::model::smooth_clamp(
            &qa.affine(1.0, prior_a)?,
            self.config.jitter,
            self.config.conc_clamp,
        )?;
        let clamped_qb = crate::model::smooth_clamp(
            &qb.broadcast_add(&prior_b)?,
            self.config.jitter,
            self.config.conc_clamp,
        )?;

        let u = self.uniform_like_with(mode, &clamped_qa, u_noise)?;
        let one_minus_u = (u.ones_like()? - &u)?;
        let inv_b = clamped_qb.recip()?;
        let inv_a = clamped_qa.recip()?;

        let pow_inner = pow_tensor(&one_minus_u, &inv_b)?;
        let base = (pow_inner.ones_like()? - &pow_inner)?;
        let v = pow_tensor(&base, &inv_a)?;

        let v = crate::model::affine_unit(&v, self.config.uniform_dist_clamp)?;

        let log_v = v.log()?;
        let one_minus_v = (v.ones_like()? - &v)?;
        let log_1mv = one_minus_v.log()?;

        let padded = prepend_zero_last_dim(&log_1mv)?;
        let cum = padded.cumsum(D::Minus1)?;

        let n = log_v.dim(D::Minus1)?;
        let cum_head = cum.narrow(D::Minus1, 0, n)?;
        let cum_tail = cum.narrow(D::Minus1, n, 1)?;

        let log_pi_k = (log_v + cum_head)?;
        let log_pi = Tensor::cat(&[&log_pi_k, &cum_tail], D::Minus1)?;

        // Python: kill −inf, log-softmax normalise, then a SMOOTH boundary
        // squash on the PROBABILITY (affine_unit keeps a non-vanishing
        // gradient where clamp_min had a dead zone), renormalise back onto
        // the simplex.
        let log_pi = log_pi.clamp(-60.0_f32, f32::INFINITY)?;
        let log_pi = log_pi.broadcast_sub(&crate::model::logsumexp_keepdim(&log_pi)?)?;
        let pi = crate::model::affine_unit(
            &log_pi.exp()?,
            self.config.posterior_dirichlet_epsilon,
        )?;
        let sum = pi.sum_keepdim(D::Minus1)?;
        pi.broadcast_div(&sum)
    }

    /// Unconditional prior draw of π (mixture weights) of shape `[n, k_atoms]`.
    ///
    /// Used as μ₀ for the Schrödinger Bridge. Collapses the local stick-breaking
    /// to its prior by passing zero qa/qb (so clamped_qa → 1.0, clamped_qb →
    /// global_beta·37 + 1.0), with the global β drawn from the Kumaraswamy prior
    /// parameterised by the learned marginal `q_a_global` / `q_b_global`.
    ///
    /// No applicant evidence involved. For the conditional bridge, the evidence
    /// enters only through the drift network's `cond` input.
    pub fn sample_prior(&self, n: usize, device: &Device) -> Result<Tensor> {
        let k = self.config.k_atoms;

        // 1. n independent global β draws by broadcasting the learned Kumaraswamy
        //    parameters to shape [n, k-1] and resampling u.
        let q_a_vec = (softplus(&self.q_a_global()?)? + 1.035)?; // [k-1]
        let q_b_vec = (softplus(&self.q_b_global()?)? + 1.035)?; // [k-1]
        let q_a = q_a_vec.broadcast_as((n, k - 1))?;
        let q_b = q_b_vec.broadcast_as((n, k - 1))?;

        let u = Tensor::rand(0f32, 1f32, (n, k - 1), device)?;
        let inv_b = q_b.recip()?;
        let inv_a = q_a.recip()?;
        let one_minus_u = u.affine(-1.0, 1.0)?;
        let pow_inner = pow_tensor(&one_minus_u, &inv_b)?;
        let base = pow_inner.affine(-1.0, 1.0)?;
        let qv = pow_tensor(&base, &inv_a)?;
        let qv = qv.clamp(
            self.config.uniform_dist_clamp as f32,
            (1.0 - self.config.uniform_dist_clamp) as f32,
        )?;

        let log_v = qv.log()?;
        let one_minus_v = qv.affine(-1.0, 1.0)?;
        let log_1mv = one_minus_v.log()?;
        let padded = prepend_zero_last_dim(&log_1mv)?;
        let cum = padded.cumsum(D::Minus1)?;
        let cum_head = cum.narrow(D::Minus1, 0, k - 1)?;
        let cum_tail = cum.narrow(D::Minus1, k - 1, 1)?;
        let log_pi_k = (log_v + cum_head)?;
        let log_beta = Tensor::cat(&[&log_pi_k, &cum_tail], D::Minus1)?;
        let beta = log_beta.exp()?;
        let sum = beta.sum_keepdim(D::Minus1)?;
        let beta = beta.broadcast_div(&sum)?; // [n, k]

        // 2. Local stick-breaking with zero qa/qb collapses to the Kumaraswamy
        //    prior (a=1, b=β·concentration_strength+1) — the unconditional
        //    prior over π.
        let zero = Tensor::zeros((n, k - 1), self.dtype, device)?;
        self.local_stick_breaking(&zero, &zero, &beta, None)
    }

    /// Draws a conditional sample (pi) given the encoder's local evidence outputs.
    /// Used to generate `x1` (target data) for the Schrödinger Bridge.
    pub fn sample_posterior(
        &self, 
        alpha_mu: &Tensor, 
        alpha_factor: &Tensor, 
        alpha_diag: &Tensor
    ) -> Result<Tensor> {
        // 1. Get the global stick-breaking prior
        let beta = self.global_stick_breaking_kumaraswamy(None)?;

        // 2. Reparameterise the local evidence
        // No bottleneck on this path — `sample_posterior` is handed the
        // encoder's evidence outputs alone, so there is no z to couple.
        let (qa, qb, _ab_logits, _) =
            self.get_local_evidence(alpha_mu, alpha_factor, alpha_diag, None, None, None)?;

        // 3. Local stick-breaking conditioned on the global beta
        let pi = self.local_stick_breaking(&qa, &qb, &beta, None)?;
        
        Ok(pi.clamp(self.config.posterior_dirichlet_epsilon as f32, f32::INFINITY)?)
    }
    /// Reparameterise the low-rank MVN for local evidence:
    /// `ab_logits = μ + √diag·η_d + F·η_f`. Deterministic mode (the Python
    /// `not self.training`) uses `ab_logits = mu` — no reparam noise.
    ///
    /// `mu`    shape [B, evidence_dim]
    /// `factor`shape [B, evidence_dim, rank]
    /// `diag`  shape [B, evidence_dim]
    /// `noise` = the bundle's `(local_evidence_d, local_evidence_f)` sites.
    fn get_local_evidence(
        &self,
        mu: &Tensor,
        factor: &Tensor,
        diag: &Tensor,
        noise: Option<(&Tensor, &Tensor)>,
        z_features: Option<&Tensor>,
        gibbs: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Option<ZEvidence>)> {
        self.get_local_evidence_with(self.mode, mu, factor, diag, noise, z_features, gibbs)
    }

    /// Mode-parameterised twin — see [`Self::uniform_like_with`].
    fn get_local_evidence_with(
        &self,
        mode: DirichletMode,
        mu: &Tensor,
        factor: &Tensor,
        diag: &Tensor,
        noise: Option<(&Tensor, &Tensor)>,
        z_features: Option<&Tensor>,
        gibbs: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Option<ZEvidence>)> {
        let ab_logits = match mode {
            DirichletMode::Deterministic => mu.clone(),
            DirichletMode::Stochastic => {
                let b = mu.dim(0)?;
                let d = mu.dim(1)?;
                let r = factor.dim(2)?;
                let safe_diag = (softplus(diag)? + self.config.jitter)?;

                let (eta_d, eta_f) = match noise {
                    Some((nd, nf))
                        if nd.dims() == [b, d] && nf.dims() == [b, r] =>
                    {
                        (nd.to_dtype(mu.dtype())?, nf.to_dtype(mu.dtype())?)
                    }
                    _ => (
                        Tensor::randn(0f32, 1f32, (b, d), mu.device())?.to_dtype(mu.dtype())?,
                        Tensor::randn(0f32, 1f32, (b, r), mu.device())?.to_dtype(mu.dtype())?,
                    ),
                };

                let std_d = safe_diag.sqrt()?;

                // PASSIVE-IRL GIBBS SHIFT. Displaces the DRAW toward the
                // observer's particle mean, in units of the posterior's own
                // scale: `η_d ← η_d + λ·(ᾱ − μ)/σ_d`.
                //
                // It moves the NOISE, not the mean, and the target arrives
                // already detached — so it changes where the sample lands
                // without adding a gradient path of its own. The path this
                // whole channel exists to feed (`π → coreg → the control head`)
                // still runs through `mu` and `std_d`, untouched.
                //
                // THE TARGET IS AN ARGUMENT, not a getter. Python reaches a
                // sampler through `passive_irl_getter()`, and its own guard
                // documents why that is dangerous: every Dirichlet built
                // outside `SoTA` falls back to a `_prior_pool` of 256 draws
                // from `N(−4, 2²)`, so an UNOBSERVED pool still shifts and
                // every probe, direct-VAE test and ad-hoc driver trains with a
                // permanent λ-shrinkage toward −4 wearing reward-information's
                // clothes. `None` here means "no observation", which is the
                // only safe default, and the caller — which is the only thing
                // that can know `n_observed ≥ 1` — supplies it or does not.
                // Same discipline as `prev_bw` and `ssm_state`.
                //
                // The shift BYPASSES THE ELBO and that is worth stating: the
                // KL prices `(μ, Σ)` while the sample comes from
                // `N(μ + λ(ᾱ − μ), Σ)`, and the displacement is in LOGIT units,
                // so a confident posterior gets a draw many σ into its own tail.
                // Hence `gibbs_noise_scale` defaulting to 0.
                let eta_d = match gibbs {
                    Some(tgt) if self.config.gibbs_noise_scale > 0.0 => {
                        let tgt = tgt.detach().to_dtype(mu.dtype())?;
                        let shift = tgt
                            .broadcast_sub(mu)?
                            .broadcast_div(&std_d.clamp(1e-6, f64::INFINITY)?)?
                            .affine(self.config.gibbs_noise_scale, 0.0)?
                            .detach();
                        eta_d.broadcast_add(&shift)?
                    }
                    _ => eta_d,
                };

                // cov_factor @ eta_f  → [B, d]
                let low_rank = factor.matmul(&eta_f.unsqueeze(D::Minus1)?)?.squeeze(D::Minus1)?;
                let diag_term = (std_d * eta_d)?;
                ((mu + &low_rank)? + diag_term)?
            }
        };

        // The latent's own contribution, added to the MEAN of the reparam
        // rather than to the noise: it shifts where the evidence sits without
        // touching its variance, so the decoder's low-rank Gaussian keeps
        // stating the uncertainty and z states an offset.
        //
        // Applied AFTER the reparam, so it lands identically in both modes.
        // The deterministic branch above uses `mu` alone, and a z-path that
        // vanished there would make every held-out simplex reading a
        // measurement of a different model.
        let (ab_logits, z_evidence) = match (&self.latent_evidence_head, z_features) {
            (Some(head), Some(zf)) if self.config.latent_simplex_coupling => {
                let dz = head.forward(zf)?;
                // The bottleneck may arrive [B, T, H] on the windowed GP path
                // while the evidence is [B, E]; take the last window step, the
                // same convention `last_step` applies to every other upstream
                // field.
                let dz = if dz.rank() == 3 {
                    let t = dz.dim(1)?;
                    dz.narrow(1, t - 1, 1)?.squeeze(1)?
                } else {
                    dz
                };
                if dz.dims() == ab_logits.dims() {
                    let a = scalar_f64(&dz.abs()?.mean_all()?)?;
                    let b = scalar_f64(&ab_logits.abs()?.mean_all()?)?;
                    let diag = ZEvidence {
                        z_absmean: a,
                        evidence_absmean: b,
                        share: a / (a + b + 1e-12),
                    };
                    ((ab_logits + dz)?, Some(diag))
                } else {
                    // Shape disagreement means the latent's simplex path is
                    // INACTIVE — say so rather than silently dropping it.
                    tracing::warn!(
                        z_evidence = ?dz.dims(), ab_logits = ?ab_logits.dims(),
                        "latent_simplex_coupling: shapes disagree, the latent's \
                         simplex path is inactive",
                    );
                    (ab_logits, None)
                }
            }
            _ => (ab_logits, None),
        };

        // VERY conservative bounds (Python moved [-7, 4] → [-2, 2]):
        // Dirichlet/Kumaraswamy concentrations are hyper-sensitive —
        // softplus(smooth_clamp(·, -7, 4)) let local_a fall to ~9e-4, so the
        // stick's (·)^(1/q_a) exponent reached ~1000 and its gradient
        // exploded (the "rogue alpha" inf-grad). [-2, 2] keeps local_a/b in
        // ~[0.13, 2.13] ⇒ 1/q ∈ ~[0.47, 7.9].
        let (local_a, local_b) = self.evidence_concentrations(&ab_logits)?;
        Ok((local_a, local_b, ab_logits, z_evidence))
    }

    /// `α → (q_a, q_b)`, the Kumaraswamy concentration pair.
    ///
    /// Chunk in half → `smooth_clamp[-2, 2]` → `softplus` + eps. Factored out
    /// because THREE callers need it — [`get_local_evidence`](Self::get_local_evidence),
    /// [`restick_from_alpha`](Self::restick_from_alpha) and
    /// [`restick_from_alpha_with_evidence`](Self::restick_from_alpha_with_evidence) —
    /// and the bounds are load-bearing rather than cosmetic (see the comment
    /// above the caller: `[-7, 4]` let `local_a` reach ~9e-4, so the stick's
    /// `(·)^(1/q_a)` exponent hit ~1000 and its gradient exploded). Three
    /// copies of that would be three chances to widen one of them.
    fn evidence_concentrations(&self, ab_logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let (ab_a, ab_b) = chunk_last_dim(ab_logits, 2)?;
        let local_a =
            (softplus(&crate::model::smooth_clamp(&ab_a, -2.0, 2.0)?)? + self.config.eps)?;
        let local_b =
            (softplus(&crate::model::smooth_clamp(&ab_b, -2.0, 2.0)?)? + self.config.eps)?;
        Ok((local_a, local_b))
    }

    /// Re-derive the simplex `π` from an (EM-updated, or particle-pool) `α` —
    /// the π reparam (`models/parent.py::_restick_from_alpha`).
    ///
    /// `α` is the local-evidence low-rank-MVN sample; map it to the
    /// Kumaraswamy concentration pair `(qa, qb)` EXACTLY as
    /// [`get_local_evidence`](Self::get_local_evidence) does (chunk in half →
    /// `smooth_clamp[-2, 2]` → `softplus` + eps), then push through
    /// [`local_stick_breaking`](Self::local_stick_breaking) to a normalised
    /// `π`. Deterministic and DIFFERENTIABLE in `α`, which is what lets the
    /// passive-IRL corner barrier take its one autograd pass through here —
    /// and what makes the Langevin move on the low-rank MVN a legitimate
    /// reparam rather than an (awkward) direct move on the simplex.
    ///
    /// `alpha`: `[.., evidence_dim]`; `beta_stick`: the global stick,
    /// broadcastable over the leading axis. Returns `[.., k_atoms]`.
    pub fn restick_from_alpha(&self, alpha: &Tensor, beta_stick: &Tensor) -> Result<Tensor> {
        Ok(self.restick_from_alpha_with_evidence(alpha, beta_stick)?.0)
    }

    /// [`restick_from_alpha`](Self::restick_from_alpha), also returning the
    /// Kumaraswamy concentration pair `(q_a, q_b)` it derived on the way.
    ///
    /// `π` alone is not enough to assemble the packed state vector:
    /// [`crate::model::concentration::pack_dirichlet_theta`] takes `qa` and
    /// `qb` as separate slots, and the only other way to obtain them is a
    /// full Dirichlet forward. Without this, an out-of-crate caller entering
    /// at the `[B, bottleneck_dim]` carrier has to re-derive them — and a
    /// second copy of `softplus(smooth_clamp(chunk(α), −2, 2)) + eps` is a
    /// second set of bounds to get wrong, silently, in a direction the
    /// stick's `(·)^(1/q_a)` exponent punishes hard.
    ///
    /// Returns `(π [.., k_atoms], q_a [.., (K−1)], q_b [.., (K−1)])`.
    pub fn restick_from_alpha_with_evidence(
        &self,
        alpha: &Tensor,
        beta_stick: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (qa, qb) = self.evidence_concentrations(alpha)?;
        let pi = self.local_stick_breaking(&qa, &qb, beta_stick, None)?;
        Ok((pi, qa, qb))
    }

    /// `π → W` — the LMC consensus half of
    /// [`coregionalisation_matrix`](Self::coregionalisation_matrix), which is
    /// the only half the passive-IRL policy readout wants.
    pub fn lmc_consensus_from_pi(&self, pi: &Tensor) -> Result<Tensor> {
        Ok(self.coregionalisation_matrix(pi)?.1)
    }

    /// LMC coregionalisation matrix `B` and consensus matrix `W`.
    ///
    /// `B` is `[B, k_atoms, k_atoms]` — the dense per-sample coregionalisation
    /// that [`crate::model::concentration::pack_dirichlet_theta`] packs as its
    /// `lmc_matrices` slot. `W` is `[B, num_latents]`, the per-sample simplex
    /// over latents, and is what
    /// [`lmc_consensus_from_pi`](Self::lmc_consensus_from_pi) hands back.
    ///
    /// Public for the `B` half: the consensus wrapper is the only half the
    /// passive-IRL readout wants, but a caller assembling the packed state
    /// vector needs the matrix itself and there is no other route to it.
    pub fn coregionalisation_matrix(&self, pi: &Tensor) -> Result<(Tensor, Tensor)> {
        let lmc = self.lmc_matrix()?; // [k, num_latents]
        let b_base = lmc.matmul(&lmc.t()?)?; // [k, k]

        // B_dense = B_base * pi[..., None] * pi[..., None, :]
        let pi_col = pi.unsqueeze(D::Minus1)?; // [B, k, 1]
        let pi_row = pi.unsqueeze(1)?; // [B, 1, k]
        let pi_outer = pi_col.broadcast_mul(&pi_row)?; // [B, k, k]
        let b_dense = b_base.unsqueeze(0)?.broadcast_mul(&pi_outer)?;

        // Add (lmc_var + jitter) to diagonal.
        let v = self.lmc_var()?.affine(1.0, self.config.jitter)?; // [k]
        let b = b_dense.dim(0)?;
        let k = b_dense.dim(1)?;
        let diag_add = Tensor::eye(k, b_dense.dtype(), b_dense.device())?
            .broadcast_mul(&v.unsqueeze(D::Minus1)?)?; // [k, k]
        let b_dense = b_dense.broadcast_add(&diag_add.unsqueeze(0)?)?;
        let _ = b;

        // Python: raw_logits = smooth_clamp(π @ Λ, −4, 4);
        //         W_per_sample = softmax(raw_logits / τ, dim=−1)  → [B, L].
        // (The old port softmaxed π⊗Λ over the ATOM axis and batch-averaged
        // to [K, L] — the new consensus is a per-sample simplex over the
        // LATENTS.)
        let raw_logits =
            crate::model::smooth_clamp(&pi.matmul(&lmc)?, -4.0, 4.0)?; // [B, L]
        let scaled = (&raw_logits / self.config.consensus_tau)?;
        let w_per_sample = candle_nn::ops::softmax(&scaled, D::Minus1)?; // [B, L]

        Ok((b_dense.contiguous()?, w_per_sample))
    }

    /// Produce spectral frequencies `omega` of shape
    /// `[B, k_atoms, M, latent_dim]`.
    fn get_omega(&self, bw: &Tensor) -> Result<Tensor> {
        // Python: 4-D bw takes the last Itô step; 2-D grows the atom axis.
        let bw = match bw.rank() {
            4 => {
                let t = bw.dim(1)?;
                bw.narrow(1, t - 1, 1)?.squeeze(1)?
            }
            2 => bw.unsqueeze(1)?,
            _ => bw.clone(),
        };

        // base = h_mu.unsqueeze(2) + atom_loc.transpose(0,1).unsqueeze(2)
        let h_mu = self.h_mu()?.unsqueeze(2)?; // [1, 1, 1, latent]
        let atom_loc = self.atom_loc()?.transpose(0, 1)?.unsqueeze(2)?; // [1, k, 1, latent]
        let base = h_mu.broadcast_add(&atom_loc)?; // [1, k, 1, latent]

        // omega = einsum("bkd, kmd -> bkmd", bw, noise_weights)
        //       = bw.unsqueeze(-2) * noise_weights.unsqueeze(0)
        // `omega_scale` is [k, 1, latent]; squeeze the singleton so it
        // multiplies the [B, k, latent] bandwidth per atom and per dim rather
        // than broadcasting into a spurious axis.
        let bw = bw.broadcast_mul(&self.omega_scale()?.squeeze(1)?)?;

        let bw_expanded = bw.unsqueeze(2)?; // [B, k, 1, latent]
        let nw_expanded = self.noise_weights.unsqueeze(0)?; // [1, k, M, latent]
        let omega = bw_expanded.broadcast_mul(&nw_expanded)?; // [B, k, M, latent]

        let omega = omega.broadcast_add(&base)?;
        // Python: smooth_clamp(omega, -30, 20) — NOT the freq_bounds clamp.
        crate::model::smooth_clamp(&omega, -30.0, 20.0)
    }

    /// Random Fourier features with `pi`-weighted cos/sin projections.
    /// Returns `[B, k * 2M]`.
    fn random_fourier_features(&self, z: &Tensor, omega: &Tensor, pi: &Tensor) -> Result<Tensor> {
        let _b = z.dim(0)?;
        let _d = z.dim(1)?;
        let m = self.config.num_fourier_features;

        // proj = einsum("bd, bkmd -> bkm", z, omega)
        //      = (z.unsqueeze(1).unsqueeze(2) * omega).sum(-1)
        let z_exp = z.unsqueeze(1)?.unsqueeze(1)?; // [B, 1, 1, latent]
        let proj = omega.broadcast_mul(&z_exp)?.sum(D::Minus1)?; // [B, k, M]

        // + noise_bias.unsqueeze(0)
        let proj = proj.broadcast_add(&self.noise_bias.unsqueeze(0)?)?;

        let scale = (m as f64).sqrt().recip();
        let pi_scl = (pi.sqrt()? * scale)?.unsqueeze(D::Minus1)?;

        let cos_proj = proj.cos()?.broadcast_mul(&pi_scl)?;
        let sin_proj = proj.sin()?.broadcast_mul(&pi_scl)?;

        let feats = Tensor::cat(&[&cos_proj, &sin_proj], D::Minus1)?; // [B, k, 2M]
        feats.flatten(1, 2)
    }

    fn compress_and_gate(&self, features: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let embedded = self.compress_spectral_features_head.forward(features)?;
        gate * embedded
    }

    /// `[current ‖ previous-step]` bandwidth (Python `_bw_with_previous`).
    /// Self-bootstraps on cold start (previous ← current) and falls back to
    /// the current bandwidth whenever the carried tensor can't broadcast
    /// (fresh process, last short batch, mode flip).
    ///
    /// # The carry is an ARGUMENT, not hidden state (2026-08-22)
    ///
    /// This used to read and write a `Mutex<Option<Tensor>>` on the module, so
    /// the forward was a function of CALL HISTORY: repeated `encode_latent`
    /// calls on one latent produced `features` that converged rather than
    /// agreed (measured deltas 7.5e-3 → 2.9e-3 → 1.5e-3 → … → 1.1e-4 at a
    /// feature scale of 6.5e-2). `pi` was unaffected, which is why it read as a
    /// randomness bug for a long time — the bandwidth is the only thing the
    /// carry reaches, through `get_omega → random_fourier_features`.
    ///
    /// Now it follows the same discipline as `ssm_state` and `ewma_state`
    /// (`variationalautoencoder.py`'s `_NEVER_FOLDED`): carried state is
    /// explicit, threaded by the caller, and never folded. `None` means "cold
    /// start", which self-bootstraps and makes a single call a pure function of
    /// its inputs.
    fn bw_with_previous(&self, bw_learned: &Tensor, prev_bw: Option<&Tensor>) -> Result<Tensor> {
        let carried = prev_bw.cloned();
        let prev = match carried {
            Some(p) if p.dims()[p.rank() - 2..] == bw_learned.dims()[bw_learned.rank() - 2..] => {
                if p.shape() == bw_learned.shape() {
                    p
                } else if p.dim(0)? == 1 {
                    p.broadcast_as(bw_learned.shape())?
                } else {
                    bw_learned.clone()
                }
            }
            _ => bw_learned.clone(),
        };
        Tensor::cat(&[bw_learned, &prev.to_dtype(bw_learned.dtype())?], D::Minus1)
    }

    fn predict_kernel_lengthscales(
        &self,
        ls: Option<&Tensor>,
        prev_bw: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // bw_base = (h_sigma * atom_scale) + jitter ; squeeze(1)
        let bw_raw = self
            .h_sigma()?
            .broadcast_mul(&self.atom_scale()?)?
            .affine(1.0, self.config.jitter)?;
        let bw_base = bw_raw.squeeze(1)?; // [k, latent_dim]

        // Python treats an EMPTY ls the same as None.
        let ls = ls.filter(|t| t.elem_count() > 0);

        let (ls_pred, bw_learned, ls_logvar) = match ls {
            None => {
                // Python: bw_base.mean(dim=-1).unsqueeze(0) → [1, k].
                let ls_pred = crate::model::smooth_clamp(
                    &bw_base.mean(D::Minus1)?.unsqueeze(0)?,
                    self.config.min_ls,
                    self.config.max_ls,
                )?;
                let ls_logvar = Tensor::zeros(
                    (1, self.config.k_atoms),
                    bw_base.dtype(),
                    bw_base.device(),
                )?;
                (ls_pred, bw_base.unsqueeze(0)?, ls_logvar)
            }
            Some(ls) => {
                let ls_pred = crate::model::smooth_clamp(
                    ls,
                    self.config.min_ls,
                    self.config.max_ls,
                )?;
                let batch_size = ls_pred.dim(0)?;

                // precision = 1 / (ls_pred.unsqueeze(-1) + eps)
                let precision = ls_pred
                    .unsqueeze(D::Minus1)?
                    .affine(1.0, self.config.eps)?
                    .recip()?; // [B, k, 1]

                let bw_expanded = bw_base.unsqueeze(0)?; // [1, k, latent]
                let bw_learned = bw_expanded.broadcast_mul(&precision)?;

                let ls_logvar = self
                    .lengthscale_uncertainty()?
                    .broadcast_as((batch_size, self.config.k_atoms))?;
                (ls_pred, bw_learned, ls_logvar)
            }
        };

        // The carry is cross-refinement-step VALUE conditioning (the Python
        // detaches; candle serving carries no graph). It goes OUT as
        // `bw_learned` — the caller threads it back in on the next step rather
        // than the module remembering it.
        let bw_32 = self.bw_with_previous(&bw_learned, prev_bw)?;
        Ok((ls_pred, bw_learned, ls_logvar, bw_32))
    }

    fn run_neural_nets_dirichlet(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, GpParams)> {
        let bn = self.bottleneck_linear.forward(x)?;
        let bn = self.bottleneck_norm.forward(&bn)?;
        let bn = bn.elu(1.0)?;

        // Dirichlet-side FiLM on the master time embedding (zero-init ⇒
        // identity; absent head or absent conditioning ⇒ identity).
        let bn = match (&self.film_projector, t_emb) {
            (Some(film), Some(t)) => {
                let fb = film.forward(t)?; // [B, 2·bottleneck]
                let w = self.config.bottleneck_dim;
                let scale = fb.narrow(D::Minus1, 0, w)?;
                let bias = fb.narrow(D::Minus1, w, w)?;
                bn.broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(&bias)?
            }
            _ => bn,
        };

        let (gp, gate) = self.kernel_network.forward(&bn, t_emb)?;
        Ok((bn, gate, gp))
    }

    /// The trait [`Dirichlet::forward`] body with the MODE AS A PARAMETER —
    /// the seam `deterministic_pass` uses for one eval-forced fold from a
    /// `&self` call chain (no ambient flips; a parameter cannot leak).
    pub fn forward_with_mode(
        &self,
        mode: DirichletMode,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        vae_out: &EncoderOutput,
        _indices: Option<&Tensor>,
        _steps: usize,
        _t: usize,
        _generative_mode: bool,
        noise: Option<&crate::model::noise::NoiseBundle>,
        prev_bw: Option<&Tensor>,
    ) -> Result<DirichletOutput> {
        // ---- inputs: per-step latent (Python `_last_step`) ---------------
        let x = if x.rank() == 3 {
            let l = x.dim(1)?;
            x.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?
        } else {
            x.clone()
        };
        let batch_size = x.dim(0)?;
        let device = x.device().clone();

        // Python `_align_batch`: evidence rows are [B₀, …] while the
        // windowed path runs [B₀·T, …] — repeat each element T times.
        let align = |t: &Tensor| -> Result<Tensor> {
            let b0 = if t.rank() >= 1 { t.dim(0)? } else { 0 };
            if b0 != 0 && b0 != batch_size && batch_size % b0 == 0 {
                repeat_interleave0(t, batch_size / b0)
            } else {
                Ok(t.clone())
            }
        };
        let last_step = |t: &Tensor| -> Result<Tensor> {
            if t.rank() == 3 {
                let l = t.dim(1)?;
                t.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()
            } else {
                Ok(t.clone())
            }
        };

        let mu_z = align(&last_step(&vae_out.mu_z)?)?;
        let logvar_z = align(&last_step(&vae_out.logvar_z)?)?;
        let real_x = align(&vae_out.real_x)?;

        let alpha_mu = {
            let t = align(&last_step(&vae_out.alpha_mu)?)?;
            if t.elem_count() == 0 {
                Tensor::zeros(
                    (batch_size, self.config.evidence_dim),
                    self.dtype,
                    &device,
                )?
            } else {
                t
            }
        };
        let alpha_diag = {
            let t = align(&last_step(&vae_out.alpha_diag)?)?;
            if t.elem_count() == 0 {
                Tensor::ones(
                    (batch_size, self.config.evidence_dim),
                    self.dtype,
                    &device,
                )?
            } else {
                t
            }
        };
        let alpha_factor = {
            // Python: 4-D alpha_factor takes the last step; then align.
            let t = &vae_out.alpha_factor;
            let t = if t.rank() == 4 {
                let l = t.dim(1)?;
                t.narrow(1, l - 1, 1)?.squeeze(1)?.contiguous()?
            } else {
                t.clone()
            };
            let t = align(&t)?;
            if t.elem_count() == 0 {
                Tensor::zeros(
                    (
                        batch_size,
                        self.config.evidence_dim,
                        self.config.alpha_factor_rank,
                    ),
                    self.dtype,
                    &device,
                )?
            } else {
                t
            }
        };

        // `ls` may be absent on the first refinement step.
        let ls_aligned = align(&last_step(&vae_out.ls)?)?;
        let ls_opt: Option<&Tensor> = if ls_aligned.elem_count() == 0 {
            None
        } else {
            Some(&ls_aligned)
        };

        // ---- 1. Global stick-breaking ------------------------------------
        let beta = self.global_stick_breaking_kumaraswamy(
            noise.and_then(|n| n.get("global_stick")),
        )?;

        // ---- 2. Neural nets: bottleneck (FiLM'd), gate, GP primitives ----
        // Legacy windowed-mixer checkpoints declare `latent_dim·T` input:
        // tile the per-step latent up to that width (degenerate no-history
        // window — the same broadcast convention as a cold start).
        let x_width = x.dim(1)?;
        let mixer_x = if x_width != self.mixer_in && self.mixer_in % x_width == 0 {
            let reps = self.mixer_in / x_width;
            x.unsqueeze(1)?
                .broadcast_as((batch_size, reps, x_width))?
                .reshape((batch_size, self.mixer_in))?
        } else {
            x.clone()
        };
        let (bottleneck, gate, gp_params) = self.run_neural_nets_dirichlet(&mixer_x, t_emb)?;

        // ---- 3. Local evidence reparam -----------------------------------
        // Bundle sites: `local_evidence_d [B, ev]` + `local_evidence_f
        // [B, rank]` (the Python split-site manifest).
        let eta_d = noise.and_then(|n| n.get("local_evidence_d"));
        let eta_f = noise.and_then(|n| n.get("local_evidence_f"));
        let evidence_eps = match (eta_d, eta_f) {
            (Some(d), Some(f)) => Some((d, f)),
            _ => None,
        };
        let (qa, qb, alpha_rs, z_evidence) = self.get_local_evidence_with(mode, 
            &alpha_mu,
            &alpha_factor,
            &alpha_diag,
            evidence_eps,
            Some(&bottleneck),
            // Bound by the caller that owns the observer — see the shift's note.
            None,
        )?;

        // ---- 4. Local stick-breaking -------------------------------------
        let pi = self.local_stick_breaking_with(mode, 
            &qa,
            &qb,
            &beta,
            noise.and_then(|n| n.get("local_stick")),
        )?;

        // ---- 5. LMC / consensus ------------------------------------------
        let (b_mat, w_con) = self.coregionalisation_matrix(&pi)?;

        // ---- 6. Lengthscales (+ the [current ‖ previous] bandwidth carry)
        let (ls_pred, bw_learned, ls_logvar, bw_32) =
            self.predict_kernel_lengthscales(ls_opt, prev_bw)?;

        // ---- 7. Frequencies and random Fourier features ------------------
        let omega = self.get_omega(&bw_learned)?;
        let raw_features = self.random_fourier_features(&x, &omega, &pi)?;
        let gated_features = self.compress_and_gate(&raw_features, &gate)?;

        Ok(DirichletOutput {
            features: gated_features,
            frequencies: omega,
            gated_weights: gate,
            ls_pred,
            bw_learned,
            ls_logvar,
            beta,
            pi,
            local_conc: alpha_rs,
            z_in: x,
            bottleneck,
            z_evidence,
            mu_z,
            logvar_z,
            real_x,
            lmc_matrices: b_mat,
            lmc_consensus: w_con,
            bandwidth_loss_term: Some(bw_32),
            gates: gp_params.gates,
            gates_logits: gp_params.gates_logits,
            mat52: gp_params.mat52,
            rbf: gp_params.rbf,
            rq: gp_params.rq,
        })
    }
}

// ============================================================================
// Dirichlet trait impl
// ============================================================================

impl<K: KernelNetworkModule> Dirichlet for AmortisedDirichlet<K> {
    fn kumaraswamy_global(
        &self,
        u_noise: Option<&Tensor>,
    ) -> Result<Option<crate::model::concentration::KumaraswamyConcentration>> {
        Ok(Some(AmortisedDirichlet::kumaraswamy_global(self, u_noise)?))
    }

    /// Forward pass matching the Python reference — the T=12 Itô-time-space
    /// contract:
    ///
    /// * `x` is the PER-STEP boundary latent `[B, latent_dim]` (3-D input
    ///   takes the last step). The windowed GP path flattens the
    ///   `[B, T_HISTORY, D]` Itô window to `[B·T, D]` and runs this forward
    ///   over EVERY window step — the T axis rides the BATCH dim.
    /// * Evidence on `vae_out` is only the LAST step's `[B₀, …]`; when x's
    ///   batch is `B₀·T`, each element is repeat-interleaved T times so the
    ///   evidence aligns row-for-row with `window.reshape(B·T, D)` (the
    ///   Python `_align_batch`).
    ///
    /// Missing evidence gets safe defaults (zero / one tensors), matching
    /// the Python fallbacks.
    fn power_iteration(&self) -> Option<bool> {
        Some(self.bottleneck_linear.power_iteration())
    }

    fn set_power_iteration(&self, on: bool) {
        self.compress_spectral_features_head.set_power_iteration(on);
        self.bottleneck_linear.set_power_iteration(on);
        // The kernel network owns three more (`sfl_lin0/1/2`) and they are what
        // `gated_weights` runs through — freezing only the two above leaves the
        // gate drifting, which is how this was found.
        self.kernel_network.set_power_iteration(on);
    }

    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        vae_out: &EncoderOutput,
        _indices: Option<&Tensor>,
        _steps: usize,
        _t: usize,
        _generative_mode: bool,
        noise: Option<&crate::model::noise::NoiseBundle>,
        prev_bw: Option<&Tensor>,
    ) -> Result<DirichletOutput> {
        self.forward_with_mode(
            self.mode, x, t_emb, vae_out, _indices, _steps, _t, _generative_mode, noise, prev_bw,
        )
    }

}

// ============================================================================
// Small tensor helpers
// ============================================================================

/// Element-wise power with a broadcastable exponent: `base ** exp`.
/// Candle lacks a tensor-exponent pow; we fall back to `(exp * log(base)).exp()`.
/// `base` is expected strictly positive.
fn pow_tensor(base: &Tensor, exp: &Tensor) -> Result<Tensor> {
    // eps-clip the base to avoid log(0) when the caller forgets to clamp.
    let tiny = 1e-30_f32;
    let safe_base = base.clamp(tiny, f32::INFINITY)?;
    let log_base = safe_base.log()?;
    let scaled = exp.broadcast_mul(&log_base)?;
    scaled.exp()
}

/// `torch.repeat_interleave(t, reps, dim=0)` — each dim-0 element repeated
/// `reps` times CONTIGUOUSLY (`[a, a, b, b]`, unlike `repeat`'s tiling).
fn repeat_interleave0(t: &Tensor, reps: usize) -> Result<Tensor> {
    let dims = t.dims().to_vec();
    let b0 = dims[0];
    let mut expanded = vec![b0, reps];
    expanded.extend_from_slice(&dims[1..]);
    let mut out_shape = vec![b0 * reps];
    out_shape.extend_from_slice(&dims[1..]);
    t.unsqueeze(1)?
        .broadcast_as(expanded)?
        .contiguous()?
        .reshape(out_shape)
}

/// Prepend a zero along the last dim. Equivalent to `F.pad(x, (1, 0), 0.0)`.
fn prepend_zero_last_dim(x: &Tensor) -> Result<Tensor> {
    let mut pad_shape: Vec<usize> = x.dims().to_vec();
    let last = pad_shape.len() - 1;
    pad_shape[last] = 1;
    let pad = Tensor::zeros(pad_shape, x.dtype(), x.device())?;
    Tensor::cat(&[&pad, x], D::Minus1)
}

/// Split a tensor into `n` equal chunks along the last dim.
fn chunk_last_dim(x: &Tensor, n: usize) -> Result<(Tensor, Tensor)> {
    assert_eq!(n, 2, "chunk_last_dim currently only supports n=2");
    let d = x.dim(D::Minus1)?;
    assert!(d % 2 == 0, "last dim must be divisible by 2");
    let half = d / 2;
    let a = x.narrow(D::Minus1, 0, half)?;
    let b = x.narrow(D::Minus1, half, half)?;
    Ok((a, b))
}

// ============================================================================
// Tests (shape-level only — no real weights)
// ============================================================================

#[cfg(test)]
mod tests {
    //! Verification of amortised dirichlet latent distributions and configurable kernel-network features.

    use super::*;

    /// Build the module at the small test dims, sharing the fixture the
    /// concentration test uses.
    #[cfg(test)]
    fn test_dirichlet(
        vb: candle_nn::VarBuilder,
        cfg: &DirichletConfig,
    ) -> AmortisedDirichlet<crate::model::KernelNetwork> {
        use crate::config::KernelNetworkConfig;
        use crate::model::KernelNetwork;
        let kn = KernelNetwork::load(
            KernelNetworkConfig {
                bottleneck_dim: cfg.bottleneck_dim,
                spectral_emb_dim: cfg.spectral_emb_dim,
                individual_kernel_dim_out: 8,
                num_primitives: 3,
                num_gates: 8,
                norm_eps: 1e-5,
                time_dim: cfg.time_dim,
            },
            vb.pp("kernel_network"),
        )
        .unwrap();
        AmortisedDirichlet::load(cfg.clone(), DirichletMode::Deterministic, vb, kn).unwrap()
    }

    #[cfg(test)]
    fn small_cfg() -> DirichletConfig {
        let mut cfg = DirichletConfig::default();
        cfg.latent_dim = 4;
        cfg.k_atoms = 5;
        cfg.evidence_dim = 8;
        cfg.alpha_factor_rank = 2;
        cfg.num_fourier_features = 6;
        cfg.spectral_emb_dim = 32;
        cfg.bottleneck_dim = 16;
        cfg.num_latents = 8;
        cfg.time_dim = 12;
        cfg
    }

    /// The latent → simplex coupling is INERT at load, bit for bit.
    ///
    /// `latent_evidence_head` is zero-init, so an architectural path that
    /// changed the simplex at step 0 would be indistinguishable from a
    /// regression in every downstream transport reading. The head still
    /// receives gradient — it is additive, not multiplicative — so this is a
    /// statement about the INIT, not about the head being inactive.
    #[test]
    fn the_latent_simplex_coupling_is_inert_at_load() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = small_cfg();
        let dir = test_dirichlet(vb, &cfg);

        let b = 3usize;
        let mu = Tensor::randn(0f32, 2f32, (b, cfg.evidence_dim), &dev).unwrap();
        let factor =
            Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim, cfg.alpha_factor_rank), &dev).unwrap();
        let diag = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        let zf = Tensor::randn(0f32, 1f32, (b, cfg.bottleneck_dim), &dev).unwrap();

        let (_, _, off, none) =
            dir.get_local_evidence(&mu, &factor, &diag, None, None, None).unwrap();
        let (_, _, on, diagnostic) =
            dir.get_local_evidence(&mu, &factor, &diag, None, Some(&zf), None).unwrap();

        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&off), bits(&on), "the zero-init head moved the concentration logits");
        assert!(none.is_none(), "no bottleneck ⇒ no measurement");
        let d = diagnostic.expect("the head ran, so it must report");
        assert_eq!(d.z_absmean, 0.0);
        assert_eq!(d.share, 0.0, "share is 0 at the zero init, by construction");
        assert!(d.evidence_absmean > 0.0);
    }

    /// Off by config ⇒ the head is registered but never called, so the flag
    /// never changes the checkpoint's key set.
    #[test]
    fn the_coupling_flag_gates_the_call_not_the_registration() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut cfg = small_cfg();
        cfg.latent_simplex_coupling = false;
        let dir = test_dirichlet(vb.clone(), &cfg);

        let b = 2usize;
        let mu = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        let factor =
            Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim, cfg.alpha_factor_rank), &dev).unwrap();
        let diag = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        let zf = Tensor::randn(0f32, 1f32, (b, cfg.bottleneck_dim), &dev).unwrap();

        let (_, _, _, d) =
            dir.get_local_evidence(&mu, &factor, &diag, None, Some(&zf), None).unwrap();
        assert!(d.is_none(), "the flag is off, so the head must not run");

        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(
            names.iter().any(|n| n.contains("latent_evidence_head.weight")),
            "the flag must not change the key set: {names:?}",
        );
    }

    /// `raw_omega_scale` decodes to EXACTLY 1.0 at init, which is what makes
    /// the parameter inert for every checkpoint written before it existed.
    ///
    /// The trap it avoids: raw zero maps to the box MIDPOINT, 2.025 on the
    /// default (0.05, 4.0) box. The formula before this parameter had no scale
    /// factor at all — an implicit 1.0 — so seeding at the raw zero would have
    /// DOUBLED the frequency spread of every existing checkpoint the moment the
    /// key appeared, silently: a missing key loads as "fresh init", the shapes
    /// all match, and nothing reports that the spectral mixture just moved.
    #[test]
    fn raw_omega_scale_is_seeded_through_its_own_inverse() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = small_cfg();
        let dir = test_dirichlet(vb, &cfg);

        let os: Vec<f32> = dir.omega_scale().unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(!os.is_empty());
        for v in &os {
            assert!((*v - 1.0).abs() < 1e-5, "omega_scale seeded at {v}, not 1.0");
        }

        let (lo, hi) = cfg.omega_scale_bounds();

        // The PRODUCTION raw value, read straight out of the live artefact:
        // `vae.dirichlet.raw_omega_scale` is -0.607158 at every one of its 480
        // entries. It decodes to exactly 1.0 under the algebraic box and to
        // 1.443 under the logistic one — so decoding this parameter with
        // `interval` rather than `interval_ns` would run every RFF frequency
        // 44% too wide, on a checkpoint whose author set it to be inert.
        let ckpt = Tensor::new(&[-0.607158f32][..], &dev).unwrap();
        let decoded = crate::model::interval_ns(&ckpt, lo, hi).unwrap()
            .to_vec1::<f32>().unwrap()[0];
        assert!((decoded - 1.0).abs() < 1e-4, "the artefact's raw decoded to {decoded}");
        let wrong = crate::model::interval(&ckpt, lo, hi).unwrap()
            .to_vec1::<f32>().unwrap()[0];
        assert!((wrong - 1.443).abs() < 1e-2, "the logistic box gives {wrong}, not 1.0");

        // The trap, asserted so nobody "simplifies" the init back to zero.
        let at_raw_zero = crate::model::interval_ns(
            &Tensor::zeros((1,), DType::F32, &dev).unwrap(), lo, hi,
        ).unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(
            (at_raw_zero - 2.025).abs() < 1e-4,
            "raw zero decodes to {at_raw_zero}; the midpoint is what a naive init would pick",
        );
    }

    /// The Gibbs shift moves the DRAW, never the gradient path.
    ///
    /// It displaces the sample toward the observer's particle mean in units of
    /// the posterior's own scale, and it arrives detached — so `π`'s gradient
    /// to `raw_lmc_matrix` and to the encoder is bit-for-bit what it was
    /// without the shift. That is the property that lets the channel be turned
    /// on without re-deriving anything downstream.
    #[test]
    fn the_gibbs_shift_moves_the_draw_and_not_the_gradient() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut cfg = small_cfg();
        cfg.gibbs_noise_scale = 0.5;
        let dir = {
            use crate::config::KernelNetworkConfig;
            use crate::model::KernelNetwork;
            let kn = KernelNetwork::load(
                KernelNetworkConfig {
                    bottleneck_dim: cfg.bottleneck_dim,
                    spectral_emb_dim: cfg.spectral_emb_dim,
                    individual_kernel_dim_out: 8,
                    num_primitives: 3,
                    num_gates: 8,
                    norm_eps: 1e-5,
                    time_dim: cfg.time_dim,
                },
                vb.pp("kernel_network"),
            )
            .unwrap();
            // STOCHASTIC: the shift lives on the reparam branch, so the
            // deterministic mode would make this test vacuous.
            AmortisedDirichlet::load(cfg.clone(), DirichletMode::Stochastic, vb, kn).unwrap()
        };

        let b = 4usize;
        let mu = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        let factor =
            Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim, cfg.alpha_factor_rank), &dev).unwrap();
        let diag = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        // FIXED noise, so the only thing that can move the draw is the shift.
        let eta_d = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
        let eta_f = Tensor::randn(0f32, 1f32, (b, cfg.alpha_factor_rank), &dev).unwrap();
        let n = Some((&eta_d, &eta_f));
        // A particle mean far from the posterior — the observer disagreeing.
        let target = Tensor::full(-4.0f32, (1, cfg.evidence_dim), &dev).unwrap();

        let (_, _, plain, _) =
            dir.get_local_evidence(&mu, &factor, &diag, n, None, None).unwrap();
        let (_, _, shifted, _) =
            dir.get_local_evidence(&mu, &factor, &diag, n, None, Some(&target)).unwrap();

        let d: f32 = (&shifted - &plain).unwrap().abs().unwrap()
            .max_all().unwrap().to_vec0().unwrap();
        assert!(d > 1e-3, "the shift moved the draw by {d}; it must move it at all");

        // And a scale of 0 is an exact no-op, so the knob is honest.
        let mut off = cfg.clone();
        off.gibbs_noise_scale = 0.0;
        let vm2 = candle_nn::VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&vm2, DType::F32, &dev);
        let dir_off = test_dirichlet_stochastic(vb2, &off);
        let (_, _, a, _) = dir_off.get_local_evidence(&mu, &factor, &diag, n, None, None).unwrap();
        let (_, _, c, _) =
            dir_off.get_local_evidence(&mu, &factor, &diag, n, None, Some(&target)).unwrap();
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&a), bits(&c), "gibbs_noise_scale = 0 must be an exact no-op");
    }

    #[cfg(test)]
    fn test_dirichlet_stochastic(
        vb: candle_nn::VarBuilder,
        cfg: &DirichletConfig,
    ) -> AmortisedDirichlet<crate::model::KernelNetwork> {
        use crate::config::KernelNetworkConfig;
        use crate::model::KernelNetwork;
        let kn = KernelNetwork::load(
            KernelNetworkConfig {
                bottleneck_dim: cfg.bottleneck_dim,
                spectral_emb_dim: cfg.spectral_emb_dim,
                individual_kernel_dim_out: 8,
                num_primitives: 3,
                num_gates: 8,
                norm_eps: 1e-5,
                time_dim: cfg.time_dim,
            },
            vb.pp("kernel_network"),
        )
        .unwrap();
        AmortisedDirichlet::load(cfg.clone(), DirichletMode::Stochastic, vb, kn).unwrap()
    }

    /// THE SIMPLEX CARRIES GRADIENT BACK TO THE LOADINGS, with the Gibbs
    /// injection live.
    ///
    /// `π → coreg(L) → a control-head-shaped read` must reach
    /// `raw_lmc_matrix`. Python gates this channel off in several places, and a
    /// gated channel is indistinguishable from a severed one from the outside —
    /// candle reports nothing at all when a graph is cut, `backward()` simply
    /// succeeds with the entry missing. So it is asserted, both with the Gibbs
    /// shift on and with it off, and the two must agree: the shift displaces
    /// the draw and must not change what the gradient can reach.
    #[test]
    fn the_control_head_reaches_the_lmc_loadings_through_the_simplex() {
        let dev = Device::Cpu;
        let target = Tensor::full(-4.0f32, (1, 8), &dev).unwrap();

        let run = |scale: f64, gibbs: Option<&Tensor>| -> (f64, f64) {
            let vm = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let mut cfg = small_cfg();
            cfg.gibbs_noise_scale = scale;
            let dir = test_dirichlet_stochastic(vb, &cfg);

            let b = 4usize;
            let mu = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
            let factor = Tensor::randn(
                0f32, 1f32, (b, cfg.evidence_dim, cfg.alpha_factor_rank), &dev,
            ).unwrap();
            let diag = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
            let eta_d = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();
            let eta_f = Tensor::randn(0f32, 1f32, (b, cfg.alpha_factor_rank), &dev).unwrap();

            let (qa, qb, _, _) = dir
                .get_local_evidence(&mu, &factor, &diag, Some((&eta_d, &eta_f)), None, gibbs)
                .unwrap();
            let beta = dir.global_stick_breaking_kumaraswamy(None).unwrap();
            let pi = dir.local_stick_breaking(&qa, &qb, &beta, None).unwrap();

            // A control-head-shaped read: the coregionalisation the drift heads
            // consume, contracted to a scalar. `B = (L Lᵀ) ⊙ (π πᵀ)`, so this
            // is the ONLY route from the simplex to the loadings.
            let bmat = dir.coregionalisation_matrix(&pi).unwrap();
            let loss = bmat.0.sqr().unwrap().mean_all().unwrap();
            let grads = loss.backward().unwrap();

            let data = vm.data().lock().unwrap();
            let g = |n: &str| -> f64 {
                let t: &Tensor = data[n].as_ref();
                grads
                    .get(t)
                    .map(|x| {
                        x.abs().unwrap().max_all().unwrap()
                            .to_dtype(DType::F64).unwrap().to_vec0::<f64>().unwrap()
                    })
                    .unwrap_or(0.0)
            };
            (g("raw_lmc_matrix"), g("raw_q_a_global"))
        };

        let (lmc_off, stick_off) = run(0.0, None);
        assert!(lmc_off > 0.0, "the simplex does not reach raw_lmc_matrix at all");
        assert!(stick_off > 0.0, "the simplex does not reach the global stick");

        let (lmc_on, stick_on) = run(0.5, Some(&target));
        assert!(
            lmc_on > 0.0,
            "the Gibbs injection severed the simplex → raw_lmc_matrix gradient",
        );
        assert!(stick_on > 0.0, "the Gibbs injection severed the stick's gradient");
    }

    /// The α→(q_a, q_b) derivation must be ONE derivation.
    ///
    /// `get_local_evidence` and `restick_from_alpha` each used to carry their
    /// own copy of `softplus(smooth_clamp(chunk(α), −2, 2)) + eps`. Now that
    /// the pair is also handed OUT — an out-of-crate caller assembling the
    /// packed state vector needs it — a divergence between the internal and
    /// external answers would be invisible: both are finite, plausible
    /// concentrations, and the stick would happily consume either.
    ///
    /// Bit-for-bit, not approximate: these are the same expression on the
    /// same tensor, so anything other than equality means the copies drifted.
    #[test]
    fn the_evidence_concentrations_are_derived_exactly_once() {
        use crate::config::KernelNetworkConfig;
        use crate::model::KernelNetwork;
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let mut cfg = DirichletConfig::default();
        cfg.latent_dim = 4;
        cfg.k_atoms = 5;
        cfg.evidence_dim = 8; // 2·(k−1)
        cfg.alpha_factor_rank = 2;
        cfg.num_fourier_features = 6;
        cfg.spectral_emb_dim = 32;
        cfg.bottleneck_dim = 16;
        cfg.num_latents = 8;
        cfg.time_dim = 12;
        let kn = KernelNetwork::load(
            KernelNetworkConfig {
                bottleneck_dim: 16,
                spectral_emb_dim: 32,
                individual_kernel_dim_out: 8,
                num_primitives: 3,
                num_gates: 8,
                norm_eps: 1e-5,
                time_dim: 12,
            },
            vb.pp("kernel_network"),
        )
        .unwrap();
        // Deterministic: `ab_logits == mu`, so the α fed to the public path
        // is exactly the one the internal path built.
        let dir =
            AmortisedDirichlet::load(cfg.clone(), DirichletMode::Deterministic, vb, kn).unwrap();

        let b = 3usize;
        let mu = Tensor::randn(0f32, 2f32, (b, cfg.evidence_dim), &dev).unwrap();
        let factor =
            Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim, cfg.alpha_factor_rank), &dev).unwrap();
        let diag = Tensor::randn(0f32, 1f32, (b, cfg.evidence_dim), &dev).unwrap();

        let (local_a, local_b, alpha, _) =
            dir.get_local_evidence(&mu, &factor, &diag, None, None, None).unwrap();
        let beta = dir.global_stick_breaking_kumaraswamy(None).unwrap();
        let (pi, qa, qb) = dir.restick_from_alpha_with_evidence(&alpha, &beta).unwrap();

        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&qa), bits(&local_a), "q_a diverged from get_local_evidence");
        assert_eq!(bits(&qb), bits(&local_b), "q_b diverged from get_local_evidence");

        // The one-return wrapper must be exactly the three-return path's π.
        assert_eq!(bits(&dir.restick_from_alpha(&alpha, &beta).unwrap()), bits(&pi));
        assert_eq!(pi.dims(), &[b, cfg.k_atoms]);
        assert_eq!(qa.dims(), &[b, cfg.k_atoms - 1]);

        // And the newly-public `B` half lines up with the `W` wrapper.
        let (b_dense, w) = dir.coregionalisation_matrix(&pi).unwrap();
        assert_eq!(b_dense.dims(), &[b, cfg.k_atoms, cfg.k_atoms]);
        assert_eq!(bits(&w), bits(&dir.lmc_consensus_from_pi(&pi).unwrap()));
    }

    /// Full trait-forward over a `[B·T, latent]` Itô-window batch: evidence
    /// supplied at `[B, …]` must be repeat-interleaved to align (the Python
    /// `_align_batch`), the consensus is the per-sample `[B·T, num_latents]`
    /// simplex, and the same noise bundle must reproduce bit-for-bit.
    #[test]
    fn forward_aligns_ito_window_batch_and_is_bundle_deterministic() {
        use crate::model::KernelNetwork;
        use crate::config::KernelNetworkConfig;
        use candle_nn::VarMap;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let mut cfg = DirichletConfig::default();
        cfg.latent_dim = 4;
        cfg.k_atoms = 5;
        cfg.evidence_dim = 8; // 2·(k−1)
        cfg.alpha_factor_rank = 2;
        cfg.num_fourier_features = 6;
        cfg.spectral_emb_dim = 32;
        cfg.bottleneck_dim = 16;
        cfg.num_latents = 8;
        cfg.dirichlet_t_window = 1;
        // The dirichlet-level FiLM head is now ALWAYS built on a trainable
        // map, so the fixture's t_emb width must match the configured
        // time_dim (it shares the width the kernel network uses below).
        cfg.time_dim = 12;

        let kn_cfg = KernelNetworkConfig {
            bottleneck_dim: 16,
            spectral_emb_dim: 32,
            individual_kernel_dim_out: 8,
            num_primitives: 3,
            num_gates: 8,
            norm_eps: 1e-5,
            time_dim: 12,
        };
        let kn = KernelNetwork::load(kn_cfg, vb.pp("kernel_network")).unwrap();
        let dir = AmortisedDirichlet::load(cfg.clone(), DirichletMode::Stochastic, vb, kn).unwrap();

        let (b0, t_win) = (2usize, 3usize);
        let bt = b0 * t_win;
        let x = Tensor::randn(0f32, 1f32, (bt, cfg.latent_dim), &dev).unwrap();

        // Evidence at the UN-interleaved batch [B₀, …].
        let enc = EncoderOutput {
            alpha_mu: Tensor::randn(0f32, 1f32, (b0, cfg.evidence_dim), &dev).unwrap(),
            alpha_factor: Tensor::randn(0f32, 0.1f32, (b0, cfg.evidence_dim, cfg.alpha_factor_rank), &dev).unwrap(),
            alpha_diag: Tensor::ones((b0, cfg.evidence_dim), DType::F32, &dev).unwrap(),
            alpha: Tensor::ones((b0, cfg.evidence_dim), DType::F32, &dev).unwrap(),
            log_pi: Tensor::zeros((b0, cfg.k_atoms), DType::F32, &dev).unwrap(),
            pi: Tensor::ones((b0, cfg.k_atoms), DType::F32, &dev).unwrap(),
            z: Tensor::randn(0f32, 1f32, (b0, cfg.latent_dim), &dev).unwrap(),
            mu_z: Tensor::randn(0f32, 1f32, (b0, cfg.latent_dim), &dev).unwrap(),
            logvar_z: Tensor::zeros((b0, cfg.latent_dim), DType::F32, &dev).unwrap(),
            ls: Tensor::ones((b0, cfg.k_atoms), DType::F32, &dev).unwrap(),
            real_x: Tensor::randn(0f32, 1f32, (b0, 1, 7), &dev).unwrap(),
        };

        let mut sites = std::collections::HashMap::new();
        sites.insert(
            "global_stick".to_string(),
            crate::model::noise::sobol_randn_like(&[cfg.k_atoms - 1], &dev, 1).unwrap(),
        );
        sites.insert(
            "local_stick".to_string(),
            crate::model::noise::sobol_randn_like(&[bt, cfg.k_atoms - 1], &dev, 2).unwrap(),
        );
        sites.insert(
            "local_evidence_d".to_string(),
            crate::model::noise::sobol_randn_like(&[bt, cfg.evidence_dim], &dev, 3).unwrap(),
        );
        sites.insert(
            "local_evidence_f".to_string(),
            crate::model::noise::sobol_randn_like(&[bt, cfg.alpha_factor_rank], &dev, 4).unwrap(),
        );
        let bundle = crate::model::noise::NoiseBundle::from_map(sites);

        let t_emb = Tensor::randn(0f32, 1f32, (bt, 12), &dev).unwrap();
        let out = Dirichlet::forward(&dir, &x, Some(&t_emb), &enc, None, 1, 0, false, Some(&bundle), None).unwrap();

        // Batch axis carries the Itô window everywhere.
        assert_eq!(out.pi.dims(), &[bt, cfg.k_atoms]);
        assert_eq!(out.z_in.dims(), &[bt, cfg.latent_dim]);
        assert_eq!(out.features.dims(), &[bt, cfg.spectral_emb_dim]);
        assert_eq!(out.lmc_consensus.dims(), &[bt, cfg.num_latents], "per-sample latent simplex");
        assert_eq!(out.lmc_matrices.dims(), &[bt, cfg.k_atoms, cfg.k_atoms]);
        // Evidence was repeat-interleaved: local_conc rows come in T-blocks.
        assert_eq!(out.local_conc.dims(), &[bt, cfg.evidence_dim]);
        // bw stash: [current ‖ previous] on the latent axis.
        let bw32 = out.bandwidth_loss_term.as_ref().unwrap();
        assert_eq!(bw32.dims(), &[bt, cfg.k_atoms, 2 * cfg.latent_dim]);

        // π is a simplex.
        let sums = out.pi.sum(D::Minus1).unwrap().to_vec1::<f32>().unwrap();
        assert!(sums.iter().all(|s| (s - 1.0).abs() < 1e-4), "pi sums {sums:?}");
        // Consensus is a simplex over latents.
        let wsums = out.lmc_consensus.sum(D::Minus1).unwrap().to_vec1::<f32>().unwrap();
        assert!(wsums.iter().all(|s| (s - 1.0).abs() < 1e-4), "W sums {wsums:?}");

        // Same bundle ⇒ bit-for-bit reproduction even in Stochastic mode.
        let out2 = Dirichlet::forward(&dir, &x, Some(&t_emb), &enc, None, 1, 0, false, Some(&bundle), None).unwrap();
        let d = out.pi.sub(&out2.pi).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(d, 0.0, "bundle-seeded forward must be deterministic");
    }

    #[test]
    fn config_default_matches_python() {
        let c = DirichletConfig::default();
        assert_eq!(c.latent_dim, 16);
        assert_eq!(c.k_atoms, 30);
        assert_eq!(c.num_fourier_features, 128);
        assert_eq!(c.spectral_emb_dim, 2048);
        assert_eq!(c.evidence_dim, 58);
        assert_eq!(c.alpha_factor_rank, 3);
        assert_eq!(c.num_latents, 8);
    }

    #[test]
    fn softplus_matches_reference_at_zero() {
        let d = Device::Cpu;
        let x = Tensor::zeros((3,), DType::F32, &d).unwrap();
        let sp = softplus(&x).unwrap();
        let v = sp.to_vec1::<f32>().unwrap();
        // softplus(0) = log(2) ≈ 0.6931
        for vi in v {
            assert!((vi - 0.6931472_f32).abs() < 1e-5);
        }
    }

    #[test]
    fn interval_clamps_to_range() {
        let d = Device::Cpu;
        // Very large positive input → near upper bound.
        let x = Tensor::new(&[100.0_f32], &d).unwrap();
        let y = interval(&x, -5.0, 5.0).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        assert!((v[0] - 5.0).abs() < 1e-3);
        // Very large negative input → near lower bound.
        let x = Tensor::new(&[-100.0_f32], &d).unwrap();
        let y = interval(&x, -5.0, 5.0).unwrap();
        let v = y.to_vec1::<f32>().unwrap();
        assert!((v[0] + 5.0).abs() < 1e-3);
    }

    #[test]
    fn prepend_zero_1d() {
        let d = Device::Cpu;
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &d).unwrap();
        let y = prepend_zero_last_dim(&x).unwrap();
        assert_eq!(y.dims(), &[4]);
        let v = y.to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn chunk_last_dim_2d() {
        let d = Device::Cpu;
        let x = Tensor::arange(0.0_f32, 8.0, &d).unwrap().reshape((2, 4)).unwrap();
        let (a, b) = chunk_last_dim(&x, 2).unwrap();
        assert_eq!(a.dims(), &[2, 2]);
        assert_eq!(b.dims(), &[2, 2]);
    }
}


/// Pairwise categorical KL loss for normalised Dirichlet mean probabilities,
/// weighted by the actual bridge coupling. Rows must be strictly positive.
pub fn bridge_weighted_divergence(policy: &Tensor, reference: &Tensor, coupling: &Tensor) -> candle_core::Result<Tensor> {
    let (n, k) = policy.dims2()?;
    let (m, rk) = reference.dims2()?;
    if k == 0 || rk != k || coupling.dims() != [n, m] { candle_core::bail!("Dirichlet/coupling shape mismatch"); }
    for rows in [policy, reference] {
        let values = rows.to_dtype(candle_core::DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0) || values.chunks(k).any(|row| (row.iter().sum::<f64>() - 1.0).abs() > 1e-5) {
            candle_core::bail!("Dirichlet means must be positive probability rows");
        }
    }
    let mass = coupling.sum_all()?;
    let value = mass.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    let min = coupling.min_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    if !value.is_finite() || value <= 0.0 || !min.is_finite() || min < 0.0 { candle_core::bail!("invalid coupling mass"); }
    let p = policy.unsqueeze(1)?;
    let log_ratio = p.log()?.broadcast_sub(&reference.unsqueeze(0)?.log()?)?;
    p.broadcast_mul(&log_ratio)?.sum(2)?.mul(coupling)?.sum_all()?.div(&mass)
}
