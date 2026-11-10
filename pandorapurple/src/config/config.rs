//! Inferno — Serializable configuration values for model construction and numerical routines.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ===========================================================================
// Module-level constants — mirror Python K_ATOMS / EVIDENCE / etc.
// ===========================================================================

pub const K_ATOMS: usize = 30;
pub const EVIDENCE: usize = (K_ATOMS - 1) * 2;
pub const LATENTS: usize = 8;
/// Counterfactual heads `K` — how many sparse counterfactuals the entmax
/// router emits per batch element (Python `n_sparse`).
pub const N_SPARSE: usize = 8;
pub const TINY_EPS: f64 = 3e-8;
pub const LARGE_EPS: f64 = 4e-2;
pub const JITTER: f64 = 1e-6;
pub const EPS: f64 = 1e-3;
pub const NEUTRAL_LOGIT: f32 = -4.0;
pub const JEFFREYS: f32 = -0.5413;

pub const MIN_LS: f64 = 0.05;
pub const MAX_LS: f64 = 15.0;

pub const RFF_DIM: usize = 128;
pub const SPECTRAL_PROJ_DIM: usize = 2048;
pub const TRAIN_BATCH: usize = 1024;
pub const TRAIN_DATA_SIZE: f32 = 1.0;
pub const GLOBAL_RANK: usize = 3;
pub const INPUT_DIM: usize = 128;
pub const BOTTLENECK: usize = 64;
pub const LATENT_DIM: usize = 16;
/// Default temporal input length for the core VAE.
pub const DATA_SEQ_LEN: usize = 32;
pub const LOGVAR_CLAMP: (f64, f64) = (-10.0, 3.0);
pub const REPARAM_CLAMPS: (f64, f64) = (-3.3, 3.3);

// NKN / kernel-network constants — shared by `DirichletConfig` and
// `KernelNetworkConfig` so a value change here propagates to both.
pub const NUM_PRIMITIVES: usize = 3;
pub const INDIVIDUAL_KERNEL_DIM_OUT: usize = 32;
pub const KERNEL_NORM_EPS: f64 = 1e-5;

// DIRICHLET PARAMS:
pub const DROPOUT: f32 = 0.0;
pub const GAMMA_INIT: f32 = 1.5417;
pub const NOISE_SCALAR: f32 = 0.316;
pub const PSI_SCALE: f32 = 1.0;
pub const POSTERIOR_DIRICHLET_EPS: f64 = 2e-5;
pub const CONC_CLAMP: f64 = 9.0;
pub const CONC_STRENGTH: f32 = 77.0;
pub const SIG_MIN: f64 = 1e-4;
pub const SIG_MAX: f64 = 5.0;
pub const MU_MIN: f64 = -17.0;
pub const MU_MAX: f64 = 17.0;
pub const EPS_CLIP: f64 = 2.7;
pub const SB_EPS: f64 = 3e-3;
pub const UNI_CLAMP: f64 = 5e-5;
pub const NUM_TAYLOR_EXPANSIONS: usize = 9;
pub const FREQ_BOUNDS: (f64, f64) = (0.01, 20.0);
pub const W_TAU: f64 = 0.86;
pub const LMC_CLAMPS: (f64, f64) = (-4.0, 4.0);
pub const PARAMS_MIN_CONSTRAINT: f64 = 1e-4;
pub const TIGHTS: (f64, f64) = (0.01, 1.5);
pub const BROADS: (f64, f64) = (0.01, 5.0);
pub const ALPHA_RANK: usize = 3;

//DECODER PARAMS:
pub const DISENTANGLE_SPLIT: [usize; 3] = [4, 30, 30];

pub const LS_HEAD_SPLIT: [usize; 2] = [4, 60];
pub const PI_HEAD_SPLIT: [usize; 2] = [30, 60];
pub const DATA_HEAD_SPLIT: [usize; 2] = [30, 60];

pub const BETA_DEFAULT: f32 = 1.0;

//BERT

//SDE


pub const GATES: usize = 8;





//Ito Time

//Runtime
// ===========================================================================
// Core model configuration
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectralVaeConfig {
    /// Applicant input feature count (Python `input_dim`).
    pub input_dim: usize,
    /// Sequence length (Python `seq_len`).
    pub seq_len: usize,
    /// Latent dim per expert (Python `latent_dim`).
    pub latent_dim: usize,
    /// Dirichlet latent window length `T`: the boundary consumes the
    /// flattened rolling window `[B, latent_dim·T]` (edge-padded until the
    /// history fills). `1` = legacy single-step input. The Python model
    /// runs `T = ITO_T_HISTORY = 12` (`z_dim 16 × 12 = 192`-wide mixer).
    #[serde(default = "default_dirichlet_t_window")]
    pub dirichlet_t_window: usize,
    /// Projection width of `latent_ssm.predictor` — the SSM's self-supervised
    /// InfoNCE head. `0` builds no head and no `predictor.*` keys.
    #[serde(default = "default_ssm_predict_dim")]
    pub ssm_predict_dim: usize,
    /// Prediction horizons, one `(q, k)` projection pair each.
    #[serde(default = "default_ssm_predict_horizons")]
    pub ssm_predict_horizons: Vec<usize>,
    /// InfoNCE temperature for that head.
    #[serde(default = "default_ssm_predict_temp")]
    pub ssm_predict_temp: f64,
    /// Number of Dirichlet atoms / cluster count (Python `k_atoms`).
    pub k_atoms: usize,
    /// `2 * (k_atoms - 1)` — evidence dim for the α-posterior.
    pub evidence_dim: usize,
    /// Master time-conditioning width (`config.model.time_dim`, Python
    /// default 64): the FourierTimeEmbedding's embed_dim AND every FiLM
    /// head's input width. Reconciled from the checkpoint at load.
    #[serde(default = "default_time_dim_model")]
    pub time_dim: usize,
    /// FourierTimeEmbedding `n_features` (Python `FOURIER_TIME_DIM = 64`).
    /// Reconciled from the checkpoint (`time_emb.mlp.0.weight` is
    /// `[time_dim, 2·time_rff_dim]`).
    #[serde(default = "default_time_rff_dim")]
    pub time_rff_dim: usize,
    /// Number of LMC latents (Python `num_latents`).
    pub num_latents: usize,
    /// Number of MoE experts; equals `num_latents` in the current model.
    pub num_experts: usize,
    /// LoRA-style rank for the alpha-factor (Python `rank`).
    pub rank: usize,
    /// Distinct knob from `rank` in Python, equal in practice.
    pub alpha_factor_rank: usize,
    /// Dirichlet bottleneck width (Python `bottleneck_dim`).
    pub bottleneck_dim: usize,
    
    /// Neutral logit used for `alpha_mu` initialisation (Python: -4.0).
    pub neutral_logit: f32,
    /// `-jeffreys_prior` used for `alpha_diag` initialisation (Python: -0.5413).
    pub alpha_diag_init: f32,

    
    pub jitter: f64,
    pub eps: f64,
    pub large_eps: f64,
    pub tiny_eps: f64,
    pub norm_eps: f64,
    
    pub min_ls: f64,
    pub max_ls: f64,

    // --- training-side / spectral dims ---
    pub training_batch_size: usize,
    pub num_fourier_features: usize,
    pub spectral_emb_dim: usize,
    pub n_data: f32,
}

fn default_dirichlet_t_window() -> usize {
    1
}

fn default_true() -> bool {
    true
}

fn default_omega_scale_bounds() -> (f64, f64) {
    (0.05, 4.0)
}

/// Width of the SSM's self-supervised InfoNCE head. `0` builds no
/// `latent_ssm.predictor.*` at all — Python's own default, and the tree every
/// pre-head checkpoint carries. **32** is what production trains and what the
/// live artefact holds (`vae.latent_ssm.predictor.q1.weight [32, 16]`).
fn default_ssm_predict_dim() -> usize {
    32
}

/// One projection pair per horizon. Predicting only `t+1` is coverable by a
/// one-lag map; `[1, 2, 4]` is not, because no memoryless function of row `t`
/// knows row `t+4`.
fn default_ssm_predict_horizons() -> Vec<usize> {
    vec![1, 2, 4]
}

fn default_ssm_predict_temp() -> f64 {
    0.1
}

fn default_n_sparse() -> usize {
    N_SPARSE
}

fn default_counterfactual_routing_dim() -> usize {
    LATENTS
}

fn default_decoder_time_dim() -> usize {
    64
}

impl Default for SpectralVaeConfig {
    fn default() -> Self {
        Self {
            input_dim: INPUT_DIM,
            seq_len: DATA_SEQ_LEN,
            latent_dim: LATENT_DIM,
            dirichlet_t_window: default_dirichlet_t_window(),
            ssm_predict_dim: default_ssm_predict_dim(),
            ssm_predict_horizons: default_ssm_predict_horizons(),
            ssm_predict_temp: default_ssm_predict_temp(),
            k_atoms: K_ATOMS,
            evidence_dim: EVIDENCE,
            time_dim: default_time_dim_model(),
            time_rff_dim: default_time_rff_dim(),
            num_latents: LATENTS,
            num_experts: LATENTS,
            rank: GLOBAL_RANK,
            alpha_factor_rank: GLOBAL_RANK,
            bottleneck_dim: BOTTLENECK,
            eps: EPS,
            neutral_logit: NEUTRAL_LOGIT,
            alpha_diag_init: JEFFREYS,
            jitter: JITTER,
            large_eps: LARGE_EPS,
            tiny_eps: TINY_EPS,
            norm_eps: KERNEL_NORM_EPS,
            min_ls: MIN_LS,
            max_ls: MAX_LS,
            training_batch_size: TRAIN_BATCH,
            num_fourier_features: RFF_DIM,
            spectral_emb_dim: SPECTRAL_PROJ_DIM,
            n_data: TRAIN_DATA_SIZE,
        }
    }
}


impl SpectralVaeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.evidence_dim != 2 * (self.k_atoms - 1) {
            bail!(
                "SpectralVaeConfig.evidence_dim ({}) must equal 2 * (k_atoms - 1) = {}",
                self.evidence_dim, 2 * (self.k_atoms - 1)
            );
        }
        if self.min_ls >= self.max_ls {
            bail!(
                "SpectralVaeConfig.min_ls ({}) must be < max_ls ({})",
                self.min_ls, self.max_ls
            );
        }
        if self.eps <= 0.0 || self.jitter <= 0.0
            || self.large_eps <= 0.0 || self.tiny_eps <= 0.0
        {
            bail!("SpectralVaeConfig: jitter / eps / large_eps / tiny_eps must all be > 0");
        }
        if self.input_dim == self.k_atoms {
            tracing::warn!(
                input_dim = self.input_dim, k_atoms = self.k_atoms,
                "SpectralVaeConfig.input_dim == k_atoms; these are independent quantities. \
                 Set input_dim ~ 2 * bottleneck_dim."
            );
        }
        if self.input_dim < 2 * self.bottleneck_dim {
            tracing::warn!(
                input_dim = self.input_dim, bottleneck_dim = self.bottleneck_dim,
                "SpectralVaeConfig.input_dim < 2 * bottleneck_dim — stem will expand before compressing"
            );
        }
        Ok(())
    }
}

// ===========================================================================
// EncoderConfig — Python `class EncoderConfig(SharingIsCaring)`
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EncoderConfig {
    // shared dims
    pub input_dim: usize,
    pub seq_len: usize,
    pub latent_dim: usize,
    pub bottleneck_dim: usize,
    pub k_atoms: usize,
    pub evidence_dim: usize,
    pub num_latents: usize,
    /// Python `SharingIsCaring` fields the encoder inherits; carried here
    /// for YAML parity (extra="forbid" on both sides).
    pub num_experts: usize,
    pub rank: usize,
    pub training_batch_size: usize,

    pub n_data: f32,

    pub logvar_clamp: (f64, f64),
    pub reparam_eps_clamp: (f64, f64),

    // shared numerical bounds
    pub jitter: f64,
    pub eps: f64,
    pub large_eps: f64,
    pub tiny_eps: f64,
    pub norm_eps: f64,
    
    pub min_ls: f64,
    pub max_ls: f64,

    // encoder-specific
    pub num_fourier_features: usize,
    pub spectral_emb_dim: usize,
    pub alpha_factor_rank: usize,
    /// FiLM head input width (mirror of `SpectralVaeConfig::time_dim`).
    #[serde(default = "default_time_dim_model")]
    pub time_dim: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: INPUT_DIM,
            seq_len: DATA_SEQ_LEN,
            latent_dim: LATENT_DIM,
            bottleneck_dim: BOTTLENECK,
            k_atoms: K_ATOMS,
            evidence_dim: EVIDENCE,
            time_dim: default_time_dim_model(),
            num_latents: LATENTS,
            num_experts: LATENTS,
            rank: GLOBAL_RANK,
            training_batch_size: TRAIN_BATCH,
            n_data: TRAIN_DATA_SIZE,

            logvar_clamp: LOGVAR_CLAMP,
            reparam_eps_clamp: REPARAM_CLAMPS,

            jitter: JITTER,
            eps: EPS,
            large_eps: LARGE_EPS,
            tiny_eps: TINY_EPS,
            norm_eps: KERNEL_NORM_EPS,
            min_ls: MIN_LS,
            max_ls: MAX_LS,
            num_fourier_features: RFF_DIM,
            spectral_emb_dim: SPECTRAL_PROJ_DIM,
            alpha_factor_rank: GLOBAL_RANK,
        }
    }
}

impl EncoderConfig {
    pub fn validate(&self) -> Result<()> {
        if self.input_dim == self.k_atoms {
            tracing::warn!(
                input_dim = self.input_dim, k_atoms = self.k_atoms,
                "EncoderConfig.input_dim == k_atoms; these are independent."
            );
        }
        if self.input_dim < 2 * self.bottleneck_dim {
            tracing::warn!(
                input_dim = self.input_dim, bottleneck_dim = self.bottleneck_dim,
                "EncoderConfig.input_dim < 2 * bottleneck_dim — stem expands before compressing"
            );
        }
        if self.min_ls >= self.max_ls {
            bail!("EncoderConfig.min_ls must be < max_ls");
        }
        Ok(())
    }
}

// ===========================================================================
// DirichletConfig — Python `class DirichletConfig(SharingIsCaring)`
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DirichletConfig {
    // shared dims
    pub input_dim: usize,
    /// See `SharedConfig::seq_len` — `None` inherits the authority.
    pub seq_len: Option<usize>,
    pub latent_dim: usize,
    /// Latent window length `T` consumed by the bottleneck mixer
    /// (`latent_dim·T`-wide input; RFF + `z_in` stay current-step).
    /// Mirror of `SpectralVaeConfig::dirichlet_t_window`.
    #[serde(default = "default_dirichlet_t_window")]
    pub dirichlet_t_window: usize,
    pub bottleneck_dim: usize,
    pub k_atoms: usize,
    pub evidence_dim: usize,
    /// Route the FiLM'd bottleneck into the concentration logits through
    /// `latent_evidence_head` — the latent's OWN contribution to the simplex.
    /// Python's default is **on**; `false` leaves the head registered but
    /// never called, exactly as Python's flag does (so the flag never changes
    /// the checkpoint's key set).
    #[serde(default = "default_true")]
    pub latent_simplex_coupling: bool,
    /// λ on the passive-IRL Gibbs shift of the evidence DRAW. `0` disables it.
    ///
    /// Off by default because the shift bypasses the ELBO — the KL prices
    /// `(μ, Σ)` while the sample comes from `N(μ + λ(ᾱ − μ), Σ)` — and the
    /// displacement is in logit units, so a confident posterior gets a draw
    /// many σ into its own tail.
    #[serde(default)]
    pub gibbs_noise_scale: f64,
    /// Box for `raw_omega_scale`, the per-atom per-dim SPREAD of the RFF
    /// frequency draw. Split into the two Python field names for YAML
    /// parity; consumers use `omega_scale_bounds()`.
    pub omega_scale_min: f64,
    pub omega_scale_max: f64,
    /// Redraw the RFF frequencies every forward from the seeded `rff_omega`
    /// noise site instead of reusing the construction-time buffer.
    pub omega_resample: bool,
    pub num_gates: usize,
    /// Sparse-process count for the shared atoms→z projection.
    pub n_sparse: usize,
    pub num_latents: usize,
    pub num_experts: usize,
    pub rank: usize,
    pub training_batch_size: usize,
    pub n_data: f32,
    pub num_fourier_features: usize,
    pub spectral_emb_dim: usize,

    // shared numerical bounds
    pub jitter: f64,
    pub eps: f64,
    pub large_eps: f64,
    pub tiny_eps: f64,
    pub norm_eps: f64,
    
    pub min_ls: f64,
    pub max_ls: f64,

    // dirichlet-specific hyperparameters
    pub dropout: f32,
    pub gamma_concentration_init: f32,
    pub noise_scalar: f32,
    pub psi_scale: f32,

    // numerical bounds (dirichlet-specific)
    pub posterior_dirichlet_epsilon: f64,

    pub conc_clamp: f64,

    pub concentration_strength: f32,
    pub sigma_lower_bound: f64,
    pub sigma_upper_bound: f64,

    pub mu_lower_bound: f64,
    pub mu_upper_bound: f64,

    pub eps_clip: f64,
    pub stick_breaking_epsilon: f64,
    pub uniform_dist_clamp: f64,

    /// Python spells this `M_series` in YAML; the alias accepts both.
    #[serde(alias = "M_series")]
    pub m_series: usize,
    pub freq_bounds: (f64, f64),

    pub consensus_tau: f64,

    pub lmc_constraint: (f64, f64),
    pub scale_constraint_lower: f64,

    pub tight_var: (f64, f64),
    pub broad_var: (f64, f64),
    pub alpha_factor_rank: usize,

    /// FiLM head input width (mirror of `SpectralVaeConfig::time_dim`).
    #[serde(default = "default_time_dim_model")]
    pub time_dim: usize,

    // NKN
    pub num_primitives: usize,
    pub individual_kernel_dim_out: usize,
}

impl Default for DirichletConfig {
    fn default() -> Self {
        Self {
            input_dim: INPUT_DIM,
            seq_len: None,
            latent_dim: LATENT_DIM,
            dirichlet_t_window: default_dirichlet_t_window(),
            bottleneck_dim: BOTTLENECK,
            k_atoms: K_ATOMS,
            evidence_dim: EVIDENCE,
            latent_simplex_coupling: true,
            gibbs_noise_scale: 0.0,
            omega_scale_min: default_omega_scale_bounds().0,
            omega_scale_max: default_omega_scale_bounds().1,
            omega_resample: false,
            num_gates: GATES,
            n_sparse: N_SPARSE,
            num_latents: LATENTS,
            num_experts: LATENTS,
            rank: GLOBAL_RANK,
            training_batch_size: TRAIN_BATCH,
            n_data: TRAIN_DATA_SIZE,
            num_fourier_features: RFF_DIM,
            spectral_emb_dim: SPECTRAL_PROJ_DIM,
            jitter: JITTER,
            eps: EPS,
            large_eps: LARGE_EPS,
            tiny_eps: TINY_EPS,
            norm_eps: KERNEL_NORM_EPS,
            min_ls: MIN_LS,
            max_ls: MAX_LS,

            dropout: DROPOUT,
            gamma_concentration_init: GAMMA_INIT,
            noise_scalar: NOISE_SCALAR,
            psi_scale: PSI_SCALE,

            posterior_dirichlet_epsilon: POSTERIOR_DIRICHLET_EPS,
            conc_clamp: CONC_CLAMP,
            concentration_strength: CONC_STRENGTH,
            sigma_lower_bound: SIG_MIN,
            sigma_upper_bound: SIG_MAX,
            mu_lower_bound: MU_MIN,
            mu_upper_bound: MU_MAX,
            
            eps_clip: EPS_CLIP,
            stick_breaking_epsilon: SB_EPS,
            uniform_dist_clamp: UNI_CLAMP,
            m_series: NUM_TAYLOR_EXPANSIONS,
            freq_bounds: FREQ_BOUNDS,
            consensus_tau: W_TAU,
            lmc_constraint: LMC_CLAMPS,
            scale_constraint_lower: PARAMS_MIN_CONSTRAINT,
            tight_var: TIGHTS,
            broad_var: BROADS,
            alpha_factor_rank: ALPHA_RANK,

            time_dim: default_time_dim_model(),

            num_primitives: NUM_PRIMITIVES,
            individual_kernel_dim_out: INDIVIDUAL_KERNEL_DIM_OUT,
        }
    }
}

impl DirichletConfig {
    /// The `(min, max)` box for `raw_omega_scale` — the pre-split field.
    #[inline]
    pub fn omega_scale_bounds(&self) -> (f64, f64) {
        (self.omega_scale_min, self.omega_scale_max)
    }

    pub fn validate(&self) -> Result<()> {
        if self.omega_scale_min <= 0.0 || self.omega_scale_max <= 0.0
            || self.omega_scale_min >= self.omega_scale_max
        {
            bail!(
                "DirichletConfig.omega_scale_min ({}) must be > 0 and < omega_scale_max ({})",
                self.omega_scale_min, self.omega_scale_max
            );
        }
        if self.n_sparse < 1 || self.n_sparse > self.k_atoms {
            bail!(
                "DirichletConfig.n_sparse ({}) must be in [1, k_atoms = {}]",
                self.n_sparse, self.k_atoms
            );
        }
        if self.num_gates < 1 || self.num_gates > 64 {
            bail!("DirichletConfig.num_gates ({}) must be in [1, 64]", self.num_gates);
        }
        if self.min_ls >= self.max_ls {
            bail!("DirichletConfig.min_ls ({}) must be < max_ls ({})",
                self.min_ls, self.max_ls);
        }
        if self.sigma_lower_bound >= self.sigma_upper_bound {
            bail!("DirichletConfig.sigma_lower_bound ({}) must be < sigma_upper_bound ({})",
                self.sigma_lower_bound, self.sigma_upper_bound);
        }
        if self.mu_lower_bound >= self.mu_upper_bound {
            bail!("DirichletConfig.mu_lower_bound ({}) must be < mu_upper_bound ({})",
                self.mu_lower_bound, self.mu_upper_bound);
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            bail!("DirichletConfig.dropout ({}) must be in [0, 1]", self.dropout);
        }
        if self.m_series < 1 {
            bail!("DirichletConfig.m_series must be ≥ 1");
        }
        if self.num_primitives < 1 {
            bail!("DirichletConfig.num_primitives must be ≥ 1");
        }
        Ok(())
    }
}

// ===========================================================================
// KernelNetworkConfig — NKN slab parameters
//
// Sits alongside DirichletConfig rather than nested under it because the
// kernel network is consumed in multiple places (Dirichlet head + the
// kernel slab in the GP path). Fields that overlap with the broader
// dimensional configs (`bottleneck_dim`, `spectral_emb_dim`, `num_gates`)
// are cross-checked in `RootConfig::validate`.
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KernelNetworkConfig {
    pub bottleneck_dim: usize,
    pub spectral_emb_dim: usize,
    pub individual_kernel_dim_out: usize,
    pub num_primitives: usize,
    /// Number of attention gates — must equal `num_latents` in the
    /// surrounding configs (one gate per LMC latent).
    pub num_gates: usize,
    /// LayerNorm / GroupNorm epsilon. Kept as f64 to match the original
    /// Python kernel-network module's dtype convention; the other eps
    /// fields in this file are f32.
    pub norm_eps: f64,
    /// Width of the master FiLM time conditioning (the FourierTimeEmbedding
    /// output the `film_projector` consumes). Python `time_dim = 256`.
    #[serde(default = "default_time_dim")]
    pub time_dim: usize,
}

fn default_time_dim() -> usize {
    default_time_dim_model()
}

/// Python `config.model.time_dim` default — the width the whole trained
/// stack was constructed at (SpectralVAE passes it into every submodule).
fn default_time_dim_model() -> usize {
    64
}

fn default_time_rff_dim() -> usize {
    64
}

impl Default for KernelNetworkConfig {
    fn default() -> Self {
        Self {
            bottleneck_dim: BOTTLENECK,
            spectral_emb_dim: SPECTRAL_PROJ_DIM,
            individual_kernel_dim_out: INDIVIDUAL_KERNEL_DIM_OUT,
            num_primitives: NUM_PRIMITIVES,
            num_gates: GATES,
            norm_eps: KERNEL_NORM_EPS,
            time_dim: default_time_dim(),
        }
    }
}

impl KernelNetworkConfig {
    /// Total width of the concatenated primitive block.
    #[inline]
    pub fn primitives_total_dim(&self) -> usize {
        self.num_primitives * self.individual_kernel_dim_out
    }

    pub fn validate(&self) -> Result<()> {
        if self.bottleneck_dim == 0 {
            bail!("KernelNetworkConfig.bottleneck_dim must be ≥ 1");
        }
        if self.spectral_emb_dim == 0 {
            bail!("KernelNetworkConfig.spectral_emb_dim must be ≥ 1");
        }
        if self.individual_kernel_dim_out == 0 {
            bail!("KernelNetworkConfig.individual_kernel_dim_out must be ≥ 1");
        }
        if self.num_primitives == 0 {
            bail!("KernelNetworkConfig.num_primitives must be ≥ 1");
        }
        if self.num_gates == 0 {
            bail!("KernelNetworkConfig.num_gates must be ≥ 1");
        }
        if self.norm_eps <= 0.0 {
            bail!("KernelNetworkConfig.norm_eps ({}) must be > 0", self.norm_eps);
        }
        Ok(())
    }
}

// ===========================================================================
// DecoderConfig — Python `class DecoderConfig(SharingIsCaring)`
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecoderConfig {
    // shared dims
    pub input_dim: usize,
    /// See `SharedConfig::seq_len` — `None` inherits the authority.
    pub seq_len: Option<usize>,
    pub latent_dim: usize,
    pub bottleneck_dim: usize,
    pub k_atoms: usize,
    pub evidence_dim: usize,
    pub num_latents: usize,
    pub num_experts: usize,
    /// Counterfactual heads `K` — the entmax router weight is
    /// `[n_sparse·num_experts, counterfactual_routing_dim]`, so this sizes
    /// the head. Reconciled from the checkpoint at load.
    #[serde(default = "default_n_sparse")]
    pub n_sparse: usize,
    /// Width of the routing signal fed to the counterfactual router (the
    /// NKN gate logits). Reconciled from the checkpoint at load.
    #[serde(default = "default_counterfactual_routing_dim")]
    pub counterfactual_routing_dim: usize,
    /// Master time-embedding width the decoder's `film_projector` consumes
    /// (Python `config.model.time_dim`; reconciled from the checkpoint).
    #[serde(default = "default_decoder_time_dim")]
    pub time_dim: usize,
    pub training_batch_size: usize,
    pub n_data: f32,
    pub alpha_factor_rank: usize,
    /// Python `SharingIsCaring` fields carried for YAML parity.
    pub rank: usize,
    pub num_fourier_features: usize,
    /// Python `DecoderConfig.top_k_sparse_processes`.
    pub top_k_sparse_processes: usize,
    pub disentangle_split: [usize; 3],
    
    pub ls_recon: [usize; 2],
    pub pi_recon: [usize; 2],
    pub data_recon: [usize; 2],

    // shared numerical bounds
    pub jitter: f64,
    pub eps: f64,
    pub large_eps: f64,
    pub tiny_eps: f64,
    pub norm_eps: f64,

    pub min_ls: f64,
    pub max_ls: f64,

    // decoder-specific
    pub spectral_compressions: Vec<usize>,
    pub spectral_emb_dim: usize,
    pub dropout: f32,
    /// VAE KL weight (β-VAE).
    pub beta: f32,
    pub conc_clamp: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            input_dim: INPUT_DIM,
            seq_len: None,
            latent_dim: LATENT_DIM,
            bottleneck_dim: BOTTLENECK,
            k_atoms: K_ATOMS,
            evidence_dim: EVIDENCE,
            num_latents: LATENTS,
            num_experts: LATENTS,
            n_sparse: N_SPARSE,
            counterfactual_routing_dim: LATENTS,
            time_dim: 64,
            training_batch_size: TRAIN_BATCH,
            n_data: TRAIN_DATA_SIZE,
            alpha_factor_rank: GLOBAL_RANK,
            rank: GLOBAL_RANK,
            num_fourier_features: RFF_DIM,
            top_k_sparse_processes: 5,
            disentangle_split: DISENTANGLE_SPLIT,
            
            ls_recon: LS_HEAD_SPLIT,
            pi_recon: PI_HEAD_SPLIT,
            data_recon: DATA_HEAD_SPLIT,

            jitter: JITTER,
            eps: EPS,
            large_eps: LARGE_EPS,
            tiny_eps: TINY_EPS,
            norm_eps: KERNEL_NORM_EPS,
            
            min_ls: MIN_LS,
            max_ls: MAX_LS,

            spectral_compressions: vec![1024, 512, 128, 64],
            
            spectral_emb_dim: SPECTRAL_PROJ_DIM,
            dropout: DROPOUT,
            beta: BETA_DEFAULT,
            conc_clamp: CONC_CLAMP,
        }
    }
}

impl DecoderConfig {
    pub fn validate(&self) -> Result<()> {
        if self.min_ls >= self.max_ls {
            bail!("DecoderConfig.min_ls must be < max_ls");
        }
        if self.spectral_compressions.is_empty() {
            bail!("DecoderConfig.spectral_compressions must not be empty");
        }
        if self.spectral_compressions.iter().any(|&c| c == 0) {
            bail!("DecoderConfig.spectral_compressions entries must all be > 0");
        }
        let mut sorted = self.spectral_compressions.clone();
        sorted.sort_by(|a, b| b.cmp(a)); // descending
        if self.spectral_compressions != sorted {
            tracing::warn!(
                compressions = ?self.spectral_compressions,
                "DecoderConfig.spectral_compressions is not monotonically decreasing; intended as a compression ladder."
            );
        }
        if self.beta <= 0.0 {
            bail!("DecoderConfig.beta ({}) must be > 0", self.beta);
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            bail!("DecoderConfig.dropout ({}) must be in [0, 1]", self.dropout);
        }
        Ok(())
    }
}



// ===========================================================================
// BuildConfig — pooler + time embedder (plus runtime-filled fields)
// ===========================================================================

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct RootConfig {
    pub vae: SpectralVaeConfig,
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub dirichlet: DirichletConfig,
    pub kernel_network: KernelNetworkConfig,
    pub ewma: super::EWMAConfig,
}
