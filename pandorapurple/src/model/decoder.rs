//! Inferno — Spectral decoders mapping latent distributions to reconstructed observations.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv_transpose1d, linear, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder,
};
use crate::model::norm::{layer_norm, LayerNorm, LayerNormConfig};

use crate::model::{DirichletOutput, softplus};
use crate::config::{DecoderConfig};

/// Sobol salt for the lengthscale reparam when no bundle site is supplied.
///
/// Distinct from every other salt in the tree so two sites never share a
/// draw; the value itself is arbitrary (b"lengths" in ASCII).
const LENGTHSCALE_FALLBACK_SALT: u64 = 0x6c65_6e67_7468_73;

// ============================================================================
// Execution mode
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderMode {
    Deterministic,
    Stochastic,
}

impl Default for DecoderMode {
    fn default() -> Self {
        DecoderMode::Deterministic
    }
}

// ============================================================================
// Output
// ============================================================================

/// Rust mirror of Python `DecoderOutput` NamedTuple (25 fields).
#[derive(Debug, Clone)]
pub struct DecoderOutput {
    // Carried forward from the Dirichlet output (shallow clones).
    pub gates: Tensor,
    pub mat52: Tensor,
    pub rbf: Tensor,
    pub rq: Tensor,
    pub pi: Tensor,
    pub lmc_matrices: Tensor,
    pub lmc_consensus: Tensor,
    pub real_x: Tensor,
    pub alpha: Tensor, // = dir_out.local_conc

    // Produced by the decoder itself.
    pub bottleneck: Tensor,           // [B, bottleneck_dim]
    pub alpha_mu: Tensor,             // [B, evidence_dim]
    pub alpha_factor: Tensor,         // [B, evidence_dim, rank]
    pub alpha_diag: Tensor,           // [B, evidence_dim]
    pub gp_features: Tensor,          // [B, num_experts, latent_dim]
    pub parameters_per_expert: Tensor,// [num_experts, B, k_atoms]
    pub recon: Tensor,                // [B, 32, input_dim]
    pub bandwidth_mod: Tensor,        // [B, num_latents]
    pub amp: Tensor,                  // [B, 32, input_dim]
    pub trend: Tensor,                // [B, 32, input_dim]
    pub res: Tensor,                  // [B, 32, input_dim]
    pub ls: Tensor,                   // [B, k_atoms]
    pub ls_logvar: Tensor,            // [B, k] — lengthscale posterior logvar
    pub bw_learned: Tensor,           // [B, k, latent_dim] learned bandwidths
    pub mu_z: Tensor,
    pub logvar_z: Tensor,

    // ── Counterfactual entmax head (new) ────────────────────────────────
    /// Pre-sigmoid gate logits carried from the kernel network when exposed
    /// (routing signal for the counterfactual head).
    pub gates_logits: Option<Tensor>,
    /// `[B, K, bottleneck_dim]` sparse counterfactual bottleneck lifts.
    pub counterfactual_bottlenecks: Option<Tensor>,
    /// `[B, K, latent_dim]` PRE-projection latent-space counterfactuals from
    /// the entmax head — what the SDE bridge / LatentSSM scores (replaces the
    /// old bottleneck-only emission).
    pub counterfactual_latents: Option<Tensor>,
    /// `[B, K, num_experts]` entmax-sparse routing weights.
    pub counterfactual_routing: Option<Tensor>,
    /// `[B, K, num_experts]` RAW router logits (pre-entmax) — the router-K
    /// KL source (the entmax weights sum to 1 per row: zero Jacobian).
    pub counterfactual_routing_logits: Option<Tensor>,

    // ── Dense-MoE path (heteroscedastic recon) ──────────────────────────
    /// `[B, num_experts]` dense mixture responsibilities (softmax of the
    /// first `num_experts` gate logits) — the MoE recon mixture weights AND
    /// the `irl_routing` gate-alignment target.
    pub moe_weights: Option<Tensor>,
    /// Heteroscedastic 3-expert reconstruction NLL (trains the log-σ heads).
    /// `None` off the training path (no real target).
    pub recon_likelihood: Option<Tensor>,
    /// `[B]` per-sample reconstruction NLL — the same per-`(b,t)` map as
    /// `recon_likelihood`, summed over features and the temporal axis
    /// instead of batch-meaned. `−recon_nll_per_sample = log p(x|z)` up to
    /// the dropped `0.5·D·log2π` constant (constant in z ⇒ cancels in
    /// self-normalised importance weights). The IWAE evidence bound's input.
    pub recon_nll_per_sample: Option<Tensor>,
    /// Dense-MoE mixture reconstruction NLL (`−log Σ π_e N(x;μ_e,σ_e)`).
    pub moe_recon_nll: Option<Tensor>,
    /// Composite predictive variance `1/P` paired with `recon` (= μ*) — the
    /// heteroscedastic heads' own uncertainty, in the SAME (scaled-outcome)
    /// units as the reconstruction. Trained by `recon_likelihood` /
    /// `moe_recon`; consumed by the likelihood network's outcome predictive.
    /// Always present: the moments need no real target, only the heads.
    pub recon_var: Tensor,
}

// ============================================================================
// Decoder trait
// ============================================================================

pub trait Decoder {
    /// Hold or advance any `SpectralNormLinear` power iteration this module
    /// owns. `false` is PyTorch's `eval()` semantics. Default is a NO-OP; a
    /// module that owns one and does not override stays call-history dependent.
    fn set_power_iteration(&self, _on: bool) {}

    /// `noise` is the per-Itô-step deterministic reparam bundle — accepted
    /// for flow parity with enc/dir; the spectral decoder has no internal
    /// sampling sites today (the `decoder_alpha` site is reserved).
    ///
    /// `t_emb` is the master Fourier time embedding `[B, time_dim]` — the
    /// decoder FiLMs its compressed bottleneck on it (Python decoder.py:210,
    /// `bottleneck·(1+scale)+bias` with zero-init projector ⇒ identity).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        dir_out: &DirichletOutput,
        indices: Option<&Tensor>,
        steps: usize,
        t: usize,
        generative_mode: bool,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<DecoderOutput>;
}

// ============================================================================
// MultiDimLayerNorm — for PyTorch's LayerNorm([32, 8])
// ============================================================================

/// LayerNorm whose `normalized_shape` spans the last `N` dims. Equivalent
/// to `torch.nn.LayerNorm([d0, d1, ...])` — stats are computed over the
/// flattened trailing block, while affine `weight` / `bias` carry the
/// original multi-dim shape.
pub struct MultiDimLayerNorm {
    weight: Tensor,
    bias: Tensor,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl MultiDimLayerNorm {
    pub fn load(normalized_shape: Vec<usize>, eps: f64, vb: VarBuilder) -> Result<Self> {
        // `get_with_hints`, NOT `get`. `VarBuilder::get` forwards
        // `Init::default()`, which is `Init::Const(0.)` — so on a FRESH
        // `VarMap` (training from scratch, as opposed to loading a
        // checkpoint, where the hint is ignored and the stored tensor wins)
        // a bare `get` initialises the affine SCALE to zero. That is not a
        // slow start, it is a hard cut, and it severs the stack in two
        // separate ways at once:
        //
        //   1. Forward: `w · x̂ + b == 0` identically, so this LayerNorm
        //      emits an exactly-zero tensor. `gelu(0) == 0`, so the NEXT
        //      convolution sees an all-zero input and its weight gradient
        //      `dL/dW = grad ⊗ input` is exactly zero — `deconv.6.weight`
        //      dies even though `deconv.6.bias` is perfectly healthy.
        //   2. Backward: `∂(w · x̂)/∂x̂ = w = 0`, so NOTHING upstream of this
        //      norm receives gradient — conv2, ln1, conv1 and fc are all
        //      cut off, whatever the loss does.
        //
        // Confusingly the norm's OWN affine params stay live throughout
        // (`dL/dw = grad ⊗ x̂`, and `x̂` is non-zero), so a naive "does
        // anything in this module get gradient?" probe answers yes. That is
        // what let it survive review.
        //
        // `nn.LayerNorm` initialises `weight = 1`, `bias = 0`, and so does
        // candle's own `layer_norm`; this type is the hand-rolled multi-dim
        // twin and must match, or the port silently disagrees with the
        // Python it mirrors.
        let weight =
            vb.get_with_hints(normalized_shape.clone(), "weight", candle_nn::Init::Const(1.))?;
        let bias =
            vb.get_with_hints(normalized_shape.clone(), "bias", candle_nn::Init::Const(0.))?;
        Ok(Self {
            weight,
            bias,
            normalized_shape,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let n = dims.len();
        let ndim = self.normalized_shape.len();
        assert!(
            n >= ndim,
            "MultiDimLayerNorm: tensor rank {} < normalized_shape rank {}",
            n,
            ndim
        );
        // Flatten trailing `ndim` dims, normalise, reshape back.
        let flat_start = n - ndim;
        let flat = x.flatten_from(flat_start)?;
        let mean = flat.mean_keepdim(D::Minus1)?;
        let centered = flat.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = var.affine(1.0, self.eps)?.sqrt()?;
        let normed = centered.broadcast_div(&denom)?;
        let reshaped = normed.reshape(dims)?;
        reshaped
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

// ============================================================================
// TemporalReconDecoder
// ============================================================================

pub struct TemporalReconDecoder {
    fc: Linear,
    conv1: ConvTranspose1d,
    ln1: MultiDimLayerNorm,
    conv2: ConvTranspose1d,
    ln2: MultiDimLayerNorm,
    conv3: ConvTranspose1d,
    init_channels: usize,
    init_length: usize,
}

impl TemporalReconDecoder {
    pub fn load(in_dim: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let init_channels = 64_usize;
        let init_length = 4_usize;

        let fc = crate::model::parametrize::sobol_linear(&vb.pp("fc"), in_dim, init_channels * init_length)?;

        let deconv = vb.pp("deconv");
        let conv_cfg = ConvTranspose1dConfig {
            padding: 1,
            output_padding: 0,
            stride: 2,
            dilation: 1,
            groups: 1,
        };

        // Sequential indices (GELU at 2 & 5 has no params):
        //   0: ConvTranspose1d(64 -> 32)
        //   1: LayerNorm([32, 8])
        //   2: GELU
        //   3: ConvTranspose1d(32 -> 32)
        //   4: LayerNorm([32, 16])
        //   5: GELU
        //   6: ConvTranspose1d(32 -> out_features)
        let conv1 = crate::model::parametrize::sobol_conv_transpose1d(&deconv.pp("0"), 64, 32, 4, conv_cfg)?;
        let ln1 = MultiDimLayerNorm::load(vec![32, 8], 1e-5, deconv.pp("1"))?;
        let conv2 = crate::model::parametrize::sobol_conv_transpose1d(&deconv.pp("3"), 32, 32, 4, conv_cfg)?;
        let ln2 = MultiDimLayerNorm::load(vec![32, 16], 1e-5, deconv.pp("4"))?;
        let conv3 = crate::model::parametrize::sobol_conv_transpose1d(&deconv.pp("6"), 32, out_features, 4, conv_cfg)?;

        Ok(Self {
            fc,
            conv1,
            ln1,
            conv2,
            ln2,
            conv3,
            init_channels,
            init_length,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let fc_out = self.fc.forward(x)?; // [B, 64*4 = 256]
        let h = fc_out.reshape((batch, self.init_channels, self.init_length))?;

        // `crate::model::conv_t::forward`, NOT `Module::forward` — candle 0.8
        // has no backward arm for `ConvTranspose1D`, so calling the module
        // directly makes every loss that reads `recon` or `recon_likelihood`
        // die at `backward()`. The replacement is numerically identical and
        // differentiates; see that module for the identity.
        let h = crate::model::conv_t::forward(&self.conv1, &h)?;
        let h = self.ln1.forward(&h)?;
        let h = h.gelu()?;

        let h = crate::model::conv_t::forward(&self.conv2, &h)?;
        let h = self.ln2.forward(&h)?;
        let h = h.gelu()?;

        let h = crate::model::conv_t::forward(&self.conv3, &h)?; // [B, out_features, 32]
        h.transpose(1, 2) // [B, 32, out_features]
    }
}

// ============================================================================
// CompressionNetwork — Linear → LayerNorm → ELU (→ Dropout skipped at eval)
// ============================================================================

struct CompressionStage {
    linear: Linear,
    norm: LayerNorm,
}

impl CompressionStage {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear.forward(x)?;
        let h = self.norm.forward(&h)?;
        h.elu(1.0)
    }
}

struct CompressionNetwork {
    stages: Vec<CompressionStage>,
}

impl CompressionNetwork {
    fn load(
        input_dim: usize,
        compressions: &[usize],
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: norm_eps,
            remove_mean: true,
            affine: true,
        };

        let mut stages = Vec::with_capacity(compressions.len());
        let mut current = input_dim;
        for (i, &c) in compressions.iter().enumerate() {
            // Each stage occupies 4 Sequential slots: Linear, LayerNorm,
            // ELU, Dropout — indices 4i, 4i+1, 4i+2 (unused), 4i+3 (unused).
            let lin_idx = (i * 4).to_string();
            let ln_idx = (i * 4 + 1).to_string();
            let linear = crate::model::parametrize::sobol_linear(&vb.pp(&lin_idx), current, c)?;
            let norm = layer_norm(c, ln_cfg, vb.pp(&ln_idx))?;
            stages.push(CompressionStage { linear, norm });
            current = c;
        }
        Ok(Self { stages })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for stage in &self.stages {
            h = stage.forward(&h)?;
        }
        Ok(h)
    }
}

// ============================================================================
// Heteroscedastic disentangle helpers
// ============================================================================

/// Smooth (tanh) saturation to `[lo, hi]`.
///
/// EXAGGERATED WHY: this is NOT interchangeable with `Tensor::clamp`, and the
/// choice is load-bearing. A hard clamp has a gradient of *exactly zero* the
/// instant a value leaves the band — so the moment a log-σ head pushes a
/// value past `2.0` or below `-4.0`, that head stops receiving ANY
/// reconstruction gradient and is frozen for the rest of training (a dead
/// variance head that silently pins σ at the boundary). tanh saturates
/// *asymptotically*: it still hands back a small restoring gradient
/// everywhere, so an over-confident head gets pulled back instead of
/// flat-lining. The whole GEOC autograd story depends on every op being
/// smooth; a hard clamp is the one non-smooth kink we deliberately refuse.
/// Same closed form as the Python `smooth_clamp`.
fn smooth_clamp(x: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    let mid = 0.5 * (hi + lo);
    let half = 0.5 * (hi - lo);
    x.affine(1.0 / half, -mid / half)?.tanh()?.affine(half, mid)
}

/// `chunk(2, last_dim)` → `(first_half, second_half)`.
///
/// EXAGGERATED WHY: the recon heads emit `2·input_dim` on purpose — the first
/// half is μ, the second is log σ, packed into ONE tensor so a single conv
/// stack produces both and they share the same receptive field. Splitting on
/// the LAST axis (not, say, reshaping) is what keeps μ and log σ
/// element-aligned per output feature: column `i` of the first half and
/// column `i` of the second half describe the same data dimension's mean and
/// scale. `chunk` returns a `Vec`; we clone the two views out because the
/// downstream precision math holds both simultaneously.
fn chunk2(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let parts = x.chunk(2, D::Minus1)?;
    Ok((parts[0].clone(), parts[1].clone()))
}

/// Numerically-stable `logsumexp` over `dim`.
///
/// EXAGGERATED WHY: the naïve `Σ exp(x)` overflows to `+inf` the moment any
/// logit exceeds ~88 (f32 exp range), and a single `inf` poisons the whole
/// mixture NLL into `nan` — which then propagates through every shared
/// decoder parameter on the backward pass. Subtracting the per-row max before
/// exponentiating bounds every exponent at `≤ 0` (so `exp ≤ 1`, never
/// overflows) and the max is added back after the log — the algebraic
/// identity `lse(x) = m + log Σ exp(x − m)` is exact, not an approximation.
/// candle ships no built-in, so we spell it out; `max_keepdim` keeps the
/// broadcast axis for the subtract, then `squeeze` drops it to match the
/// reduced sum.
fn log_sum_exp(x: &Tensor, dim: usize) -> Result<Tensor> {
    let m = x.max_keepdim(dim)?;
    let sumexp = x.broadcast_sub(&m)?.exp()?.sum(dim)?;
    sumexp.log()? + m.squeeze(dim)?
}

/// The heteroscedastic disentangle output: three INDEPENDENT Gaussian experts
/// over the recon target, plus their precision-weighted composite mean.
///
/// EXAGGERATED WHY (the whole point of the rewrite): the old code fused the
/// heads multiplicatively as `recon = trend + amp · res`. That coupling is a
/// modelling lie — it makes the residual head's gradient depend on the
/// amplitude head's current value and vice versa, so the two can't be
/// regularised or scored independently, and the implied density has no
/// closed-form score. The new form combines the experts ADDITIVELY in
/// log-density (a product of Gaussians), which has two consequences we
/// actually want: (1) the composite is itself Gaussian with precision
/// `P = Σ 1/σ²` and mean `μ* = (Σ μ/σ²)/P` — the precision-weighted average,
/// where a confident (low-σ) expert dominates and an unsure one is ignored,
/// exactly the Bayesian fusion you'd write by hand; and (2) the posterior
/// score is the closed form `∇log p(z) = −P·(z − μ*)`, which the SDE bridge
/// can consume analytically. No expert multiplies another anywhere.
struct Disentangled {
    mu_trend: Tensor,
    log_s_trend: Tensor,
    mu_mode: Tensor,
    log_s_mode: Tensor,
    mu_res: Tensor,
    log_s_res: Tensor,
    /// Precision-weighted composite mean `μ*` — the reconstruction target.
    mu_star: Tensor,
    /// Composite predictive variance `1/P`, `P = Σ_h 1/σ_h²` — the exact
    /// product-of-Gaussians posterior variance that goes with `mu_star`.
    /// Computed here since the heads landed and DISCARDED until the
    /// likelihood network needed a predictive in outcome units: the NLL
    /// collapses (μ*, P) to a scalar, so the moments never left this struct.
    var_star: Tensor,
}

// ============================================================================
// SpectralDecoder
// ============================================================================

pub struct SpectralDecoder {
    pub config: DecoderConfig,
    mode: DecoderMode,

    compression_network: CompressionNetwork,

    ls_head_recon: TemporalReconDecoder,
    pi_head_recon: TemporalReconDecoder,
    data_head_recon: TemporalReconDecoder,

    /// Spectral-normed on the trainable path
    /// (`expert_variational_heads.{i}.weight_orig`); folded on cleaned
    /// serving checkpoints.
    expert_variational_heads: Vec<crate::model::parametrize::SpectralNormLinear>,
    expert_logit_heads: Vec<Linear>, // len = num_experts
    /// Per-expert reconstruction heads (`latent_dim → 2·input_dim`): each
    /// expert decodes its OWN latent feature to a Gaussian (μ, log σ) over
    /// the data. Feeds the MoE mixture likelihood so the dense MoE path +
    /// gp_features are trained against a real data density, not left dangling.
    moe_recon_heads: Vec<Linear>, // len = num_experts

    mu_alpha: Linear,
    factor_alpha: Linear,
    diag_alpha: Linear,

    lengthscale_mu: Linear,
    lengthscale_logvar: Linear,
    scale_head_per_expert: Linear,

    /// Sparse entmax counterfactual head (`counterfactual_head.*`) — present
    /// on checkpoints trained with it; emits the latent-space counterfactuals.
    counterfactual_head: Option<crate::model::counterfactual::EntmaxCounterfactualHead>,
    counterfactual_routing_dim: usize,

    /// FiLM head on the master time embedding (`film_projector`, zero-init ⇒
    /// identity; Python decoder.py:101). `None` only on pre-migration
    /// tensor-backed checkpoints.
    film_projector: Option<Linear>,

    device: Device,
    dtype: DType,
}

impl SpectralDecoder {
    /// Load from a [`VarBuilder`] rooted at the Python `decoder` submodule.
    pub fn load(config: DecoderConfig, mode: DecoderMode, vb: VarBuilder) -> Result<Self> {
        let compression_network = CompressionNetwork::load(
            config.spectral_emb_dim,
            &config.spectral_compressions,
            config.norm_eps,
            vb.pp("compression_network"),
        )?;

        // EXAGGERATED WHY the out-width is `2·input_dim`, not `config.*_recon[1]`:
        // the recon head emits (μ, log σ) stacked, so its output MUST be twice
        // the data width. The `*_recon[1]` slot held the literal `60`, which is
        // only `2·input_dim` when `input_dim == 30` — a value frozen into the
        // config from one dataset. Deriving it as `2·config.input_dim` keeps it
        // correct after the loader reconciles `input_dim` from the checkpoint
        // (a model trained on a different feature count would otherwise build a
        // conv whose ConvTranspose1d weight `[32, out, 4]` mismatches the
        // stored shape). `*_recon[0]` (the bottleneck-split IN width: 4/30/30)
        // is still read from config — that one genuinely IS a fixed split.
        let recon_out = 2 * config.input_dim;
        let ls_head_recon =
            TemporalReconDecoder::load(config.ls_recon[0], recon_out, vb.pp("ls_head_recon"))?;
        let pi_head_recon =
            TemporalReconDecoder::load(config.pi_recon[0], recon_out, vb.pp("pi_head_recon"))?;
        let data_head_recon =
            TemporalReconDecoder::load(config.data_recon[0], recon_out, vb.pp("data_head_recon"))?;

        let evh = vb.pp("expert_variational_heads");
        let mut expert_variational_heads = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            // spectral_norm in Python (decoder.py:111) — trainable path
            // registers `expert_variational_heads.{i}.weight_orig`.
            let l = crate::model::parametrize::SpectralNormLinear::new(
                evh.pp(&i.to_string()),
                config.bottleneck_dim,
                config.k_atoms,
                candle_nn::init::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0 / (config.bottleneck_dim as f64).sqrt(),
                },
            )?;
            expert_variational_heads.push(l);
        }

        let elh = vb.pp("expert_logit_heads");
        let mut expert_logit_heads = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            let l = crate::model::parametrize::sobol_linear(&elh.pp(&i.to_string()), config.k_atoms, config.latent_dim)?;
            expert_logit_heads.push(l);
        }

        // Per-expert MoE recon heads — `latent_dim → 2·input_dim` (μ, log σ),
        // one per expert. PyTorch names ModuleList children by integer index,
        // so `.pp(&i.to_string())` reproduces `moe_recon_heads.{i}.weight`.
        let mrh = vb.pp("moe_recon_heads");
        let mut moe_recon_heads = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            moe_recon_heads.push(crate::model::parametrize::sobol_linear(&mrh.pp(&i.to_string()), config.latent_dim, 2 * config.input_dim)?);
        }

        let mu_alpha = crate::model::parametrize::sobol_linear(&vb.pp("mu_alpha"), config.bottleneck_dim, config.evidence_dim)?;
        let factor_alpha = crate::model::parametrize::sobol_linear(&vb.pp("factor_alpha"), config.bottleneck_dim, config.alpha_factor_rank * config.evidence_dim)?;
        let diag_alpha = crate::model::parametrize::sobol_linear(&vb.pp("diag_alpha"), config.bottleneck_dim, config.evidence_dim)?;

        let lengthscale_mu = crate::model::parametrize::sobol_linear(&vb.pp("lengthscale_mu"), config.bottleneck_dim, config.k_atoms)?;
        let lengthscale_logvar = crate::model::parametrize::sobol_linear(&vb.pp("lengthscale_logvar"), config.bottleneck_dim, config.k_atoms)?;
        let scale_head_per_expert = crate::model::parametrize::sobol_linear(&vb.pp("scale_head_per_expert"), config.k_atoms, config.num_latents)?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Counterfactual entmax head. Its router weight is
        // `[n_sparse·num_experts, routing_dim]`, so BOTH knobs are structural:
        // the loader reconciles them off that weight before we get here (see
        // `SpectralVae::load`), which is why they come from config rather than
        // the hardcoded 8/8 that only fitted one trained geometry.
        let counterfactual_routing_dim = config.counterfactual_routing_dim;
        // FiLM time head (zero-init ⇒ identity conditioning at start).
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
        // ALWAYS built on a trainable map (Python registers it
        // unconditionally in __init__); absent only on tensor-backed
        // checkpoints from before the head existed.
        let counterfactual_head = crate::model::parametrize::build_or_absent(
            &vb,
            "counterfactual_head.router.weight",
            || {
                crate::model::counterfactual::EntmaxCounterfactualHead::new(
                    vb.pp("counterfactual_head"),
                    config.num_experts,
                    config.n_sparse,
                    config.latent_dim,
                    config.bottleneck_dim,
                    Some(counterfactual_routing_dim),
                )
            },
        )?;

        Ok(Self {
            config,
            mode,
            compression_network,
            ls_head_recon,
            pi_head_recon,
            data_head_recon,
            expert_variational_heads,
            expert_logit_heads,
            moe_recon_heads,
            mu_alpha,
            factor_alpha,
            diag_alpha,
            lengthscale_mu,
            lengthscale_logvar,
            scale_head_per_expert,
            counterfactual_head,
            counterfactual_routing_dim,
            film_projector,
            device,
            dtype,
        })
    }

    /// The EFFECTIVE expert-variational-head weights (spectral norm applied)
    /// — the `ExpertDiversityTerm`'s gram-penalty inputs.
    pub fn expert_head_weights(&self) -> Result<Vec<Tensor>> {
        self.expert_variational_heads
            .iter()
            .map(|h| h.effective_weight())
            .collect()
    }

    /// Attach a (freshly initialised) counterfactual head — for models built
    /// from a VarMap rather than a checkpoint.
    pub fn with_counterfactual_head(
        mut self,
        head: crate::model::counterfactual::EntmaxCounterfactualHead,
        routing_dim: usize,
    ) -> Self {
        self.counterfactual_head = Some(head);
        self.counterfactual_routing_dim = routing_dim;
        self
    }

    pub fn mode(&self) -> DecoderMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: DecoderMode) {
        self.mode = mode;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn disentangle(&self, bottleneck: &Tensor) -> Result<Disentangled> {
        // Split the bottleneck along the LAST axis into the three head inputs.
        // EXAGGERATED WHY these widths and this order: the split `[4, 30, 30]`
        // sums to `bottleneck_dim = 64` and MUST match the recon heads' IN
        // dims exactly — `z_ls` (4) feeds `ls_head_recon` which was built with
        // `ls_recon[0] = 4`, etc. Getting the offsets wrong wouldn't error, it
        // would silently route the wrong 30 bottleneck columns into the "mode"
        // head and quietly wreck training — so the offsets are cumulative
        // (`0`, `a`, `a+b`) rather than hardcoded, tied to the same config the
        // heads were sized from.
        let [a, b, c] = [
            self.config.ls_recon[0],
            self.config.pi_recon[0],
            self.config.data_recon[0],
        ];
        let z_ls = bottleneck.narrow(D::Minus1, 0, a)?;
        let z_pi = bottleneck.narrow(D::Minus1, a, b)?;
        let z_dt = bottleneck.narrow(D::Minus1, a + b, c)?;

        // Each head emits `2·input_dim` → chunk into (μ, log σ).
        let (mu_trend, log_s_trend) = chunk2(&self.ls_head_recon.forward(&z_ls)?)?;
        let (mu_mode, log_s_mode) = chunk2(&self.pi_head_recon.forward(&z_pi)?)?;
        let (mu_res, log_s_res) = chunk2(&self.data_head_recon.forward(&z_dt)?)?;

        // Bound each log σ to [-4, 2] SMOOTHLY (see `smooth_clamp` — a hard
        // clamp here would freeze any saturated variance head).
        let log_s_trend = smooth_clamp(&log_s_trend, -4.0, 2.0)?;
        let log_s_mode = smooth_clamp(&log_s_mode, -4.0, 2.0)?;
        let log_s_res = smooth_clamp(&log_s_res, -4.0, 2.0)?;

        // Precision `1/σ² = exp(-2 log σ)`. EXAGGERATED WHY we work in log σ and
        // exponentiate here rather than predicting σ directly: a raw σ head can
        // emit a negative (invalid) scale and needs a softplus that saturates;
        // predicting log σ makes the head's whole real line valid, and
        // `exp(-2·)` turns it into a strictly-positive precision in one smooth
        // op — no branch, no floor, differentiable everywhere.
        let prec_trend = log_s_trend.affine(-2.0, 0.0)?.exp()?;
        let prec_mode = log_s_mode.affine(-2.0, 0.0)?.exp()?;
        let prec_res = log_s_res.affine(-2.0, 0.0)?.exp()?;
        let p_composite = ((&prec_trend + &prec_mode)? + &prec_res)?;

        // μ* = Σ(μ·prec) / P — the precision-weighted reference. EXAGGERATED
        // WHY division not a learned gate: this is the exact posterior mean of
        // a product of Gaussians, so a confident expert (large prec) is
        // trusted and an unsure one is down-weighted AUTOMATICALLY, with zero
        // extra parameters. A learned mixing weight would have to relearn this
        // 1/σ² relationship the closed form already gives for free.
        let weighted = (((&mu_trend * &prec_trend)? + (&mu_mode * &prec_mode)?)?
            + (&mu_res * &prec_res)?)?;
        let mu_star = weighted.broadcast_div(&p_composite)?;
        // 1/P: each per-expert precision is exp of a [−4, 2]-bounded log σ,
        // so P ≥ 3·e⁻⁴ > 0 and the reciprocal needs no floor.
        let var_star = p_composite.recip()?;

        Ok(Disentangled {
            mu_trend,
            log_s_trend,
            mu_mode,
            log_s_mode,
            mu_res,
            log_s_res,
            mu_star,
            var_star,
        })
    }

    /// Heteroscedastic reconstruction NLL over the three disentangled experts
    /// — the term that actually TRAINS the log σ heads.
    ///
    /// EXAGGERATED WHY this replaces a plain SSE against `mu_star`: an SSE
    /// `‖x − μ*‖²` has NO dependence on any σ, so under it the three variance
    /// heads receive exactly zero gradient and are dead weight. The Gaussian
    /// NLL `0.5·Σ (x−μ)²/σ² + Σ log σ` couples them: the first term rewards a
    /// head for being precise where it's accurate, the `Σ log σ` normaliser
    /// punishes it for being over-confident everywhere — together they teach
    /// each expert its own honest uncertainty. The constant `0.5·D·log2π` is
    /// dropped because it's gradient-free (adding a constant to a minimised
    /// loss changes nothing). Lower = higher likelihood; still a MINIMISED NLL.
    ///
    /// Returns `(scalar, per_sample)`, both from ONE shared per-`(b,t)` map:
    ///   * `scalar` — mean over (B, T) of the feature-summed map; identical to
    ///     the historical loss term, so the loss landscape is untouched.
    ///   * `per_sample` — `[B]`: the map SUMMED over T, the joint log-density
    ///     of the window (each temporal position is a Gaussian factor). This
    ///     is what an IWAE bound needs per proposal sample; a batch-meaned NLL
    ///     cannot be turned back into per-sample importance weights.
    /// Relation: `scalar == per_sample.mean() / T` (T = the heads' 32-step
    /// temporal axis). Sign: `log p(x|z) = −per_sample` up to the dropped
    /// constant, which is constant in z and cancels in self-normalised weights.
    fn recon_nll(&self, real_x: &Tensor, d: &Disentangled) -> Result<(Tensor, Tensor)> {
        // Per-expert prec-weighted squared error; real_x [B,1,input_dim]
        // broadcasts over the heads' temporal axis [B,32,input_dim].
        let term = |mu: &Tensor, log_s: &Tensor| -> Result<Tensor> {
            let prec = log_s.affine(-2.0, 0.0)?.exp()?;
            real_x.broadcast_sub(mu)?.sqr()?.broadcast_mul(&prec)
        };
        let data = ((term(&d.mu_trend, &d.log_s_trend)? + term(&d.mu_mode, &d.log_s_mode)?)?
            + term(&d.mu_res, &d.log_s_res)?)?;
        // 0.5·Σ_over_features → [B, T]; likewise the Σ log σ normaliser.
        let data_bt = data.affine(0.5, 0.0)?.sum(D::Minus1)?;
        let norm_bt = ((&d.log_s_trend + &d.log_s_mode)? + &d.log_s_res)?.sum(D::Minus1)?;
        // Scalar: mean over batch+time, same reduction order as before.
        let scalar = (data_bt.mean_all()? + norm_bt.mean_all()?)?;
        // Per-sample: Σ over the temporal axis → [B].
        let per_sample = (data_bt + norm_bt)?.sum(1)?;
        Ok((scalar, per_sample))
    }

    /// Dense-MoE mixture reconstruction NLL: each expert decodes its own
    /// latent feature to a Gaussian over the data, and the dense
    /// `moe_weights` are the mixture responsibilities.
    
    fn moe_recon_nll(
        &self,
        real_x: &Tensor,
        gp_features: &Tensor, // [B, E, latent_dim]
        moe_weights: &Tensor, // [B, E]
    ) -> Result<Tensor> {
        let e = self.moe_recon_heads.len();
        let mut parts = Vec::with_capacity(e);
        for (i, head) in self.moe_recon_heads.iter().enumerate() {
            let f_i = gp_features.narrow(1, i, 1)?.squeeze(1)?; // [B, latent_dim]
            parts.push(head.forward(&f_i)?); // [B, 2·input_dim]
        }
        let moe_params = Tensor::stack(&parts, 1)?; // [B, E, 2·input_dim]
        let (moe_mu, moe_log_s) = chunk2(&moe_params)?; // [B, E, input_dim]
        let moe_log_s = smooth_clamp(&moe_log_s, -4.0, 2.0)?;

        // log N(x; μ_e, σ_e) per expert = -0.5·log2π - log σ - 0.5·(x-μ)²/σ².
        let neg_half_log2pi = -0.5 * (2.0 * std::f64::consts::PI).ln();
        let prec = moe_log_s.affine(-2.0, 0.0)?.exp()?;
        let sq = real_x
            .broadcast_sub(&moe_mu)?
            .sqr()?
            .broadcast_mul(&prec)?
            .affine(-0.5, 0.0)?;
        let per_dim = (moe_log_s.affine(-1.0, neg_half_log2pi)? + sq)?; // [B,E,input_dim]
        let log_lik = per_dim.sum(D::Minus1)?; // [B, E]
        let log_pi = moe_weights.clamp(1e-9, f64::INFINITY)?.log()?; // [B, E]
        // −mean_B logsumexp_E (log π + log N).
        log_sum_exp(&(log_pi + log_lik)?, 1)?.mean_all()?.affine(-1.0, 0.0)
    }



    /// Draw the Dirichlet evidence `α` straight off the fused bottleneck —
    /// `[B, bottleneck_dim] → (mu, factor, diag, alpha)`.
    ///
    /// The companion to `Encoder::latent_from_bottleneck`: the SAME `[B, 64]`
    /// carrier that yields `z` also yields `α`, because all three heads are
    /// `Linear(bottleneck_dim → ·)`. Together they give the whole state cheaply:
    ///
    /// ```text
    ///   [B,64] ─┬─▶ latent_mu / latent_logvar ─▶ z
    ///           └─▶ mu_alpha / factor_alpha / diag_alpha ─▶ α
    ///                                    α ─▶ restick_from_alpha ─▶ π
    ///                                    π ─▶ coregionalisation    ─▶ B
    /// ```
    ///
    /// `α` is the low-rank Gaussian reparam `μ + √diag·η_d + F·η_f` with
    /// `Σ = diag + FFᵀ` — never forming `Σ`, matching the Python
    /// `get_local_evidence`. In [`DecoderMode::Deterministic`] it is `μ` and the
    /// noise is skipped entirely, mirroring the Python's `if not self.training`.
    ///
    /// Noise is Sobol, not `Tensor::randn`: the vendored candle CPU backend
    /// ignores `Device::set_seed`, so QMC is the only reproducible route. The
    /// two sites get distinct salts so `η_d` and `η_f` are independent.
    pub fn alpha_from_bottleneck(
        &self,
        bottleneck: &Tensor,
        seed: u64,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (mu, factor, diag) = self.get_alpha_mvn_heads(bottleneck)?;
        let alpha = match self.mode {
            DecoderMode::Deterministic => mu.clone(),
            DecoderMode::Stochastic => {
                use crate::model::noise::{site_seed, sobol_randn_like};
                let eta_d =
                    sobol_randn_like(mu.dims(), mu.device(), site_seed("local_evidence_d", seed))?
                        .to_dtype(mu.dtype())?;
                let eta_f = sobol_randn_like(
                    factor.dims()[..factor.rank() - 1]
                        .iter()
                        .copied()
                        .chain(std::iter::once(self.config.alpha_factor_rank))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    mu.device(),
                    site_seed("local_evidence_f", seed),
                )?
                .to_dtype(mu.dtype())?;
                // μ + √diag·η_d + F·η_f
                let lowrank = factor.matmul(&eta_f.unsqueeze(candle_core::D::Minus1)?)?
                    .squeeze(candle_core::D::Minus1)?;
                mu.add(&diag.sqrt()?.mul(&eta_d)?)?.add(&lowrank)?
            }
        };
        Ok((mu, factor, diag, alpha))
    }

    fn get_alpha_mvn_heads(
        &self,
        bottleneck: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mu = self.mu_alpha.forward(bottleneck)?;
        let factor_flat = self.factor_alpha.forward(bottleneck)?;
        let b = factor_flat.dim(0)?;
        let factor = factor_flat.reshape((b, self.config.evidence_dim, self.config.alpha_factor_rank))?;
        let diag_raw = self.diag_alpha.forward(bottleneck)?;
        let diag = softplus(&diag_raw)?.affine(1.0, self.config.jitter as f64)?;
        Ok((mu, factor, diag))
    }

    /// Lengthscale reparam. Returns `ls_sample` of shape `[B, k_atoms]`.
    ///
    /// Already a bottleneck-entry head — `lengthscale_mu` and
    /// `lengthscale_logvar` are both `Linear(bottleneck_dim → k_atoms)` — so
    /// it is public for the same reason
    /// [`alpha_from_bottleneck`](Self::alpha_from_bottleneck) is: a caller
    /// entering at the `[B, bottleneck_dim]` carrier needs `ls` for
    /// [`crate::model::concentration::pack_dirichlet_theta`], and the only
    /// other route to it is a full spectral decoder forward, which is the
    /// ~8 ms path the whole cheap entry point exists to avoid.
    ///
    /// In [`DecoderMode::Deterministic`] this is `exp(clamp(μ))` — no reparam
    /// noise, so it is a deterministic function of the bottleneck.
    pub fn predict_lengthscale(&self, bottleneck: &Tensor) -> Result<Tensor> {
        self.predict_lengthscale_with(self.mode, bottleneck, None)
    }

    /// Mode-parameterised twin — the seam `deterministic_pass` uses for one
    /// eval-forced fold from a `&self` call chain (no ambient flips).
    ///
    /// `eps_noise` is the bundle's `lengthscale` site, `[B, k_atoms]` — the
    /// shape `lengthscale_mu` emits. It had a slot in
    /// [`noise_site_shapes`](crate::model::vae::noise_site_shapes) from the
    /// start and nothing ever read it: this reparam drew a fresh
    /// `Tensor::randn`, which reaches candle's process-global stream (the
    /// vendored CPU backend ignores `Device::set_seed`). That made every
    /// `DecoderMode::Stochastic` forward a function of call history rather
    /// than of its arguments, and the leak was not confined to the
    /// lengthscale: `ls_sample` drives `bw_learned`, which the refinement
    /// loop carries into the next step's Dirichlet, so it reached `z`.
    ///
    /// `None` falls back to a Sobol draw at a FIXED salt, not to `randn` —
    /// the same choice `StateSpaceOutput::stochastic_zero_state` makes, for
    /// the same reason. It is therefore CONSTANT across calls; a caller that
    /// wants independent draws per step passes the per-step bundle, which
    /// `refinement_loop` already builds.
    pub fn predict_lengthscale_with(
        &self,
        mode: DecoderMode,
        bottleneck: &Tensor,
        eps_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mu = self.lengthscale_mu.forward(bottleneck)?;
        let logvar = self.lengthscale_logvar.forward(bottleneck)?;
        let sigma = logvar
            .affine(0.5, 0.0)?
            .exp()?
            .affine(1.0, self.config.eps)?;

        let log_ls_sample = match mode {
            DecoderMode::Deterministic => mu.clone(),
            DecoderMode::Stochastic => {
                let eps = match eps_noise {
                    // Shape-CHECKED rather than trusted: a bundle drawn at a
                    // different batch would otherwise broadcast into a
                    // silently wrong reparam instead of failing.
                    Some(e) if e.dims() == mu.dims() => e.to_dtype(mu.dtype())?,
                    _ => crate::model::noise::sobol_randn_like(
                        mu.dims(),
                        mu.device(),
                        LENGTHSCALE_FALLBACK_SALT,
                    )?
                    .reshape(mu.dims())?
                    .to_dtype(mu.dtype())?,
                };
                (&mu + (sigma * eps)?)?
            }
        };

        let safe_log_ls = log_ls_sample.clamp(
            self.config.min_ls.ln() as f32,
            self.config.max_ls.ln() as f32,
        )?;
        safe_log_ls.exp()
    }

    /// The trait [`Decoder::forward`] body with the MODE AS A PARAMETER —
    /// see [`Self::predict_lengthscale_with`].
    pub fn forward_with_mode(
        &self,
        mode: DecoderMode,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        dir_out: &DirichletOutput,
        _indices: Option<&Tensor>,
        _steps: usize,
        _t: usize,
        _generative_mode: bool,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<DecoderOutput> {
        // 1. Compression network: [B, spectral_emb_dim] → [B, bottleneck_dim]
        let bottleneck = self.compression_network.forward(x)?;

        // 1b. FiLM time conditioning on the compressed bottleneck
        // (decoder.py:210: `bottleneck·(1+scale)+bias`, zero-init projector
        // ⇒ exact identity at start; identity when unbound).
        let bottleneck = match (&self.film_projector, t_emb) {
            (Some(film), Some(t)) => {
                let fb = film.forward(&t.to_dtype(bottleneck.dtype())?)?; // [B, 2·bn]
                let parts = fb.chunk(2, D::Minus1)?;
                let (mut scale, mut bias) = (parts[0].clone(), parts[1].clone());
                while scale.rank() < bottleneck.rank() {
                    scale = scale.unsqueeze(scale.rank() - 1)?;
                    bias = bias.unsqueeze(bias.rank() - 1)?;
                }
                bottleneck
                    .broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(&bias)?
            }
            _ => bottleneck,
        };

        // 2. Disentangle into the three heteroscedastic experts. EXAGGERATED
        //    WHY the field aliasing: downstream state still speaks
        //    trend/amp/res/recon, but their MEANING changed — `recon` is now
        //    the precision-weighted composite `μ*` (was `trend + amp·res`),
        //    and trend/amp/res are the three experts' MEANS (μ_trend/μ_mode/
        //    μ_res), no products. The names are kept only so consumers don't
        //    churn; the semantics match the new Python disentangle exactly.
        let d = self.disentangle(&bottleneck)?;
        let recon = d.mu_star.clone();
        let trend = d.mu_trend.clone();
        let amp = d.mu_mode.clone();
        let res = d.mu_res.clone();

        // 3. Expert variational heads (per-expert logits), shape
        //    [num_experts, B, k_atoms].
        let mut variational: Vec<Tensor> = Vec::with_capacity(self.expert_variational_heads.len());
        for head in &self.expert_variational_heads {
            variational.push(head.forward(&bottleneck)?);
        }
        let parameters_per_expert = Tensor::stack(&variational, 0)?;

        // 4. Alpha MVN heads.
        let (alpha_mu, alpha_factor, alpha_diag) = self.get_alpha_mvn_heads(&bottleneck)?;

        // 5. Lengthscale reparam.
        let ls_sample = self.predict_lengthscale_with(
            mode,
            &bottleneck,
            noise.and_then(|n| n.get("lengthscale")),
        )?;

        // 6. Bandwidth modulation: scale_head_per_expert(ls_sample) * 2.0
        let bw_mod = self
            .scale_head_per_expert
            .forward(&ls_sample)?
            .affine(2.0, 0.0)?;

        // 7. Expert logit heads, stacked along a new dim → [B, num_experts, latent_dim].
        let mut logits: Vec<Tensor> = Vec::with_capacity(self.expert_logit_heads.len());
        for (head, var_state) in self.expert_logit_heads.iter().zip(variational.iter()) {
            logits.push(head.forward(var_state)?);
        }
        let gp_features = Tensor::stack(&logits, 1)?;

        // 7b. Dense-MoE responsibilities + reconstruction likelihoods.
        // moe_weights = softmax over the first `num_experts` gate logits, then
        // renormalised. EXAGGERATED WHY sliced to `num_experts` (not all gates,
        // which is what the old vae-fold `softmax(gates_logits)` did): the
        // mixture has exactly `num_experts` components, so the responsibility
        // vector MUST be `num_experts`-wide. Here `num_gates == num_experts == 8`
        // so the two happen to coincide, but slicing makes the contract explicit
        // and correct the day a config gives more gates than experts — matching
        // the Python `routing_signal[..., :num_experts]`.
        let ne = self.config.num_experts;
        let moe_weights = {
            let gl = &dir_out.gates_logits;
            let sliced = if gl.dim(1)? >= ne {
                gl.narrow(D::Minus1, 0, ne)?
            } else {
                // Fewer gates than experts → uniform (softmax of zeros).
                Tensor::zeros((gl.dim(0)?, ne), gp_features.dtype(), gp_features.device())?
            };
            let sm = candle_nn::ops::softmax(&sliced, D::Minus1)?;
            let denom = sm.sum_keepdim(D::Minus1)?.affine(1.0, 1e-9)?;
            sm.broadcast_div(&denom)?
        };

        // The two data-likelihood stashes are only meaningful when a real
        // target is present (the training path). EXAGGERATED WHY the guard: on
        // a stateless/generative pass `real_x` is a zero placeholder, and an
        // NLL against zeros is noise — computing it would feed the loss stack a
        // meaningless gradient. Matches the Python `if real_x is not None`.
        let (recon_likelihood, recon_nll_per_sample, moe_recon_nll) =
            if dir_out.real_x.elem_count() > 0 {
                let (scalar, per_sample) = self.recon_nll(&dir_out.real_x, &d)?;
                (
                    Some(scalar),
                    Some(per_sample),
                    Some(self.moe_recon_nll(&dir_out.real_x, &gp_features, &moe_weights)?),
                )
            } else {
                (None, None, None)
            };

        // 8. Sparse entmax counterfactual path: gp_features are the expert
        // outputs [B, E, latent]; the routing signal is the NKN's REAL
        // pre-sigmoid gate logits, now carried on DirichletOutput (the old
        // logit(clamp(σ(g))) reconstruction is gone — no saturation clamp
        // artefacts).
        let gates_logits = dir_out.gates_logits.clone(); // [B, num_gates]

        let mut counterfactual_bottlenecks = None;
        let mut counterfactual_latents = None;
        let mut counterfactual_routing = None;
        let mut counterfactual_routing_logits = None;
        if let Some(head) = &self.counterfactual_head {
            let b = gp_features.dim(0)?;
            // Routing signal = gates_logits when the width matches; else the
            // uniform (zeros) fallback — entmax15 of zeros is the simplex
            // centroid, matching the Python guard.
            let routing_signal = if gates_logits.dim(1)? == self.counterfactual_routing_dim {
                gates_logits.clone()
            } else {
                Tensor::zeros(
                    (b, self.counterfactual_routing_dim),
                    gp_features.dtype(),
                    gp_features.device(),
                )?
            };
            let (cf_b, cf_l, cf_r, cf_lg) = head.forward(&gp_features, &routing_signal)?;
            counterfactual_routing_logits = Some(cf_lg);
            counterfactual_bottlenecks = Some(cf_b);
            counterfactual_latents = Some(cf_l);
            counterfactual_routing = Some(cf_r);
        }

        // Build the output, carrying dir_out fields by clone.
        Ok(DecoderOutput {
            gates: dir_out.gates.clone(),
            mat52: dir_out.mat52.clone(),
            rbf: dir_out.rbf.clone(),
            rq: dir_out.rq.clone(),
            pi: dir_out.pi.clone(),
            lmc_matrices: dir_out.lmc_matrices.clone(),
            lmc_consensus: dir_out.lmc_consensus.clone(),
            real_x: dir_out.real_x.clone(),
            alpha: dir_out.local_conc.clone(),
            bottleneck,
            alpha_mu,
            alpha_factor,
            alpha_diag,
            gp_features,
            parameters_per_expert,
            recon,
            recon_var: d.var_star.clone(),
            bandwidth_mod: bw_mod,
            amp,
            trend,
            res,
            ls: ls_sample,
            ls_logvar: dir_out.ls_logvar.clone(),
            bw_learned: dir_out.bw_learned.clone(),
            mu_z: dir_out.mu_z.clone(),
            logvar_z: dir_out.logvar_z.clone(),

            gates_logits: Some(gates_logits),
            counterfactual_bottlenecks,
            counterfactual_latents,
            counterfactual_routing,
            counterfactual_routing_logits,
            moe_weights: Some(moe_weights),
            recon_likelihood,
            recon_nll_per_sample,
            moe_recon_nll,
        })
    }
}

// ============================================================================
// Decoder trait impl
// ============================================================================

impl Decoder for SpectralDecoder {
    fn set_power_iteration(&self, on: bool) {
        for h in &self.expert_variational_heads {
            h.set_power_iteration(on);
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        dir_out: &DirichletOutput,
        _indices: Option<&Tensor>,
        _steps: usize,
        _t: usize,
        _generative_mode: bool,
        _noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<DecoderOutput> {
        self.forward_with_mode(
            self.mode, x, t_emb, dir_out, _indices, _steps, _t, _generative_mode, _noise,
        )
    }

}



// ============================================================================
// Tests (shape & helper only)
// ============================================================================

#[cfg(test)]
mod tests {
    //! Verification of spectral decoders mapping latent distributions to reconstructed observations.

    use super::*;

    /// Shared fixture for the recon-NLL tests: a fresh decoder, a bottleneck
    /// disentangled through the REAL conv recon heads (so gradients traverse
    /// the `conv_t` composite ConvTranspose1d path), and a real target of the
    /// decoder's own `[B, 1, input_dim]` convention.
    fn recon_fixture(
        dev: &Device,
        b: usize,
    ) -> (SpectralDecoder, candle_nn::VarMap, Tensor, Disentangled) {
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, dev);
        let cfg = DecoderConfig::default();
        let (input_dim, bdim) = (cfg.input_dim, cfg.bottleneck_dim);
        let dec = SpectralDecoder::load(cfg, DecoderMode::Deterministic, vb).unwrap();
        let bottleneck = Tensor::randn(0f32, 1f32, (b, bdim), dev).unwrap();
        let d = dec.disentangle(&bottleneck).unwrap();
        let real_x = Tensor::randn(0f32, 1f32, (b, 1, input_dim), dev).unwrap();
        (dec, vm, real_x, d)
    }

    /// The scalar half of the refactored `recon_nll` must be BYTE-IDENTICAL
    /// to the historical expression — the per-sample vector is a new output,
    /// not a change to the loss landscape. Recomputed inline, asserted with
    /// `==` (no tolerance): the refactor uses the same ops in the same order.
    #[test]
    fn recon_nll_scalar_is_byte_identical_to_the_old_expression() {
        let dev = Device::Cpu;
        let (dec, _vm, real_x, d) = recon_fixture(&dev, 3);

        // The pre-refactor formula, verbatim.
        let term = |mu: &Tensor, log_s: &Tensor| -> Result<Tensor> {
            let prec = log_s.affine(-2.0, 0.0)?.exp()?;
            real_x.broadcast_sub(mu)?.sqr()?.broadcast_mul(&prec)
        };
        let data = ((term(&d.mu_trend, &d.log_s_trend).unwrap()
            + term(&d.mu_mode, &d.log_s_mode).unwrap())
        .unwrap()
            + term(&d.mu_res, &d.log_s_res).unwrap())
        .unwrap();
        let old_data = data
            .affine(0.5, 0.0)
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .mean_all()
            .unwrap();
        let old_norm = ((&d.log_s_trend + &d.log_s_mode).unwrap() + &d.log_s_res)
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .mean_all()
            .unwrap();
        let old = (old_data + old_norm).unwrap().to_scalar::<f32>().unwrap();

        let (scalar, _per) = dec.recon_nll(&real_x, &d).unwrap();
        let new = scalar.to_scalar::<f32>().unwrap();
        assert_eq!(new, old, "scalar recon NLL drifted: old={old}, new={new}");
    }

    /// per_sample is `[B]` and relates to the scalar by exactly the documented
    /// identity: scalar == per_sample.mean() / T (sum-over-T vs mean-over-T of
    /// the same [B, T] map). f32 tolerance only for the reduction reorder.
    #[test]
    fn recon_nll_per_sample_shape_and_mean_relation() {
        let dev = Device::Cpu;
        let b = 4usize;
        let (dec, _vm, real_x, d) = recon_fixture(&dev, b);
        let t = d.mu_trend.dim(1).unwrap(); // the heads' temporal axis (32)

        let (scalar, per) = dec.recon_nll(&real_x, &d).unwrap();
        assert_eq!(per.dims(), &[b], "per_sample must be [B]");

        let s = scalar.to_scalar::<f32>().unwrap();
        let m = per.mean_all().unwrap().to_scalar::<f32>().unwrap() / t as f32;
        assert!(
            (s - m).abs() <= 1e-5 * s.abs().max(1.0),
            "scalar={s} != per_sample.mean()/T={m}"
        );
    }

    /// Corrupting ONE sample's target must move ONLY that sample's NLL —
    /// every other entry stays bit-identical. This is the guard against an
    /// accidental cross-batch reduction (broadcast_sub misalignment, a
    /// keepdim slip, a mean smuggled in before the batch axis is resolved).
    #[test]
    fn recon_nll_per_sample_entries_are_batch_independent() {
        let dev = Device::Cpu;
        let b = 3usize;
        let (dec, _vm, real_x, d) = recon_fixture(&dev, b);

        let (_, per) = dec.recon_nll(&real_x, &d).unwrap();
        let base = per.to_vec1::<f32>().unwrap();

        // Shift sample 1's target far off the reconstruction; leave 0 and 2.
        let mask = Tensor::new(&[0f32, 10.0, 0.0], &dev)
            .unwrap()
            .reshape((b, 1, 1))
            .unwrap();
        let corrupted = real_x.broadcast_add(&mask).unwrap();
        let (_, per2) = dec.recon_nll(&corrupted, &d).unwrap();
        let bumped = per2.to_vec1::<f32>().unwrap();

        assert_eq!(base[0], bumped[0], "sample 0 leaked");
        assert_eq!(base[2], bumped[2], "sample 2 leaked");
        assert!(
            bumped[1] > base[1],
            "corrupted sample's NLL did not rise: {} -> {}",
            base[1],
            bumped[1]
        );
    }

    /// backward() from per_sample.sum() must reach every parameter of all
    /// three recon heads — i.e. the new reduction still exercises the
    /// `conv_t` composite ConvTranspose1d backward (candle's native op has
    /// none) end to end. Same step-0 rationale as
    /// `every_temporal_recon_decoder_parameter_receives_gradient_at_init`.
    #[test]
    fn recon_nll_per_sample_gradient_reaches_all_recon_head_params() {
        let dev = Device::Cpu;
        let (dec, vm, real_x, d) = recon_fixture(&dev, 3);

        let (_, per) = dec.recon_nll(&real_x, &d).unwrap();
        let grads = per.sum(0).unwrap().backward().unwrap();

        let mut dead: Vec<String> = vm
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| {
                n.starts_with("ls_head_recon")
                    || n.starts_with("pi_head_recon")
                    || n.starts_with("data_head_recon")
            })
            .filter(|(_, v)| {
                grads
                    .get(v.as_tensor())
                    .map(|g| {
                        g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() == 0.0
                    })
                    .unwrap_or(true)
            })
            .map(|(n, _)| n.clone())
            .collect();
        dead.sort();
        assert!(
            dead.is_empty(),
            "recon-head params with no gradient through per_sample: {dead:?}"
        );
    }

    #[test]
    fn log_sum_exp_is_stable_and_correct() {
        let dev = Device::Cpu;
        // Correctness vs hand value: lse([0, log2]) = log(1 + 2) = log3.
        let x = Tensor::new(&[[0f32, std::f32::consts::LN_2]], &dev).unwrap();
        let lse = log_sum_exp(&x, 1).unwrap().to_vec1::<f32>().unwrap();
        assert!((lse[0] - 3f32.ln()).abs() < 1e-5, "lse={}", lse[0]);

        // Stability: a large logit that would overflow naïve exp (e^100=inf)
        // must still give a finite, near-exact answer (≈ the max).
        let big = Tensor::new(&[[100f32, 0.0, -100.0]], &dev).unwrap();
        let lse_big = log_sum_exp(&big, 1).unwrap().to_vec1::<f32>().unwrap();
        assert!(lse_big[0].is_finite() && (lse_big[0] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn smooth_clamp_saturates_within_band() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[-1000f32, 0.0, 1000.0], &dev).unwrap();
        let y = smooth_clamp(&x, -4.0, 2.0).unwrap().to_vec1::<f32>().unwrap();
        // Reaches the band edges (f32 tanh(±1000) rounds to exactly ±1) and
        // NEVER exceeds them — the whole point vs an unbounded log σ.
        assert!(y[0] >= -4.0 && y[0] <= -3.99, "low={}", y[0]);
        assert!(y[2] <= 2.0 && y[2] >= 1.99, "high={}", y[2]);
        assert!(y[1].abs() < 0.6, "mid={}", y[1]); // x=0 → tanh(1/3)·3−1 ≈ −0.035
    }

    #[test]
    fn config_default_sums() {
        let c = DecoderConfig::default();
        let sum: usize = c.disentangle_split.iter().sum();
        assert_eq!(sum, c.bottleneck_dim);
        assert_eq!(c.spectral_compressions.last().copied(), Some(c.bottleneck_dim));
    }

    #[test]
    fn multidim_layer_norm_zero_mean_unit_var() {
        let d = Device::Cpu;
        // shape [2, 4, 3] → normalise over last two dims (4x3 = 12 per sample).
        let x = Tensor::arange(0.0_f32, 24.0, &d).unwrap().reshape((2, 4, 3)).unwrap();
        let weight = Tensor::ones((4, 3), DType::F32, &d).unwrap();
        let bias = Tensor::zeros((4, 3), DType::F32, &d).unwrap();
        let ln = MultiDimLayerNorm {
            weight,
            bias,
            normalized_shape: vec![4, 3],
            eps: 1e-5,
        };
        let y = ln.forward(&x).unwrap();
        // Check mean per batch ≈ 0.
        let flat = y.flatten_from(1).unwrap();
        let mean = flat.mean(D::Minus1).unwrap().to_vec1::<f32>().unwrap();
        for m in mean {
            assert!(m.abs() < 1e-4, "mean {} not ~0", m);
        }
    }

    #[test]
    fn softplus_zero() {
        let d = Device::Cpu;
        let x = Tensor::zeros((3,), DType::F32, &d).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        for v in y {
            assert!((v - 0.6931472).abs() < 1e-5);
        }
    }

    #[test]
    fn disentangle_split_indices() {
        // Just sanity-check the split arithmetic used in disentangle().
        let split = [4_usize, 30, 30];
        assert_eq!(split[0] + split[1] + split[2], 64);
    }

    // ── MultiDimLayerNorm initialisation ────────────────────────────────
    //
    // These two are the regression net for a defect that presented as
    // "only 9 of the 36 recon-head parameters receive gradient", and whose
    // cause was a bare `VarBuilder::get` on the affine params. Both run
    // against a FRESH `VarMap`, because that is the only path where the
    // init hint is consulted at all — load a checkpoint and the stored
    // tensor wins, which is why this never showed up in serving.

    /// A LayerNorm must start as the identity transform: scale 1, shift 0.
    ///
    /// The tighter of the two tests — it fails at the exact line that is
    /// wrong, with no forward or backward pass in between to interpret.
    #[test]
    fn multidim_layer_norm_initialises_to_the_identity_transform() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ln = MultiDimLayerNorm::load(vec![4, 3], 1e-5, vb.pp("ln")).unwrap();

        let w = ln.weight.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = ln.bias.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            w.iter().all(|v| *v == 1.0),
            "LayerNorm scale must init to 1 (it is `Init::Const(0.)` if you \
             used `vb.get`); got {w:?}"
        );
        assert!(b.iter().all(|v| *v == 0.0), "LayerNorm shift must init to 0; got {b:?}");
    }

    /// Every parameter in the recon head receives gradient on the FIRST
    /// backward pass.
    ///
    /// # Why the whole head and not just the norm
    ///
    /// A zero affine scale is invisible to any probe that asks "does this
    /// module get gradient?", because the norm's own `weight`/`bias` stay
    /// live either way (`dL/dw = grad ⊗ x̂`, and `x̂` is normalised, hence
    /// non-zero). What dies is everything AROUND it: the next conv's weight
    /// (its input is identically zero, so `dL/dW = grad ⊗ 0`) and every
    /// parameter upstream (`∂(w·x̂)/∂x̂ = w = 0` blocks the chain). So the
    /// assertion has to span the stack.
    ///
    /// # Why the first step and not a few hundred
    ///
    /// Because "count of parameters with non-zero gradient" recovers on its
    /// own, and means nothing when it does. With the scale pinned at zero,
    /// this count is 3/12 at step 0 but 12/12 by step 50 — while the scale
    /// is still ~1e-5 against a correct value of 1, and the loss has not
    /// moved. The gradient becomes nominally non-zero long before the head
    /// becomes functional, so a late count is a false all-clear. Step 0 is
    /// the only reading that is unambiguous.
    #[test]
    fn every_temporal_recon_decoder_parameter_receives_gradient_at_init() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let head = TemporalReconDecoder::load(30, 60, vb.pp("h")).unwrap();

        let x = Tensor::randn(0f32, 1f32, (4, 30), &dev).unwrap();
        let loss = head.forward(&x).unwrap().sqr().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();

        let mut dead: Vec<String> = vm
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, v)| {
                grads
                    .get(v.as_tensor())
                    .map(|g| {
                        g.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() == 0.0
                    })
                    .unwrap_or(true)
            })
            .map(|(n, _)| n.clone())
            .collect();
        dead.sort();

        assert!(
            dead.is_empty(),
            "{} of 12 head parameters got no gradient: {dead:?}\n\
             `fc.*` + `deconv.0/1/3` + `deconv.6.weight` dead while \
             `deconv.4.*` and `deconv.6.bias` live is the signature of a \
             zero-initialised LayerNorm scale.",
            dead.len(),
        );
    }
}

/// Reconstruction loss weighted by the actual component coupling [N,M].
/// Predictions [N,D] and targets [M,D] may use different component counts.
/// The plan remains in the graph; no bridge objective is constructed.
pub fn bridge_weighted_reconstruction(prediction: &Tensor, target: &Tensor, coupling: &Tensor) -> Result<Tensor> {
    let (n, d) = prediction.dims2()?;
    let (m, td) = target.dims2()?;
    if d == 0 || td != d || coupling.dims() != [n, m] { candle_core::bail!("reconstruction/coupling shape mismatch"); }
    let mass = coupling.sum_all()?;
    let value = mass.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    let min = coupling.min_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    if !value.is_finite() || value <= 0.0 || !min.is_finite() || min < 0.0 { candle_core::bail!("invalid coupling mass"); }
    let cost = prediction.unsqueeze(1)?.broadcast_sub(&target.unsqueeze(0)?)?.sqr()?.mean(2)?;
    cost.mul(coupling)?.sum_all()?.div(&mass)
}
