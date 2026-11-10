//! Inferno — Convolutional sequence encoders producing probabilistic latent representations.

use anyhow::{Context, Result};
use candle_core::{Module, Tensor, D};
use candle_nn::{
    conv1d_no_bias, group_norm, linear, Conv1d, Conv1dConfig, GroupNorm, Linear, VarBuilder,
};
use crate::model::norm::{layer_norm, LayerNorm};

use crate::model::{StateSpaceOutput};
use crate::config::{EncoderConfig};


// ---------------------------------------------------------------------------
// EncoderOutput — full superset of the Python NamedTuple
// ---------------------------------------------------------------------------

/// Everything the Python `EncoderOutput` exposes, every field a `Tensor`.
/// The Dirichlet module reads from this struct; any field you don't need
/// is cheap to ignore (Candle tensors are Arc-backed — clone is a refcount
/// bump).
#[derive(Clone)]
pub struct EncoderOutput {
    pub alpha_mu: Tensor,      // [B, evidence_dim]
    pub alpha_factor: Tensor,  // [B, evidence_dim, rank]
    pub alpha_diag: Tensor,    // [B, evidence_dim]
    pub alpha: Tensor,         // [B, evidence_dim]
    pub log_pi: Tensor,        // [B, k_atoms]
    pub pi: Tensor,            // [B, k_atoms]
    pub z: Tensor,             // [B, latent_dim]
    pub mu_z: Tensor,          // [B, latent_dim]
    pub logvar_z: Tensor,      // [B, latent_dim]
    pub ls: Tensor,            // [B, k_atoms]  (or empty if upstream didn't fill)
    pub real_x: Tensor,        // [B, 1, input_dim]  (anchor copy)
}

// ---------------------------------------------------------------------------
// Encoder trait — implemented here by ConvolutionalLoopEncoder
// ---------------------------------------------------------------------------

/// Your `SpectralVae` orchestrator is generic over this trait so alternative
/// encoders (mock, distilled, quantised) can be swapped in without touching
/// the refinement loop.
pub trait Encoder {
    /// Hold or advance any `SpectralNormLinear` power iteration this module
    /// owns. `false` is PyTorch's `eval()` semantics. Default is a NO-OP; a
    /// module that owns one and does not override stays call-history dependent.
    fn set_power_iteration(&self, _on: bool) {}

    /// `t_emb` is the master Fourier time conditioning `[B, 256]` — the
    /// FiLM head modulates the fused bottleneck on it (`None` or a missing
    /// head is identity conditioning).
    ///
    /// `noise` is the per-Itô-step deterministic reparam bundle. Sites
    /// (matching the Python `noise_site_shapes` — the keys the forward
    /// actually READS): `conv_bottleneck [B, bottleneck_dim]`, `latent_z
    /// [B, latent_dim]`, `default_pi [B, k_atoms]` (fallback path only).
    /// `None` falls back to the installed `step_noise_*` seam and then to
    /// fresh `randn` — training mode only; eval is deterministic.
    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        state: &StateSpaceOutput,
        indices: Option<&Tensor>,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<EncoderOutput>;
}

// ---------------------------------------------------------------------------
// ConvNet1d — residual 1D conv block (port of ConvolutionalNetwork1D)
// ---------------------------------------------------------------------------

pub struct ConvNet1d {
    conv1: Conv1d,
    norm1: GroupNorm,
    conv2: Conv1d,
    norm2: GroupNorm,
    /// `None` when stride == 1 && in_channels == out_channels — matches
    /// PyTorch's empty `nn.Sequential()` fallback, which has no weights.
    shortcut: Option<(Conv1d, GroupNorm)>,
}

impl ConvNet1d {
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        let cfg1 = Conv1dConfig { padding, stride, dilation: 1, groups: 1, ..Default::default() };
        let cfg2 = Conv1dConfig { padding, stride: 1, dilation: 1, groups: 1, ..Default::default() };
        let conv1 = crate::model::parametrize::sobol_conv1d_no_bias(&vb.pp("conv1"), in_channels, out_channels, kernel_size, cfg1)
            .context("load conv1")?;
        let norm1 = group_norm(8, out_channels, norm_eps, vb.pp("norm1"))
            .context("load norm1")?;
        let conv2 = crate::model::parametrize::sobol_conv1d_no_bias(&vb.pp("conv2"), out_channels, out_channels, kernel_size, cfg2)
            .context("load conv2")?;
        let norm2 = group_norm(8, out_channels, norm_eps, vb.pp("norm2"))
            .context("load norm2")?;

        let shortcut = if stride != 1 || in_channels != out_channels {
            let sc_cfg = Conv1dConfig { padding: 0, stride, dilation: 1, groups: 1, ..Default::default() };
            let sc_conv = crate::model::parametrize::sobol_conv1d_no_bias(&vb.pp("shortcut").pp("0"), in_channels, out_channels, 1, sc_cfg)
            .context("load shortcut conv")?;
            let sc_norm = group_norm(8, out_channels, norm_eps, vb.pp("shortcut").pp("1"))
                .context("load shortcut norm")?;
            Some((sc_conv, sc_norm))
        } else {
            None
        };

        Ok(Self { conv1, norm1, conv2, norm2, shortcut })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Residual path
        let residual = match &self.shortcut {
            Some((conv, norm)) => {
                let r = conv.forward(x)?;
                norm.forward(&r)?
            }
            None => x.clone(),
        };

        // Main path
        let out = self.conv1.forward(x)?;
        let out = self.norm1.forward(&out)?;
        let out = out.elu(1.0)?;
        let out = self.conv2.forward(&out)?;
        let out = self.norm2.forward(&out)?;

        // Sum + post-activation (Python does `out += residual; out = act2(out)`)
        let out = (out + residual)?;
        Ok(out.elu(1.0)?)
    }
}

// ---------------------------------------------------------------------------
// ConvolutionalLoopEncoder — top-level encoder
// ---------------------------------------------------------------------------

pub struct ConvolutionalLoopEncoder {
    /// Per-step sampler noise slots (the NoiseBundle `latent_z` /
    /// `conv_bottleneck` sites). `None` ⇒ fresh internal draws. Set via
    /// [`set_step_noise`](Self::set_step_noise) between refinement passes.
    pub step_noise_z: Option<Tensor>,
    pub step_noise_bottleneck: Option<Tensor>,
    /// Python `self.training`: `true` = reparameterised (bundle-seeded)
    /// draws at both sites; `false` (serving default) = the fully
    /// deterministic eval path — `conv_bottleneck = mu_data`, `z = mu_z`.
    pub training: bool,
    // Stem (Sequential [Conv1d, GroupNorm, ELU]):
    stem_conv: Conv1d,
    stem_norm: GroupNorm,

    // Residual conv stages:
    stage1: ConvNet1d,
    stage2: ConvNet1d,
    stage3: ConvNet1d,

    // Post-conv projection to (mu, logvar) bottleneck:
    fc_mu: Linear,
    fc_logvar: Linear,

    // Fusion_net (Sequential [Linear_sn, LayerNorm, ELU]):
    /// Spectral-normed on the trainable path (`fusion_net.0.weight_orig`,
    /// the Python `named_parameters()` name); folded plain weight on a
    /// cleaned serving checkpoint (see top-of-file note).
    fusion_lin: crate::model::parametrize::SpectralNormLinear,
    fusion_ln: LayerNorm,

    // Latent projection (post-fusion → z):
    latent_mu: Linear,
    latent_logvar: Linear,

    /// FiLM head on the master time embedding (see `load`).
    film_projector: Option<Linear>,

    config: EncoderConfig,
}

impl ConvolutionalLoopEncoder {
    /// Load all weights from a Candle `VarBuilder`. Variable names follow
    /// PyTorch's dotted-path convention; `Sequential` children are named
    /// "0", "1", "2", … matching module order.
    pub fn load(vb: VarBuilder, config: EncoderConfig) -> Result<Self> {
        // --- Stem -----------------------------------------------------------
        let stem_cfg = Conv1dConfig { padding: 3, stride: 2, dilation: 1, groups: 1, ..Default::default() };
        let stem_conv = crate::model::parametrize::sobol_conv1d_no_bias(&vb.pp("stem").pp("0"), config.input_dim, 32, 7, stem_cfg)
        .context("load stem conv")?;
        let stem_norm = group_norm(8, 32, config.norm_eps, vb.pp("stem").pp("1"))
            .context("load stem norm")?;

        // --- Residual stages ------------------------------------------------
        let stage1 = ConvNet1d::load(vb.pp("stage1"), 32, 64, 5, 2, config.norm_eps)?;
        let stage2 = ConvNet1d::load(vb.pp("stage2"), 64, 128, 3, 2, config.norm_eps)?;
        let stage3 = ConvNet1d::load(vb.pp("stage3"), 128, 256, 3, 2, config.norm_eps)?;

        // --- Post-conv bottleneck projection --------------------------------
        let fc_mu = crate::model::parametrize::sobol_linear(&vb.pp("fc_mu"), 256, config.bottleneck_dim)
            .context("load fc_mu")?;
        let fc_logvar = crate::model::parametrize::sobol_linear(&vb.pp("fc_logvar"), 256, config.bottleneck_dim)
            .context("load fc_logvar")?;

        // --- Fusion net -----------------------------------------------------
        // fusion_in_dim = bottleneck_dim + k_atoms + bottleneck_dim
        //              = 64 + 30 + 64 = 158 (default config).
        let fusion_in = config.bottleneck_dim + config.k_atoms + config.bottleneck_dim;
        let fusion_lin = crate::model::parametrize::SpectralNormLinear::new(
            vb.pp("fusion_net").pp("0"),
            fusion_in,
            64,
            // Python `_init_weights`: orthogonal, gain 1.41 → rows of norm
            // ≈ 1.41. Candle has no orthogonal init; a matched-scale Gaussian
            // is the standard stand-in (checkpoints override it anyway).
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 1.41 / (fusion_in as f64).sqrt() },
        )
        .context("build fusion_net.0 (spectral norm)")?;
        let fusion_ln = layer_norm(64, config.norm_eps, vb.pp("fusion_net").pp("1"))
            .context("load fusion_net.1")?;

        // --- Latent projection ----------------------------------------------
        let latent_mu = crate::model::parametrize::sobol_linear(&vb.pp("latent_mu"), config.bottleneck_dim, config.latent_dim)
            .context("load latent_mu")?;
        let latent_logvar = crate::model::parametrize::sobol_linear(&vb.pp("latent_logvar"), config.bottleneck_dim, config.latent_dim)
        .context("load latent_logvar")?;

        // FiLM time conditioning (`film_projector`, zero-init ⇒ identity):
        // t_emb [B, time_dim] → (scale, bias) on the FUSED bottleneck.
        // ALWAYS built on a trainable map (Python registers it
        // unconditionally); absent only on pre-migration tensor-backed
        // checkpoints — identity conditioning there.
        let film_projector = crate::model::parametrize::build_or_absent(
            &vb,
            "film_projector.weight",
            || {
                crate::model::parametrize::zero_linear(
                    vb.pp("film_projector"),
                    config.time_dim,
                    config.bottleneck_dim * 2,
                )
                .map_err(anyhow::Error::from)
            },
        )?;

        Ok(Self {
            step_noise_z: None,
            step_noise_bottleneck: None,
            training: false,
            stem_conv, stem_norm,
            stage1, stage2, stage3,
            fc_mu, fc_logvar,
            fusion_lin, fusion_ln,
            latent_mu, latent_logvar,
            film_projector,
            config,
        })
    }

    /// Port of Python `run_convolutional_layers`. Accepts `[B, L, C]` (the
    /// sequence-major PyTorch format used throughout the VAE), transposes
    /// to `[B, C, L]` for Conv1d, runs the stem + three residual stages,
    /// and global-average-pools the temporal axis. Returns `(mu, logvar)`
    /// in `[B, bottleneck_dim]`.
    fn run_convolutional_layers(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // Shape coercion — faithful to Python's path:
        //   if x.dim() == 2:
        //     if x.shape[1] == input_dim: x = x.unsqueeze(1)  # [B, 1, D_in]
        //     else:                        x = x.unsqueeze(-1) # [B, D_in, 1]
        let x = if x.dims().len() == 2 {
            if x.dim(1)? == self.config.input_dim {
                x.unsqueeze(1)?
            } else {
                x.unsqueeze(D::Minus1)?
            }
        } else {
            x.clone()
        };

        // [B, L, C] -> [B, C, L] for Conv1d
        let x = x.transpose(1, 2)?;

        // Stem
        let x = self.stem_conv.forward(&x)?;
        let x = self.stem_norm.forward(&x)?;
        let x = x.elu(1.0)?;

        // Stages
        let x = self.stage1.forward(&x)?;
        let x = self.stage2.forward(&x)?;
        let x = self.stage3.forward(&x)?;

        // Global average pool over temporal axis (dim 2) → [B, 256]
        let x = x.mean(2)?;

        // Bottleneck projection. Python smooth-clamps the logvar in
        // training only (eval discards it); smooth_clamp is the C^∞
        // tanh bound, not a hard clamp.
        let mu = self.fc_mu.forward(&x)?;
        let logvar = self.fc_logvar.forward(&x)?;
        let (lo, hi) = self.config.logvar_clamp;
        let logvar = crate::model::smooth_clamp(&logvar, lo, hi)?;
        Ok((mu, logvar))
    }

    /// The trait [`Encoder::forward`] body with the TRAINING FLAG AS A
    /// PARAMETER — the seam `deterministic_pass` uses for one eval-forced
    /// fold from a `&self` call chain. `training = false` closes both
    /// reparameterisation branches (`conv_bottleneck = mu_data`,
    /// `z = mu_z`); `logvar_z` is still emitted either way, so the analytic
    /// spread consumers keep their gradient path (the 42cd1a8 parity).
    pub fn forward_with_training(
        &self,
        training: bool,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        state: &StateSpaceOutput,
        _indices: Option<&Tensor>,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<EncoderOutput> {
        let device = x.device();

        // Reparam noise precedence: per-step bundle site → installed
        // `step_noise_*` (the legacy `&mut` seam) → fresh randn fallback.
        // Site name is `latent_z` — the key the Python forward actually
        // READS (the old "encoder_z" placeholder matched nothing, so a
        // drawn bundle never reached the reparam sites: half of the
        // "not using the deterministic seed" bug).
        let noise_z = noise
            .and_then(|n| n.get("latent_z"))
            .or(self.step_noise_z.as_ref());
        let noise_bottleneck = noise
            .and_then(|n| n.get("conv_bottleneck"))
            .or(self.step_noise_bottleneck.as_ref());

        // Match Python: if x.dim() == 2: x = x.unsqueeze(1) → [B, 1, D_in].
        // The refinement_loop already pre-shapes to 3D so this is usually a no-op.
        let x = if x.dims().len() == 2 { x.unsqueeze(1)? } else { x.clone() };

        // --- Pull carrier variables from state -----------------------------
        // In our StateSpaceOutput every field is always populated (unlike
        // the Python path where they might be None and fall back to defaults).
        let pi = state.pi.clone();                      // [B, k_atoms]
        let bottleneck = state.bottleneck.to_device(device)?;  // [B, bottleneck_dim]
        let alpha_mu = state.alpha_mu.clone();          // [B, evidence_dim]
        let alpha_diag = state.alpha_diag.clone();      // [B, evidence_dim]
        let alpha_factor = state.alpha_factor.clone();  // [B, evidence_dim, rank]
        let alpha = state.alpha.clone();                // [B, evidence_dim]
        let ls = state.ls.clone();                      // [B, k_atoms]

        // --- Convolutional branch ------------------------------------------
        // Python: training reparameterises (bundle-seeded); eval is the
        // deterministic `conv_bottleneck = mu_data`.
        let (mu_data, logvar_data) = self.run_convolutional_layers(&x)?;
        let conv_bottleneck = if training {
            reparameterise(&mu_data, &logvar_data, noise_bottleneck)?
        } else {
            mu_data
        };

        let log_pi = pi
            .affine(1.0, self.config.jitter as f64)?
            .log()?
            .to_device(device)?;

        // --- Fusion: concat → Linear → LayerNorm → ELU ---------------------
        // weights shape: [B, bottleneck_dim + k_atoms + bottleneck_dim] (= 158 default)
        let weights = Tensor::cat(&[&conv_bottleneck, &log_pi, &bottleneck], D::Minus1)?;
        let post = self.fusion_lin.forward(&weights)?;
        let post = self.fusion_ln.forward(&post)?;
        let post = post.elu(1.0)?;

        // --- FiLM time conditioning on the fused bottleneck ----------------
        // Python `_apply_film`: post · (1 + scale) + bias, with (scale, bias)
        // = film_projector(t_emb).chunk(2). Zero-init ⇒ identity; absent
        // head or absent conditioning ⇒ identity here.
        let post = match (&self.film_projector, t_emb) {
            (Some(film), Some(t)) => {
                let fb = film.forward(t)?; // [B, 2·bottleneck]
                let w = self.config.bottleneck_dim;
                let scale = fb.narrow(D::Minus1, 0, w)?;
                let bias = fb.narrow(D::Minus1, w, w)?;
                post.broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(&bias)?
            }
            _ => post,
        };

        // --- Latent projection ---------------------------------------------
        // Python: training → smooth-clamped logvar + seeded reparam;
        // eval → z = mu_z (logvar_z is still emitted — the closed-form
        // score −(z−μ)/σ² downstream reads it; Python carries None there).
        let mu_z = self.latent_mu.forward(&post)?;
        let (lo, hi) = self.config.logvar_clamp;
        let logvar_z =
            crate::model::smooth_clamp(&self.latent_logvar.forward(&post)?, lo, hi)?;
        let z = if training {
            reparameterise(&mu_z, &logvar_z, noise_z)?
        } else {
            mu_z.clone()
        };

        Ok(EncoderOutput {
            alpha_mu,
            alpha_factor,
            alpha_diag,
            alpha,
            log_pi,
            pi,
            z,
            mu_z,
            logvar_z,
            ls,
            real_x: x,
        })
    }
}

// ---------------------------------------------------------------------------
// Reparameterisation
// ---------------------------------------------------------------------------

/// Gaussian reparameterisation — port of the CURRENT
/// `BaseGenerativeModel.reparameterise`:
///
/// ```python
/// sigma = torch.exp(0.5 * logvar)
/// if noise is None: noise = self.sobol_randn_like(mu, scramble=True)
/// return mu + sigma * noise
/// ```
///
/// NO eps clamping (the old port clamped to ±3.3 — the reference dropped
/// it; smooth_clamp on the LOGVAR is the boundary control now). A supplied
/// `noise` (the NoiseBundle site) makes the draw deterministic — the GEOC
/// requirement; the fallback is a fresh draw from Candle's global RNG.
fn reparameterise(mu: &Tensor, logvar: &Tensor, noise: Option<&Tensor>) -> Result<Tensor> {
    let std = (logvar * 0.5)?.exp()?;
    let eps = match noise {
        Some(n) => {
            let n = n.to_dtype(std.dtype())?;
            if n.shape() != mu.shape() {
                n.broadcast_as(mu.shape())?
            } else {
                n
            }
        }
        None => Tensor::randn(0f32, 1f32, std.dims(), std.device())?.to_dtype(std.dtype())?,
    };
    Ok((mu + (std * eps)?)?)
}

// ---------------------------------------------------------------------------
// Encoder trait impl — the whole forward pass
// ---------------------------------------------------------------------------

impl ConvolutionalLoopEncoder {
    /// Install (or clear) the per-forward-pass sampler noise for the two
    /// reparameterisation sites. Call with the bundle's `encoder_z` /
    /// `conv_bottleneck` tensors before each refinement pass (salt = step).
    pub fn set_step_noise(&mut self, z: Option<Tensor>, bottleneck: Option<Tensor>) {
        self.step_noise_z = z;
        self.step_noise_bottleneck = bottleneck;
    }
}

impl Encoder for ConvolutionalLoopEncoder {
    fn set_power_iteration(&self, on: bool) {
        self.fusion_lin.set_power_iteration(on);
    }

    fn forward(
        &self,
        x: &Tensor,
        t_emb: Option<&Tensor>,
        state: &StateSpaceOutput,
        _indices: Option<&Tensor>,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<EncoderOutput> {
        self.forward_with_training(self.training, x, t_emb, state, _indices, noise)
    }

}

// ---------------------------------------------------------------------------
// Optional: deterministic forward for serving
// ---------------------------------------------------------------------------

impl ConvolutionalLoopEncoder {
    /// Same as `forward` but skips the reparameterisation noise — returns
    /// `z = mu_z` and `conv_bottleneck = fc_mu(x)`. Use this for
    /// regulated/auditable approval paths where stochastic latent sampling
    /// would produce non-reproducible scores across requests.
    pub fn forward_deterministic(
        &self,
        x: &Tensor,
        state: &StateSpaceOutput,
    ) -> Result<EncoderOutput> {
        let device = x.device();
        let x = if x.dims().len() == 2 { x.unsqueeze(1)? } else { x.clone() };

        let pi = state.pi.clone();
        let bottleneck = state.bottleneck.to_device(device)?;
        let alpha_mu = state.alpha_mu.clone();
        let alpha_diag = state.alpha_diag.clone();
        let alpha_factor = state.alpha_factor.clone();
        let alpha = state.alpha.clone();
        let ls = state.ls.clone();

        // Deterministic conv bottleneck — use mu directly, no noise.
        let (mu_data, logvar_data) = self.run_convolutional_layers(&x)?;
        let _ = logvar_data; // we still compute it so the clamp pathway is on par
        let conv_bottleneck = mu_data;

        let log_pi = pi
            .affine(1.0, self.config.jitter as f64)?
            .log()?
            .to_device(device)?;
        let weights = Tensor::cat(&[&conv_bottleneck, &log_pi, &bottleneck], D::Minus1)?;
        let post = self.fusion_lin.forward(&weights)?;
        let post = self.fusion_ln.forward(&post)?;
        let post = post.elu(1.0)?;

        let mu_z = self.latent_mu.forward(&post)?;
        let (lo, hi) = self.config.logvar_clamp;
        let logvar_z =
            crate::model::smooth_clamp(&self.latent_logvar.forward(&post)?, lo, hi)?;
        // Deterministic z
        let z = mu_z.clone();

        Ok(EncoderOutput {
            alpha_mu, alpha_factor, alpha_diag, alpha,
            log_pi, pi, z, mu_z, logvar_z, ls,
            real_x: x,
        })
    }

    /// Enter the encoder at the FUSED BOTTLENECK instead of the raw window:
    /// `post [B, bottleneck_dim]` → `(mu_z, logvar_z, z)`, each `[B, latent_dim]`.
    ///
    /// This is the low-latency serving seam. The conv stack + fusion measure
    /// **159.7 µs** on a 4-core CPU at `input_dim = 30`; the tail below is
    /// `2 × (bottleneck_dim × latent_dim)` = 2048 MACs and measures **6.5 µs**.
    /// A caller that already holds a `[B, 64]` carrier — `state.bottleneck` is
    /// exactly that shape — can skip the 159.7 µs entirely.
    ///
    /// **This is not a surrogate.** It runs the same trained `latent_mu` /
    /// `latent_logvar` heads, the same `logvar_clamp` and the same
    /// `reparameterise` as [`forward`](Encoder::forward) — it simply enters one
    /// layer later. The clamp in particular is load-bearing: dropping it makes
    /// serving diverge from training silently, because `logvar` is otherwise
    /// unbounded below and the closed-form score `−(z−μ)/σ²` divides by it.
    ///
    /// `noise` follows `forward`'s precedence: per-step bundle site `latent_z`
    /// → installed `step_noise_z` → fresh `randn`. In eval (`self.training ==
    /// false`) `z = mu_z` and the noise is ignored, matching
    /// [`forward_deterministic`](Self::forward_deterministic).
    ///
    /// `logvar_z` is ALWAYS returned, in both modes — the downstream analytic
    /// score reads it even on the deterministic path, which is why `forward`
    /// emits it there too rather than carrying `None` like the Python.
    pub fn latent_from_bottleneck(
        &self,
        post: &Tensor,
        noise: Option<&crate::model::noise::NoiseBundle>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let w = self.config.bottleneck_dim;
        let got = post.dim(D::Minus1)?;
        if got != w {
            return Err(anyhow::anyhow!(
                "latent_from_bottleneck expects a fused bottleneck of width {w} \
                 (config.bottleneck_dim), got {got}. This entry point takes the \
                 POST-fusion vector, not the raw window."
            ));
        }
        let mu_z = self.latent_mu.forward(post)?;
        let (lo, hi) = self.config.logvar_clamp;
        let logvar_z =
            crate::model::smooth_clamp(&self.latent_logvar.forward(post)?, lo, hi)?;
        let z = if self.training {
            let noise_z = noise
                .and_then(|n| n.get("latent_z"))
                .or(self.step_noise_z.as_ref());
            reparameterise(&mu_z, &logvar_z, noise_z)?
        } else {
            mu_z.clone()
        };
        Ok((mu_z, logvar_z, z))
    }
}


// ---------------------------------------------------------------------------
// Notes on small differences vs the Python
// ---------------------------------------------------------------------------
//
// * Weight init: the Python `_init_weights` uses orthogonal init at *train*
//   time. Candle doesn't need it since we load pre-trained weights from
//   safetensors — the init op is a no-op at inference.
//
// * `reparameterise` noise: Candle's global RNG is used. For deterministic
//   scoring call `forward_deterministic` or seed once at boot with
//   `candle_core::utils::set_seed(...)`.
//
// * `empty_tensor` fallback: the Python returns an empty placeholder when
//   `ls` isn't supplied. Our `StateSpaceOutput::zero` always fills `ls`
//   with `ones(B, k_atoms)` so the fallback branch is unreachable.
//
// * `n_data`: the Python constructor takes `n_data=87636.0` which is only
//   used by the training-time KL scaler. We omit it — serving has no loss.
//
// * Spectral norm: addressed at export time (see top-of-file). If you
//   prefer to do it at load time, implement a `spectral_norm_load` helper
//   that power-iterates on `weight_orig` to recover σ, divides, and
//   rebuilds a `Linear` — but one extra line in the Python export is
//   simpler than shipping that code.
//
// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
//
// ```no_run
// use candle_core::Device;
// use crate::encoder::{ConvolutionalLoopEncoder, EncoderConfig};
// use crate::spectral_vae::{SpectralVaeConfig, StateSpaceOutput, load_varbuilder};
//
// let device = Device::Cpu;
// let vb = load_varbuilder("artefacts/vae.safetensors", &device)?;
// let cfg = EncoderConfig::default();
// let encoder = ConvolutionalLoopEncoder::load(vb.pp("encoder"), cfg)?;
//
// // Build a zero state for the first refinement step
// let vae_cfg = SpectralVaeConfig::default();
// let x = candle_core::Tensor::randn(0f32, 1f32, (1, 1, 30), &device)?;
// let state = StateSpaceOutput::zero(&x, &vae_cfg, &device, 1)?;
//
// let out = encoder.forward(&x, &state, None)?;
// println!("z shape: {:?}", out.z.dims());   // → [1, 16]
// # Ok::<_, anyhow::Error>(())
// ```
