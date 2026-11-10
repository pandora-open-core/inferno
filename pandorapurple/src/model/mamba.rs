//! Inferno — Selective state-space sequence layers on Candle tensors.

use candle_core::{Result, Tensor, D};
use std::collections::HashMap;
use candle_nn::init::Init;
use candle_nn::ops::silu;
use candle_nn::{Linear, Module, VarBuilder};
use crate::model::norm::{layer_norm, LayerNorm};

/// Carried recurrence state: `(h [B, d_inner, d_state], conv_cache [B, d_inner, d_conv−1])`.
pub type SsmState = (Tensor, Tensor);

/// 0-d tensor → host `f64`, for diagnostics only.
fn scalar(t: &Tensor) -> Result<f64> {
    t.to_dtype(candle_core::DType::F64)?.to_vec0::<f64>()
}

/// Per-column unbiased standard deviation over dim 0 — `torch.std(dim=0)`.
fn std_dim0(t: &Tensor) -> Result<Tensor> {
    let n = t.dim(0)? as f64;
    let mu = t.mean_keepdim(0)?;
    let ss = t.broadcast_sub(&mu)?.sqr()?.sum(0)?;
    ss.affine(1.0 / (n - 1.0).max(1.0), 0.0)?.sqrt()
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = x.abs()?.neg()?;
    relu.add(&neg_abs.exp()?.affine(1.0, 1.0)?.log()?)
}

pub struct MambaBlock {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub d_inner: usize,
    pub dt_rank: usize,
    /// Zero-order hold on `B`. See `forward`.
    pub zoh_b: bool,
    /// Duration of one scan step. `Δ` is multiplied by it, and `dt_proj.bias`
    /// is initialised divided by it, so the two cancel at init.
    pub physical_dt: f64,
    /// The calibrated residual scale — a PLAIN FIELD, not a `Var`.
    ///
    /// Python makes it a non-persistent buffer for two reasons this mirrors: it
    /// is a pure function of the config knobs rather than learned state, so it
    /// is recomputed at every init and a stale value from an older checkpoint
    /// can never override a changed `out_rms`; and folding it into `out_proj`'s
    /// weights instead would leave `‖out_proj‖ ~ 1600` at `d_init = 0`, which
    /// breaks everything that READS that norm — AdamW's step is ~lr regardless
    /// of gradient magnitude, so entries of ~70 would move by 1e-5 relative per
    /// step and the layer would be effectively frozen.
    pub out_gain: f64,
    in_proj: Linear,   // d_model → 2·d_inner (no bias)
    conv_w: Tensor,    // conv1d.weight [d_inner, 1, d_conv] (depthwise)
    conv_b: Tensor,    // conv1d.bias   [d_inner]
    x_proj: Linear,    // d_inner → dt_rank + 2·d_state (no bias)
    dt_proj: Linear,   // dt_rank → d_inner
    a_log: Tensor,     // [d_inner, d_state]
    d_skip: Tensor,    // D: [d_inner]
    out_proj: Linear,  // d_inner → d_model (no bias, zero-init)
}

/// The parameterisation THIS port implements. **Now v2**, matching Python.
///
/// The v1 tree — Euler discretisation of `B`, `D` initialised to 1.0,
/// `dt_rank = ceil(d_model/16)`, zero-init `out_proj`, no physical step — is
/// what Python measured DEAD and what this crate shipped until now.
///
/// The version guard is not belt-and-braces. `dt_proj.bias` is `[d_inner]` in
/// both versions, so a mismatched artefact does not reliably fail on shape:
/// `dt_proj.weight` would (rank 4 vs 1), but the `D` initialisation, the ZOH
/// branch and the output gain are all *values*. A checkpoint that is partially
/// v2 — or one whose `dt_rank` happens to agree — would load into the wrong
/// arithmetic and train, with the block silently behaving as a gated conv MLP.
pub const PARAM_VERSION: u32 = 2;

/// What a checkpoint says about its own parameterisation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParamVersion {
    /// No buffer at all — a v1 artefact, from before Python began stamping it.
    Absent,
    /// Stamped, and equal to [`PARAM_VERSION`].
    Current,
    /// Stamped, and something else.
    Other(u32),
}

/// Read a checkpoint's stamped parameterisation without judging it.
///
/// Separate from [`assert_param_version`] because a LOADER may legitimately
/// choose to re-initialise the SSM subtree on a mismatch rather than fail —
/// which is what Python's `ssm_reinit_on_version_mismatch` does, and it is the
/// right policy for weights whose arithmetic has changed underneath them. That
/// decision belongs to the loader; this function only reports.
pub fn param_version_of(vb: &VarBuilder) -> Result<ParamVersion> {
    if !vb.contains_tensor("param_version") {
        return Ok(ParamVersion::Absent);
    }
    // `to_vec1` rather than `to_scalar`: Python stores this as
    // `torch.tensor([2])`, so the buffer is rank-1 `[1]` and `to_scalar`
    // refuses anything but rank 0.
    let got = vb
        .get(1, "param_version")?
        .flatten_all()?
        .to_dtype(candle_core::DType::F64)?
        .to_vec1::<f64>()?
        .first()
        .copied()
        .unwrap_or(PARAM_VERSION as f64) as u32;
    Ok(if got == PARAM_VERSION { ParamVersion::Current } else { ParamVersion::Other(got) })
}

/// Refuse a checkpoint this code cannot load correctly.
///
/// Both `Absent` and `Other` are refusals. Absence is a v1 artefact, whose SSM
/// weights were trained through Euler-discretised `B` against a `D = 1` skip
/// that swamped the state read-out — they are not v2 weights that happen to be
/// unlabelled, and loading them would be training the wrong arithmetic on the
/// wrong initialisation.
///
/// A loader that wants Python's `ssm_reinit_on_version_mismatch` behaviour
/// should call [`param_version_of`] and rebuild the SSM subtree fresh instead
/// of calling this.
pub fn assert_param_version(vb: &VarBuilder) -> Result<()> {
    match param_version_of(vb)? {
        ParamVersion::Current => Ok(()),
        ParamVersion::Absent => Err(candle_core::Error::Msg(format!(
            "mamba: this checkpoint carries no param_version, so it is v1 — trained through \
             Euler-discretised B against a D = 1 skip. This port implements v{PARAM_VERSION}. \
             The differences are VALUES, not shapes, so loading it would not fail; it would \
             train v1 weights through v2 arithmetic. Re-initialise the SSM subtree instead."
        ))),
        ParamVersion::Other(got) => Err(candle_core::Error::Msg(format!(
            "mamba: checkpoint carries param_version {got}, this port implements \
             {PARAM_VERSION}. Re-initialise the SSM subtree rather than loading it."
        ))),
    }
}

/// `e^x − 1`, stable at small `|x|`.
///
/// candle has no `expm1`, and the naive `exp(x) − 1` cancels: at the slow end of
/// the timescale bank `ΔA ~ −1e-2`, where `exp` returns `0.99005` and the
/// subtraction keeps only the digits that differ from 1. The series
/// `x + x²/2 + x³/6` is exact to f32 there and the branch is a `where`, so both
/// arms stay differentiable.
fn expm1(x: &Tensor) -> Result<Tensor> {
    let small = x.abs()?.le(&x.affine(0.0, 1e-4)?)?.to_dtype(x.dtype())?;
    let big = small.affine(-1.0, 1.0)?;
    let series = x
        .add(&x.sqr()?.affine(0.5, 0.0)?)?
        .add(&x.powf(3.0)?.affine(1.0 / 6.0, 0.0)?)?;
    let exact = x.exp()?.affine(1.0, -1.0)?;
    series.mul(&small)?.add(&exact.mul(&big)?)
}

/// Knobs the v2 block reads. Python's `SSMConfig` subset that reaches the block.
#[derive(Clone, Copy, Debug)]
pub struct MambaConfig {
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    /// `None` ⇒ `max(1, ceil(d_model/16))`. Production forces **4**; the
    /// derived formula gives 1 at `d_model = 16`, which is a different
    /// `dt_proj.weight` shape.
    pub dt_rank: Option<usize>,
    pub dt_min: f64,
    pub dt_max: f64,
    pub dt_init_floor: f64,
    /// `D`, the depthwise-conv skip. **0.0** in v2 — see the `new` docs.
    pub d_init: f64,
    /// Zero-order hold on `B` rather than Euler. **true** in v2.
    pub zoh_b: bool,
    /// Duration of ONE scan step, in the clock's units. `1.0` reproduces
    /// index-counting exactly; a live model sources it from
    /// `DataClock::ito_dt`.
    pub physical_dt: f64,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_state: 16,
            d_conv: 4,
            expand: 2,
            dt_rank: Some(4),
            dt_min: 1e-2,
            dt_max: 1e-1,
            dt_init_floor: 1e-4,
            d_init: 0.0,
            zoh_b: true,
            physical_dt: 1.0,
        }
    }
}

impl MambaBlock {
    /// Short-form constructor over the v2 defaults, for callers that only have
    /// `(d_state, d_conv, expand)` to say.
    ///
    /// `dt_rank` comes from [`MambaConfig::default`] (**4**) and is NOT derived.
    /// The derived formula `ceil(d_model/16)` is a v1 property: it returns 1 at
    /// `d_model = 16`, which is the production latent width, so deriving here
    /// builds `x_proj` as `[33, 32]` against a checkpoint written `[36, 32]` and
    /// every artefact load fails on shape. Pass `dt_rank: None` through
    /// [`MambaBlock::with_config`] if the derived rank is genuinely wanted.
    pub fn new(vb: VarBuilder, d_model: usize, d_state: usize, d_conv: usize, expand: usize) -> Result<Self> {
        Self::with_config(
            vb,
            d_model,
            MambaConfig { d_state, d_conv, expand, ..Default::default() },
        )
    }

    /// The v2 block.
    ///
    /// Five things differ from v1, and Python numbered them as the causes of a
    /// block it measured DEAD:
    ///
    /// 1. **`D` starts at 0, not 1.** At `d_model = 16` a unit skip is ~1300x
    ///    the state read-out, so the block degenerates into a gated conv MLP and
    ///    the whole selective scan is decoration. Learnable, so the network can
    ///    restore the skip if the conv path earns it.
    /// 2. **`Δ` is initialised log-uniform in `[dt_min, dt_max]`** through the
    ///    softplus inverse, rather than by candle's default `Linear` bias. The
    ///    whole fix is "the bias needs to sit three to five units lower".
    ///    `dt_rank^-0.5` on the WEIGHT keeps the input-dependent modulation
    ///    unit-scaled as the rank widens, so widening it changes the degrees of
    ///    freedom without also changing `Δ`'s scale.
    /// 4. **`out_proj` is calibrated, not zero-init.** Zero-init made the
    ///    insertion an exact identity — right for hot-patching a pretrained
    ///    pipeline, wrong for training, because `dL/dy_inner` is then exactly
    ///    zero and the entire interior is gradient-starved until `out_proj`
    ///    crawls off zero on downstream gradient alone. See
    ///    [`LatentSSM::calibrate_out_proj`].
    /// 5. **`B` is discretised by zero-order hold**, not Euler. See `forward`.
    ///
    /// (3 was a data-pipeline cause and does not live in this block.)
    ///
    /// `physical_dt` is the sixth change and is not one of the five: it makes
    /// `Δ` a duration rather than an index count. The bias init divides by it
    /// and the forward multiplies by it, so the two cancel exactly at
    /// initialisation and the only thing that changes is what `Δ` MEANS.
    pub fn with_config(vb: VarBuilder, d_model: usize, cfg: MambaConfig) -> Result<Self> {
        let d_inner = cfg.expand * d_model;
        let dt_rank = cfg
            .dt_rank
            .unwrap_or_else(|| ((d_model as f64) / 16.0).ceil().max(1.0) as usize)
            .max(1);
        let (d_state, d_conv) = (cfg.d_state, cfg.d_conv);

        // Check the stamp when the artefact carries one; do NOT register it.
        //
        // Python's `param_version` is a persistent BUFFER — it appears in
        // `state_dict()` and not in `named_parameters()`. candle's `VarMap`
        // conflates the two: anything a `VarBuilder` registers is a `Var`, so
        // stamping it here would add a 221st parameter to a tree Python reports
        // as 220 and break `sota_param_tree_matches_python_routing`,
        // `full_param_name_set_matches_python` and the routing digest — three
        // parity checks, to persist a constant.
        //
        // So the guard reads and never writes. It protects the direction that
        // matters, Python → Rust, since Python is the source of truth for
        // weights. A Rust-saved checkpoint carries no stamp and is indexed by
        // `absent`, which is why refusing absence belongs to a LOADER that knows
        // it is reading a Python artefact — see `assert_param_version`.
        if vb.contains_tensor("param_version") {
            assert_param_version(&vb)?;
        }

        // PyTorch's `nn.Linear`/`nn.Conv1d` default — `kaiming_uniform_(a=√5)`,
        // which reduces to `U(±1/√fan_in)`.
        //
        // Spelled out rather than left to `VarBuilder::get`, which defaults to
        // `Init::Const(0.)`. That default made the WHOLE BLOCK output exactly
        // zero on a fresh VarMap: `in_proj = 0` gives `x_in = z_gate = 0`, the
        // conv sees zeros, and the gate `silu(0) = 0` annihilates whatever the
        // scan produced. v1 shipped that way and its `identity_at_init` test
        // PASSED — because a dead block is trivially an identity. Anything
        // asserting "the block is inert at init" has to distinguish a calibrated
        // small residual from a structurally zero one, which is what
        // `the_interior_receives_gradient_at_init` now does.
        let unif = |fan_in: usize| Init::Uniform {
            lo: -(fan_in as f64).powf(-0.5),
            up: (fan_in as f64).powf(-0.5),
        };
        let in_proj = Linear::new(
            crate::model::parametrize::sobol_from_init(
                &vb.pp("in_proj"), "weight", (2 * d_inner, d_model), unif(d_model),
            )?,
            None,
        );
        let conv_w = crate::model::parametrize::sobol_from_init(
            &vb.pp("conv1d"), "weight", (d_inner, 1, d_conv), unif(d_conv),
        )?;
        let conv_b = vb.pp("conv1d").get_with_hints(d_inner, "bias", Init::Const(0.0))?;
        let x_proj = Linear::new(
            crate::model::parametrize::sobol_from_init(
                &vb.pp("x_proj"),
                "weight",
                (dt_rank + 2 * d_state, d_inner),
                unif(d_inner),
            )?,
            None,
        );

        // ---- cause 2: Δ init -------------------------------------------------
        let dt_w = crate::model::parametrize::sobol_uniform(
            &vb.pp("dt_proj"),
            "weight",
            (d_inner, dt_rank),
            -(dt_rank as f64).powf(-0.5),
            (dt_rank as f64).powf(-0.5),
        )?;
        // Log-uniform timescales in `[dt_min, dt_max]`, expressed as a UNIFORM
        // draw in softplus-inverse space — which is the same distribution.
        //
        // `softplus⁻¹(x) = x + log(−expm1(−x))`, and over this band that is
        // numerically `ln x`: at `x = 0.01` it gives −4.60 against `ln = −4.605`,
        // at `x = 0.1` it gives −2.25 against −2.303. So a log-uniform draw in
        // `Δ` followed by `softplus⁻¹` and a uniform draw in bias space put the
        // bias in the same interval with the same density to within that
        // correction — and the interval IS cause 2 ("the bias needs to sit three
        // to five units lower", roughly [−4.6, −2.25]).
        //
        // Divided by `physical_dt` first: the forward multiplies `Δ` by it, so
        // the sampled band — which is a band in STEPS — must be compensated here
        // or the block starts with a horizon `1/dt` times longer and `dA → 1`,
        // a perfect integrator that never forgets. The two cancel at init.
        let inv_softplus = |x: f64| -> f64 { x + (-(-x).exp_m1()).ln() };
        let pdt = cfg.physical_dt.max(1e-12);
        let dt_b = crate::model::parametrize::sobol_uniform(
            &vb.pp("dt_proj"),
            "bias",
            d_inner,
            inv_softplus((cfg.dt_min / pdt).max(cfg.dt_init_floor)),
            inv_softplus((cfg.dt_max / pdt).max(cfg.dt_init_floor)),
        )?;
        let dt_proj = Linear::new(dt_w, Some(dt_b));

        // A: diagonal, negative-real, stored as log.
        //
        // KNOWN DEVIATION, and it only bites on a fresh Rust build. Python seeds
        // the HiPPO-ish ladder `log(1..d_state)` repeated across `d_inner`, so
        // every row carries the same `N` horizons. candle's `Init` cannot
        // express a deterministic ladder, and `VarBuilder` hands back a `Tensor`
        // rather than the `Var`, so there is nothing to write through after the
        // fact without threading the `VarMap` into four call sites.
        //
        // A uniform draw over `[0, ln d_state]` spans the SAME horizon range and
        // preserves the per-channel diversity that is the point — what it does
        // not preserve is the ladder's structure across rows. That matters for
        // an exact Python-init comparison and not for training, and a loaded
        // checkpoint carries the trained values either way. Revisit if a
        // fresh-init parity fixture is ever wanted.
        let a_log = crate::model::parametrize::sobol_uniform(&vb, "A_log", (d_inner, d_state), 0.0, (d_state as f64).ln())?;

        // ---- cause 1: D starts at 0 -----------------------------------------
        let d_skip = vb.get_with_hints(d_inner, "D", Init::Const(cfg.d_init))?;
        // ---- cause 4: out_proj is NOT zero-init ------------------------------
        let out_proj = Linear::new(
            crate::model::parametrize::sobol_from_init(
                &vb.pp("out_proj"),
                "weight",
                (d_model, d_inner),
                unif(d_inner),
            )?,
            None,
        );

        Ok(Self {
            d_model,
            d_state,
            d_conv,
            d_inner,
            dt_rank,
            zoh_b: cfg.zoh_b,
            physical_dt: cfg.physical_dt,
            out_gain: 1.0,
            in_proj,
            conv_w,
            conv_b,
            x_proj,
            dt_proj,
            a_log,
            d_skip,
            out_proj,
        })
    }

    pub fn init_state(&self, batch: usize, device: &candle_core::Device, dtype: candle_core::DType) -> Result<SsmState> {
        Ok((
            Tensor::zeros((batch, self.d_inner, self.d_state), dtype, device)?,
            Tensor::zeros((batch, self.d_inner, self.d_conv.saturating_sub(1)), dtype, device)?,
        ))
    }

    /// Depthwise causal conv over `[B, d_inner, (k−1)+L]` → `[B, d_inner, L]`.
    fn depthwise_conv(&self, xc: &Tensor) -> Result<Tensor> {
        let w = self.conv_w.to_dtype(xc.dtype())?;
        let y = xc.conv1d(&w, 0, 1, 1, self.d_inner)?; // padding 0, groups = d_inner
        y.broadcast_add(&self.conv_b.to_dtype(xc.dtype())?.reshape((1, self.d_inner, 1))?)
    }

    /// `x: [B, L, d_model]` (+ optional carried state) → `([B, L, d_model], state)`.
    pub fn forward(&self, x: &Tensor, state: Option<SsmState>) -> Result<(Tensor, SsmState)> {
        let (b, l, _) = x.dims3()?;
        let (mut h, conv_cache) = match state {
            Some(s) => s,
            None => self.init_state(b, x.device(), x.dtype())?,
        };

        let xz = self.in_proj.forward(x)?; // [B, L, 2·d_inner]
        let x_in = xz.narrow(D::Minus1, 0, self.d_inner)?;
        let z_gate = xz.narrow(D::Minus1, self.d_inner, self.d_inner)?;

        // ---- depthwise causal conv with explicit cache ----------------------
        let xc = x_in.transpose(1, 2)?.contiguous()?; // [B, d_inner, L]
        let xc = Tensor::cat(&[&conv_cache, &xc], D::Minus1)?; // [B, d_inner, (k−1)+L]
        let cache_len = self.d_conv.saturating_sub(1);
        let total = xc.dim(D::Minus1)?;
        let new_conv_cache = xc.narrow(D::Minus1, total - cache_len, cache_len)?.contiguous()?;
        let x_conv = silu(&self.depthwise_conv(&xc)?.transpose(1, 2)?)?; // [B, L, d_inner]

        // ---- input-dependent Δ, B, C -----------------------------------------
        let x_dbl = self.x_proj.forward(&x_conv)?; // [B, L, dt_rank + 2N]
        let dt = x_dbl.narrow(D::Minus1, 0, self.dt_rank)?;
        let bm = x_dbl.narrow(D::Minus1, self.dt_rank, self.d_state)?;
        let cm = x_dbl.narrow(D::Minus1, self.dt_rank + self.d_state, self.d_state)?;
        // PHYSICAL STEP. `Δ` is dimensionless — "how much of a step" — and a
        // step used to be one scan index, so the block's memory horizon was
        // measured in indices and moved whenever the window's discretisation
        // moved. Multiplying by the clock's per-bin `dt` makes `Δ` a duration,
        // so `τ = 1/(Δ|A|)` is a time and stays put.
        //
        // The bias init is compensated by the same factor, so the BASE `Δ` is
        // preserved exactly. The INPUT-DEPENDENT half is not, and that is worth
        // being precise about: `softplus(Wx + b)·dt` cannot equal
        // `softplus(Wx + b_old)` for all x, because softplus is not homogeneous.
        // With `b` compensated it sits high enough that softplus is nearly
        // linear there, so the modulation passes through roughly proportionally
        // and ends up scaled alongside the base. That is the intended reading —
        // the modulation is a duration too — but it is a real change in how
        // strongly the input can move `Δ`, not a no-op.
        let dt = softplus(&self.dt_proj.forward(&dt)?)?.affine(self.physical_dt, 0.0)?;
        let a = self.a_log.to_dtype(x.dtype())?.exp()?.neg()?; // [d_inner, N]

        // ---- discretise + sequential selective scan --------------------------
        // dA = exp(Δ ⊗ A), dBx = Δ ⊗ B ⊗ x_conv  → both [B, L, d_inner, N]
        let da = {
            let t = dt
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&a.reshape((1, 1, self.d_inner, self.d_state))?)?;
            t.exp()?
        };
        // ZOH, NOT EULER (cause 5). The exact zero-order-hold discretisation of
        // `ḣ = Ah + Bx` over a step `Δ` is `h ← e^{ΔA}h + A⁻¹(e^{ΔA} − 1)Bx`;
        // `Δ·B·x` is only its `Δ → 0` limit. At the fast end of the bank
        // (`ΔA = −1.6`) Euler OVER-INJECTS BY 2x — `(e^{ΔA} − 1)/(ΔA) = 0.499` —
        // a per-channel gain error the scan then compounds across the rollout.
        //
        // `A` is strictly negative (`= −exp(A_log)`), so the division needs no
        // guard; `expm1` keeps the small-Δ end (`ΔA ~ −1e-2`) from cancelling.
        let dta = dt
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&a.reshape((1, 1, self.d_inner, self.d_state))?)?;
        let db = if self.zoh_b {
            expm1(&dta)?
                .broadcast_div(&a.reshape((1, 1, self.d_inner, self.d_state))?)?
        } else {
            dt.unsqueeze(D::Minus1)?.broadcast_as(dta.shape())?.contiguous()?
        };
        let dbx = db
            .broadcast_mul(&bm.unsqueeze(2)?)?
            .broadcast_mul(&x_conv.unsqueeze(D::Minus1)?)?;

        let mut ys = Vec::with_capacity(l);
        for i in 0..l {
            // h ← dA_i ⊙ h + dBx_i ;  y_i = Σ_n h·C_i
            let da_i = da.narrow(1, i, 1)?.squeeze(1)?; // [B, d_inner, N]
            let dbx_i = dbx.narrow(1, i, 1)?.squeeze(1)?;
            h = da_i.mul(&h)?.add(&dbx_i)?;
            let c_i = cm.narrow(1, i, 1)?.squeeze(1)?.unsqueeze(D::Minus1)?; // [B, N, 1]
            ys.push(h.contiguous()?.matmul(&c_i.contiguous()?)?.squeeze(D::Minus1)?); // [B, d_inner]
        }
        let y = Tensor::stack(&ys, 1)?; // [B, L, d_inner]

        let y = y.add(&x_conv.broadcast_mul(&self.d_skip.to_dtype(x.dtype())?)?)?; // skip
        let y = y.mul(&silu(&z_gate)?)?; // gated
        Ok((
            self.out_proj.forward(&y)?.affine(self.out_gain, 0.0)?,
            (h, new_conv_cache),
        ))
    }
}

// ===========================================================================
// Self-supervised next-step head — `LatentSSM.predictor` in Python.
// ===========================================================================

/// Knobs for the InfoNCE head. `predict_dim = 0` (or an empty horizon list)
/// builds no head at all and no `predictor.*` keys, byte-identical to a tree
/// from before this existed — the same optional-block contract Python uses.
#[derive(Clone, Debug)]
pub struct PredictorConfig {
    /// Projection width. `0` ⇒ no head.
    pub predict_dim: usize,
    /// InfoNCE temperature.
    pub temp: f64,
    /// One projection PAIR per horizon. A single shared pair would let the
    /// `k = 1` map — which a one-lag model solves — cover `k = 4` as well, and
    /// covering `k = 4` without memory is the thing multi-horizon forbids.
    pub horizons: Vec<usize>,
    /// `sg` on the key. `z_pre[t+1]` is NOT independent of `z_post[t]` (the
    /// encoder is called with the previous state), so without this the term is
    /// satisfiable by stamping a watermark on `z_post[t]`, letting the encoder
    /// carry it into `z_pre[t+1]`, and reading it back — predicting itself
    /// through a one-step laundering loop.
    pub detach_key: bool,
    /// Subtract the per-step batch mean from the key. Not optional in
    /// practice: each rollout step reads a different input row, so every key
    /// carries "I am step t", and a query that knows its own t can ride that to
    /// a real score without learning anything about content.
    pub center_key: bool,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self { predict_dim: 32, temp: 0.1, horizons: vec![1, 2, 4], detach_key: true, center_key: true }
    }
}

/// One `(q, k)` projection pair per horizon. Python holds these in an
/// `nn.ModuleDict` keyed `q{h}` / `k{h}`; the parameter names here match.
pub struct SsmPredictor {
    cfg: PredictorConfig,
    /// `(horizon, q, k)`, in the config's horizon order.
    heads: Vec<(usize, Linear, Linear)>,
}

impl SsmPredictor {
    pub fn new(vb: VarBuilder, d_model: usize, cfg: PredictorConfig) -> Result<Self> {
        let mut heads = Vec::with_capacity(cfg.horizons.len());
        for &h in &cfg.horizons {
            heads.push((
                h,
                crate::model::parametrize::sobol_linear(&vb.pp(format!("q{h}")), d_model, cfg.predict_dim)?,
                crate::model::parametrize::sobol_linear(&vb.pp(format!("k{h}")), d_model, cfg.predict_dim)?,
            ));
        }
        Ok(Self { cfg, heads })
    }

    pub fn horizons(&self) -> impl Iterator<Item = usize> + '_ {
        self.heads.iter().map(|(h, _, _)| *h)
    }

    pub fn predict_dim(&self) -> usize {
        self.cfg.predict_dim
    }
}

/// `F.normalize(t, dim=-1)` — unit rows, with Python's `eps = 1e-12` floor.
fn normalize_last(t: &Tensor) -> Result<Tensor> {
    let n = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.clamp(1e-12, f64::INFINITY)?;
    t.broadcast_div(&n)
}

/// Pre-norm residual wrapper around [`MambaBlock`], sized to the latent dim —
/// the drop-in for the `encoder_z → Dirichlet` boundary. Accepts `[B, L, D]`
/// or `[B, D]` (treated as L=1) and returns the same shape + the carried state.
pub struct LatentSSM {
    norm: LayerNorm,
    mamba: MambaBlock,
    /// The self-supervised InfoNCE head. `None` when the config asks for no
    /// head, or when a tensor-backed store predates it.
    predictor: Option<SsmPredictor>,
}

impl LatentSSM {
    /// No InfoNCE head — Python's `predict_dim = 0` default.
    pub fn new(vb: VarBuilder, d_model: usize, d_state: usize, d_conv: usize, expand: usize) -> Result<Self> {
        Self::with_config(
            vb,
            d_model,
            MambaConfig { d_state, d_conv, expand, ..Default::default() },
            None,
        )
    }

    /// The config-driven form. The only way to reach a non-default `dt_rank`,
    /// `physical_dt`, discretisation, or the InfoNCE head from outside this
    /// module.
    ///
    /// `predict` follows the optional-block contract: a `None` (or a
    /// zero-width / horizon-less) config registers no `predictor.*` keys, so a
    /// tree built without it is byte-identical to one from before the head
    /// existed. A store that CARRIES the head is adopted regardless — a
    /// checkpoint's own key set outranks a default.
    pub fn with_config(
        vb: VarBuilder,
        d_model: usize,
        cfg: MambaConfig,
        predict: Option<PredictorConfig>,
    ) -> Result<Self> {
        let predict = predict.filter(|p| p.predict_dim > 0 && !p.horizons.is_empty());
        let predictor = match predict {
            Some(p) => {
                let probe = format!("predictor.q{}.weight", p.horizons[0]);
                tracing::debug!(
                    probe = %probe, in_store = vb.contains_tensor(&probe),
                    "latent_ssm predictor",
                );
                // Strict when the store has it (a width mismatch must fail
                // loudly), permissive on a tensor-backed store that predates it.
                crate::model::parametrize::build_or_absent(&vb, &probe, || {
                    SsmPredictor::new(vb.pp("predictor"), d_model, p.clone())
                })?
            }
            None => None,
        };
        Ok(Self {
            norm: layer_norm(d_model, 1e-5, vb.pp("norm"))?,
            mamba: MambaBlock::with_config(vb.pp("mamba"), d_model, cfg)?,
            predictor,
        })
    }

    /// The head, when one was built.
    pub fn predictor(&self) -> Option<&SsmPredictor> {
        self.predictor.as_ref()
    }

    /// Set `out_gain` so a unit-Gaussian input produces a residual branch whose
    /// RMS is `out_rms` times the input's — the block starts at a KNOWN fraction
    /// of the trunk whatever the other knobs are set to.
    ///
    /// This replaces v1's zero-init `out_proj` (cause 4). Zero-init made the
    /// insertion an exact identity, which was the right call for hot-patching a
    /// pretrained pipeline and the wrong one for training: it also makes
    /// `dL/dy_inner` exactly zero, so the whole interior — `in_proj`, `conv1d`,
    /// `x_proj`, `dt_proj`, `A_log`, `D` — is gradient-starved until `out_proj`
    /// crawls off zero on downstream gradient alone. Python measured `‖out_proj‖`
    /// taking ~500 steps to reach 0.16, with the interior learning at ~1/100
    /// speed the whole time.
    ///
    /// A CALIBRATION rather than a hardcoded `std`, because the forward scale is
    /// a product of six config-settable knobs — and `d_init` especially, since
    /// setting `D` to 0 drops the block's raw output ~1300x. A constant would
    /// silently drift the moment any one of them moved.
    ///
    /// # What `out_rms` is measured against, because it is not what it looks like
    ///
    /// The block reads `norm(z)`, so LayerNorm has already removed `z`'s scale
    /// and the residual's magnitude is set by the weights ALONE — it does not
    /// track `‖z‖`. (A pre-norm residual block is scale-free by construction.)
    /// So this calibrates against a UNIT-RMS trunk, and the ratio actually
    /// realised in a live model is `out_rms / rms(z)`. Python measured
    /// `rms(z) ≈ 0.15`, so the 0.03 default realises ~0.20 — six times the
    /// number it names. That is a figure to read off a live diagnostic, not one
    /// to correct here: making the residual track `‖z‖` would make the block
    /// scale-adaptive, which is a real behaviour change and not one a
    /// calibration should smuggle in.
    ///
    /// `out_rms = 0.0` restores the exact-identity behaviour. Returns the gain.
    pub fn calibrate_out_proj(&mut self, out_rms: f64, n: usize, seed: u64) -> Result<f64> {
        if out_rms <= 0.0 {
            self.mamba.out_gain = 0.0;
            return Ok(0.0);
        }
        self.mamba.out_gain = 1.0;
        let d = self.mamba.d_model;
        // A deterministic standard-normal probe. Box–Muller over a small LCG
        // rather than `Tensor::randn`, because the calibration has to be
        // reproducible across runs and candle's global RNG is not seedable per
        // call — a gain that moves between two builds of the same config would
        // make every `ssm/*` diagnostic incomparable.
        let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || -> f64 {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut buf = Vec::with_capacity(n * d);
        while buf.len() < n * d {
            let (u1, u2) = (next().max(1e-12), next());
            let r = (-2.0 * u1.ln()).sqrt();
            buf.push(r * (std::f64::consts::TAU * u2).cos());
            buf.push(r * (std::f64::consts::TAU * u2).sin());
        }
        buf.truncate(n * d);

        let dev = self.mamba.d_skip.device().clone();
        let dt = self.mamba.d_skip.dtype();
        let x = Tensor::from_vec(buf, (n, 1, d), &dev)?.to_dtype(dt)?;
        let (y, _) = self.mamba.forward(&self.norm.forward(&x)?, None)?;
        let rms = |t: &Tensor| -> Result<f64> {
            Ok(t.sqr()?.mean_all()?.sqrt()?.to_dtype(candle_core::DType::F64)?.to_vec0::<f64>()?)
        };
        let (cur, reference) = (rms(&y)?, rms(&x)?);
        let gain = (out_rms * reference) / cur.max(1e-12);
        self.mamba.out_gain = gain;
        Ok(gain)
    }

    /// The block's current residual gain.
    pub fn out_gain(&self) -> f64 {
        self.mamba.out_gain
    }

    pub fn forward(&self, z: &Tensor, state: Option<SsmState>) -> Result<(Tensor, SsmState)> {
        let squeeze = z.rank() == 2;
        let z3 = if squeeze { z.unsqueeze(1)? } else { z.clone() };
        let (y, state) = self.mamba.forward(&self.norm.forward(&z3)?, state)?;
        let out = z3.add(&y)?; // residual — identity at init
        Ok((if squeeze { out.squeeze(1)? } else { out }, state))
    }

    /// The self-supervised InfoNCE term — Python `LatentSSM.predictive_loss`.
    ///
    /// WHY THIS EXISTS. Nothing else in this stack rewards the latent
    /// trajectory for carrying history, so it does not. Python measured this
    /// block moving `z` by 0.21%, 99.8% of THAT a batch-constant vector, with
    /// no loss term referencing it at all — `out_proj` grew 0.159 → 0.161 over
    /// 120 steps on downstream gradient alone. This term is the supervision.
    ///
    /// WHY THE QUERY IS THE RESIDUAL, NOT THE OUTPUT. `z_post = z_pre + y` with
    /// `‖y‖/‖z‖ ~ 3%`, so a query built from `z_post` is, to three digits, a
    /// query built from `z_pre` — and "predict `z_pre[t+1]` from `z_pre[t]`" is
    /// a task a STATELESS map solves exactly as well, which would make the term
    /// satisfiable without ever using the recurrence. The query is `y` alone,
    /// so the only tensor this loss can shape is the SSM's own contribution.
    ///
    /// WHY CONTRASTIVE AND NOT REGRESSION. An L2 regression onto the next
    /// latent has a trivial optimum — constant output — and that is not
    /// hypothetical, it IS the measured state of this block. InfoNCE inverts
    /// the incentive: a batch-constant query gives every row the SAME
    /// distribution over keys, and the mean of `−log p_i` over a fixed `p` is
    /// `≥ ln(N)` by Jensen, with equality only when `p` is uniform. So the loss
    /// is FLOORED at `ln(N)` once `y` collapses, unimprovable, and the only way
    /// down is a sample-dependent `y`. Normalised by `ln(N)` so it lands in
    /// ~`[0, 1]` and its weight is comparable rather than a scale.
    ///
    /// `n_real` is the unpadded suffix length. `z_history` is LEFT-padded by
    /// repeating the earliest step, and padded rows are IDENTICAL, so a
    /// `(t, t+h)` pair inside the pad is a duplicate — a false negative that
    /// poisons the softmax, because two rows of the batch carry the same key
    /// and the "correct" one is unidentifiable.
    ///
    /// Returns `(loss, diagnostics)`; the loss is `None` whenever the term is
    /// inapplicable, with `ssm/skip` saying why (2 = no head, 1 = bad shapes,
    /// 3 = too short).
    pub fn predictive_loss(
        &self,
        z_post: &Tensor,
        z_pre: &Tensor,
        n_real: Option<usize>,
    ) -> Result<(Option<Tensor>, HashMap<String, f64>)> {
        let mut diag: HashMap<String, f64> = HashMap::new();
        let Some(pred) = self.predictor.as_ref() else {
            diag.insert("ssm/skip".into(), 2.0);
            return Ok((None, diag));
        };
        if z_post.rank() != 3 || z_post.dims() != z_pre.dims() {
            diag.insert("ssm/skip".into(), 1.0);
            return Ok((None, diag));
        }
        let (b, t, _) = z_post.dims3()?;
        let n = n_real.map_or(t, |v| v.min(t));
        if n < 2 {
            diag.insert("ssm/skip".into(), 3.0);
            return Ok((None, diag));
        }
        let lo = t - n;

        let y = z_post.sub(z_pre)?.narrow(1, lo, n)?; // [B, n, D] — the residual
        let mut k_src = z_pre.narrow(1, lo, n)?;
        if pred.cfg.detach_key {
            k_src = k_src.detach();
        }
        if pred.cfg.center_key {
            // Per-step batch mean. Removes ALL batch-constant per-step
            // structure, so the objective is unimprovable by the very constant
            // bias this block otherwise degenerates into.
            k_src = k_src.broadcast_sub(&k_src.mean_keepdim(0)?)?;
        }

        let mut losses: Vec<Tensor> = Vec::new();
        let (mut acc_sum, mut chance_sum, mut key_std_sum) = (0.0f64, 0.0f64, 0.0f64);
        let (mut pairs, mut used) = (0.0f64, 0usize);

        for (h, q_head, k_head) in &pred.heads {
            let h = *h;
            if n < h + 1 {
                continue;
            }
            let span = n - h;
            let n_pairs = b * span;
            if n_pairs < 2 {
                continue;
            }
            let p_dim = pred.cfg.predict_dim;

            let q = normalize_last(&q_head.forward(&y.narrow(1, 0, span)?.contiguous()?)?)?
                .reshape((n_pairs, p_dim))?;
            let k = normalize_last(&k_head.forward(&k_src.narrow(1, h, span)?.contiguous()?)?)?
                .reshape((n_pairs, p_dim))?;

            let logits = q.matmul(&k.t()?.contiguous()?)?.affine(1.0 / pred.cfg.temp, 0.0)?;
            let labels = Tensor::arange(0u32, n_pairs as u32, logits.device())?;
            let ln_n = (n_pairs as f64).ln();
            let loss_h = candle_nn::loss::cross_entropy(&logits, &labels)?.affine(1.0 / ln_n, 0.0)?;
            diag.insert(format!("ssm/infonce_h{h}"), scalar(&loss_h)?);
            losses.push(loss_h);

            used += 1;
            pairs += n_pairs as f64;
            let hit = logits
                .argmax(D::Minus1)?
                .eq(&labels)?
                .to_dtype(logits.dtype())?
                .mean_all()?;
            acc_sum += scalar(&hit)?;
            chance_sum += 1.0 / n_pairs as f64;
            // Keys are unit-norm, so this is the spread ON the sphere: it goes
            // to 0 exactly when every key coincides.
            key_std_sum += scalar(&std_dim0(&k)?.mean_all()?)?;
        }

        if losses.is_empty() {
            diag.insert("ssm/skip".into(), 3.0);
            return Ok((None, diag));
        }

        let loss = Tensor::stack(&losses, 0)?.mean_all()?;

        // The probe's two headline quantities: `rel_delta` says how much the
        // block moves z at all, `const_share` how much of that movement is the
        // same vector for every sample. `const_share → 1` IS the pathology.
        let flat = y.flatten_to(1)?;
        let rows = flat.dim(0)? as f64;
        let mu = flat.mean_keepdim(0)?;
        let sq = scalar(&flat.sqr()?.sum_all()?)?;
        let z_ref = scalar(&z_pre.narrow(1, lo, n)?.sqr()?.sum_all()?)?.sqrt();
        diag.insert("ssm/infonce".into(), scalar(&loss)?);
        diag.insert("ssm/acc".into(), acc_sum / used as f64);
        diag.insert("ssm/chance".into(), chance_sum / used as f64);
        diag.insert("ssm/key_std".into(), key_std_sum / used as f64);
        diag.insert("ssm/n_pairs".into(), pairs);
        diag.insert("ssm/rel_delta".into(), sq.sqrt() / z_ref.max(1e-12));
        diag.insert("ssm/const_share".into(), scalar(&mu.sqr()?.sum_all()?)? * rows / sq.max(1e-12));

        Ok((Some(loss), diag))
    }
}

/// An ARMED one-slot holder for the scan state.
///
/// It exists because the mixture extractor is called from two places with
/// incompatible meanings. `Flow::rollout` walks a genuine sequence, where
/// carrying `h` from one step to the next is the whole point. One-off drift
/// evaluations — a loss term, a `u*` probe, the expert marginal fit — are
/// unrelated single points, and carrying state across them splices them into a
/// trajectory that never happened, producing a plausible `h` built from
/// evaluations that have no temporal relationship at all.
///
/// The Rust callers thread `Option<SsmState>` by hand and are therefore always
/// "armed" wherever they choose to thread, which is the same hazard with no
/// name. Arming it explicitly makes the sequence-vs-point distinction a thing
/// the code says rather than a thing the caller remembers.
#[derive(Default)]
pub struct SsmCarrier {
    armed: bool,
    slot: Option<SsmState>,
    batch: usize,
}

impl SsmCarrier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a sequence. Until this is called, `take` yields nothing and `put`
    /// discards — so a one-off evaluation cannot pick up someone else's state
    /// by forgetting to disarm.
    pub fn arm(&mut self) {
        self.armed = true;
    }

    /// End the sequence and drop whatever was carried.
    pub fn disarm(&mut self) {
        self.armed = false;
        self.slot = None;
        self.batch = 0;
    }

    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// The carried state, if this carrier is armed AND the batch still matches.
    ///
    /// A batch-size change drops the state rather than broadcasting it: `h` is
    /// `[B, d_inner, d_state]` and a silent reshape would mix one request's
    /// history into another's.
    pub fn take(&mut self, batch: usize) -> Option<SsmState> {
        if !self.armed || self.batch != batch {
            return None;
        }
        self.slot.take()
    }

    /// Store the state for the next step of the sequence.
    ///
    /// DETACHED, and that is not optional: candle's `Tensor::clone` shares
    /// storage, so a carrier holding a live tensor would keep the whole step's
    /// graph alive across the rollout and leak the previous step's autograd
    /// history into the next one's backward.
    pub fn put(&mut self, state: SsmState, batch: usize) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        self.slot = Some((state.0.detach(), state.1.detach()));
        self.batch = batch;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Verification of selective state-space sequence layers on candle tensors.

    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// The head's parameter tree, pinned by NAME and SHAPE against the live
    /// artefact — `vae.latent_ssm.predictor.{q,k}{1,2,4}.{weight,bias}`.
    #[test]
    fn the_predictor_tree_matches_the_artefact() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::with_config(
            vb, 16, MambaConfig::default(), Some(PredictorConfig::default()),
        ).unwrap();

        assert_eq!(ssm.predictor().unwrap().predict_dim(), 32);
        assert_eq!(ssm.predictor().unwrap().horizons().collect::<Vec<_>>(), vec![1, 2, 4]);

        let data = vm.data().lock().unwrap();
        let pred: Vec<&String> = data.keys().filter(|k| k.starts_with("predictor.")).collect();
        assert_eq!(pred.len(), 12, "six (q, k) pairs: {pred:?}");
        for h in [1usize, 2, 4] {
            for side in ["q", "k"] {
                assert_eq!(data[&format!("predictor.{side}{h}.weight")].dims(), &[32, 16]);
                assert_eq!(data[&format!("predictor.{side}{h}.bias")].dims(), &[32]);
            }
        }
    }

    /// The optional-block contract: no config, no keys. A tree built without
    /// the head must be byte-identical to one from before it existed, or every
    /// pre-head checkpoint starts failing to load.
    #[test]
    fn no_predictor_config_registers_no_predictor_keys() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::new(vb, 16, 16, 4, 2).unwrap();

        assert!(ssm.predictor().is_none());
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(!names.iter().any(|n| n.contains("predictor")), "{names:?}");
        assert_eq!(names.len(), 11, "the pre-head tree is 11 tensors: {names:?}");
    }

    /// The Jensen floor, which is the whole reason this term is contrastive.
    ///
    /// A batch-constant residual gives every row of the batch the SAME
    /// distribution over keys, and the mean of `−log p_i` over a fixed `p` is
    /// `≥ ln(N)`. Normalised by `ln(N)`, the loss is therefore FLOORED at 1.0
    /// and descent buys exactly nothing by collapsing — the property an L2
    /// regression onto the next latent does not have, since its optimum IS the
    /// constant output.
    #[test]
    fn a_collapsed_residual_cannot_beat_the_ln_n_floor() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::with_config(
            vb, 16, MambaConfig::default(), Some(PredictorConfig::default()),
        ).unwrap();

        let z_pre = Tensor::randn(0f32, 1f32, (6, 8, 16), &dev).unwrap();
        // y ≡ 1: the block contributes the same vector for every sample and step.
        let z_post = z_pre.affine(1.0, 1.0).unwrap();

        let (loss, diag) = ssm.predictive_loss(&z_post, &z_pre, None).unwrap();
        let v = scalar(&loss.unwrap()).unwrap();
        assert!(v >= 1.0 - 1e-4, "collapsed residual scored {v}, below the ln(N) floor");
        // const_share → 1 IS the pathology, and this is what reports it.
        assert!(diag["ssm/const_share"] > 0.99, "{:?}", diag["ssm/const_share"]);
        assert!(diag.contains_key("ssm/infonce_h4"));
    }

    /// A varied residual reaches BOTH the head and the block's interior. The
    /// term exists to supply gradient to a recurrence nothing else rewards, so
    /// a version that only trains its own projections would be pointless.
    #[test]
    fn the_predictive_term_reaches_the_head_and_the_recurrence() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::with_config(
            vb, 16, MambaConfig::default(), Some(PredictorConfig::default()),
        ).unwrap();

        let z_pre = Tensor::randn(0f32, 1f32, (4, 6, 16), &dev).unwrap();
        let (z_post, _) = ssm.forward(&z_pre, None).unwrap();
        let (loss, diag) = ssm.predictive_loss(&z_post, &z_pre, None).unwrap();
        let grads = loss.unwrap().backward().unwrap();

        let q1 = &ssm.predictor().unwrap().heads[0].1;
        for (name, t) in [
            ("predictor.q1", q1.weight()),
            ("mamba.x_proj", ssm.mamba.x_proj.weight()),
            ("mamba.out_proj", ssm.mamba.out_proj.weight()),
        ] {
            let g = grads.get(t).unwrap_or_else(|| panic!("{name} got no gradient"));
            let n = scalar(&g.abs().unwrap().max_all().unwrap()).unwrap();
            assert!(n > 0.0, "{name} is gradient-starved under the predictive term");
        }
        assert!(diag["ssm/acc"] >= 0.0 && diag["ssm/chance"] > 0.0);
    }

    /// Every inapplicable case is reported, not silently zero — a skipped term
    /// that returns 0.0 is indistinguishable from a satisfied one.
    #[test]
    fn the_skip_codes_say_why_the_term_did_not_run() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let with = LatentSSM::with_config(
            vb, 16, MambaConfig::default(), Some(PredictorConfig::default()),
        ).unwrap();
        let vm2 = VarMap::new();
        let without = LatentSSM::new(
            VarBuilder::from_varmap(&vm2, DType::F32, &dev), 16, 16, 4, 2,
        ).unwrap();

        let z = Tensor::randn(0f32, 1f32, (4, 6, 16), &dev).unwrap();

        // 2 = no head at all.
        let (l, d) = without.predictive_loss(&z, &z, None).unwrap();
        assert!(l.is_none() && d["ssm/skip"] == 2.0);

        // 1 = shapes the term cannot read.
        let flat = z.flatten_to(1).unwrap();
        let (l, d) = with.predictive_loss(&flat, &flat, None).unwrap();
        assert!(l.is_none() && d["ssm/skip"] == 1.0);

        // 3 = the unpadded suffix is too short to form a single (t, t+h) pair.
        let (l, d) = with.predictive_loss(&z, &z, Some(1)).unwrap();
        assert!(l.is_none() && d["ssm/skip"] == 3.0);
    }

    /// `n_real` takes the unpadded SUFFIX. `z_history` is left-padded by
    /// repeating the earliest step, and identical padded rows make a (t, t+h)
    /// pair a duplicate — two rows of the batch carrying the same key, so the
    /// "correct" one is unidentifiable and the softmax is poisoned.
    #[test]
    fn n_real_trims_the_left_pad_rather_than_the_tail() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::with_config(
            vb, 16, MambaConfig::default(), Some(PredictorConfig::default()),
        ).unwrap();

        let real = Tensor::randn(0f32, 1f32, (3, 4, 16), &dev).unwrap();
        // Left-pad by repeating step 0 three times: [p, p, p, real...].
        let pad = real.narrow(1, 0, 1).unwrap();
        let padded = Tensor::cat(&[&pad, &pad, &pad, &real], 1).unwrap();
        let post = padded.affine(1.0, 0.0).unwrap().add(
            &Tensor::randn(0f32, 0.1f32, padded.dims(), &dev).unwrap(),
        ).unwrap();

        let (_, trimmed) = ssm.predictive_loss(&post, &padded, Some(4)).unwrap();
        let (_, whole) = ssm.predictive_loss(&post, &padded, None).unwrap();
        // 7 steps vs 4 ⇒ strictly more pairs when the pad is included.
        assert!(whole["ssm/n_pairs"] > trimmed["ssm/n_pairs"]);
        assert_eq!(trimmed["ssm/n_pairs"], (3 * 3 + 3 * 2 + 0) as f64, "h=4 cannot fit in n=4");
    }

    /// The production latent width, pinned by SHAPE against the artefact.
    ///
    /// `dt_rank` was left derived at the `LatentSSM` call site after the v2
    /// port, so at `d_model = 16` it came out 1 and the block built
    /// `x_proj = [33, 32]` / `dt_proj.weight = [32, 1]`. Every checkpoint is
    /// written at `dt_rank = 4`, so the engine died on
    /// `shape mismatch for latent_ssm.mamba.x_proj.weight, expected: [33, 32],
    /// got: [36, 32]` before it served a single request. The derived formula is
    /// a v1 property; v2 reads the config.
    #[test]
    fn the_production_latent_width_matches_the_artefact() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        // (d_model, d_state, d_conv, expand) exactly as `SpectralVae::load` builds it.
        let ssm = LatentSSM::new(vb, 16, 16, 4, 2).unwrap();

        assert_eq!(ssm.mamba.dt_rank, 4, "dt_rank must come from config, not ceil(d_model/16)");
        assert_eq!(ssm.mamba.x_proj.weight().dims(), &[36, 32]);
        assert_eq!(ssm.mamba.dt_proj.weight().dims(), &[32, 4]);
    }

    /// The v2 parameterisation, asserted by name. Replaces the v1-contract test,
    /// which was written to fail exactly here — see its docstring in the P0
    /// commit: "when it fails after the v2 port, replace it, do not relax it".
    #[test]
    fn the_v2_contract_holds_where_v1s_did_not() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::new(vb, 16, 16, 4, 2).unwrap();

        assert_eq!(PARAM_VERSION, 2);

        // Cause 1: D starts at 0, not 1. At d_model = 16 a unit skip is ~1300x
        // the state read-out and the block degenerates to a gated conv MLP.
        let d: Vec<f32> = ssm.mamba.d_skip.flatten_all().unwrap().to_vec1().unwrap();
        assert!(d.iter().all(|v| *v == 0.0), "v2 inits D = 0; v1 used 1.0");

        // Cause 2: the Δ bias sits in the log-uniform band, roughly [−4.6, −2.25]
        // — "three to five units lower" than a default Linear bias.
        let b: Vec<f32> = ssm.mamba.dt_proj.bias().unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let (lo, hi) = b.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), v| {
            (l.min(*v), h.max(*v))
        });
        assert!(lo > -4.7 && hi < -2.2, "Δ bias band is [{lo}, {hi}], expected ~[−4.6, −2.25]");

        // Cause 5: ZOH is on.
        assert!(ssm.mamba.zoh_b, "v2 discretises B by zero-order hold");

        // A_log spans the horizon range rather than sitting at one horizon —
        // A_log = 0 means |A| = 1 for every state channel, whatever the data says.
        let al: Vec<f32> = ssm.mamba.a_log.flatten_all().unwrap().to_vec1().unwrap();
        let spread = al.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v))
            - al.iter().fold(f32::INFINITY, |m, v| m.min(*v));
        assert!(spread > 1.0, "A_log must span horizons, spread {spread}");

        // param_version is NOT a Var. Python has it as a buffer, outside
        // `named_parameters()`; registering it here would add a parameter to a
        // tree the parity tests pin at Python's count.
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(
            !names.iter().any(|n| n.contains("param_version")),
            "the version stamp must stay out of the VarMap: {names:?}"
        );
    }

    /// **Cause 5, as a number.** Euler over-injects by exactly 2x at `ΔA = −1.6`.
    ///
    /// The most valuable assertion in this port, because it is a measurable
    /// factor rather than a shape: `(e^{ΔA} − 1)/(ΔA) = 0.4988` at that point, so
    /// the Euler `Δ·B·x` is `1/0.4988 ≈ 2.0x` the exact zero-order-hold
    /// injection — a per-channel gain error the scan then compounds across the
    /// rollout.
    #[test]
    fn euler_over_injects_by_two_at_the_fast_end_of_the_bank() {
        let dev = Device::Cpu;
        for (dta_val, want) in [(-1.6_f64, 2.0_f64), (-0.01, 1.005), (-3.0, 3.157)] {
            let dta = Tensor::new(&[dta_val], &dev).unwrap();
            // ZOH factor (e^{ΔA} − 1)/(ΔA) against Euler's implicit 1.
            let zoh: f64 = expm1(&dta)
                .unwrap()
                .div(&dta)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap()[0];
            let ratio = 1.0 / zoh;
            assert!(
                (ratio - want).abs() < 0.01,
                "at ΔA = {dta_val} Euler is {ratio}x the ZOH injection, expected ~{want}"
            );
        }

        // And the two branches really do differ in the block: same weights, ZOH
        // on and off, different output.
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F64, &dev);
        let zoh = MambaBlock::with_config(
            vb.pp("m"),
            8,
            MambaConfig { d_state: 4, dt_rank: Some(2), ..Default::default() },
        )
        .unwrap();
        let vb2 = VarBuilder::from_varmap(&vm, DType::F64, &dev);
        let euler = MambaBlock::with_config(
            vb2.pp("m"),
            8,
            MambaConfig { d_state: 4, dt_rank: Some(2), zoh_b: false, ..Default::default() },
        )
        .unwrap();

        let x = Tensor::randn(0f64, 1f64, (2, 5, 8), &dev).unwrap();
        let (yz, _) = zoh.forward(&x, None).unwrap();
        let (ye, _) = euler.forward(&x, None).unwrap();
        let d = yz.sub(&ye).unwrap().abs().unwrap().max_all().unwrap().to_vec0::<f64>().unwrap();
        assert!(d > 1e-9, "the two discretisations must actually differ, got {d}");
    }

    /// `expm1` stays accurate where `exp(x) − 1` cancels.
    #[test]
    fn expm1_beats_the_naive_difference_at_the_slow_end() {
        let dev = Device::Cpu;
        for x in [-1e-2_f64, -1e-4, -1e-6, -1e-8] {
            let t = Tensor::new(&[x], &dev).unwrap();
            let got = expm1(&t).unwrap().to_vec1::<f64>().unwrap()[0];
            // Reference series, far more accurate than exp(x) − 1 at these x.
            let want = x + x * x / 2.0 + x * x * x / 6.0 + x.powi(4) / 24.0;
            assert!(
                (got - want).abs() / want.abs() < 1e-9,
                "expm1({x}) = {got}, expected ≈ {want}"
            );
        }
    }

    /// Cause 4: the interior is NOT gradient-starved at init.
    ///
    /// Stated as a gradient assertion rather than a value one, because gradient
    /// starvation is the actual defect — v1's zero-init `out_proj` made the
    /// block an exact identity AND made `dL/dy_inner` exactly zero, so the whole
    /// interior learned only once `out_proj` had crawled off zero.
    #[test]
    fn the_interior_receives_gradient_at_init() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // F32: candle's LayerNorm has no F64 kernel.
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::new(vb, 8, 4, 4, 2).unwrap();

        let z = Tensor::randn(0f32, 1f32, (3, 4, 8), &dev).unwrap();
        let (out, _) = ssm.forward(&z, None).unwrap();
        let loss = out.sqr().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();

        for (name, t) in [
            ("in_proj", &ssm.mamba.in_proj.weight().clone()),
            ("x_proj", &ssm.mamba.x_proj.weight().clone()),
            ("A_log", &ssm.mamba.a_log),
            ("conv1d", &ssm.mamba.conv_w),
        ] {
            let g = grads.get(t).unwrap_or_else(|| panic!("{name} got no gradient at all"));
            let n = g.abs().unwrap().max_all().unwrap()
                .to_dtype(DType::F64).unwrap().to_vec0::<f64>().unwrap();
            assert!(n > 0.0, "{name} is gradient-starved at init (v1's defect)");
        }
    }

    /// The calibration hits its target ratio and is REPRODUCIBLE — a gain that
    /// moved between two builds of one config would make every `ssm/*`
    /// diagnostic incomparable across runs.
    #[test]
    fn the_out_gain_calibration_is_on_target_and_deterministic() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut ssm = LatentSSM::new(vb, 8, 4, 4, 2).unwrap();
        // Deterministic PROBE. The realised-ratio assertion below is a
        // finite-sample RMS estimate; read off an unseeded probe it drifts
        // whenever anything upstream consumes a different number of numbers
        // from candle's process-global stream.
        //
        // The WEIGHTS are deliberately left at their real init. Overwriting
        // them with a uniform draw was tried and is wrong: `A_log` carries a
        // HiPPO-ish ladder and `D` inits at 1.0, so a flat overwrite builds a
        // degenerate SSM and the calibration is then measured on a model that
        // does not exist.
        // GAUSSIAN, via Box–Muller — not a uniform draw. `calibrate_out_proj`
        // fits the gain against a `randn` probe, so measuring it on a uniform
        // one compares two different distributions and reads ~1.7x high.
        let gauss = |seed: u64, n: usize| -> Vec<f32> {
            let mut s = seed;
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12)
            };
            (0..n)
                .map(|_| {
                    let (u1, u2) = (u(), u());
                    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
                })
                .collect()
        };
        // 4096, the production default, not 512. The realised-ratio assertion
        // below compares the calibration's estimate against a DIFFERENT probe,
        // so the gap between them is finite-sample error — and the weights are
        // now drawn from a low-discrepancy sequence, whose structure the
        // estimator sees more of at a small n than an iid draw's would show.
        let g1 = ssm.calibrate_out_proj(0.03, 4096, 0).unwrap();
        let g2 = ssm.calibrate_out_proj(0.03, 4096, 0).unwrap();
        assert_eq!(g1, g2, "same seed ⇒ same gain");
        assert!(g1 > 0.0 && g1.is_finite());

        // The realised residual RMS against a unit trunk.
        let x = Tensor::from_vec(gauss(0xA5A5_1234, 256 * 8), (256, 1, 8), &dev).unwrap();
        let (y, _) = ssm.mamba.forward(&ssm.norm.forward(&x).unwrap(), None).unwrap();
        let rms = |t: &Tensor| t.sqr().unwrap().mean_all().unwrap().sqrt().unwrap()
            .to_dtype(DType::F64).unwrap().to_vec0::<f64>().unwrap();
        let ratio = rms(&y) / rms(&x);
        assert!((ratio - 0.03).abs() < 0.01, "realised residual ratio {ratio}, target 0.03");

        // out_rms = 0 restores the exact identity.
        ssm.calibrate_out_proj(0.0, 64, 0).unwrap();
        assert_eq!(ssm.out_gain(), 0.0);
        let z = Tensor::randn(0f32, 1f32, (2, 3, 8), &dev).unwrap();
        let (o2, _) = ssm.forward(&z, None).unwrap();
        let d = o2.sub(&z).unwrap().abs().unwrap().max_all().unwrap()
            .to_dtype(DType::F64).unwrap().to_vec0::<f64>().unwrap();
        assert!(d < 1e-9, "out_rms = 0 must restore the exact identity, off by {d}");
    }

    /// The carrier is INERT until armed, and drops state on a batch change.
    #[test]
    fn the_carrier_is_inert_until_armed_and_drops_a_batch_change() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::new(vb, 8, 4, 4, 2).unwrap();

        let z = Tensor::randn(0f32, 1f32, (2, 3, 8), &dev).unwrap();
        let (_, st) = ssm.forward(&z, None).unwrap();

        let mut c = SsmCarrier::new();
        assert!(!c.is_armed());
        c.put(st.clone(), 2).unwrap();
        assert!(c.take(2).is_none(), "an unarmed carrier must not hand out state");

        c.arm();
        c.put(st.clone(), 2).unwrap();
        assert!(c.take(3).is_none(), "a batch change must drop the state, not reshape it");
        c.put(st.clone(), 2).unwrap();
        assert!(c.take(2).is_some());

        c.arm();
        c.put(st, 2).unwrap();
        c.disarm();
        assert!(c.take(2).is_none(), "disarm drops what was carried");
    }

    /// A v2 checkpoint must be refused, not loaded.    /// A v2 checkpoint must be refused, not loaded. The differences are VALUES
    /// as well as shapes, so nothing downstream would notice.
    #[test]
    fn a_future_param_version_is_refused_at_load() {
        let dev = Device::Cpu;
        let mut map = std::collections::HashMap::new();
        map.insert("param_version".to_string(), Tensor::new(&[3f32], &dev).unwrap());
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        assert_eq!(param_version_of(&vb).unwrap(), ParamVersion::Other(3));
        let err = assert_param_version(&vb).unwrap_err().to_string();
        assert!(err.contains("param_version 3"), "{err}");

        // v2 is what this port implements.
        let mut m2 = std::collections::HashMap::new();
        m2.insert("param_version".to_string(), Tensor::new(&[2f32], &dev).unwrap());
        let vb2 = VarBuilder::from_tensors(m2, DType::F32, &dev);
        assert_eq!(param_version_of(&vb2).unwrap(), ParamVersion::Current);
        assert!(assert_param_version(&vb2).is_ok());

        // An UNVERSIONED checkpoint is v1 and is now REFUSED — its SSM weights
        // were trained through Euler-discretised B against a D = 1 skip, so they
        // are not v2 weights that happen to be unlabelled.
        let empty: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let vb3 = VarBuilder::from_tensors(empty, DType::F32, &dev);
        assert_eq!(param_version_of(&vb3).unwrap(), ParamVersion::Absent);
        let err = assert_param_version(&vb3).unwrap_err().to_string();
        assert!(err.contains("no param_version"), "{err}");
    }

    #[test]
    fn streaming_matches_full_sequence() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let ssm = LatentSSM::new(vb, 8, 4, 4, 2).unwrap();
        // give out_proj real weights so the test is non-trivial
        {
            let data = vm.data().lock().unwrap();
            let w = data.iter().find(|(n, _)| n.contains("out_proj")).unwrap().1.clone();
            let rnd = Tensor::randn(0f32, 0.1f32, w.shape().dims(), &dev).unwrap();
            w.set(&rnd).unwrap();
        }

        let z = Tensor::randn(0f32, 1f32, (2, 6, 8), &dev).unwrap();
        let (full, _) = ssm.forward(&z, None).unwrap();

        // streaming: one step at a time, threading state
        let mut state: Option<SsmState> = None;
        let mut outs = Vec::new();
        for i in 0..6 {
            let zi = z.narrow(1, i, 1).unwrap();
            let (oi, s) = ssm.forward(&zi, state).unwrap();
            outs.push(oi);
            state = Some(s);
        }
        let streamed = Tensor::cat(&outs, 1).unwrap();
        let err = full.sub(&streamed).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(err < 1e-4, "streaming should equal full-sequence scan, off by {err}");
    }

}
