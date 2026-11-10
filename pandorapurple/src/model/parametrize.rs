//! Inferno — Weight and spectral normalisation for parameterised neural layers.

use std::sync::RwLock;

use candle_core::{Result, Tensor, D};
use candle_nn::init::Init;
use candle_nn::VarBuilder;

fn l2_normalize(t: &Tensor) -> Result<Tensor> {
    let n = t.sqr()?.sum_all()?.sqrt()?.clamp(1e-12, f64::INFINITY)?;
    t.broadcast_div(&n)
}

/// `spectral_norm(nn.Linear)` — power-iteration spectral normalisation with
/// the pytorch parameter layout.
pub struct SpectralNormLinear {
    /// Raw weight `[out, in]` (`weight_orig`). `None` on the folded path.
    weight_orig: Option<Tensor>,
    /// Folded weight `[out, in]` (`weight`) — the cleaned-checkpoint path.
    folded: Option<Tensor>,
    bias: Option<Tensor>,
    /// Power-iteration state (pytorch's `weight_u` / `weight_v` buffers) —
    /// advanced in-place (detached) by one iteration per forward WHEN
    /// [`advance`](Self::set_power_iteration) is set.
    u: RwLock<Tensor>,
    v: RwLock<Tensor>,
    /// Whether a forward advances the power iteration.
    ///
    /// # Why this is a switch and not always-on (2026-08-22)
    ///
    /// PyTorch's `spectral_norm` advances `u`/`v` only in TRAINING mode; in
    /// `eval()` it reuses the buffers, so the layer is a pure function of its
    /// input. This port advanced unconditionally, which made every module
    /// downstream of one a function of CALL HISTORY: the σ estimate converges
    /// over successive forwards, so repeated evaluation of the same input
    /// drifts and then settles.
    ///
    /// Measured through `AmortisedDirichlet::bottleneck_linear` — a
    /// `SpectralNormLinear` — with everything else already pinned:
    /// `gated_weights` moved by `9.2e-6` between two identical calls while
    /// `bw_learned`, `frequencies`, `ls_pred` and `pi` were bit-identical, and
    /// the resulting `features` delta decayed monotonically over ~15 calls
    /// rather than differing at random. Single-threaded, so not a reduction-order
    /// artefact.
    ///
    /// Default `true` preserves training semantics exactly. A serving or
    /// parity path calls `set_power_iteration(false)` and becomes reproducible.
    advance: std::sync::atomic::AtomicBool,
}

/// A deterministic starting vector for the power iteration.
///
/// `Tensor::randn` here would read candle's PROCESS-GLOBAL generator, and
/// `u`/`v` are BUFFERS rather than `Var`s — they are not in the `VarMap`, so
/// no amount of reseeding parameters afterwards can reach them. Two processes
/// constructing the same model would then compute different `σ` on the first
/// forward, and every layer behind a spectral norm would differ. That is
/// exactly what a stage-wise parity fixture cannot tolerate, and it is
/// invisible until something compares two processes.
///
/// The start vector has no statistical meaning: the iteration converges to the
/// dominant singular vector of each layer's own `W`, so all that is required is
/// that it not be orthogonal to it — which a fixed pseudo-random vector is not,
/// with probability 1. Keying on both dims means differently-shaped layers
/// start from different vectors, which costs nothing and makes the intent
/// legible.
fn power_start(
    n: usize,
    other: usize,
    role: u64,
    dev: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64
        ^ (n as u64).wrapping_mul(0x1000_0000_1b3)
        ^ (other as u64).rotate_left(17)
        ^ role.rotate_left(41);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push((((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5) as f32);
    }
    Tensor::from_vec(out, n, dev)?.to_dtype(dtype)
}

/// Initial detached power-iteration buffers for an owning checkpoint store.
/// This matches the deterministic buffers used by `SpectralNormLinear::new`.
pub fn initial_power_buffers(out_dim: usize, in_dim: usize, dev: &candle_core::Device,
    dtype: candle_core::DType) -> Result<(Tensor, Tensor)> {
    Ok((l2_normalize(&power_start(out_dim, in_dim, 0, dev, dtype)?)?,
        l2_normalize(&power_start(in_dim, out_dim, 1, dev, dtype)?)?))
}

impl SpectralNormLinear {
    /// `vb` is rooted AT the layer (e.g. `…fusion_net.0`). `init` seeds
    /// `weight_orig` on a fresh map.
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, init: Init) -> Result<Self> {
        let (weight_orig, folded) = if vb.contains_tensor("weight") {
            // Folded / cleaned checkpoint: use the plain weight verbatim.
            (None, Some(vb.get((out_dim, in_dim), "weight")?))
        } else {
            (
                Some(sobol_from_init(&vb, "weight_orig", (out_dim, in_dim), init)?),
                None,
            )
        };
        let bias = Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?);

        let dev = vb.device().clone();
        let dtype = vb.dtype();
        // Buffers: loaded when the (uncleaned) checkpoint carries them,
        // fresh normalized draws otherwise. Never registered as Vars.
        let u = if vb.contains_tensor("weight_u") {
            vb.get(out_dim, "weight_u")?
        } else {
            l2_normalize(&power_start(out_dim, in_dim, 0, &dev, dtype)?)?
        };
        let v = if vb.contains_tensor("weight_v") {
            vb.get(in_dim, "weight_v")?
        } else {
            l2_normalize(&power_start(in_dim, out_dim, 1, &dev, dtype)?)?
        };
        Ok(Self {
            advance: std::sync::atomic::AtomicBool::new(true),
            weight_orig,
            folded,
            bias,
            u: RwLock::new(u),
            v: RwLock::new(v),
        })
    }

    /// The effective weight `[out, in]`: folded verbatim, or
    /// `weight_orig / σ` with σ from ONE power iteration (pytorch default
    /// `n_power_iterations=1`). The iteration runs on the DETACHED weight and
    /// updates the `u`/`v` state; σ itself is grad-carrying through
    /// `weight_orig` (`σ = uᵀ W v` with `u`,`v` detached) — the exact pytorch
    /// gradient structure.
    pub fn effective_weight(&self) -> Result<Tensor> {
        if let Some(w) = &self.folded {
            return Ok(w.clone());
        }
        let w = self.weight_orig.as_ref().expect("raw or folded");
        let wd = w.detach();
        let (u_new, v_new) = {
            let u = self.u.read().unwrap().detach();
            // v ← normalize(Wᵀ u); u ← normalize(W v)
            let v = l2_normalize(&wd.t()?.broadcast_matmul(&u.unsqueeze(1)?)?.squeeze(1)?)?;
            let u2 = l2_normalize(&wd.broadcast_matmul(&v.unsqueeze(1)?)?.squeeze(1)?)?;
            (u2, v)
        };
        if self.advance.load(std::sync::atomic::Ordering::Relaxed) {
            let mut u = self.u.write().unwrap();
            let mut v = self.v.write().unwrap();
            // Owning SDK stores may register buffers as shared Var handles so
            // checkpoints observe their updates. Keep them detached from the
            // loss graph and preserve plain tensor-backed legacy behavior.
            if u.is_variable() { candle_core::Var::from_tensor(&u)?.set(&u_new)?; }
            else { *u = u_new.clone(); }
            if v.is_variable() { candle_core::Var::from_tensor(&v)?.set(&v_new)?; }
            else { *v = v_new.clone(); }
        }
        // σ = uᵀ W v — grad flows through W only.
        let sigma = u_new
            .unsqueeze(0)?
            .matmul(&w.matmul(&v_new.unsqueeze(1)?)?)?
            .reshape(())?
            .clamp(1e-12, f64::INFINITY)?;
        w.broadcast_div(&sigma)
    }

    /// Advance the power iteration on every forward (`true`, the training
    /// default) or hold the buffers fixed (`false`, PyTorch's `eval()`).
    pub fn set_power_iteration(&self, on: bool) {
        self.advance.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn power_iteration(&self) -> bool {
        self.advance.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.effective_weight()?.to_dtype(x.dtype())?;
        let y = x.broadcast_matmul(&w.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(x.dtype())?),
            None => Ok(y),
        }
    }
}

/// `parametrizations.weight_norm(nn.Linear)` — `W = g · v / ‖v‖_row`.
pub struct WeightNormLinear {
    /// `g` `[out, 1]` (`parametrizations.weight.original0`).
    g: Option<Tensor>,
    /// `v` `[out, in]` (`parametrizations.weight.original1`).
    v: Option<Tensor>,
    /// Folded plain weight (cleaned checkpoint).
    folded: Option<Tensor>,
    bias: Option<Tensor>,
}

impl WeightNormLinear {
    /// `g_init` seeds the per-row gain on a fresh map (Python applies
    /// weight_norm AFTER an orthogonal init with some gain `γ`, making
    /// `g = ‖w_row‖ ≈ γ`).
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        v_init: Init,
        g_init: f64,
    ) -> Result<Self> {
        let (g, v, folded) = if vb.contains_tensor("weight") {
            (None, None, Some(vb.get((out_dim, in_dim), "weight")?))
        } else {
            let p = vb.pp("parametrizations").pp("weight");
            (
                Some(p.get_with_hints((out_dim, 1), "original0", Init::Const(g_init))?),
                Some(sobol_from_init(&p, "original1", (out_dim, in_dim), v_init)?),
                None,
            )
        };
        let bias = Some(vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?);
        Ok(Self { g, v, folded, bias })
    }

    pub fn effective_weight(&self) -> Result<Tensor> {
        if let Some(w) = &self.folded {
            return Ok(w.clone());
        }
        let g = self.g.as_ref().expect("raw or folded");
        let v = self.v.as_ref().expect("raw or folded");
        let row_norm = v
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?; // [out, 1]
        v.broadcast_div(&row_norm)?.broadcast_mul(g)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.effective_weight()?.to_dtype(x.dtype())?;
        let y = x.broadcast_matmul(&w.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(x.dtype())?),
            None => Ok(y),
        }
    }
}

/// Zero-initialised `nn.Linear` — the FiLM-projector pattern (`W = 0`,
/// `b = 0` ⇒ exact identity conditioning at init).
pub fn zero_linear(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<candle_nn::Linear> {
    let w = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.0))?;
    let b = vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?;
    Ok(candle_nn::Linear::new(w, Some(b)))
}

/// The QMC seed for one parameter: FNV-1a over its FULL name.
///
/// Derived rather than passed so it cannot be forgotten, duplicated, or
/// silently reused when a call site is copied — and so it is stable under
/// reordering, since it depends on the parameter's identity and nothing else.
/// Two parameters with the same name under different prefixes get different
/// sequences, which is what stops a whole tree of same-shaped heads sharing
/// one draw.
fn param_seed(vb: &VarBuilder, name: &str, shape: &candle_core::Shape) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    for b in vb.prefix().as_bytes().iter().chain(b".").chain(name.as_bytes()) {
        eat(*b);
    }
    // The SHAPE too, so two same-named parameters in sibling sub-trees cannot
    // share a sequence when they are built from bare (prefix-less) builders —
    // which is exactly what a unit test does, and what made the kernel's local
    // atom support come out identical to the shared one.
    for d in shape.dims() {
        for k in 0..8 {
            eat((d >> (k * 8)) as u8);
        }
    }
    h
}

/// A parameter whose init is drawn from the SEEDED Sobol/QMC engine rather
/// than candle's device RNG.
///
/// # Why not `Init::Randn`
///
/// `Init::Randn` reaches the device RNG, and the vendored CPU backend REFUSES
/// `Device::set_seed` — the warning every daemon start prints. So a
/// distributional init is not reproducible: two runs of "the same" fresh model
/// differ, and an init-sensitive result cannot be attributed to anything. The
/// QMC engine is deterministic in its seed by construction, and is already what
/// every other reproducible draw in this crate uses.
///
/// # Mechanics
///
/// candle's `Init` is a closed enum, so the draw cannot be injected through it.
/// The parameter is therefore created at a constant and the draw written into
/// its storage with `slice_set`, which is in place — the returned tensor shares
/// that storage with the registered `Var`, so both see it.
///
/// A CHECKPOINT ALWAYS WINS: when the store carries `name`, no draw is made.
///
/// Rank-0 parameters fall back to `mean` and are left alone: a scalar "random"
/// init is a single number with no symmetry to break, and `slice_set` has no
/// axis to write along.
pub fn sobol_randn<S: Into<candle_core::Shape>>(
    vb: &VarBuilder,
    name: &str,
    shape: S,
    mean: f64,
    stdev: f64,
) -> Result<Tensor> {
    let shape: candle_core::Shape = shape.into();
    let had = vb.contains_tensor(name);
    let t = vb.get_with_hints(shape.clone(), name, Init::Const(mean))?;
    if had || shape.rank() == 0 {
        return Ok(t);
    }
    let draw = crate::model::noise::sobol_randn_like_per_dim(shape.dims(), vb.device(), param_seed(vb, name, &shape))?
        .affine(stdev, mean)?
        .to_dtype(vb.dtype())?;
    t.slice_set(&draw, 0, 0)?;
    Ok(t)
}

/// [`sobol_randn`]'s bounded twin: a uniform `[lo, up)` QMC draw, for
/// parameters whose support is bounded and which must therefore not be given
/// Gaussian tails.
pub fn sobol_uniform<S: Into<candle_core::Shape>>(
    vb: &VarBuilder,
    name: &str,
    shape: S,
    lo: f64,
    up: f64,
) -> Result<Tensor> {
    let shape: candle_core::Shape = shape.into();
    let had = vb.contains_tensor(name);
    let mid = 0.5 * (lo + up);
    let t = vb.get_with_hints(shape.clone(), name, Init::Const(mid))?;
    if had || shape.rank() == 0 {
        return Ok(t);
    }
    let draw = crate::model::noise::sobol_uniform_like_per_dim(shape.dims(), vb.device(), param_seed(vb, name, &shape))?
        .affine(up - lo, lo)?
        .to_dtype(vb.dtype())?;
    t.slice_set(&draw, 0, 0)?;
    Ok(t)
}

/// Route a caller-supplied [`Init`] through the seeded engine.
///
/// The general primitive behind [`sobol_randn`] / [`sobol_uniform`], for the
/// constructors that take an `Init` from ABOVE — `SpectralNormLinear`,
/// `WeightNormLinear` and friends. Fixing it here means every caller is fixed
/// at once, and none of them has to know the draw changed.
///
/// Deterministic inits (`Const`, `Kaiming`) pass straight through: there is
/// nothing to seed.
pub fn sobol_from_init<S: Into<candle_core::Shape>>(
    vb: &VarBuilder,
    name: &str,
    shape: S,
    init: Init,
) -> Result<Tensor> {
    let shape: candle_core::Shape = shape.into();
    match init {
        Init::Randn { mean, stdev } => sobol_randn(vb, name, shape, mean, stdev),
        Init::Uniform { lo, up } => sobol_uniform(vb, name, shape, lo, up),
        other => vb.get_with_hints(shape, name, other),
    }
}

/// `candle_nn::linear`, seeded.
///
/// Same distributions candle uses — Kaiming-normal weights at
/// `√2/√fan_in` (the ReLU gain) and a `±1/√fan_in` uniform bias — drawn from
/// the QMC engine instead of the device RNG. Swapping the source, not the
/// scale, so no layer's init statistics move.
pub fn sobol_linear(vb: &VarBuilder, in_dim: usize, out_dim: usize) -> Result<candle_nn::Linear> {
    let ws = sobol_randn(vb, "weight", (out_dim, in_dim), 0.0, (2.0 / in_dim as f64).sqrt())?;
    let b = 1.0 / (in_dim as f64).sqrt();
    let bs = sobol_uniform(vb, "bias", out_dim, -b, b)?;
    Ok(candle_nn::Linear::new(ws, Some(bs)))
}

/// [`sobol_linear`] without the bias term.
pub fn sobol_linear_no_bias(
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
) -> Result<candle_nn::Linear> {
    let ws = sobol_randn(vb, "weight", (out_dim, in_dim), 0.0, (2.0 / in_dim as f64).sqrt())?;
    Ok(candle_nn::Linear::new(ws, None))
}

/// `candle_nn::conv1d`, seeded. Kaiming-normal weights, `±1/√in_channels`
/// uniform bias — candle's own scales, drawn from the QMC engine.
pub fn sobol_conv1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv1dConfig,
) -> Result<candle_nn::Conv1d> {
    let shape = (out_channels, in_channels / cfg.groups, kernel_size);
    let ws = sobol_randn(vb, "weight", shape, 0.0, (2.0 / in_channels as f64).sqrt())?;
    let b = 1.0 / (in_channels as f64).sqrt();
    let bs = sobol_uniform(vb, "bias", out_channels, -b, b)?;
    Ok(candle_nn::Conv1d::new(ws, Some(bs), cfg))
}

/// [`sobol_conv1d`] without the bias term.
pub fn sobol_conv1d_no_bias(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv1dConfig,
) -> Result<candle_nn::Conv1d> {
    let shape = (out_channels, in_channels / cfg.groups, kernel_size);
    let ws = sobol_randn(vb, "weight", shape, 0.0, (2.0 / in_channels as f64).sqrt())?;
    Ok(candle_nn::Conv1d::new(ws, None, cfg))
}

/// `candle_nn::conv_transpose1d`, seeded. Candle bounds BOTH weight and bias by
/// `1/√(out_channels·kernel)` here, unlike the forward conv — kept exactly.
pub fn sobol_conv_transpose1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::ConvTranspose1dConfig,
) -> Result<candle_nn::ConvTranspose1d> {
    let b = 1.0 / ((out_channels * kernel_size) as f64).sqrt();
    let shape = (in_channels, out_channels / cfg.groups, kernel_size);
    let ws = sobol_uniform(vb, "weight", shape, -b, b)?;
    let bs = sobol_uniform(vb, "bias", out_channels, -b, b)?;
    Ok(candle_nn::ConvTranspose1d::new(ws, Some(bs), cfg))
}

/// `candle_nn::embedding`, seeded. Unit-normal rows, as candle draws them.
pub fn sobol_embedding(
    vb: &VarBuilder,
    in_size: usize,
    out_size: usize,
) -> Result<candle_nn::Embedding> {
    let ws = sobol_randn(vb, "weight", (in_size, out_size), 0.0, 1.0)?;
    Ok(candle_nn::Embedding::new(ws, out_size))
}

/// Optional-module builder for the "absent on pre-migration checkpoints"
/// pattern: STRICT when the store carries the probe tensor (a shape mismatch
/// must fail loudly), permissive otherwise — a fresh `VarMap` creates (build
/// always succeeds and registers), a tensor-backed store without the module
/// yields `None`.
pub fn build_or_absent<T, E>(
    vb: &VarBuilder,
    probe: &str,
    build: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<Option<T>, E> {
    if vb.contains_tensor(probe) {
        return Ok(Some(build()?));
    }
    match build() {
        Ok(t) => Ok(Some(t)),
        Err(_) => Ok(None), // tensor-backed store without the module
    }
}

#[cfg(test)]
mod tests {
    //! Verification of weight and spectral normalisation for parameterised neural layers.

    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn spectral_norm_registers_weight_orig_and_normalizes() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let l = SpectralNormLinear::new(
            vb.pp("sn"),
            6,
            4,
            Init::Randn { mean: 0.0, stdev: 1.0 },
        )
        .unwrap();
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(names.contains(&"sn.weight_orig".to_string()));
        assert!(names.contains(&"sn.bias".to_string()));
        assert!(!names.iter().any(|n| n.contains("weight_u")), "buffers must not be Vars");

        // After a few iterations the largest singular value of the effective
        // weight approaches 1 (power iteration convergence).
        let mut w = l.effective_weight().unwrap();
        for _ in 0..20 {
            w = l.effective_weight().unwrap();
        }
        // ‖W v‖ where v is the converged right vector ≈ σ_max(W) ≈ 1.
        let v = l.v.read().unwrap().clone();
        let s = w
            .matmul(&v.unsqueeze(1).unwrap())
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((s - 1.0).abs() < 5e-2, "σ_max after normalisation = {s}");
    }

    #[test]
    fn spectral_norm_gradient_reaches_weight_orig() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let l = SpectralNormLinear::new(vb.pp("g"), 3, 2, Init::Randn { mean: 0.0, stdev: 1.0 })
            .unwrap();
        let x = Tensor::randn(0f32, 1f32, (4, 3), &dev).unwrap();
        let y = l.forward(&x).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = y.backward().unwrap();
        let wo = vm.data().lock().unwrap().get("g.weight_orig").unwrap().clone();
        let g = grads.get(wo.as_tensor()).expect("grad on weight_orig");
        let n = g.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(n.is_finite() && n > 0.0, "grad norm {n}");
    }

    #[test]
    fn weight_norm_row_norms_equal_g() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let l = WeightNormLinear::new(
            vb.pp("wn"),
            5,
            3,
            Init::Randn { mean: 0.0, stdev: 0.3 },
            0.8,
        )
        .unwrap();
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(names.contains(&"wn.parametrizations.weight.original0".to_string()));
        assert!(names.contains(&"wn.parametrizations.weight.original1".to_string()));

        let w = l.effective_weight().unwrap();
        let rows: Vec<f32> = w
            .sqr()
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .sqrt()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(rows.iter().all(|r| (r - 0.8).abs() < 1e-5), "{rows:?}");
    }

    #[test]
    fn folded_checkpoint_path_uses_plain_weight_verbatim() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (2, 3), &dev).unwrap();
        let b = Tensor::zeros(2, DType::F32, &dev).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("weight".to_string(), w.clone());
        map.insert("bias".to_string(), b);
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let l = SpectralNormLinear::new(vb, 3, 2, Init::Const(0.0)).unwrap();
        let diff = l
            .effective_weight()
            .unwrap()
            .sub(&w)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "folded weight must pass through untouched");
    }
    /// **The gate for the 2026-08-22 determinism fix.** An unfrozen
    /// `SpectralNormLinear` advances its power iteration once per forward, so
    /// the SAME input gives a DIFFERENT output that converges over successive
    /// calls; frozen, it is a pure function.
    ///
    /// Both halves are asserted. Only checking the frozen case would pass on a
    /// layer that had stopped iterating altogether, which is a different bug
    /// with the same symptom here and a worse one in training.
    #[test]
    fn freezing_the_power_iteration_makes_the_layer_a_pure_function() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let l = SpectralNormLinear::new(vb.pp("l"), 8, 5, Init::Randn { mean: 0.0, stdev: 0.5 })
            .unwrap();
        let x = Tensor::randn(0f32, 1f32, (4, 8), &dev).unwrap();
        let d = |a: &Tensor, b: &Tensor| -> f32 {
            a.sub(b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };

        // Advancing (the training default): successive forwards DISAGREE, and
        // the disagreement SHRINKS — power iteration converging, not noise.
        assert!(l.power_iteration(), "advancing must be the default");
        let a = l.forward(&x).unwrap();
        let b = l.forward(&x).unwrap();
        let d1 = d(&a, &b);
        assert!(d1 > 0.0, "an advancing layer must not already be converged");
        // Convergence is asserted over a HORIZON, not step to step. Power
        // iteration from a random start is not monotone — the first couple of
        // steps can widen before the dominant singular direction takes over —
        // so a consecutive-pair comparison is flaky by construction. What is
        // true, and what distinguishes this from nondeterminism, is that the
        // deltas go to zero.
        let mut prev = b;
        let mut last = d1;
        for _ in 0..24 {
            let n = l.forward(&x).unwrap();
            last = d(&prev, &n);
            prev = n;
        }
        assert!(
            last < d1 * 0.25,
            "the iterate must CONVERGE ({last:.3e} !< {:.3e}) — that is what identifies this \
             as a converging sequence rather than nondeterminism",
            d1 * 0.25
        );

        // Frozen: bit-identical, forever.
        l.set_power_iteration(false);
        assert!(!l.power_iteration());
        let f0 = l.forward(&x).unwrap();
        for _ in 0..5 {
            let f = l.forward(&x).unwrap();
            assert_eq!(d(&f0, &f), 0.0, "a frozen layer must be bit-identical");
        }
    }

}
