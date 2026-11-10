//! Inferno — Load caller-supplied model tensors from safetensors checkpoints.

// ---------------------------------------------------------------------------
// Weight loading helper
// ---------------------------------------------------------------------------

/// Convenience: load `TimeEmbedding` + your submodules from a single
/// safetensors file. Submodule loads are delegated to user-supplied
/// closures so this file stays decoupled from their implementation.
///
/// Example usage:
/// ```text
/// # use candle_core::Device;
/// # use crate::spectral_vae::*;
/// let device = Device::cuda_if_available(0)?;
/// let config = SpectralVaeConfig::default();
/// let vb = load_varbuilder("artefacts/vae.safetensors", &device)?;
//
/// let encoder  = MyEncoder::load(vb.pp("encoder"), &config)?;
/// let dirich   = MyDirichlet::load(vb.pp("dirichlet"), &config)?;
/// let decoder  = MyDecoder::load(vb.pp("decoder"), &config)?;
/// let time_emb = TimeEmbedding::load(vb.clone(), config.latent_dim)?;
///
/// let vae = SpectralVae::new(encoder, dirich, decoder, time_emb, config, device);
/// # Ok::<_, anyhow::Error>(())
/// ```
use candle_core::{Device, DType, Result};
use candle_nn::VarBuilder;

pub fn load_varbuilder<'a>(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, device)
    };
    Ok(vb?)
}

/// Checkpoint-tolerant loader: accepts BOTH the cleaned export
/// (`deepkernels-weights export --root vae`) AND the raw training
/// checkpoint, by normalising at load time:
///
///   * a `vae.` key prefix is stripped (the module tree must sit at the
///     file root for `vb.pp("dirichlet")…` to walk);
///   * weight-norm parametrizations are FOLDED, not renamed —
///     `{m}.parametrizations.weight.original0` is the magnitude `g` and
///     `original1` the direction `v`, so `{m}.weight = g · v/‖v‖`
///     (row-wise norm, torch's `dim=0`). Neither original IS the weight;
///     a rename would load garbage.
///   * spectral-norm triples `{m}.weight_orig/_u/_v` fold to
///     `{m}.weight = W/σ`, `σ = uᵀWv`.
///
/// Prefer the export tool for serving artefacts (smaller file, one-time
/// cost); this path is the safety net for the raw-checkpoint mistake.
pub fn load_varbuilder_cleaned<'a>(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<VarBuilder<'a>> {
    Ok(VarBuilder::from_tensors(
        load_cleaned_map(path, device, Some("vae."))?,
        DType::F32,
        device,
    ))
}

/// The fold-all core: load EVERY tensor from the (possibly combined)
/// checkpoint, fold weight-norm/spectral-norm parametrizations, and
/// optionally re-root under `strip_prefix` (keys without the prefix pass
/// through unchanged, so a combined `vae.* + gp.*` file serves both
/// consumers from ONE artefact — no manual weight separation).
pub fn load_cleaned_map(
    path: impl AsRef<std::path::Path>,
    device: &Device,
    strip_prefix: Option<&str>,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| {
        candle_core::Error::Msg(format!("reading {}: {e}", path.as_ref().display()))
    })?;
    load_cleaned_map_from_bytes(&bytes, device, strip_prefix)
}

/// [`load_cleaned_map`] over in-memory safetensors bytes — the blob-store /
/// queue path (dk-inference, Redis tensor payloads), where the artefact
/// arrives as `Vec<u8>` rather than a file. Identical fold/strip behaviour.
/// One safetensors view → a candle tensor, or `None` for a buffer candle
/// cannot hold.
///
/// `None` (rather than an error) for BOOL masks / `_dummy` flags / integer
/// index buffers and zero-element tensors: real PyTorch exports carry those
/// and they are never forward-pass weights. A genuinely-needed key that got
/// skipped fails loudly later at `VarBuilder::get`, not here.
///
/// Shared so the checkpoint reader and the weight loader cannot disagree
/// about which dtypes are admissible.
pub fn tensor_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
    device: &Device,
) -> Result<Option<candle_core::Tensor>> {
    let dtype = match view.dtype() {
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::U32 => DType::U32,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        other => {
            tracing::warn!(
                key = %name, dtype = ?other,
                "skipping buffer with a dtype candle can't hold (not a forward-pass weight)"
            );
            return Ok(None);
        }
    };
    let shape = view.shape();
    if shape.iter().product::<usize>() == 0 {
        tracing::warn!(key = %name, shape = ?shape, "skipping zero-element tensor in checkpoint");
        return Ok(None);
    }
    let t = candle_core::Tensor::from_raw_buffer(view.data(), dtype, shape, device)
        .map_err(|e| candle_core::Error::Msg(format!("decoding `{name}`: {e}")))?;
    Ok(Some(t))
}

pub fn load_cleaned_map_from_bytes(
    bytes: &[u8],
    device: &Device,
    strip_prefix: Option<&str>,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    use std::collections::HashMap;
    // Decode + fold on CPU, move to `device` at the very end. We read the
    // safetensors VIEWS ourselves rather than `candle_core::safetensors::load`
    // because that loader converts every tensor eagerly and dies on the
    // FIRST tensor whose dtype candle can't hold — a training checkpoint's
    // BOOL masks / `_dummy` flags / integer index buffers ("unsupported
    // safetensor dtype BOOL"). Those are never forward-pass weights, so we
    // skip any dtype outside candle's set (and any zero-element buffer — a
    // raw PyTorch export straight onto CUDA otherwise trips
    // `CUDA_ERROR_INVALID_VALUE` on cuMemAlloc(0)). A genuinely-needed key
    // that got skipped fails loudly later at `VarBuilder::get`, not here.
    let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| {
        candle_core::Error::Msg(format!("parsing safetensors bytes: {e}"))
    })?;
    let mut map: HashMap<String, candle_core::Tensor> = HashMap::new();
    for (name, view) in st.tensors() {
        let Some(t) = tensor_from_view(&view, &name, device)? else {
            continue;
        };
        let key = match strip_prefix {
            Some(p) => name.strip_prefix(p).unwrap_or(&name).to_string(),
            None => name,
        };
        map.insert(key, t);
    }

    // ---- weight norm: w = g · v / ‖v‖ (per output row) --------------------
    let g_keys: Vec<String> = map
        .keys()
        .filter(|k| k.ends_with(".parametrizations.weight.original0"))
        .cloned()
        .collect();
    for gk in g_keys {
        let stem = gk.trim_end_matches(".original0");
        let vk = format!("{stem}.original1");
        let module = stem.trim_end_matches(".parametrizations.weight");
        let plain = format!("{module}.weight");
        if map.contains_key(&plain) || !map.contains_key(&vk) {
            continue;
        }
        let g = map[&gk].to_dtype(DType::F32)?;
        let v = map[&vk].to_dtype(DType::F32)?;
        let rows = v.dim(0)?;
        let norm = v
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?; // [rows, 1]
        let w = v
            .broadcast_div(&norm)?
            .broadcast_mul(&g.reshape((rows, 1))?)?;
        map.remove(&gk);
        map.remove(&vk);
        map.insert(plain, w);
    }

    // ---- spectral norm: w = W / (uᵀ W v) ----------------------------------
    let o_keys: Vec<String> =
        map.keys().filter(|k| k.ends_with(".weight_orig")).cloned().collect();
    for ok in o_keys {
        let module = ok.trim_end_matches(".weight_orig");
        let (uk, vk) = (format!("{module}.weight_u"), format!("{module}.weight_v"));
        let plain = format!("{module}.weight");
        if map.contains_key(&plain) || !map.contains_key(&uk) || !map.contains_key(&vk) {
            continue;
        }
        let w0 = map[&ok].to_dtype(DType::F32)?;
        let u = map[&uk].to_dtype(DType::F32)?.flatten_all()?;
        let v = map[&vk].to_dtype(DType::F32)?.flatten_all()?;
        let wv = w0.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // [out]
        let sigma = u.mul(&wv)?.sum_all()?.to_scalar::<f32>()? as f64;
        if sigma.abs() < 1e-12 || !sigma.is_finite() {
            continue;
        }
        let w = w0.affine(1.0 / sigma, 0.0)?;
        map.remove(&ok);
        map.remove(&uk);
        map.remove(&vk);
        map.insert(plain, w);
    }

    // ---- single H2D hop per tensor, with the offender named on failure ----
    if !matches!(device, Device::Cpu) {
        let keys: Vec<String> = map.keys().cloned().collect();
        for k in keys {
            let moved = map[&k]
                .to_device(device)
                .map_err(|e| candle_core::Error::Msg(format!("moving `{k}` to {device:?}: {e}")))?;
            map.insert(k, moved);
        }
    }

    Ok(map)
}

/// The GP subtree of the combined checkpoint, re-rooted and ALIASED to the
/// serving stack's names (derived from the live checkpoint manifest):
///
///   * `variational_strategy._external_w_base` → `variational_strategy.W_base`
///     (the Python Strategy registers the standalone W_base as the
///     `_external_w_base` buffer);
///   * `mean_module.{raw_bary_*, raw_bias_sigma, atom_coords}` →
///     `mean_module.shared.*` — BACK-COMPAT ONLY. These parameters live on the
///     shared `StochasticControlState` in both Python and Rust, so the live
///     checkpoint already writes `shared.`-prefixed keys and this alias is a
///     no-op on it; it only rescues checkpoints saved while the Rust mean
///     still owned flat copies. The scalars are also normalised to rank-0,
///     matching the 0-d registration on both sides — a tensor-backed
///     VarBuilder validates shape and errors on a rank mismatch rather than
///     falling back to the declared init;
///   * a missing `W_binary_t` is synthesized as ZEROS like `W_binary` — the
///     time-gated residual head starts inactive (`W_eff = W_binary + g·0`),
///     which is the semantically-safe init, and a tensor-backed VarBuilder
///     errors on absent keys rather than init-filling.
///
/// Constraint/prior aux tensors (`*_constraint.*`, `*_prior.*`) ride along
/// harmlessly — lookups are by name.
pub fn load_gp_map(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    apply_gp_aliases(load_cleaned_map(path, device, Some("gp."))?)
}

/// [`load_gp_map`] over in-memory safetensors bytes (see
/// [`load_cleaned_map_from_bytes`]).
pub fn load_gp_map_from_bytes(
    bytes: &[u8],
    device: &Device,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    apply_gp_aliases(load_cleaned_map_from_bytes(bytes, device, Some("gp."))?)
}

/// The GP-map back-compat aliases shared by the path and bytes loaders.
fn apply_gp_aliases(
    mut map: std::collections::HashMap<String, candle_core::Tensor>,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
    if !map.contains_key("variational_strategy.W_base") {
        if let Some(w) = map.get("variational_strategy._external_w_base").cloned() {
            map.insert("variational_strategy.W_base".into(), w);
        }
    }
    for leaf in ["raw_bary_eps", "raw_bary_tau1", "raw_bary_tau2", "raw_bary_weights", "raw_bias_sigma", "atom_coords"] {
        let shared = format!("mean_module.shared.{leaf}");
        if !map.contains_key(&shared) {
            if let Some(t) = map.get(&format!("mean_module.{leaf}")).cloned() {
                map.insert(shared.clone(), t);
            }
        }
        // The scalars are 0-d on both sides; a `[1]` value is an old Rust save.
        if leaf != "raw_bary_weights" && leaf != "atom_coords" {
            if let Some(t) = map.get(&shared) {
                if t.rank() == 1 && t.dim(0)? == 1 {
                    let t = t.reshape(())?;
                    map.insert(shared, t);
                }
            }
        }
    }
    if !map.contains_key("variational_strategy.W_binary_t") {
        if let Some(wb) = map.get("variational_strategy.W_binary") {
            map.insert("variational_strategy.W_binary_t".into(), wb.zeros_like()?);
        }
    }
    Ok(map)
}
#[cfg(test)]
mod tests {
    //! Verification of load caller-supplied model tensors from safetensors checkpoints.

    use super::*;
    use candle_core::Tensor;

    #[test]
    fn cleaned_loader_folds_weight_norm_and_strips_prefix() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("lvc-{}.safetensors", std::process::id()));
        // raw-checkpoint layout: vae.-prefixed, weight-norm parametrized
        let g = Tensor::new(&[[10.0f32], [5.0]], &dev).unwrap(); // [2, 1]
        let v = Tensor::new(&[[3.0f32, 4.0], [0.0, 5.0]], &dev).unwrap(); // rows ‖·‖ = 5
        let b = Tensor::new(&[0.1f32, 0.2], &dev).unwrap();
        candle_core::safetensors::save(
            &std::collections::HashMap::from([
                (
                    "vae.dirichlet.kernel_network.mat52.parametrizations.weight.original0"
                        .to_string(),
                    g,
                ),
                (
                    "vae.dirichlet.kernel_network.mat52.parametrizations.weight.original1"
                        .to_string(),
                    v,
                ),
                ("vae.dirichlet.kernel_network.mat52.bias".to_string(), b),
            ]),
            &dir,
        )
        .unwrap();

        let vb = load_varbuilder_cleaned(&dir, &dev).unwrap();
        // exactly the lookup that failed in serving:
        let w = vb
            .pp("dirichlet")
            .pp("kernel_network")
            .pp("mat52")
            .get((2, 2), "weight")
            .unwrap();
        let got: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![6.0, 8.0, 0.0, 5.0]); // g·v/‖v‖ = 10/5·[3,4], 5/5·[0,5]
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn cleaned_loader_skips_bool_and_unsupported_dtype_buffers() {
        // candle can't AUTHOR a BOOL tensor, so write the file with the raw
        // safetensors crate: one BOOL mask + one I64 index buffer alongside a
        // real F32 weight. The old `candle_core::safetensors::load` died on
        // the BOOL ("unsupported safetensor dtype BOOL"); the view reader must
        // skip both non-weight buffers and keep the weight.
        let dir =
            std::env::temp_dir().join(format!("lvc-bool-{}.safetensors", std::process::id()));

        let weight: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mask: Vec<u8> = vec![1u8, 0, 1]; // BOOL is 1 byte/elem in safetensors
        let idx: Vec<u8> = [0i64, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            ("weight".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &weight).unwrap()),
            ("attn_mask".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::BOOL, vec![3], &mask).unwrap()),
            ("_dummy_idx".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![2], &idx).unwrap()),
        ];
        std::fs::write(&dir, safetensors::tensor::serialize(views, None).unwrap()).unwrap();

        let map = load_cleaned_map(&dir, &Device::Cpu, None).unwrap();
        assert!(map.contains_key("weight"), "the real F32 weight must survive");
        assert!(!map.contains_key("attn_mask"), "BOOL buffer must be skipped, not error the load");
        // I64 is candle-representable, so the index buffer is KEPT (harmless —
        // unused keys never touched by VarBuilder::get).
        assert!(map.contains_key("_dummy_idx"));
        let got: Vec<f32> = map["weight"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_file(&dir).ok();
    }

    /// The barycentre scalars are rank-0 on the SHARED control state in both
    /// Python and Rust. The loader must leave the live layout untouched, and
    /// still rescue the two legacy shapes: a flat `mean_module.*` name, and a
    /// `[1]` value written by the older Rust mean.
    #[test]
    fn gp_map_normalises_barycentre_scalars_onto_the_shared_state() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("lvc-bary-{}.safetensors", std::process::id()));
        candle_core::safetensors::save(
            &std::collections::HashMap::from([
                // live layout: shared-prefixed, rank 0 — must pass through
                (
                    "gp.mean_module.shared.raw_bary_eps".to_string(),
                    Tensor::new(-2.5f32, &dev).unwrap(),
                ),
                // legacy: flat name, rank 0
                (
                    "gp.mean_module.raw_bary_tau1".to_string(),
                    Tensor::new(0.5f32, &dev).unwrap(),
                ),
                // legacy: flat name, `[1]` from the old Rust save path
                (
                    "gp.mean_module.raw_bary_tau2".to_string(),
                    Tensor::new(&[0.25f32], &dev).unwrap(),
                ),
            ]),
            &dir,
        )
        .unwrap();

        let map = load_gp_map(&dir, &dev).unwrap();
        for leaf in ["raw_bary_eps", "raw_bary_tau1", "raw_bary_tau2"] {
            let t = &map[&format!("mean_module.shared.{leaf}")];
            assert_eq!(t.rank(), 0, "{leaf} must land rank-0 on the shared state");
        }
        assert_eq!(
            map["mean_module.shared.raw_bary_eps"].to_scalar::<f32>().unwrap(),
            -2.5
        );
        assert_eq!(
            map["mean_module.shared.raw_bary_tau2"].to_scalar::<f32>().unwrap(),
            0.25
        );
        std::fs::remove_file(&dir).ok();
    }
}
