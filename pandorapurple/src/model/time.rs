//! Inferno — Time embeddings and sequence-position utilities.

use candle_core::{Result, Tensor, D};
use candle_nn::init::Init;
use candle_nn::ops::silu;
use candle_nn::{linear, Linear, Module, VarBuilder};
use crate::model::norm::{layer_norm, LayerNorm};

use crate::model::utils::softplus;

// ---------------------------------------------------------------------------
// SinusoidalPositionEmbeddings
// ---------------------------------------------------------------------------

/// `t [B] → [B, dim]` interleaved `sin ‖ cos` embeddings (DDPM-style).
pub struct SinusoidalPositionEmbeddings {
    pub dim: usize,
}

impl SinusoidalPositionEmbeddings {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward(&self, time: &Tensor) -> Result<Tensor> {
        let half = self.dim / 2;
        let dev = time.device();
        let log_base = (10000f64).ln() / (half.max(2) - 1) as f64;
        let freqs: Vec<f32> = (0..half).map(|i| (-(i as f64) * log_base).exp() as f32).collect();
        let freqs = Tensor::from_vec(freqs, half, dev)?.to_dtype(time.dtype())?;
        let emb = time.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?; // [B, half]
        Tensor::cat(&[&emb.sin()?, &emb.cos()?], D::Minus1)
    }
}

// ---------------------------------------------------------------------------
// ZTimeEmbedding
// ---------------------------------------------------------------------------

/// Fuses a scalar time into the latent additively:
/// `z + MLP(sinusoidal(t))`, MLP = `Linear(d, 2d) → SiLU → Linear(2d, d)`.
pub struct ZTimeEmbedding {
    pub latent_dim: usize,
    sinu: SinusoidalPositionEmbeddings,
    net0: Linear, // time_net.0
    net2: Linear, // time_net.2
}

impl ZTimeEmbedding {
    pub fn new(vb: VarBuilder, latent_dim: usize) -> Result<Self> {
        Ok(Self {
            latent_dim,
            sinu: SinusoidalPositionEmbeddings::new(latent_dim),
            net0: crate::model::parametrize::sobol_linear(&vb.pp("time_net").pp("0"), latent_dim, latent_dim * 2)?,
            net2: crate::model::parametrize::sobol_linear(&vb.pp("time_net").pp("2"), latent_dim * 2, latent_dim)?,
        })
    }

    /// `latent_z: [B, ..., latent_dim]`, `time: [B]` → same shape as `latent_z`.
    pub fn forward(&self, latent_z: &Tensor, time: &Tensor) -> Result<Tensor> {
        let mut t_emb = self.net2.forward(&silu(&self.net0.forward(&self.sinu.forward(time)?)?)?)?;
        while t_emb.rank() < latent_z.rank() {
            t_emb = t_emb.unsqueeze(1)?;
        }
        latent_z.broadcast_add(&t_emb)
    }
}

// ---------------------------------------------------------------------------
// FourierTimeEmbedding — the global master time embedding
// ---------------------------------------------------------------------------

// Shallow Clone: parameter tensors share storage, so a clone is the SAME
// master clock (the stealth-ref pattern — e.g. the likelihood network's
// time conditioning), never a second registration.
#[derive(Clone)]
pub struct FourierTimeEmbedding {
    pub embed_dim: usize,
    pub n_features: usize,
    omega_t: Tensor, // [n_features] buffer
    mlp0: Linear,    // mlp.0
    mlp2: Linear,    // mlp.2
}

impl FourierTimeEmbedding {
    pub fn new(vb: VarBuilder, embed_dim: usize, n_features: usize, sigma: f64) -> Result<Self> {
        // omega_t is a BUFFER: load from the checkpoint when present so the
        // random projection matches training; fresh draw otherwise.
        let omega_t = if vb.contains_tensor("omega_t") {
            vb.get(n_features, "omega_t")?.detach()
        } else {
            Tensor::randn(0f32, sigma as f32, n_features, vb.device())?.to_dtype(vb.dtype())?
        };
        Ok(Self {
            embed_dim,
            n_features,
            omega_t,
            mlp0: crate::model::parametrize::sobol_linear(&vb.pp("mlp").pp("0"), 2 * n_features, embed_dim)?,
            mlp2: crate::model::parametrize::sobol_linear(&vb.pp("mlp").pp("2"), embed_dim, embed_dim)?,
        })
    }

    /// `t: [B, 1]` → `[B, embed_dim]`.
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let proj = t
            .broadcast_mul(&self.omega_t.unsqueeze(0)?)?
            .affine(2.0 * std::f64::consts::PI, 0.0)?; // [B, n_features]
        let feats = Tensor::cat(&[&proj.sin()?, &proj.cos()?], D::Minus1)?;
        self.mlp2.forward(&silu(&self.mlp0.forward(&feats)?)?)
    }
}

// ---------------------------------------------------------------------------
// LatentTimeRes — dense diffusion TimeResBlock
// ---------------------------------------------------------------------------

pub struct LatentTimeRes {
    linear1: Linear,
    linear2: Linear,
    time_proj: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl LatentTimeRes {
    pub fn new(vb: VarBuilder, hidden_dim: usize, time_emb_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: crate::model::parametrize::sobol_linear(&vb.pp("linear1"), hidden_dim, hidden_dim)?,
            linear2: crate::model::parametrize::sobol_linear(&vb.pp("linear2"), hidden_dim, hidden_dim)?,
            time_proj: crate::model::parametrize::sobol_linear(&vb.pp("time_proj"), time_emb_dim, hidden_dim)?,
            norm1: layer_norm(hidden_dim, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(hidden_dim, 1e-5, vb.pp("norm2"))?,
        })
    }

    /// `x: [B, hidden]`, `t_emb: [B, time_emb_dim]` → `[B, hidden]`.
    pub fn forward(&self, x: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        let h = silu(&self.norm1.forward(&self.linear1.forward(x)?)?)?;
        let time_context = self.time_proj.forward(&silu(t_emb)?)?;
        let h = h.add(&time_context)?;
        let h = silu(&self.norm2.forward(&self.linear2.forward(&h)?)?)?;
        h.add(x)
    }
}

// ---------------------------------------------------------------------------
// SpatioTemporalScale — the GP wrapper's learnable time scale
// ---------------------------------------------------------------------------

/// The spatiotemporal GP wrapper's single parameter: `raw_time_scale` with
/// `time_scale = softplus(raw_time_scale)`. `inject_time` appends
/// `τ = t / time_scale` as the trailing input column so
/// `k_spt((x,τ),(x',τ')) = k_x·k_t` emerges from input augmentation alone.
pub struct SpatioTemporalScale {
    raw_time_scale: Tensor, // scalar
}

impl SpatioTemporalScale {
    pub fn new(vb: VarBuilder, init_time_scale: f64) -> Result<Self> {
        // softplus(raw) = init ⇒ raw = log(exp(init) − 1); for large init this
        // is numerically just `init`.
        let raw_init = if init_time_scale > 30.0 {
            init_time_scale
        } else {
            (init_time_scale.exp() - 1.0).ln()
        };
        Ok(Self { raw_time_scale: vb.get_with_hints((), "raw_time_scale", Init::Const(raw_init))? })
    }

    pub fn time_scale(&self) -> Result<Tensor> {
        softplus(&self.raw_time_scale.unsqueeze(0)?)
            .map_err(|e| candle_core::Error::Msg(format!("time_scale softplus: {e}")))?
            .squeeze(0)
    }

    /// Append `τ = t/softplus(raw_time_scale)` as the trailing column of `x`.
    /// `x: [B, D]` (or `[B, H, D]`), scalar `t` → `[.., D+1]`.
    pub fn inject_time(&self, x: &Tensor, t: f64) -> Result<Tensor> {
        let ts = self.time_scale()?.to_dtype(x.dtype())?.to_scalar::<f32>()? as f64;
        let mut shape = x.dims().to_vec();
        let last = shape.len() - 1;
        shape[last] = 1;
        let tau = Tensor::full((t / ts) as f32, shape, x.device())?.to_dtype(x.dtype())?;
        Tensor::cat(&[x, &tau], D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    //! Verification of time embeddings and sequence-position utilities.

    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn time_modules_shapes() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

        let t = Tensor::from_vec(vec![0.1f32, 0.5, 0.9], 3, &dev).unwrap();

        let sinu = SinusoidalPositionEmbeddings::new(16);
        assert_eq!(sinu.forward(&t).unwrap().dims(), &[3, 16]);

        let zt = ZTimeEmbedding::new(vb.pp("zt"), 16).unwrap();
        let z = Tensor::randn(0f32, 1f32, (3, 16), &dev).unwrap();
        assert_eq!(zt.forward(&z, &t).unwrap().dims(), &[3, 16]);

        let ft = FourierTimeEmbedding::new(vb.pp("ft"), 256, 64, 10.0).unwrap();
        let tb = t.unsqueeze(1).unwrap();
        assert_eq!(ft.forward(&tb).unwrap().dims(), &[3, 256]);

        let res = LatentTimeRes::new(vb.pp("res"), 32, 64).unwrap();
        let x = Tensor::randn(0f32, 1f32, (3, 32), &dev).unwrap();
        let temb = Tensor::randn(0f32, 1f32, (3, 64), &dev).unwrap();
        assert_eq!(res.forward(&x, &temb).unwrap().dims(), &[3, 32]);

        let st = SpatioTemporalScale::new(vb.pp("gp"), 64.0).unwrap();
        let xin = Tensor::randn(0f32, 1f32, (3, 5), &dev).unwrap();
        assert_eq!(st.inject_time(&xin, 2.0).unwrap().dims(), &[3, 6]);

        // omega_t must NOT be a trainable var (it's a buffer).
        let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
        assert!(!names.iter().any(|n| n.contains("omega_t")), "{names:?}");
    }
}
