//! Inferno — Trainable Kumaraswamy concentration parameters for bounded latent variables.

use candle_core::{Result, Tensor, D};
use crate::model::kumaraswamy_icdf;
#[derive(Debug, Clone, Default)]
pub struct KumaraswamyConcentration {
    pub q_a: Option<Tensor>,             // [K-1] (or [B, K-1])
    pub q_b: Option<Tensor>,             // [K-1]
    pub log_one_minus_v: Option<Tensor>, // stick log(1−v)
    pub v: Option<Tensor>,               // Kumaraswamy samples (local)
    pub u: Option<Tensor>,               // uniform noise used to draw v
    pub log_pi: Option<Tensor>,          // log stick weights (local)
    pub pi_posterior: Option<Tensor>,    // softmax posterior probs (local)
    pub beta: Option<Tensor>,            // global SB weights
    pub log_beta: Option<Tensor>,        // (global)
    pub gamma: Option<Tensor>,           // (global)
}

pub type KumaraswamyGlobal = KumaraswamyConcentration;
pub type KumaraswamyLocal = KumaraswamyConcentration;

impl KumaraswamyConcentration {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Packs the raw parameter features `(q_a | q_b)` along the last dim.
    pub fn pack_features(&self) -> Result<Tensor> {
        match (&self.q_a, &self.q_b) {
            (Some(a), Some(b)) => Tensor::cat(&[a, b], D::Minus1),
            (Some(a), None) => Ok(a.clone()),
            (None, Some(b)) => Ok(b.clone()),
            (None, None) => Err(candle_core::Error::Msg("no active tensors to pack".into())),
        }
    }

    /// Reparameterised stick samples for a batch via the Kumaraswamy
    /// inverse-CDF: draws `v_a`, `v_b` `[B, K−1]`, pads the truncation stick
    /// to `[B, K]` with 1.0, and concatenates to `[B, 2K]` — the Python
    /// `pack_samples`.
    pub fn pack_samples(&self, batch_size: usize) -> Result<Tensor> {
        let (Some(qa), Some(qb)) = (&self.q_a, &self.q_b) else {
            return Err(candle_core::Error::Msg(
                "both q_a and q_b must be populated to sample".into(),
            ));
        };
        let k1 = qa.dim(D::Minus1)?;
        let qa_b = qa.reshape((1, k1))?.broadcast_as((batch_size, k1))?.contiguous()?;
        let qb_b = qb.reshape((1, k1))?.broadcast_as((batch_size, k1))?.contiguous()?;
        let pad = Tensor::ones((batch_size, 1), qa.dtype(), qa.device())?;
        let mut halves = Vec::with_capacity(2);
        for _ in 0..2 {
            let u = qa_b.rand_like(0.0, 1.0)?;
            let v = kumaraswamy_icdf(&u, &qa_b, &qb_b, 1e-12)?; // [B, K−1]
            halves.push(Tensor::cat(&[&v, &pad], D::Minus1)?); // [B, K]
        }
        Tensor::cat(&[&halves[0], &halves[1]], D::Minus1) // [B, 2K]
    }
}

