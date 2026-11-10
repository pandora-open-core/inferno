//! Inferno — Configuration sections for neural models and numerical components.

use serde::{Serialize, Deserialize};
use anyhow::{bail, Result};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EWMAConfig {
    pub ewma: bool,
    /// Half-lives in ROLLOUT STEPS, one EWMA level each. Must be ascending —
    /// the momentum readout is (first level − last level), so the order
    /// decides its SIGN.
    pub ewma_half_lives: Vec<f64>,
    pub ewma_ctx_gain: f64,
    pub ewma_to_policy: bool,
}

impl Default for EWMAConfig {
    fn default() -> Self {
        Self {
            // ON by default (shipped Python config.yaml ewma: true) — the
            // z → z_ssm chain adds the bank's momentum context, and the
            // latent policy reads z_ewma_history under ewma_to_policy.
            ewma: true,
            ewma_half_lives: vec![1.0, 2.0, 4.0, 8.0],
            ewma_ctx_gain: 1.0,
            ewma_to_policy: true,
        }
    }
}

impl EWMAConfig {
    pub fn validate(&self) -> Result<()> {
        if self.ewma_half_lives.is_empty() {
            bail!("ewma_half_lives must contain at least one scale");
        }
        if self.ewma_half_lives.iter().any(|&h| h <= 0.0) {
            bail!(
                "ewma_half_lives must all be > 0; got {:?}. The half-life sets \
                 alpha = 1 - 0.5^(1/h), which is undefined at 0.",
                self.ewma_half_lives
            );
        }
        if self.ewma_half_lives.windows(2).any(|w| w[0] > w[1]) {
            bail!(
                "ewma_half_lives must be ascending; got {:?}. An unsorted bank silently \
                 reports the trend backwards.",
                self.ewma_half_lives
            );
        }
        if self.ewma_ctx_gain < 0.0 {
            bail!("EWMAConfig.ewma_ctx_gain must be ≥ 0");
        }
        Ok(())
    }
}
