//! Inferno — Additional loss terms for Gaussian-process objectives.

use candle_core::{Result, Tensor};

/// A term added to the ELBO at FULL, undivided weight.
pub trait AddedLossTerm: Send + Sync {
    fn name(&self) -> &str;
    /// The PENALTY (`−weight·Σ log p(θ)`), a rank-0 tensor. The ELBO
    /// SUBTRACTS it, so returning a penalty adds the log-prior back.
    fn loss(&self) -> Result<Tensor>;
}

/// `(name, log-density closure)` — the closure receives nothing and reads the
/// constrained value off its captured parameter, mirroring gpytorch's
/// `closure(module)`.
pub type PriorEntry = (
    &'static str,
    std::sync::Arc<dyn Fn() -> Result<Tensor> + Send + Sync>,
);

/// Priors on the GP heads, at a weight that can actually bind — the port of
/// `models/strategy.py::HeadPriorLoss`.
pub struct HeadPriorLoss {
    name: String,
    entries: Vec<PriorEntry>,
    /// Scalar multiplier on the summed negative log-prior. Python's default
    /// `1e-2`: `log_likelihood` is per-datapoint and O(1)–O(10), so a
    /// FULL-weight prior over a handful of nuisance scalars would rival the
    /// entire data term.
    pub weight: f64,
    device: candle_core::Device,
}

impl HeadPriorLoss {
    pub fn new(
        name: impl Into<String>,
        entries: Vec<PriorEntry>,
        weight: f64,
        device: candle_core::Device,
    ) -> Self {
        Self {
            name: name.into(),
            entries,
            weight,
            device,
        }
    }
}

impl AddedLossTerm for HeadPriorLoss {
    fn name(&self) -> &str {
        &self.name
    }

    fn loss(&self) -> Result<Tensor> {
        let mut total: Option<Tensor> = None;
        for (_n, lp) in &self.entries {
            // A prior that cannot be evaluated must not take the step down
            // with it — it is a regulariser, not the objective.
            let Ok(v) = lp() else { continue };
            let v = v.sum_all()?;
            total = Some(match total {
                None => v,
                Some(t) => t.add(&v)?,
            });
        }
        match total {
            None => Tensor::zeros((), candle_core::DType::F32, &self.device),
            // −weight·Σ log p(θ): the ELBO subtracts it ⇒ +weight·log p.
            Some(t) => t.affine(-self.weight, 0.0),
        }
    }
}

/// `log N(x; μ, σ)` summed — a Normal prior's log-density.
pub fn normal_log_prob(x: &Tensor, mu: f64, sigma: f64) -> Result<Tensor> {
    let z = x.affine(1.0 / sigma, -mu / sigma)?;
    let c = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln();
    z.sqr()?.affine(-0.5, c)
}

/// Jeffreys prior for a SCALE parameter: `p(x) ∝ 1/x`, i.e. flat in `log x`,
/// so `log p = −log x`.
///
/// Improper on `(0, ∞)` — which is why it is only usable against a hard floor.
/// Paired with the temporal lengthscale's `GreaterThan(0.5)` it is proper
/// ENOUGH: scale-invariant, expressing no preference among admissible
/// magnitudes, while the CONSTRAINT (not the prior) forbids collapse. That
/// division is deliberate — a lengthscale that must not go below 0.5 should be
/// told so structurally, not nudged by a density.
pub fn log_uniform_log_prob(x: &Tensor) -> Result<Tensor> {
    x.clamp(1e-12, f64::INFINITY)?.log()?.affine(-1.0, 0.0)
}

#[cfg(test)]
mod tests {
    //! Verification of additional loss terms for gaussian-process objectives.

    use super::*;
    use candle_core::{DType, Device};
    use std::sync::Arc;

    /// The sign convention: `loss()` is a PENALTY, so a value AT the prior's
    /// mode must be penalised LESS than one far from it. Getting this backwards
    /// would push the parameter away from its prior.
    #[test]
    fn the_penalty_is_lower_at_the_prior_mode() {
        let dev = Device::Cpu;
        let at_mode = Tensor::full(-2.75f32, 1, &dev).unwrap();
        let far = Tensor::full(6.0f32, 1, &dev).unwrap();

        let pen = |t: Tensor| -> f64 {
            let l = HeadPriorLoss::new(
                "head_priors",
                vec![(
                    "fsa_scale",
                    Arc::new(move || normal_log_prob(&t, -2.75, 1.5)) as _,
                )],
                1e-2,
                dev.clone(),
            );
            l.loss().unwrap().to_scalar::<f32>().unwrap() as f64
        };
        assert!(
            pen(at_mode) < pen(far),
            "loss() must be a PENALTY: {} at the mode vs {} far from it",
            pen(Tensor::full(-2.75f32, 1, &dev).unwrap()),
            pen(Tensor::full(6.0f32, 1, &dev).unwrap())
        );
    }

    /// A prior whose evaluation fails is skipped, not fatal — it is a
    /// regulariser, not the objective.
    #[test]
    fn an_unevaluable_prior_is_skipped_not_fatal() {
        let dev = Device::Cpu;
        let ok = Tensor::full(1.0f32, 1, &dev).unwrap();
        let l = HeadPriorLoss::new(
            "mixed",
            vec![
                (
                    "broken",
                    Arc::new(|| Err(candle_core::Error::Msg("nope".into()))) as _,
                ),
                (
                    "fine",
                    Arc::new(move || normal_log_prob(&ok, 0.0, 1.0)) as _,
                ),
            ],
            1.0,
            dev.clone(),
        );
        let v = l.loss().unwrap().to_scalar::<f32>().unwrap();
        assert!(
            v.is_finite(),
            "the healthy entry must still contribute: {v}"
        );
    }

    /// With no evaluable entries the term is exactly zero (never NaN).
    #[test]
    fn an_empty_term_is_exactly_zero() {
        let dev = Device::Cpu;
        let l = HeadPriorLoss::new("empty", vec![], 1e-2, dev);
        assert_eq!(l.loss().unwrap().to_scalar::<f32>().unwrap(), 0.0);
    }

    /// The Jeffreys prior is flat in `log x`: equal RATIOS cost equally.
    #[test]
    fn the_log_uniform_prior_is_scale_invariant_in_log_x() {
        let dev = Device::Cpu;
        let lp = |v: f32| {
            log_uniform_log_prob(&Tensor::full(v, 1, &dev).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
        };
        // log p(2x) − log p(x) = −log 2 for ANY x.
        let d1 = lp(2.0) - lp(1.0);
        let d2 = lp(64.0) - lp(32.0);
        assert!(
            (d1 - d2).abs() < 1e-5,
            "must be scale-invariant: {d1} vs {d2}"
        );
        assert!((d1 + 2.0f64.ln()).abs() < 1e-5, "and equal −log 2: {d1}");
    }

    /// Gradient reaches the parameter — a prior that is in the graph but
    /// receives nothing is the `register_prior` failure mode in miniature.
    #[test]
    fn the_penalty_carries_gradient_to_its_parameter() {
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let raw = vb
            .get_with_hints(1, "raw_fsa_scale", candle_nn::init::Init::Const(0.0))
            .unwrap();
        let r2 = raw.clone();
        let l = HeadPriorLoss::new(
            "head_priors",
            vec![(
                "fsa_scale",
                Arc::new(move || normal_log_prob(&r2, -2.75, 1.5)) as _,
            )],
            1e-2,
            dev,
        );
        let grads = l.loss().unwrap().backward().unwrap();
        let g = grads.get(&raw).expect("the prior must reach its parameter");
        let n = g
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(n > 0.0, "gradient magnitude {n}");
    }
}
