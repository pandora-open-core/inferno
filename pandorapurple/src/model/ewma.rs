//! Inferno — Exponentially weighted moving statistics on tensor sequences.

use candle_core::{DType, Result, Tensor, D};

/// The bank. Construct once and share; it owns no state between calls.
#[derive(Clone, Debug)]
pub struct EwmaEncoder {
    d_model: usize,
    /// `[1, K, 1]` — `α = 1 − 0.5^(1/h)`.
    alpha: Tensor,
    half_lives: Vec<f64>,
    ctx_gain: f64,
}

impl EwmaEncoder {
    /// `half_lives` are in ROLLOUT STEPS and must be positive.
    ///
    /// `α = 1 − 0.5^(1/h)`: the level retains half its weight after `h` steps.
    /// `h = 1 → α = 0.5`, `h = 8 → α = 0.083`. Stated this way the knob is
    /// directly readable, unlike the SSM's `Δ·|A|` where the horizon is a
    /// product of two learned tensors and legible only after the fact.
    pub fn new(
        d_model: usize,
        half_lives: &[f64],
        ctx_gain: f64,
        dtype: DType,
        dev: &candle_core::Device,
    ) -> Result<Self> {
        let hl: Vec<f64> = half_lives.iter().copied().filter(|h| *h > 0.0).collect();
        if hl.is_empty() {
            return Err(candle_core::Error::Msg(
                "EwmaEncoder: half_lives must contain at least one positive value".into(),
            ));
        }
        // ASCENDING is required, not merely conventional: `momentum` reads
        // `state[0] − state[K−1]` as fastest-minus-slowest, so a descending list
        // silently flips the sign of every trend the bank reports.
        if hl.windows(2).any(|w| w[1] <= w[0]) {
            return Err(candle_core::Error::Msg(format!(
                "EwmaEncoder: half_lives must be strictly ascending — `momentum` reads \
                 fastest-minus-slowest by POSITION, so a descending list flips the sign of \
                 every trend it reports. Got {hl:?}"
            )));
        }
        let a: Vec<f64> = hl.iter().map(|h| 1.0 - 0.5f64.powf(1.0 / h)).collect();
        let alpha = Tensor::from_vec(a, (1, hl.len(), 1), dev)?.to_dtype(dtype)?;
        Ok(Self { d_model, alpha, half_lives: hl, ctx_gain })
    }

    /// The production default: half-lives `(1, 2, 4, 8)`, unit context gain.
    pub fn default_bank(d_model: usize, dtype: DType, dev: &candle_core::Device) -> Result<Self> {
        Self::new(d_model, &[1.0, 2.0, 4.0, 8.0], 1.0, dtype, dev)
    }

    pub fn n_scales(&self) -> usize {
        self.half_lives.len()
    }

    /// `D·(K+1)` — the observation plus one level per scale.
    pub fn feature_dim(&self) -> usize {
        self.d_model * (self.n_scales() + 1)
    }

    pub fn half_lives(&self) -> &[f64] {
        &self.half_lives
    }

    /// The momentum gain. `window_momentum` returns the UNSCALED momentum
    /// (matching Python) while `step` pre-scales — a windowed caller applies
    /// this itself.
    pub fn ctx_gain(&self) -> f64 {
        self.ctx_gain
    }

    /// Every level seeded AT the first observation — `[B, K, D]`.
    ///
    /// **Not zeros.** A zero-seeded EWMA spends its first several steps climbing
    /// out of an origin the latent never occupied, and within a 12-step window
    /// the slow scales never arrive at all — the bank would report a transient
    /// of its own initialisation as the trajectory's momentum. Seeding at `μ_0`
    /// makes every scale exact on a constant path, and makes the streaming and
    /// windowed paths agree on a left-padded window.
    pub fn init_state(&self, mu: &Tensor) -> Result<Tensor> {
        let (b, d) = mu.dims2()?;
        mu.detach()
            .unsqueeze(1)?
            .broadcast_as((b, self.n_scales(), d))?
            .contiguous()
    }

    /// One streaming update → `(features [B, (K+1)·D], ctx [B, D], state [B, K, D])`.
    ///
    /// `ctx` is the momentum read, pre-scaled by `ctx_gain` and ready to ADD to
    /// the SSM's input. Everything is detached: the bank observes, it does not
    /// participate in the gradient.
    pub fn step(&self, mu: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, d) = mu.dims2().map_err(|_| {
            candle_core::Error::Msg(format!("EwmaEncoder::step: mu must be [B, D], got {:?}", mu.dims()))
        })?;
        if d != self.d_model {
            return Err(candle_core::Error::Msg(format!(
                "EwmaEncoder::step: mu is {d}-wide, the bank was built for {}",
                self.d_model
            )));
        }
        let mu = mu.detach();
        // A carried state from another batch is the same class of bug a
        // broadcast time grid is: shapes that fit instead of raising. RESTART
        // the bank rather than filter this batch through another's history.
        let state = match state {
            Some(s) if s.dim(0)? == b => s.detach(),
            _ => self.init_state(&mu)?,
        };
        let a = self.alpha.to_dtype(mu.dtype())?.to_device(mu.device())?;
        let state = a
            .broadcast_mul(&mu.unsqueeze(1)?)?
            .add(&a.affine(-1.0, 1.0)?.broadcast_mul(&state)?)?; // [B, K, D]

        let feats = Tensor::cat(&[&mu, &state.flatten_from(1)?], D::Minus1)?;
        let ctx = self.momentum(&state)?.affine(self.ctx_gain, 0.0)?;
        Ok((feats, ctx, state))
    }

    /// The same recursion over a finished `[B, T, D]` window → `[B, T, F]`.
    pub fn encode_window(&self, z_mu_history: &Tensor) -> Result<Tensor> {
        let (_b, t, _d) = z_mu_history.dims3().map_err(|_| {
            candle_core::Error::Msg(format!(
                "EwmaEncoder::encode_window: expected [B, T, D], got {:?}",
                z_mu_history.dims()
            ))
        })?;
        let mut state: Option<Tensor> = None;
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let (f, _, s) = self.step(&z_mu_history.narrow(1, i, 1)?.squeeze(1)?, state.as_ref())?;
            out.push(f);
            state = Some(s);
        }
        Tensor::stack(&out, 1)
    }

    /// Fastest level minus slowest, `[B, D]` — the trend the window is on.
    ///
    /// **Exactly zero on a constant path**, so adding it to the SSM's input is a
    /// no-op for a latent that is not moving. That property is what makes it
    /// safe to switch on: it can only inject what it measured, and it cannot
    /// manufacture a signal.
    pub fn momentum(&self, state: &Tensor) -> Result<Tensor> {
        let k = self.n_scales();
        if k < 2 {
            return state.narrow(1, 0, 1)?.squeeze(1)?.zeros_like();
        }
        state
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .sub(&state.narrow(1, k - 1, 1)?.squeeze(1)?)
    }

    /// Per-position momentum over a window — `[B, T, D]`.
    pub fn window_momentum(&self, z_mu_history: &Tensor) -> Result<Tensor> {
        let (_b, t, _d) = z_mu_history.dims3()?;
        let mut state: Option<Tensor> = None;
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let (_, _, s) = self.step(&z_mu_history.narrow(1, i, 1)?.squeeze(1)?, state.as_ref())?;
            out.push(self.momentum(&s)?);
            state = Some(s);
        }
        Tensor::stack(&out, 1)
    }
}

#[cfg(test)]
mod tests {
    //! Verification of exponentially weighted moving statistics on tensor sequences.

    use super::*;
    use candle_core::Device;

    fn dev() -> Device {
        Device::Cpu
    }
    fn bank(d: usize) -> EwmaEncoder {
        EwmaEncoder::default_bank(d, DType::F64, &dev()).unwrap()
    }
    fn m(v: &[f64], s: (usize, usize)) -> Tensor {
        Tensor::from_vec(v.to_vec(), s, &dev()).unwrap()
    }
    fn maxabs(t: &Tensor) -> f64 {
        t.abs().unwrap().max_all().unwrap().to_vec0::<f64>().unwrap()
    }

    /// The half-life IS the half-life: after `h` steps of a step-input the level
    /// has covered exactly half the distance.
    #[test]
    fn the_half_life_is_the_half_life() {
        let b = bank(1);
        // Start at 0, then feed a constant 1. After h steps level k should sit
        // at 1 − 0.5 = 0.5 of the way for half-life h.
        for (k, h) in [(0usize, 1.0f64), (1, 2.0), (2, 4.0), (3, 8.0)] {
            let mut st = b.init_state(&m(&[0.0], (1, 1))).unwrap();
            let one = m(&[1.0], (1, 1));
            for _ in 0..(h as usize) {
                let (_, _, s) = b.step(&one, Some(&st)).unwrap();
                st = s;
            }
            let lvl: Vec<f64> = st.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                (lvl[k] - 0.5).abs() < 1e-12,
                "half-life {h}: level {k} reached {} after {h} steps, expected 0.5",
                lvl[k]
            );
        }
    }

    /// A constant path is a fixed point — every level stays at the observation
    /// and the momentum is EXACTLY zero. That is what makes adding the context
    /// to the SSM's input safe: on a latent that is not moving it is a no-op.
    #[test]
    fn a_constant_path_is_a_fixed_point_with_zero_momentum() {
        let b = bank(3);
        let mu = m(&[0.4, -1.2, 3.0, 0.4, -1.2, 3.0], (2, 3));
        let mut st: Option<Tensor> = None;
        for _ in 0..20 {
            let (feats, ctx, s) = b.step(&mu, st.as_ref()).unwrap();
            assert_eq!(maxabs(&ctx), 0.0, "momentum on a constant path must be EXACTLY 0");
            // features are [μ | levels], and every level equals μ here.
            assert_eq!(feats.dims(), &[2, b.feature_dim()]);
            st = Some(s);
        }
        let s = st.unwrap();
        for k in 0..b.n_scales() {
            let lvl = s.narrow(1, k, 1).unwrap().squeeze(1).unwrap();
            assert!(maxabs(&lvl.sub(&mu).unwrap()) < 1e-12, "level {k} drifted off a constant");
        }
    }

    /// Momentum takes the sign of the trend, and the filter is CAUSAL — a
    /// change at step `t` cannot move any output before `t`.
    #[test]
    fn momentum_has_the_sign_of_the_trend_and_the_filter_is_causal() {
        let b = bank(1);
        let rising: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let w = Tensor::from_vec(rising.clone(), (1, 8, 1), &dev()).unwrap();
        let mom: Vec<f64> = b.window_momentum(&w).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(mom[1..].iter().all(|v| *v > 0.0), "a rising path has positive momentum: {mom:?}");

        let falling: Vec<f64> = rising.iter().rev().copied().collect();
        let wf = Tensor::from_vec(falling, (1, 8, 1), &dev()).unwrap();
        let momf: Vec<f64> =
            b.window_momentum(&wf).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(momf[1..].iter().all(|v| *v < 0.0), "a falling path has negative momentum");

        // Causality: perturb the LAST row only; nothing before it may move.
        let mut bumped = rising.clone();
        bumped[7] += 100.0;
        let wb = Tensor::from_vec(bumped, (1, 8, 1), &dev()).unwrap();
        let a = b.encode_window(&w).unwrap();
        let c = b.encode_window(&wb).unwrap();
        let head_a = a.narrow(1, 0, 7).unwrap();
        let head_c = c.narrow(1, 0, 7).unwrap();
        assert_eq!(
            maxabs(&head_a.sub(&head_c).unwrap()),
            0.0,
            "a change at step 7 leaked backwards — the row-stopping head is scored on \
             predicting exactly that"
        );
    }

    /// The streaming and windowed entry points are the SAME recursion.
    #[test]
    fn streaming_and_windowed_agree_exactly() {
        let b = bank(2);
        let vals: Vec<f64> = (0..12 * 2).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.3).collect();
        let w = Tensor::from_vec(vals, (1, 12, 2), &dev()).unwrap();

        let windowed = b.encode_window(&w).unwrap();
        let mut st: Option<Tensor> = None;
        let mut rows = Vec::new();
        for i in 0..12 {
            let (f, _, s) = b.step(&w.narrow(1, i, 1).unwrap().squeeze(1).unwrap(), st.as_ref()).unwrap();
            rows.push(f);
            st = Some(s);
        }
        let streamed = Tensor::stack(&rows, 1).unwrap();
        assert_eq!(maxabs(&windowed.sub(&streamed).unwrap()), 0.0);
    }

    /// Seeding at `μ_0` rather than at zero is what makes a left-padded window
    /// reproduce the unpadded run.
    ///
    /// With a zero seed the slow scales spend the whole 12-step window climbing
    /// out of an origin the latent never occupied, and the bank reports a
    /// transient of its own initialisation as the trajectory's momentum.
    #[test]
    fn a_left_padded_window_reproduces_the_unpadded_run() {
        let b = bank(2);
        let core: Vec<f64> = (0..6 * 2).map(|i| ((i % 5) as f64) * 0.4 - 1.0).collect();
        let w = Tensor::from_vec(core.clone(), (1, 6, 2), &dev()).unwrap();

        // Duplicate the FIRST row four times in front: the seed already sits at
        // that row, so the padding must be inert.
        let first: Vec<f64> = core[..2].to_vec();
        let mut padded: Vec<f64> = Vec::new();
        for _ in 0..4 {
            padded.extend_from_slice(&first);
        }
        padded.extend_from_slice(&core);
        let wp = Tensor::from_vec(padded, (1, 10, 2), &dev()).unwrap();

        let a = b.encode_window(&w).unwrap();
        let c = b.encode_window(&wp).unwrap().narrow(1, 4, 6).unwrap();
        assert!(maxabs(&a.sub(&c).unwrap()) < 1e-12, "left-padding with the first row must be inert");
    }

    /// It owns no parameters and carries no gradient — the whole point of it
    /// being the timescale source rather than the SSM's learned bank.
    #[test]
    fn it_owns_nothing_and_carries_no_gradient() {
        let b = bank(3);
        let vm = candle_nn::VarMap::new();
        // Nothing was ever registered: the bank takes no VarBuilder at all.
        assert!(vm.data().lock().unwrap().is_empty());

        let mu = Tensor::randn(0f64, 1f64, (2, 3), &dev()).unwrap();
        let v = candle_core::Var::from_tensor(&mu).unwrap();
        let (feats, ctx, _) = b.step(v.as_tensor(), None).unwrap();
        // Detached throughout, so backward finds nothing to attribute.
        assert!(feats.sum_all().unwrap().backward().unwrap().get(&v).is_none());
        assert!(ctx.sum_all().unwrap().backward().unwrap().get(&v).is_none());
    }

    /// Feature width is `D·(K+1)` — the observation plus ONE level per scale.
    /// Deliberately not the momentum or the deviations: those are fixed linear
    /// combinations a linear readout forms for free.
    #[test]
    fn the_feature_width_is_the_mean_plus_one_level_per_scale() {
        let b = bank(5);
        assert_eq!(b.n_scales(), 4);
        assert_eq!(b.feature_dim(), 5 * 5);
        let mu = Tensor::randn(0f64, 1f64, (3, 5), &dev()).unwrap();
        let (f, _, s) = b.step(&mu, None).unwrap();
        assert_eq!(f.dims(), &[3, 25]);
        assert_eq!(s.dims(), &[3, 4, 5]);
        // The first D columns ARE μ.
        assert_eq!(maxabs(&f.narrow(1, 0, 5).unwrap().sub(&mu).unwrap()), 0.0);
    }

    /// A carried state from another batch RESTARTS the bank rather than
    /// broadcasting — shapes that fit instead of raising are the same class of
    /// bug as a broadcast time grid.
    #[test]
    fn a_state_from_another_batch_restarts_the_bank() {
        let b = bank(2);
        let mu4 = Tensor::randn(0f64, 1f64, (4, 2), &dev()).unwrap();
        let (_, _, s4) = b.step(&mu4, None).unwrap();

        let mu2 = m(&[1.0, 2.0, 3.0, 4.0], (2, 2));
        let (_, ctx, s2) = b.step(&mu2, Some(&s4)).unwrap();
        assert_eq!(s2.dims(), &[2, 4, 2]);
        // A restart seeds at μ, so the first step's momentum is exactly 0.
        assert_eq!(maxabs(&ctx), 0.0, "a restarted bank must not report a trend it never saw");
    }

    /// Descending half-lives are refused: `momentum` reads fastest-minus-slowest
    /// BY POSITION, so a descending list silently flips every trend's sign.
    #[test]
    fn descending_half_lives_are_refused() {
        let e = EwmaEncoder::new(4, &[8.0, 4.0, 1.0], 1.0, DType::F64, &dev()).unwrap_err();
        assert!(e.to_string().contains("ascending"), "{e}");
        assert!(EwmaEncoder::new(4, &[], 1.0, DType::F64, &dev()).is_err());
        assert!(EwmaEncoder::new(4, &[1.0, 2.0], 1.0, DType::F64, &dev()).is_ok());
    }
}
