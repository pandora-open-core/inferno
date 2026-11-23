import Pudding.Langevin.Passive

namespace Pudding.Langevin

/-- Deterministic clock scaling preserves the frozen OU variance equation.
This is conditional algebra, not a stochastic time-change theorem. -/
theorem clock_variance (k diffusion slope variance : ℝ)
    (h : k * variance = diffusion) :
    -2 * (slope * k) * variance + 2 * slope * diffusion = 0 := by
  calc
    -2 * (slope * k) * variance + 2 * slope * diffusion =
      2 * slope * (diffusion - k * variance) := by ring
    _ = 0 := by rw [h]; ring

/-- Probability-flow velocity has the McKean–Vlasov covariance derivative
when the supplied score covariance identity is valid. -/
theorem meanfield_covariance (k interaction diffusion variance coefficient : ℝ)
    (h : coefficient * variance = -(k + interaction) * variance + diffusion) :
    2 * coefficient * variance = -2 * (k + interaction) * variance + 2 * diffusion := by
  calc
    2 * coefficient * variance = 2 * (coefficient * variance) := by ring
    _ = -2 * (k + interaction) * variance + 2 * diffusion := by rw [h]; ring

/-- Off-diagonal detailed balance implies the two-state stationarity equation. -/
theorem two_state_stationarity (p q forward backward : ℝ)
    (h : p * forward = q * backward) :
    p * (1 - forward) + q * backward = p := by
  calc
    p * (1 - forward) + q * backward = p - p * forward + q * backward := by ring
    _ = p := by rw [h]; ring

/-- A common potential gauge cancels from a debiased objective readout. -/
theorem divergence_self (objective : ℝ) :
    objective - objective / 2 - objective / 2 = 0 := by ring

end Pudding.Langevin
