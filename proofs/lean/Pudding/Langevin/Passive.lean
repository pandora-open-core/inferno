import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Pudding.Langevin

/-- Drift in the requested constant-diffusion convention. -/
noncomputable def passiveDrift (beta rewardSlope priorScore : ℝ) : ℝ :=
  beta / 2 * rewardSlope + priorScore

/-- A density derivative supplied by analytic hypotheses has zero probability current.
This algebraic implication does not assert existence, normalisation or an SDE theorem. -/
theorem zero_current (beta rewardSlope priorScore density derivative : ℝ)
    (h : derivative = (priorScore + beta / 2 * rewardSlope) * density) :
    passiveDrift beta rewardSlope priorScore * density - derivative = 0 := by
  rw [h]
  unfold passiveDrift
  ring

/-- The squared candidate has twice the score required by the stated noise. -/
theorem squared_score (beta rewardSlope priorScore : ℝ) :
    2 * priorScore + beta * rewardSlope =
      2 * passiveDrift beta rewardSlope priorScore := by
  unfold passiveDrift
  ring

/-- Potential gauges cancel in a coupling logit. -/
theorem coupling_gauge (f g c gauge epsilon : ℝ) :
    ((f + gauge) + (g - gauge) - c) / epsilon =
      (f + g - c) / epsilon := by
  congr 1
  ring

/-- Noise-space displacement implements the requested coordinate increment,
provided the denominator is the actual positive standard deviation. -/
theorem diagonal_displacement (m sigma noise delta : ℝ) (h : sigma ≠ 0) :
    m + sigma * (noise + delta / sigma) = m + sigma * noise + delta := by
  field_simp [h]
  ring

/-- Frozen OU Euler variance solves the second-moment fixed point, not the SDE's
variance. The stability inequalities are additional analytic premises. -/
theorem euler_variance_fixed_point (k dt variance : ℝ)
    (h : (2 * k - k ^ 2 * dt) * variance = 2) :
    (1 - k * dt) ^ 2 * variance + 2 * dt = variance := by
  calc
    (1 - k * dt) ^ 2 * variance + 2 * dt =
        variance + dt * (2 - (2 * k - k ^ 2 * dt) * variance) := by ring
    _ = variance := by rw [h]; ring

end Pudding.Langevin
