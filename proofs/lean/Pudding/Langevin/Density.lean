import Pudding.Langevin.Passive
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace Pudding.Langevin

/-- Exact factor convention of the candidate density, with positive prior value. -/
theorem gibbs_factor (prior reward beta : ℝ) (hprior : 0 < prior) :
    Real.exp (Real.log prior + beta / 2 * reward) =
      prior * Real.exp (beta / 2 * reward) := by
  rw [Real.exp_add, Real.exp_log hprior]

/-- Coordinate derivative of the candidate is drift times density. This establishes
zero current under differentiability premises; global SDE invariance still needs
normalisation, nonexplosion and boundary hypotheses. -/
theorem candidate_derivative (logPrior reward : ℝ → ℝ)
    (x beta priorScore rewardSlope normaliser : ℝ)
    (hPrior : HasDerivAt logPrior priorScore x)
    (hReward : HasDerivAt reward rewardSlope x) :
    HasDerivAt
      (fun y => Real.exp (logPrior y + beta / 2 * reward y) / normaliser)
      (passiveDrift beta rewardSlope priorScore *
        (Real.exp (logPrior x + beta / 2 * reward x) / normaliser)) x := by
  convert ((hPrior.add (hReward.const_mul (beta / 2))).exp.div_const normaliser) using 1 <;>
    dsimp [passiveDrift] <;> ring

end Pudding.Langevin
