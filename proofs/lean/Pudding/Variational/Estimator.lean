import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
namespace Pudding.Variational
/-- A deterministic QMC error budget. Neither Sobol generation nor discrepancy
    estimates are postulated as already verified Lean implementations. -/
theorem qmc_bias_budget (estimate squashed target discrepancy squashBias : ℝ)
    (hq : |estimate-squashed| ≤ discrepancy)
    (hs : |squashed-target| ≤ squashBias) :
    |estimate-target| ≤ discrepancy+squashBias := by
  calc
    |estimate-target| = |(estimate-squashed)+(squashed-target)| := by ring_nf
    _ ≤ |estimate-squashed|+|squashed-target| := abs_add _ _
    _ ≤ discrepancy+squashBias := add_le_add hq hs
end Pudding.Variational
