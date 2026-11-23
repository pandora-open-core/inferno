import Pudding.Alignment
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
namespace Pudding.Condense
/-- Coordinate error for top-n selection and renormalisation. kept and removed
    are the CONDITIONAL barycentres. R bounds the atom coordinate, d is the
    discarded fraction of total input mass, including its 1e-6 simplex tolerance.
    This does not compare the unnormalised input sum with the output. -/
theorem renormalised_error (kept removed d R : ℝ)
    (hk : |kept| ≤ R) (hr : |removed| ≤ R) (hd : 0 ≤ d) :
    |((1-d)*kept+d*removed)-kept| ≤ 2*R*d := by
  rw [show ((1-d)*kept+d*removed)-kept = d*(removed-kept) by ring, abs_mul,
    abs_of_nonneg hd]
  have hb : |removed-kept| ≤ 2*R := by
    have := abs_sub removed kept
    linarith
  calc
    d * |removed-kept| ≤ d * (2*R) := mul_le_mul_of_nonneg_left hb hd
    _ = 2*R*d := by ring

/-- Transfer only along a separately bounded downstream map. -/
theorem downstream_error (error gain R d : ℝ) (hg : 0 ≤ gain)
    (h : error ≤ 2*R*d) : gain*error ≤ gain*(2*R*d) :=
  mul_le_mul_of_nonneg_left h hg
end Pudding.Condense
