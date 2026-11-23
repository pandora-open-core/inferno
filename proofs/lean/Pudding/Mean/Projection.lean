import Pudding.Transport.Problem

namespace Pudding.Mean
open scoped BigOperators
variable {K L : Type*} [Fintype K] [Fintype L]

noncomputable def responsibilities (w : K → L → ℝ) (k : K) (l : L) : ℝ :=
  w k l / ∑ j, w k j
noncomputable def profiles (w : K → L → ℝ) (l : L) (k : K) : ℝ :=
  w k l / ∑ j, w j l
noncomputable def project (pi : K → ℝ) (resp : K → L → ℝ) (l : L) : ℝ :=
  ∑ k, pi k * resp k l

theorem responsibility_rows (w : K → L → ℝ) (h : ∀ k, (∑ l, w k l) ≠ 0) (k : K) :
    ∑ l, responsibilities w k l = 1 := by
  simp only [responsibilities, div_eq_mul_inv, ← Finset.sum_mul]; exact mul_inv_cancel₀ (h k)

theorem profile_rows (w : K → L → ℝ) (h : ∀ l, (∑ k, w k l) ≠ 0) (l : L) :
    ∑ k, profiles w l k = 1 := by
  simp only [profiles, div_eq_mul_inv, ← Finset.sum_mul]; exact mul_inv_cancel₀ (h l)

/-- The actual guarantee is preservation of input mass, not both matrix marginals. -/
theorem projection_mass (pi : K → ℝ) (resp : K → L → ℝ)
    (h : ∀ k, ∑ l, resp k l = 1) : ∑ l, project pi resp l = ∑ k, pi k := by
  unfold project
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum, h, mul_one]

/-- With no denominator floor active, input clamping accounts for all extra mass. -/
theorem clamped_projection_mass (pi : Transport.Probability K) (resp : K → L → ℝ)
    (floor : ℝ) (h : ∀ k, ∑ l, resp k l = 1) :
    1 ≤ ∑ l, project (fun k => max (pi.mass k) floor) resp l := by
  rw [projection_mass _ _ h, ← pi.total]
  exact Finset.sum_le_sum (fun k _ => le_max_left _ _)

/-- The two normalisations are conditionals of the SAME weighted joint.
    This is the defensible rectangular interpretation of the two stochastic maps. -/
theorem paired_conditionals (w : K → L → ℝ) (k : K) (l : L)
    (hrow : (∑ j, w k j) ≠ 0) (hcol : (∑ i, w i l) ≠ 0)
    (htotal : (∑ i, ∑ j, w i j) ≠ 0) :
    ((∑ j, w k j)/(∑ i, ∑ j, w i j)) * responsibilities w k l =
    ((∑ i, w i l)/(∑ i, ∑ j, w i j)) * profiles w l k := by
  unfold responsibilities profiles
  rw [div_mul_div_cancel₀' hrow, div_mul_div_cancel₀' hcol]

end Pudding.Mean
