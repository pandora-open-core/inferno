import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
namespace Pudding.Alignment
open scoped BigOperators
/-- A single time column, with inactive denominator floor. Empty-bin uniform
    fill is a separate source branch. No negative responsibility is admissible. -/
noncomputable def normalise {n : ℕ} (mass : Fin n → ℝ) (i : Fin n) : ℝ := mass i / ∑ j, mass j

theorem normalised_sum (mass : Fin n → ℝ) (h : (∑ j, mass j) ≠ 0) :
    (∑ i, normalise mass i) = 1 := by
  simp only [normalise, div_eq_mul_inv, ← Finset.sum_mul]
  simpa only [div_eq_mul_inv] using div_self h

theorem normalised_nonneg (mass : Fin n → ℝ) (h : ∀ i, 0 ≤ mass i)
    (hz : 0 < ∑ j, mass j) (i : Fin n) : 0 ≤ normalise mass i :=
  div_nonneg (h i) (le_of_lt hz)

/-- A scalar coordinate of the barycentric map stays in the support interval. -/
theorem barycentric_bounds (w x : Fin n → ℝ) (lo hi : ℝ)
    (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1)
    (hx : ∀ i, lo ≤ x i ∧ x i ≤ hi) :
    lo ≤ ∑ i, w i*x i ∧ (∑ i, w i*x i) ≤ hi := by
  constructor
  · calc
      lo = ∑ i, w i*lo := by rw [← Finset.sum_mul, hs, one_mul]
      _ ≤ _ := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hx i).1 (hw i)
  · calc
      _ ≤ ∑ i, w i*hi := Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hx i).2 (hw i)
      _ = hi := by rw [← Finset.sum_mul, hs, one_mul]
end Pudding.Alignment
