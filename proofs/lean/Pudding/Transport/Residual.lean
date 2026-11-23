import Pudding.Transport.Update

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

/-- Exact relation from unrelaxed potential residual to row mass. -/
theorem row_residual_identity (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε : ℝ) (hε : ε ≠ 0) (u : Potentials I J) (i : I)
    (hz : 0 < ∑ j, b j * Real.exp ((u.g j - cost i j) / ε)) :
    (∑ j, plan a b cost ε u i j) =
      a i * Real.exp ((u.f i - Tf b cost ε u.g i) / ε) := by
  have factor (j : J) : plan a b cost ε u i j =
      a i * Real.exp (u.f i / ε) * (b j * Real.exp ((u.g j - cost i j) / ε)) := by
    unfold plan
    rw [show (u.f i + u.g j - cost i j) / ε = u.f i / ε + (u.g j - cost i j) / ε by ring]
    rw [Real.exp_add]; ring
  simp_rw [factor]
  rw [← Finset.mul_sum]
  unfold Tf
  rw [show (u.f i - -ε * Real.log (∑ j, b j * Real.exp ((u.g j - cost i j) / ε))) / ε =
    u.f i / ε + Real.log (∑ j, b j * Real.exp ((u.g j - cost i j) / ε)) by field_simp <;> ring]
  rw [Real.exp_add, Real.exp_log hz]; ring

/-- A supplied numerical error bound adds to the exact residual; no optimality claim is implicit. -/
theorem residual_with_roundoff (exact reported η tol : ℝ)
    (herr : exact ≤ reported + η) (hsuccess : reported ≤ tol) : exact ≤ tol + η := by
  linarith

theorem column_residual_identity (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε : ℝ) (hε : ε ≠ 0) (u : Potentials I J) (j : J)
    (hz : 0 < ∑ i, a i * Real.exp ((u.f i - cost i j) / ε)) :
    (∑ i, plan a b cost ε u i j) =
      b j * Real.exp ((u.g j - Tg a cost ε u.f j) / ε) := by
  have h := row_residual_identity b a (fun j i => cost i j) ε hε
    (Potentials.mk u.g u.f) j hz
  convert h using 1 <;> simp only [plan, Tg, Tf]
  apply Finset.sum_congr rfl
  intro i _
  congr 1 <;> ring

/-- Exact fixed points satisfy BOTH marginals. -/
theorem fixed_point_feasible (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε : ℝ) (hε : ε ≠ 0) (u : Potentials I J)
    (ha : ∀ i, 0 < a i) (hb : ∀ j, 0 < b j)
    (hzf : ∀ i, 0 < ∑ j, b j * Real.exp ((u.g j - cost i j) / ε))
    (hzg : ∀ j, 0 < ∑ i, a i * Real.exp ((u.f i - cost i j) / ε))
    (hf : ∀ i, u.f i = Tf b cost ε u.g i)
    (hg : ∀ j, u.g j = Tg a cost ε u.f j) :
    Feasible a b (plan a b cost ε u) := by
  refine ⟨fun i j => (plan_positive a b cost ε u ha hb i j).le, ?_, ?_⟩
  · intro i
    rw [row_residual_identity a b cost ε hε u i (hzf i), hf i]
    simp
  · intro j
    rw [column_residual_identity a b cost ε hε u j (hzg j), hg j]
    simp

/-- Multiplicative marginal certificate from the actual unrelaxed residual.
    This remains useful without a general convergence theorem. -/
theorem row_residual_bounds (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε δ : ℝ) (hε : 0 < ε) (u : Potentials I J) (i : I) (ha : 0 ≤ a i)
    (hz : 0 < ∑ j, b j * Real.exp ((u.g j - cost i j) / ε))
    (herr : |u.f i - Tf b cost ε u.g i| ≤ δ) :
    a i * Real.exp (-δ/ε) ≤ ∑ j, plan a b cost ε u i j ∧
    (∑ j, plan a b cost ε u i j) ≤ a i * Real.exp (δ/ε) := by
  rw [row_residual_identity a b cost ε hε.ne' u i hz]
  rw [abs_le] at herr
  exact ⟨mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr
    (div_le_div_of_nonneg_right herr.1 hε.le)) ha,
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr
    (div_le_div_of_nonneg_right herr.2 hε.le)) ha⟩

/-- A successful balanced return can be certified when roundoff bounds are
    independently supplied. observedRow is the numerical row sum, not an
    assumed exact coupling, and no objective-gap certificate is asserted. -/
theorem successful_return_row (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε tol η ζ : ℝ) (hε : 0 < ε) (u : Potentials I J) (i : I)
    (ha : 0 ≤ a i) (hz : 0 < ∑ j, b j * Real.exp ((u.g j - cost i j) / ε))
    (reported observedRow : ℝ) (hsuccess : reported ≤ tol)
    (hupdate : |u.f i-Tf b cost ε u.g i| ≤ reported+η)
    (hround : |observedRow-(∑ j, plan a b cost ε u i j)| ≤ ζ) :
    a i*Real.exp (-(tol+η)/ε)-ζ ≤ observedRow ∧
      observedRow ≤ a i*Real.exp ((tol+η)/ε)+ζ := by
  have hr : |u.f i-Tf b cost ε u.g i| ≤ tol+η := by linarith
  have hbnd := row_residual_bounds a b cost ε (tol+η) hε u i ha hz hr
  rw [abs_le] at hround
  constructor <;> linarith [hbnd.1, hbnd.2]

end Pudding.Transport
