import Pudding.Transport.Finite

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

/-- Deterministic map represented as a coupling of the quadrature source. -/
def mapCoupling [DecidableEq J] (a : Probability I) (assign : I → J) : I → J → ℝ :=
  fun i j => if assign i = j then a.mass i else 0

/-- Prescribed cell masses certify both marginals of the deterministic map. -/
theorem map_feasible [DecidableEq J] (a : Probability I) (b : Probability J) (assign : I → J)
    (cells : ∀ j, (∑ i, if assign i = j then a.mass i else 0) = b.mass j) :
    Feasible a.mass b.mass (mapCoupling a assign) := by
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    dsimp [mapCoupling]
    split_ifs
    · exact a.nonneg i
    · exact le_refl 0
  · intro i
    simp [mapCoupling, eq_comm]
  · intro j
    simpa only [mapCoupling] using cells j

/-- Finite quadrature dual bound, separate from entropic transport. -/
theorem map_dual_bound (a : Probability I) (b : Probability J)
    (cost p : I → J → ℝ) (phi : I → ℝ) (psi : J → ℝ)
    (hp : Feasible a.mass b.mass p)
    (dual : ∀ i j, phi i + psi j ≤ cost i j) :
    (∑ i, phi i * a.mass i) + (∑ j, psi j * b.mass j) ≤
      ∑ i, ∑ j, cost i j * p i j := by
  calc
    _ = ∑ i, ∑ j, (phi i + psi j) * p i j := by
      simp_rw [add_mul, Finset.sum_add_distrib]
      congr 1
      · simp_rw [← Finset.mul_sum, hp.2.1]
      · rw [Finset.sum_comm]
        simp_rw [← Finset.mul_sum, hp.2.2]
    _ ≤ _ := Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
      mul_le_mul_of_nonneg_right (dual i j) (hp.1 i j)

/-- Equality with a feasible dual certifies unregularised finite optimality.
The continuous cell/integration correspondence must be established separately. -/
theorem map_optimal (a : Probability I) (b : Probability J)
    (cost p : I → J → ℝ) (phi : I → ℝ) (psi : J → ℝ)
    (hp : Feasible a.mass b.mass p)
    (dual : ∀ i j, phi i + psi j ≤ cost i j)
    (tight : (∑ i, ∑ j, cost i j * p i j) =
      (∑ i, phi i * a.mass i) + (∑ j, psi j * b.mass j))
    (q : I → J → ℝ) (hq : Feasible a.mass b.mass q) :
    Feasible a.mass b.mass p ∧
      (∑ i, ∑ j, cost i j * p i j) ≤ ∑ i, ∑ j, cost i j * q i j := by
  refine ⟨hp, ?_⟩
  rw [tight]
  exact map_dual_bound a b cost q phi psi hq dual

end Pudding.Transport
