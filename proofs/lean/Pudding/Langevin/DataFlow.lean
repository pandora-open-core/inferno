import Pudding.Transport.Finite

namespace Pudding.Langevin
open scoped BigOperators

/-- Frozen augmented-state transition; includes pool and observation ordering. -/
structure Kernel (I J : Type*) [Fintype I] [Fintype J] where
  transition : I → J → ℝ
  nonneg : ∀ i j, 0 ≤ transition i j
  total : ∀ i, ∑ j, transition i j = 1

noncomputable def Kernel.push {I J : Type*} [Fintype I] [Fintype J]
    (k : Kernel I J) (p : Transport.Probability I) : Transport.Probability J where
  mass j := ∑ i, p.mass i * k.transition i j
  nonneg j := Finset.sum_nonneg fun i _ => mul_nonneg (p.nonneg i) (k.nonneg i j)
  total := by
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, k.total, mul_one]
    exact p.total

/-- Two stochastic legs preserve probability, without assuming stationarity. -/
theorem loop_mass {I J : Type*} [Fintype I] [Fintype J]
    (outward : Kernel I J) (feedback : Kernel J I) (p : Transport.Probability I) :
    ∑ i, (feedback.push (outward.push p)).mass i = 1 :=
  (feedback.push (outward.push p)).total

/-- Algebraic endpoint optimality transfer from an independently established
KL Pythagorean identity. Analytic disintegration and integrability are external. -/
theorem bridge_gap_nonneg (candidate competitor divergence : ℝ)
    (identity : competitor = divergence + candidate) (h : 0 ≤ divergence) :
    candidate ≤ competitor := by
  rw [identity]
  exact le_add_of_nonneg_left h

/-- Zero gap forces zero divergence; uniqueness also needs KL separation. -/
theorem bridge_zero_gap (candidate competitor divergence : ℝ)
    (identity : competitor = divergence + candidate) (h : competitor = candidate) :
    divergence = 0 := by
  have hc : divergence + candidate = 0 + candidate := by
    rw [← identity, h, zero_add]
  exact add_right_cancel hc

end Pudding.Langevin
