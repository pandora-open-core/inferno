import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

structure Probability (I : Type*) [Fintype I] where
  mass : I → ℝ
  nonneg : ∀ i, 0 ≤ mass i
  total : ∑ i, mass i = 1

def Feasible (a : I → ℝ) (b : J → ℝ) (p : I → J → ℝ) : Prop :=
  (∀ i j, 0 ≤ p i j) ∧ (∀ i, ∑ j, p i j = a i) ∧ (∀ j, ∑ i, p i j = b j)


end Pudding.Transport
