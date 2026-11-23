import Pudding.Transport.Uniqueness
import Pudding.Transport.Anchor

namespace Pudding.Transport
open scoped BigOperators
variable {I L : Type*} [Fintype I] [Fintype L]

/-- Finite squared W2 cost for a DECLARED squared ground metric.
    The metric laws (in particular intrinsic sphere distance) are separate evidence.
    Unlike the entropic bridge objective, this contains no KL regularisation. -/
noncomputable def wassersteinSq (groundSq : I → I → ℝ) (a b : Probability I) : ℝ :=
  sInf {v | ∃ p, Feasible a.mass b.mass p ∧ v = ∑ i, ∑ j, groundSq i j * p i j}

noncomputable def barycenterObjective (D : Probability I → Probability I → ℝ)
    (weights : Probability L) (samples : L → Probability I) (centre : Probability I) : ℝ :=
  ∑ l, weights.mass l * D centre (samples l)

def IsBarycenter (D : Probability I → Probability I → ℝ)
    (weights : Probability L) (samples : L → Probability I) (centre : Probability I) : Prop :=
  ∀ other, barycenterObjective D weights samples centre ≤ barycenterObjective D weights samples other

/-- Shape: K atoms × T bins, each column a probability, not a doubly stochastic map. -/
structure MassField (K T : ℕ) where
  column : Fin T → Probability (Fin K)

/-- Frozen n-active-atom support. A changing centre is still a measure on these atoms. -/
structure DynamicCentre (n : ℕ) where
  support : Fin n → SphereAtom
  atTime : ℕ → Probability (Fin n)

theorem dynamic_no_exit {n : ℕ} (c : DynamicCentre n) (t : ℕ) :
    (∀ i, 0 ≤ (c.atTime t).mass i) ∧ (∑ i, (c.atTime t).mass i = 1) :=
  ⟨(c.atTime t).nonneg, (c.atTime t).total⟩

/-- Static anchor and dynamic centre are separate data, with hard admissibility.
    Choosing a reference kernel from them still requires a model-specific map. -/
structure AnchoredGeometry (n : ℕ) where
  centre : DynamicCentre n
  anchor : Fin n → Fin n → ℝ
  balanced : DoublyStochastic anchor
  referenceCost : (Fin n → Fin n → ℝ) → Probability (Fin n) → Fin n → Fin n → ℝ

noncomputable def dynamicCost {n : ℕ} (g : AnchoredGeometry n) (t : ℕ) : Fin n → Fin n → ℝ :=
  g.referenceCost g.anchor (g.centre.atTime t)

/-- No rectangular row/column-one map exists unless the cardinalities agree. -/
theorem balanced_cardinalities {J : Type*} [Fintype J] (p : I → J → ℝ)
    (hrow : ∀ i, ∑ j, p i j = 1) (hcol : ∀ j, ∑ i, p i j = 1) :
    Fintype.card I = Fintype.card J := by
  have h : (Fintype.card I : ℝ) = Fintype.card J := by
    calc
      (Fintype.card I : ℝ) = ∑ i, ∑ j, p i j := by simp [hrow]
      _ = ∑ j, ∑ i, p i j := Finset.sum_comm
      _ = Fintype.card J := by simp [hcol]
  exact_mod_cast h

end Pudding.Transport
