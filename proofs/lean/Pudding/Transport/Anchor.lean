import Pudding.Transport.Problem

namespace Pudding.Transport
open scoped BigOperators
variable {I S : Type*} [Fintype I] [Fintype S]

/-- Convex mixtures of balanced sparse maps are balanced. Sparsity alone is not the proof. -/
theorem anchor_balanced (w : Probability S) (p : S → I → I → ℝ)
    (hp : ∀ s, DoublyStochastic (p s)) :
    DoublyStochastic (fun i j => ∑ s, w.mass s * p s i j) := by
  refine ⟨?_, ?_, ?_⟩
  · intro i j; exact Finset.sum_nonneg (fun s _ => mul_nonneg (w.nonneg s) ((hp s).1 i j))
  · intro i
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, (fun s => (hp s).2.1 i)]
    simpa using w.total
  · intro j
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, (fun s => (hp s).2.2 j)]
    simpa using w.total

/-- Exact sparse construction: every permutation contributes one nonzero per row/column. -/
noncomputable def permutationMap [DecidableEq I] (σ : Equiv.Perm I) (i j : I) : ℝ :=
  if σ i = j then 1 else 0

theorem permutation_balanced [DecidableEq I] (σ : Equiv.Perm I) :
    DoublyStochastic (permutationMap σ) := by
  refine ⟨?_, ?_, ?_⟩
  · intro i j; simp only [permutationMap]; split <;> norm_num
  · intro i; simp [permutationMap]
  · intro j
    have : (fun i => if σ i = j then (1 : ℝ) else 0) =
        (fun i => if i = σ.symm j then 1 else 0) := by
      funext i
      simp [Equiv.apply_eq_iff_eq_symm_apply]
    simp only [permutationMap]; rw [this]; simp

/-- Outside the union of the selected permutations the anchor is exactly zero. -/
theorem anchor_sparse [DecidableEq I] (w : Probability S) (σ : S → Equiv.Perm I)
    (i j : I) (h : ∀ s, σ s i ≠ j) :
    ∑ s, w.mass s * permutationMap (σ s) i j = 0 := by
  simp [permutationMap, h]

end Pudding.Transport
