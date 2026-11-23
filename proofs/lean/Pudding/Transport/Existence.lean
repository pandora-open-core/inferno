import Pudding.Transport.Problem
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.Instances.Real.Lemmas

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

/-- Coordinate bounds follow from positive mass and fixed probability row sums. -/
theorem feasible_entry_le_one (a : Probability I) (b : Probability J)
    (p : I → J → ℝ) (hp : Feasible a.mass b.mass p) (i : I) (j : J) : p i j ≤ 1 := by
  have hrow : p i j ≤ ∑ k, p i k := Finset.single_le_sum (fun k _ => hp.1 i k) (Finset.mem_univ j)
  have ha : a.mass i ≤ ∑ k, a.mass k := Finset.single_le_sum (fun k _ => a.nonneg k) (Finset.mem_univ i)
  rw [hp.2.1 i] at hrow
  rw [a.total] at ha
  exact hrow.trans ha

theorem feasible_closed (a : Probability I) (b : Probability J) :
    IsClosed {p : I → J → ℝ | Feasible a.mass b.mass p} := by
  have hn : IsClosed {p : I → J → ℝ | ∀ i j, 0 ≤ p i j} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter (fun i => isClosed_iInter (fun j =>
      isClosed_le continuous_const ((continuous_apply j).comp (continuous_apply i))))
  have hr : IsClosed {p : I → J → ℝ | ∀ i, ∑ j, p i j = a.mass i} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter (fun i => isClosed_eq
      (continuous_finset_sum _ (fun j _ => (continuous_apply j).comp (continuous_apply i))) continuous_const)
  have hc : IsClosed {p : I → J → ℝ | ∀ j, ∑ i, p i j = b.mass j} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter (fun j => isClosed_eq
      (continuous_finset_sum _ (fun i _ => (continuous_apply j).comp (continuous_apply i))) continuous_const)
  exact hn.inter (hr.inter hc)

theorem feasible_compact (a : Probability I) (b : Probability J) :
    IsCompact {p : I → J → ℝ | Feasible a.mass b.mass p} := by
  apply IsCompact.of_isClosed_subset (s := Set.Icc (fun _ _ => (0 : ℝ)) (fun _ _ => (1 : ℝ)))
    isCompact_Icc (feasible_closed a b)
  intro p hp
  exact ⟨hp.1, fun i j => feasible_entry_le_one a b p hp i j⟩

/-- The zero convention agrees with Lean's x log x, including x=0. -/
theorem klTerm_eq (x q : ℝ) : klTerm x q = x * Real.log (x/q) - x + q := by
  unfold klTerm; split_ifs with h
  · subst x; simp
  · rfl

theorem continuous_klTerm (q : ℝ) (hq : 0 < q) : Continuous (fun x => klTerm x q) := by
  have heq : (fun x => klTerm x q) =
      (fun x => q * ((x/q) * Real.log (x/q)) - x + q) := by
    funext x; rw [klTerm_eq]; field_simp <;> ring
  rw [heq]
  exact ((continuous_const.mul (Real.continuous_mul_log.comp
    (continuous_id.div_const q))).sub continuous_id).add continuous_const

theorem objective_continuous (a : Probability I) (b : Probability J)
    (cost : I → J → ℝ) (ε : ℝ) (ha : ∀ i, 0 < a.mass i) (hb : ∀ j, 0 < b.mass j) :
    Continuous (objective a b cost ε) := by
  unfold objective
  apply continuous_finset_sum
  intro i _
  apply continuous_finset_sum
  intro j _
  have hp : Continuous (fun p : I → J → ℝ => p i j) :=
    (continuous_apply j).comp (continuous_apply i)
  have hc : Continuous (fun p : I → J → ℝ => cost i j*p i j) := continuous_const.mul hp
  exact hc.add (continuous_const.mul
    ((continuous_klTerm _ (mul_pos (ha i) (hb j))).comp hp))

/-- A finite balanced entropy problem has a minimiser, including boundary couplings. -/
theorem minimiser_exists (a : Probability I) (b : Probability J) (cost : I → J → ℝ)
    (ε : ℝ) (ha : ∀ i, 0 < a.mass i) (hb : ∀ j, 0 < b.mass j) :
    ∃ p, Minimises a b cost ε p := by
  obtain ⟨p, hp, hmin⟩ := (feasible_compact a b).exists_isMinOn
    ⟨_, product_feasible a b⟩ (objective_continuous a b cost ε ha hb).continuousOn
  exact ⟨p, hp, hmin⟩

end Pudding.Transport
