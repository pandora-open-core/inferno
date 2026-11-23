import Pudding.Transport.Finite
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Push

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]


theorem product_feasible (a : Probability I) (b : Probability J) :
    Feasible a.mass b.mass (fun i j => a.mass i * b.mass j) := by
  refine ⟨fun i j => mul_nonneg (a.nonneg i) (b.nonneg j), ?_, ?_⟩
  · intro i; simp [← Finset.mul_sum, b.total]
  · intro j; simp [← Finset.sum_mul, a.total]

theorem total_mass {a : Probability I} {b : Probability J} {p : I → J → ℝ}
    (hp : Feasible a.mass b.mass p) : ∑ i, ∑ j, p i j = 1 := by
  simp_rw [hp.2.1]; exact a.total

/-- Generalised KL, including the zero convention. q is positive in the problem. -/
noncomputable def klTerm (x q : ℝ) : ℝ :=
  (if x = 0 then 0 else x * Real.log (x / q)) - x + q

noncomputable def objective (a : Probability I) (b : Probability J)
    (cost : I → J → ℝ) (ε : ℝ) (p : I → J → ℝ) : ℝ :=
  ∑ i, ∑ j, (cost i j * p i j + ε * klTerm (p i j) (a.mass i * b.mass j))

def Minimises (a : Probability I) (b : Probability J) (cost : I → J → ℝ)
    (ε : ℝ) (p : I → J → ℝ) : Prop :=
  Feasible a.mass b.mass p ∧ ∀ q, Feasible a.mass b.mass q →
    objective a b cost ε p ≤ objective a b cost ε q

/-- Both marginals are part of this definition; column normalisation alone is insufficient. -/
def DoublyStochastic (p : I → I → ℝ) : Prop :=
  Feasible (fun _ => 1) (fun _ => 1) p

theorem uniform_scaling {N : ℝ} (hN : N ≠ 0) {p : I → I → ℝ}
    (hp : Feasible (fun _ => 1 / N) (fun _ => 1 / N) p) (hpos : 0 ≤ N) :
    DoublyStochastic (fun i j => N * p i j) := by
  refine ⟨fun i j => mul_nonneg hpos (hp.1 i j), ?_, ?_⟩
  · intro i; rw [← Finset.mul_sum, hp.2.1]; field_simp
  · intro j; rw [← Finset.mul_sum, hp.2.2]; field_simp

/-- A sphere surface has intrinsic dimension two, represented by three coordinates. -/
structure SphereAtom where
  coord : Fin 3 → ℝ
  unit : ∑ d, coord d ^ 2 = 1

/-- The support is frozen; only probability weights evolve. No coordinate interpolation. -/
structure SparseSphere (n : ℕ) where
  atoms : Fin n → SphereAtom
  measure : Probability (Fin n)

noncomputable def mix (a b : Probability I) (t : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1) : Probability I where
  mass i := (1-t) * a.mass i + t * b.mass i
  nonneg i := add_nonneg (mul_nonneg (sub_nonneg.mpr ht1) (a.nonneg i)) (mul_nonneg ht (b.nonneg i))
  total := by simp [Finset.sum_add_distrib, ← Finset.mul_sum, a.total, b.total]

/-- This is mass conservation on the support, not a theorem about an SDE remaining on a sphere. -/
theorem mix_no_mass_exit (a b : Probability I) (t : ℝ) (ht : 0 ≤ t) (ht1 : t ≤ 1) :
    ∑ i, (mix a b t ht ht1).mass i = 1 := (mix a b t ht ht1).total

end Pudding.Transport
