import Pudding.Transport.Problem

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

structure Potentials (I J : Type*) where
  f : I → ℝ
  g : J → ℝ

noncomputable def plan (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε : ℝ) (u : Potentials I J) (i : I) (j : J) : ℝ :=
  a i * b j * Real.exp ((u.f i + u.g j - cost i j) / ε)

def gauge (s : ℝ) (u : Potentials I J) : Potentials I J :=
  ⟨fun i => u.f i - s, fun j => u.g j + s⟩

theorem gauge_invariant (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε s : ℝ) (u : Potentials I J) :
    plan a b cost ε (gauge s u) = plan a b cost ε u := by
  funext i j; unfold plan gauge; congr 2; congr 1; ring

theorem plan_positive (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε : ℝ) (u : Potentials I J) (ha : ∀ i, 0 < a i) (hb : ∀ j, 0 < b j) :
    ∀ i j, 0 < plan a b cost ε u i j := by
  intro i j; exact mul_pos (mul_pos (ha i) (hb j)) (Real.exp_pos _)

end Pudding.Transport
