import Pudding.Transport.Gauge

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

noncomputable def Tf (b : J → ℝ) (cost : I → J → ℝ) (ε : ℝ) (g : J → ℝ) (i : I) : ℝ :=
  -ε * Real.log (∑ j, b j * Real.exp ((g j - cost i j) / ε))
noncomputable def Tg (a : I → ℝ) (cost : I → J → ℝ) (ε : ℝ) (f : I → ℝ) (j : J) : ℝ :=
  -ε * Real.log (∑ i, a i * Real.exp ((f i - cost i j) / ε))

def relax (r : ℝ) (u v : Potentials I J) : Potentials I J :=
  ⟨fun i => (1-r) * u.f i + r * v.f i, fun j => (1-r) * u.g j + r * v.g j⟩

noncomputable def rawUpdate (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε r : ℝ) (u : Potentials I J) : Potentials I J :=
  relax r u ⟨Tf b cost ε u.g, Tg a cost ε u.f⟩

noncomputable def mean (f : I → ℝ) : ℝ := (∑ i, f i) / Fintype.card I

noncomputable def step (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε r : ℝ) (u : Potentials I J) : Potentials I J :=
  let v := rawUpdate a b cost ε r u
  gauge (mean v.f) v

theorem step_plan (a : I → ℝ) (b : J → ℝ) (cost : I → J → ℝ)
    (ε r : ℝ) (u : Potentials I J) :
    plan a b cost ε (step a b cost ε r u) = plan a b cost ε (rawUpdate a b cost ε r u) :=
  gauge_invariant _ _ _ _ _ _

/-- Code checks the residual BEFORE applying relaxation. No success after budget exhaustion. -/
inductive SolveResult (I J : Type*) where
  | success (state : Potentials I J) (iterations : ℕ) (residual : ℝ)
  | invalidInput | nonfinite | budgetExhausted

end Pudding.Transport
