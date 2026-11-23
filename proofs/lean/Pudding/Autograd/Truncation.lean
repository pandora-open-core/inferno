import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Tactic.Linarith
namespace Pudding.Autograd
variable {X : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
/-- X is the gauge complement, not the full potential space. A positive gap
    bound proves invertibility in finite dimension without assuming an inverse. -/
theorem restricted_bijective [FiniteDimensional ℝ X] (S : X →ₗ[ℝ] X)
    (gamma : ℝ) (hg : 0 < gamma)
    (hgap : ∀ x, gamma * ‖x‖ ≤ ‖S x‖) : Function.Bijective S := by
  have hi : Function.Injective S := by
    intro x y hxy
    apply sub_eq_zero.mp
    apply norm_eq_zero.mp
    have h := hgap (x-y)
    rw [map_sub, hxy, sub_self, norm_zero] at h
    have hn := norm_nonneg (x-y)
    nlinarith
  exact ⟨hi, LinearMap.injective_iff_surjective.mp hi⟩

/-- Krylov truncation certificate once a positive gauge-complement gap is
    established. Projected residual and solution must be in the same space X. -/
theorem truncated_solve_error (S : X →ₗ[ℝ] X) (gamma : ℝ) (hg : 0 < gamma)
    (hgap : ∀ x, gamma * ‖x‖ ≤ ‖S x‖)
    (approx exact rhs : X) (hexact : S exact = rhs) :
    ‖approx-exact‖ ≤ ‖S approx-rhs‖ / gamma := by
  have h := hgap (approx-exact)
  rw [map_sub, hexact] at h
  apply (le_div_iff₀ hg).mpr
  nlinarith
end Pudding.Autograd
