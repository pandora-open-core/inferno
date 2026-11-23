import Mathlib.LinearAlgebra.Basis.Defs
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Data.Real.Basic
namespace Pudding.Autograd
/-- Entrywise weighted similarity: R=P/a, A=P/sqrt(ab). sa and sb
    represent positive square-root marginals, with a=sa^2 and b=sb^2. -/
theorem weighted_similarity_entry (p sa sb dg : ℝ) (ha : sa ≠ 0) (hb : sb ≠ 0) :
    sa * ((p / sa^2) * dg) = (p / (sa*sb)) * (sb*dg) := by
  field_simp [ha, hb]
  <;> ring

section Schur
variable {U V : Type*} [AddCommGroup U] [Module ℝ U] [AddCommGroup V] [Module ℝ V]
/-- A is the plan scaled by its ACTUAL positive marginals. B is its transpose.
    The plus signs are those of I-DT for the simultaneous softmin map. -/
def schur (A : V →ₗ[ℝ] U) (B : U →ₗ[ℝ] V) (u : U) : U := u - A (B u)
/-- Exact elimination, valid before any invertibility claim. -/
theorem schur_elimination (A : V →ₗ[ℝ] U) (B : U →ₗ[ℝ] V)
    (u : U) (v s : V) (r : U) :
    (u + A v = r ∧ v + B u = s) ↔
      (schur A B u = r - A s ∧ v = s - B u) := by
  constructor
  · rintro ⟨h1, h2⟩
    have hv : v = s - B u := eq_sub_iff_add_eq.mpr h2
    refine ⟨?_, hv⟩
    rw [hv, map_sub] at h1
    dsimp [schur]
    rw [← h1]
    abel
  · rintro ⟨h1, hv⟩
    subst v
    constructor
    · rw [map_sub]
      dsimp [schur] at h1
      rw [← sub_add_cancel r (A s), ← h1]
      abel
    · abel
/-- The gauge must be removed: the full Schur operator has a null vector. -/
theorem schur_gauge (A : V →ₗ[ℝ] U) (B : U →ₗ[ℝ] V)
    (a : U) (b : V) (ha : A b = a) (hb : B a = b) : schur A B a = 0 := by
  simp [schur, hb, ha]
/-- Relaxation cancels from the tangent equation when its scalar is nonzero. -/
theorem relaxation_cancels (r : ℝ) (hr : r ≠ 0) (x y : U) :
    r • x = r • y ↔ x = y := by
  constructor
  · intro h
    have := congrArg (fun z : U => r⁻¹ • z) h
    simpa [smul_smul, hr] using this
  · exact congrArg (fun z => r • z)
end Schur
/-- Abstract adjoint identity: after solving M* lambda = cotangent, the
    parameter VJP is B* lambda. Existence requires a gauge-restricted inverse. -/
theorem implicit_vjp {U W : Type*} (pairU : U → U → ℝ) (pairW : W → W → ℝ)
    (M Mt : U → U) (B : W → U) (Bt : U → W)
    (hM : ∀ x y, pairU y (M x) = pairU (Mt y) x)
    (hB : ∀ x y, pairU y (B x) = pairW (Bt y) x)
    (dx : W) (du lam cot : U) (ht : M du = B dx) (ha : Mt lam = cot) :
    pairU cot du = pairW (Bt lam) dx := by
  rw [← ha, ← hM, ht, hB]
end Pudding.Autograd
