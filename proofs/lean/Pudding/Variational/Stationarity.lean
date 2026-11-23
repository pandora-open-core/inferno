import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Push
namespace Pudding.Variational
open scoped BigOperators
/-- Smoothness descent with a biased direction g+e, step 1/L. g2 and e2
    denote squared norms and cross is their inner product. -/
theorem smooth_biased_step (f next g2 e2 cross L : ℝ) (hL : 0 < L)
    (hsmooth : next ≤ f - (g2+cross)/L +
      L/2 * ((g2+2*cross+e2)/(L*L))) :
    (1/(2*L))*g2 ≤ f-next+(1/(2*L))*e2 := by
  have hL0 : L ≠ 0 := ne_of_gt hL
  have hid : f - (g2+cross)/L + L/2*((g2+2*cross+e2)/(L*L)) =
      f-g2/(2*L)+e2/(2*L) := by field_simp; ring
  rw [hid] at hsmooth
  simp only [div_eq_mul_inv, one_mul] at hsmooth ⊢
  nlinarith

/-- Finite-horizon biased descent bound. The per-step inequality must be
    derived from smoothness and measured gradient errors for the selected run.
    g2 denotes squared gradient norm; err includes QMC/IFT/numerical bias. -/
theorem descent_budget (f g2 err : ℕ → ℝ) (c lower : ℝ) (N : ℕ)
    (hstep : ∀ k, c*g2 k ≤ f k-f (k+1)+err k)
    (hlower : lower ≤ f N) :
    c * (∑ k ∈ Finset.range N, g2 k) ≤
      f 0-lower + ∑ k ∈ Finset.range N, err k := by
  have htel : ∀ n, c * (∑ k ∈ Finset.range n, g2 k) ≤
      f 0-f n + ∑ k ∈ Finset.range n, err k := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      simp only [Finset.sum_range_succ]
      have hs := hstep n
      nlinarith
  have := htel N
  linarith
/-- A small sum budget forces an iterate with a small squared gradient. -/
theorem exists_stationary_iterate (g2 : ℕ → ℝ) (N : ℕ) (hN : 0 < N)
    (bound : ℝ) (h : (∑ k ∈ Finset.range N, g2 k) ≤ (N : ℝ)*bound) :
    ∃ k ∈ Finset.range N, g2 k ≤ bound := by
  by_contra hn
  push_neg at hn
  have hs : (∑ k ∈ Finset.range N, bound) < ∑ k ∈ Finset.range N, g2 k :=
    Finset.sum_lt_sum_of_nonempty (by simpa using (Nat.ne_of_gt hN)) (fun k hk => hn k hk)
  have hs' : (N : ℝ)*bound < ∑ k ∈ Finset.range N, g2 k := by simpa using hs
  exact (not_lt_of_ge h) hs'
end Pudding.Variational
