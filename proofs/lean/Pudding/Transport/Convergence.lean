import Pudding.Transport.Residual
import Mathlib.Topology.MetricSpace.Contracting

namespace Pudding.Transport
open scoped BigOperators Topology

/-- The implemented simultaneous recurrence on ONE active atom, arbitrary finite cost.
    This is a restricted production recurrence, not alternating Sinkhorn. -/
theorem singleton_update (ε r c f g : ℝ) (hε : ε ≠ 0) :
    rawUpdate (fun _ : Fin 1 => 1) (fun _ : Fin 1 => 1) (fun _ _ => c) ε r
      ⟨fun _ => f, fun _ => g⟩ =
    ⟨fun _ => (1-r)*f + r*(c-g), fun _ => (1-r)*g + r*(c-f)⟩ := by
  have h (x : ℝ) : -ε * ((x-c)/ε) = c-x := by field_simp <;> ring
  simp [rawUpdate, relax, Tf, Tg, Real.log_exp, h]

/-- After gauge fixing f=0, the remaining scalar error has factor 1-2r. -/
theorem singleton_error (r c g : ℝ) :
    ((1-r)*g + r*c + r*(c-g)) - c = (1-2*r)*(g-c) := by ring

theorem relaxation_contracts (r : ℝ) (h0 : 0 < r) (h1 : r < 1) : |1-2*r| < 1 := by
  rw [abs_lt]; constructor <;> linarith

/-- Closed form tends to zero for every admissible production relaxation. -/
theorem singleton_error_converges (r e : ℝ) (h0 : 0 < r) (h1 : r < 1) :
    Filter.Tendsto (fun n : ℕ => (1-2*r)^n * e) Filter.atTop (nhds 0) := by
  have h := tendsto_pow_atTop_nhds_zero_of_abs_lt_one (relaxation_contracts r h0 h1)
  simpa using h.mul_const e

/-- Independently useful error envelope: a local one-step contraction gives a geometric bound.
    Establishing such a bound for a general bridge quotient remains an open obligation. -/
theorem geometric_envelope (e : ℕ → ℝ) (q : ℝ) (hq : 0 ≤ q)
    (hstep : ∀ n, e (n+1) ≤ q * e n) : ∀ n, e n ≤ q^n * e 0 := by
  intro n; induction n with
  | zero => simp
  | succ n ih =>
    calc
      e (n+1) ≤ q * e n := hstep n
      _ ≤ q * (q^n * e 0) := mul_le_mul_of_nonneg_left ih hq
      _ = q^(n+1) * e 0 := by rw [pow_succ]; ring

/-- A genuine conditional restriction: prove a Lipschitz constant <1 on a
    complete gauge representation, then this gives existence, uniqueness,
    convergence and an a-posteriori error certificate. The hypothesis is
    one-step analytic control, never the desired convergence conclusion. -/
theorem convergence_under_contraction {X : Type*} [MetricSpace X] [CompleteSpace X]
    [Nonempty X] (F : X → X) (q : NNReal) (hq : q < 1) (hF : LipschitzWith q F) :
    ∃ z, F z = z ∧ (∀ y, F y = y → y = z) ∧
      (∀ x, Filter.Tendsto (fun n => F^[n] x) Filter.atTop (nhds z)) ∧
      (∀ x, dist x z ≤ dist x (F x) / (1 - (q : ℝ))) := by
  let h : ContractingWith q F := ⟨hq, hF⟩
  refine ⟨h.fixedPoint F, h.fixedPoint_isFixedPt, ?_, ?_, ?_⟩
  · intro y hy; exact h.fixedPoint_unique hy
  · intro x; exact h.tendsto_iterate_fixedPoint x
  · intro x; exact h.dist_fixedPoint_le x

end Pudding.Transport
