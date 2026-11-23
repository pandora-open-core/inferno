import Pudding.Amortised.Readout
import Pudding.Variational.Stationarity
import Mathlib.Tactic.Linarith
namespace Pudding.Amortised
open scoped BigOperators
structure Budget where
  approximation : ℝ
  optimisation : ℝ
  bridge : ℝ
  discretisation : ℝ
  numerical : ℝ
  surrogate : ℝ

def Budget.total (b : Budget) : ℝ :=
  b.approximation + b.optimisation + b.bridge + b.discretisation + b.numerical + b.surrogate

/-- Six actual intermediate predictions expose the provenance of each error.
    This is pointwise, deterministic and conditional; no identifiability claim. -/
theorem prediction_budget {Y : Type*} [PseudoMetricSpace Y] (p : Fin 7 → Y) (b : Budget)
    (h0 : dist (p 0) (p 1) ≤ b.approximation)
    (h1 : dist (p 1) (p 2) ≤ b.optimisation)
    (h2 : dist (p 2) (p 3) ≤ b.bridge)
    (h3 : dist (p 3) (p 4) ≤ b.discretisation)
    (h4 : dist (p 4) (p 5) ≤ b.numerical)
    (h5 : dist (p 5) (p 6) ≤ b.surrogate) :
    dist (p 0) (p 6) ≤ b.total := by
  have t1 := dist_triangle (p 0) (p 1) (p 2)
  have t2 := dist_triangle (p 0) (p 2) (p 3)
  have t3 := dist_triangle (p 0) (p 3) (p 4)
  have t4 := dist_triangle (p 0) (p 4) (p 5)
  have t5 := dist_triangle (p 0) (p 5) (p 6)
  dsimp [Budget.total]
  linarith

/-- An engineering calibration connecting squared gradient to prediction error
    is an EXTRA hypothesis. Stationarity alone cannot supply it. -/
theorem calibrated_stationary_prediction (g2 : ℕ → ℝ) (N : ℕ) (hN : 0 < N)
    (bound K : ℝ) (hK : 0 ≤ K) (err : ℕ → ℝ)
    (calibration : ∀ k ∈ Finset.range N, err k ≤ K * g2 k)
    (training : (∑ k ∈ Finset.range N, g2 k) ≤ (N : ℝ)*bound) :
    ∃ k ∈ Finset.range N, err k ≤ K*bound := by
  obtain ⟨k, hk, hg⟩ := Variational.exists_stationary_iterate g2 N hN bound training
  exact ⟨k, hk, (calibration k hk).trans (mul_le_mul_of_nonneg_left hg hK)⟩

/-- Deterministic tracking in the prediction metric. ref may change with time;
    no claim of convergence to one fixed theta, or uniformity over inputs. -/
theorem prediction_tracking {Y : Type*} [PseudoMetricSpace Y]
    (pred ref : ℕ → Y) (b : ℕ → Budget)
    (bounded : ∀ k, dist (pred k) (ref k) ≤ (b k).total)
    (vanishing : ∀ ε : ℝ, 0 < ε → ∃ N, ∀ k ≥ N, (b k).total < ε) :
    ∀ ε : ℝ, 0 < ε → ∃ N, ∀ k ≥ N, dist (pred k) (ref k) < ε := by
  intro ε hε
  obtain ⟨N, hN⟩ := vanishing ε hε
  exact ⟨N, fun k hk => lt_of_le_of_lt (bounded k) (hN k hk)⟩
end Pudding.Amortised
