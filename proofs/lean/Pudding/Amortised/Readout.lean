import Pudding.Amortised.Carrier
import Mathlib.Topology.MetricSpace.Pseudo.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Ring.List
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
namespace Pudding.Amortised
/-- H(E(x)), with separate encoder and decoder heads. Noise is an explicit input
    of H when sampling is selected; eval mode fixes it rather than resampling. -/
def thetaHat {X Y : Type*} (E : X → Carrier n) (H : Carrier n → Y) (x : X) : Y := H (E x)

theorem positive_transform (floor raw : ℝ) (hf : 0 ≤ floor) :
    0 < floor + Real.exp raw := add_pos_of_nonneg_of_pos hf (Real.exp_pos raw)

/-- Stability on an admissible region; it must be calibrated for the chosen
    transforms, concentrations, resticking and coregionalisation heads. -/
theorem readout_transfer {X Y : Type*} [PseudoMetricSpace X] [PseudoMetricSpace Y]
    (H : X → Y) (region : Set X) (K : ℝ) (x y : X) (hx : x ∈ region) (hy : y ∈ region)
    (stable : ∀ a ∈ region, ∀ b ∈ region, dist (H a) (H b) ≤ K * dist a b) :
    dist (H x) (H y) ≤ K * dist x y := stable x hx y hy

/-- Real semantics of the concentration head after smooth clamping. -/
theorem positive_softplus (eps raw : ℝ) (he : 0 ≤ eps) :
    0 < Real.log (1 + Real.exp raw) + eps := by
  have hp : 0 < Real.log (1 + Real.exp raw) := Real.log_pos (by linarith [Real.exp_pos raw])
  exact add_pos_of_pos_of_nonneg hp he

/-- Stick fractions have already been produced by the selected sampler/global
    beta rule. This normalisation result does not verify that sampler. -/
def restick : List ℝ → List ℝ
  | [] => [1]
  | v :: vs => v :: (restick vs).map (fun p => (1-v)*p)

theorem restick_sum (vs : List ℝ) : (restick vs).sum = 1 := by
  induction vs with
  | nil => simp [restick]
  | cons v vs ih =>
      simp only [restick, List.sum_cons]
      rw [List.sum_map_mul_left (restick vs) (fun p : ℝ => p) (1-v)]
      simp only [List.map_id_fun', id_eq, ih]
      ring

theorem restick_nonneg (vs : List ℝ) (hv : ∀ v ∈ vs, 0 ≤ v ∧ v ≤ 1) :
    ∀ p ∈ restick vs, 0 ≤ p := by
  induction vs with
  | nil => simp [restick]
  | cons v vs ih =>
      intro p hp
      simp only [restick, List.mem_cons, List.mem_map] at hp
      rcases hp with hEq | ⟨q, hq, rfl⟩
      · simpa only [hEq] using (hv v (by simp)).1
      · exact mul_nonneg (sub_nonneg.mpr (hv v (by simp)).2)
          (ih (fun q hq => hv q (by simp [hq])) q hq)
end Pudding.Amortised
