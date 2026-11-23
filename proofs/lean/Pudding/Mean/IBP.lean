import Pudding.Mean.Projection

noncomputable section
namespace Pudding.Mean
open scoped BigOperators

/-- Exact real-arithmetic counterpart of gp::barycenter::ibp_step, including
    direct assignment and the old d used in elog_b. This is not balanced Jacobi. -/
structure IBPState (T K : ℕ) where
  f : Fin T → Fin K → ℝ
  g : Fin T → Fin K → ℝ
  elogB : Fin K → ℝ
  d : Option (Fin K → ℝ)

noncomputable def softAggregate {K : ℕ} (ε : ℝ) (v : Fin K → ℝ) : ℝ :=
  ε * Real.log (∑ k, Real.exp (v k / ε))

noncomputable def ibpStep {T K : ℕ} (logA : Fin T → Fin K → ℝ)
    (weights : Fin T → ℝ) (negativeCost : Fin K → Fin K → ℝ)
    (ε τ1 τ2 leak : ℝ) (st : IBPState T K) : IBPState T K :=
  let t1 := τ1 / (τ1+ε)
  let t2 := τ2 / (τ2+ε)
  let f := fun t k => t1 * (ε*logA t k - softAggregate ε (fun j => st.g t j+negativeCost k j))
  let h := fun t k => softAggregate ε (fun j => f t j+negativeCost j k)
  let elog := fun k => (∑ t, weights t*h t k) + (st.d.map (fun d => d k)).getD 0
  let d := st.d.map (fun old k => (1-leak)/2 * (old k+t2*(elog k -
    softAggregate ε (fun j => old j+negativeCost k j))))
  ⟨f, fun t k => t2*(elog k-h t k), elog, d⟩

/-- Code reduces the last cost axis for the d channel; use a symmetric negative
    cost (as produced by squared distances minus diagonal jitter). -/
def SymmetricCost {K : ℕ} (c : Fin K → Fin K → ℝ) : Prop := ∀ i j, c i j = c j i

def ibpRelax {T K : ℕ} (r : ℝ) (old new : IBPState T K) : IBPState T K :=
  let blend := fun x y : ℝ => (1-r)*x+r*y
  ⟨fun t k => blend (old.f t k) (new.f t k),
   fun t k => blend (old.g t k) (new.g t k),
   fun k => blend (old.elogB k) (new.elogB k),
   match old.d, new.d with
   | some a, some b => some (fun k => blend (a k) (b k))
   | _, _ => none⟩

/-- The production mean checks changes AFTER this full-state relaxation,
    unlike dk-descent's pre-update unrelaxed check. -/
noncomputable def meanStep {T K : ℕ} (logA : Fin T → Fin K → ℝ)
    (weights : Fin T → ℝ) (negativeCost : Fin K → Fin K → ℝ)
    (ε τ1 τ2 leak relaxation : ℝ) (st : IBPState T K) : IBPState T K :=
  ibpRelax relaxation st (ibpStep logA weights negativeCost ε τ1 τ2 leak st)

theorem unbalanced_damping (ε τ : ℝ) (hε : 0 < ε) (hτ : 0 < τ) :
    0 < τ/(τ+ε) ∧ τ/(τ+ε) < 1 := by
  exact ⟨div_pos hτ (add_pos hτ hε), (div_lt_one (add_pos hτ hε)).mpr (by linarith)⟩

/-- Normalised readout preserves mass even if the iterative state has not converged.
    This is an admissibility result, not a bridge optimality certificate. -/
noncomputable def softmaxProbability {A : Type*} [Fintype A] [Nonempty A]
    (v : A → ℝ) : Transport.Probability A := by
  have hz : 0 < ∑ i, Real.exp (v i) := Finset.sum_pos (fun i _ => Real.exp_pos (v i)) Finset.univ_nonempty
  exact ⟨fun i => Real.exp (v i)/(∑ j, Real.exp (v j)),
    fun i => (div_pos (Real.exp_pos _) hz).le,
    by simp only [div_eq_mul_inv, ← Finset.sum_mul]; exact mul_inv_cancel₀ hz.ne'⟩

/-- Source-level defaults, not authenticated live runtime settings. -/
structure MeanDefaults where
  atoms : ℕ := 30
  timeBins : ℕ := 12
  sparseLatents : ℕ := 8
  atomCoordinates : ℕ := 2
  epsilonInit : ℝ := 0.05
  tau1Init : ℝ := 1
  tau2Init : ℝ := 1
  debiasLeak : ℝ := 0.05
  selfJitter : ℝ := 0.0001
  maxIterations : ℕ := 200
  tolerance : ℝ := 0.000001
  forwardRelaxation : ℝ := 1
  checkEvery : ℕ := 5

/-- The production telemetry residual observes only one component of the state. -/
def ObservedChange {T K : ℕ} (old new : IBPState T K) (tol : ℝ) : Prop :=
  ∀ k, |new.elogB k-old.elogB k| < tol

/-- A small slaved-component change does not imply a fixed point of the full map. -/
theorem observed_change_does_not_identify_state :
    ∃ old new : IBPState 1 1, ObservedChange old new 1 ∧ old.g ≠ new.g := by
  refine ⟨⟨fun _ _ => 0, fun _ _ => 0, fun _ => 0, none⟩,
    ⟨fun _ _ => 0, fun _ _ => 1, fun _ => 0, none⟩, ?_, ?_⟩
  · intro k; norm_num
  · intro h
    have := congrFun (congrFun h 0) 0
    norm_num at this

end Pudding.Mean
