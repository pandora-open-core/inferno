import Pudding.Transport.Convergence

namespace Pudding.Transport
open scoped BigOperators Topology
variable {I J : Type*} [Fintype I] [Fintype J]

/-- A nontrivial finite-support restriction: arbitrary additive row/column costs,
    positive probability masses, and potentials in the additive invariant family. -/
theorem additive_Tf (b : Probability J) (c : I → ℝ) (d : J → ℝ)
    (ε G : ℝ) (hε : ε ≠ 0) (i : I) :
    Tf b.mass (fun i j => c i + d j) ε (fun j => d j + G) i = c i - G := by
  have h (j : J) : ((d j+G)-(c i+d j))/ε = (G-c i)/ε := by ring
  simp only [Tf, h]
  rw [← Finset.sum_mul, b.total, one_mul, Real.log_exp]
  field_simp <;> ring

theorem additive_Tg (a : Probability I) (c : I → ℝ) (d : J → ℝ)
    (ε F : ℝ) (hε : ε ≠ 0) (j : J) :
    Tg a.mass (fun i j => c i + d j) ε (fun i => c i + F) j = d j - F :=
  by simpa only [Tg, Tf, add_comm] using additive_Tf a d c ε F hε j

theorem mean_add_const [Nonempty I] (c : I → ℝ) (F : ℝ) :
    mean (fun i => c i+F) = mean c+F := by
  have hn : (Fintype.card I : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp only [mean, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp <;> ring

noncomputable def centredAdditive (c : I → ℝ) (d : J → ℝ) (s : ℝ) : Potentials I J :=
  ⟨fun i => c i-mean c, fun j => d j+mean c+s⟩

/-- This theorem connects the scalar contraction to the ACTUAL gauge-fixed recurrence. -/
theorem additive_step [Nonempty I] (a : Probability I) (b : Probability J)
    (c : I → ℝ) (d : J → ℝ) (ε r s : ℝ) (hε : ε ≠ 0) :
    step a.mass b.mass (fun i j => c i+d j) ε r (centredAdditive c d s) =
      centredAdditive c d ((1-2*r)*s) := by
  have hf : (centredAdditive c d s).f = fun i => c i + (-mean c) := by
    funext i; simp [centredAdditive]; ring
  have hg : (centredAdditive c d s).g = fun j => d j + (mean c+s) := by
    funext j; simp [centredAdditive]; ring
  have hraw : rawUpdate a.mass b.mass (fun i j => c i+d j) ε r (centredAdditive c d s) =
      Potentials.mk (fun i => c i+(-mean c-r*s))
        (fun j => d j+(mean c+(1-r)*s)) := by
    unfold rawUpdate relax
    rw [hf, hg]
    simp only [additive_Tf b c d ε (mean c+s) hε, additive_Tg a c d ε (-mean c) hε]
    congr 1 <;> funext k <;> ring
  unfold step
  rw [hraw]
  simp only [mean_add_const]
  unfold gauge centredAdditive
  congr 1 <;> funext k <;> ring

theorem additive_iterate [Nonempty I] (a : Probability I) (b : Probability J)
    (c : I → ℝ) (d : J → ℝ) (ε r s : ℝ) (hε : ε ≠ 0) (n : ℕ) :
    (step a.mass b.mass (fun i j => c i+d j) ε r)^[n] (centredAdditive c d s) =
      centredAdditive c d ((1-2*r)^n*s) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih, additive_step a b c d ε r _ hε]
    congr 1; rw [pow_succ]; ring

/-- Every coupling entry converges to a_i b_j in this additive invariant family. -/
theorem additive_plan_converges [Nonempty I] (a : Probability I) (b : Probability J)
    (c : I → ℝ) (d : J → ℝ) (ε r s : ℝ) (hε : ε ≠ 0)
    (hr0 : 0 < r) (hr1 : r < 1) (i : I) (j : J) :
    Filter.Tendsto (fun n => plan a.mass b.mass (fun i j => c i+d j) ε
      ((step a.mass b.mass (fun i j => c i+d j) ε r)^[n] (centredAdditive c d s)) i j)
      Filter.atTop (nhds (a.mass i*b.mass j)) := by
  have heq (n : ℕ) : plan a.mass b.mass (fun i j => c i+d j) ε
      ((step a.mass b.mass (fun i j => c i+d j) ε r)^[n] (centredAdditive c d s)) i j =
      a.mass i*b.mass j*Real.exp (((1-2*r)^n*s)/ε) := by
    rw [additive_iterate a b c d ε r s hε n]
    unfold plan centredAdditive
    congr 2; ring
  simp_rw [heq]
  have hz : Filter.Tendsto (fun n : ℕ => ((1-2*r)^n*s)/ε)
      Filter.atTop (nhds (0 : ℝ)) := by
    simpa using (singleton_error_converges r s hr0 hr1).div_const ε
  have h := ((Real.continuous_exp.tendsto (0 : ℝ)).comp hz).const_mul (a.mass i*b.mass j)
  simpa using h

end Pudding.Transport
