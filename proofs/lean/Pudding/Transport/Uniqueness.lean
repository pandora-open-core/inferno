import Pudding.Transport.Existence

namespace Pudding.Transport
open scoped BigOperators
variable {I J : Type*} [Fintype I] [Fintype J]

lemma klTerm_scaled (x q : ℝ) (hq : q ≠ 0) :
    klTerm x q = q * ((x/q) * Real.log (x/q)) - x + q := by
  rw [klTerm_eq]; field_simp <;> ring

theorem klTerm_strictConvex (q : ℝ) (hq : 0 < q) :
    StrictConvexOn ℝ (Set.Ici 0) (fun x => klTerm x q) := by
  refine ⟨convex_Ici 0, ?_⟩
  intro x hx y hy hxy a b ha hb hab
  have hn : x/q ≠ y/q := fun h => hxy ((div_left_inj' hq.ne').mp h)
  have h := Real.strictConvexOn_mul_log.2
    (show x/q ∈ Set.Ici 0 from div_nonneg hx hq.le)
    (show y/q ∈ Set.Ici 0 from div_nonneg hy hq.le) hn ha hb hab
  simp only [smul_eq_mul] at h ⊢
  calc
    klTerm (a*x+b*y) q = q * ((a*(x/q)+b*(y/q)) *
        Real.log (a*(x/q)+b*(y/q))) - (a*x+b*y) + q := by
      rw [klTerm_scaled _ _ hq.ne']; congr 3 <;> ring
    _ < q * (a*((x/q)*Real.log (x/q))+b*((y/q)*Real.log (y/q))) - (a*x+b*y) + q := by
      linarith [mul_lt_mul_of_pos_left h hq]
    _ = a * klTerm x q + b * klTerm y q := by
      rw [klTerm_scaled _ _ hq.ne', klTerm_scaled _ _ hq.ne']
      nlinarith [congrArg (fun z : ℝ => z*q) hab]

lemma costTerm_strictConvex (c ε q : ℝ) (hε : 0 < ε) (hq : 0 < q) :
    StrictConvexOn ℝ (Set.Ici 0) (fun x => c*x + ε*klTerm x q) := by
  refine ⟨convex_Ici 0, ?_⟩
  intro x hx y hy hxy a b ha hb hab
  have h := (klTerm_strictConvex q hq).2 hx hy hxy ha hb hab
  simp only [smul_eq_mul] at h ⊢
  nlinarith [mul_lt_mul_of_pos_left h hε]

/-- The full objective is strictly convex on nonnegative finite matrices. -/
theorem objective_strictConvex (a : Probability I) (b : Probability J)
    (cost : I → J → ℝ) (ε : ℝ) (hε : 0 < ε)
    (ha : ∀ i, 0 < a.mass i) (hb : ∀ j, 0 < b.mass j) :
    StrictConvexOn ℝ {p : I → J → ℝ | ∀ i j, 0 ≤ p i j} (objective a b cost ε) := by
  refine ⟨?_, ?_⟩
  · intro p hp q hq u v hu hv huv i j
    exact add_nonneg (mul_nonneg hu (hp i j)) (mul_nonneg hv (hq i j))
  · intro p hp q hq hpq u v hu hv huv
    obtain ⟨i, j, hij⟩ : ∃ i j, p i j ≠ q i j := by
      by_contra h; push_neg at h; exact hpq (funext fun i => funext (h i))
    have hc k l := costTerm_strictConvex (cost k l) ε (a.mass k*b.mass l) hε (mul_pos (ha k) (hb l))
    have hle k l := (hc k l).convexOn.2 (hp k l) (hq k l) hu.le hv.le huv
    have hlt := (hc i j).2 (hp i j) (hq i j) hij hu hv huv
    simp only [smul_eq_mul] at hle hlt
    have hrow k :
        (∑ l, (cost k l * (u*p k l+v*q k l) + ε*klTerm (u*p k l+v*q k l) (a.mass k*b.mass l))) ≤
        ∑ l, (u*(cost k l*p k l+ε*klTerm (p k l) (a.mass k*b.mass l)) +
          v*(cost k l*q k l+ε*klTerm (q k l) (a.mass k*b.mass l))) :=
      Finset.sum_le_sum (fun l _ => hle k l)
    have hrowlt :
        (∑ l, (cost i l * (u*p i l+v*q i l) + ε*klTerm (u*p i l+v*q i l) (a.mass i*b.mass l))) <
        ∑ l, (u*(cost i l*p i l+ε*klTerm (p i l) (a.mass i*b.mass l)) +
          v*(cost i l*q i l+ε*klTerm (q i l) (a.mass i*b.mass l))) :=
      Finset.sum_lt_sum (fun l _ => hle i l) ⟨j, Finset.mem_univ j, hlt⟩
    have htotal := Finset.sum_lt_sum (fun k _ => hrow k) ⟨i, Finset.mem_univ i, hrowlt⟩
    simpa [objective, Finset.sum_add_distrib, mul_add, Finset.mul_sum, Pi.add_apply, Pi.smul_apply, smul_eq_mul] using htotal

theorem minimiser_unique (a : Probability I) (b : Probability J) (cost : I → J → ℝ)
    (ε : ℝ) (hε : 0 < ε) (ha : ∀ i, 0 < a.mass i) (hb : ∀ j, 0 < b.mass j)
    {p q : I → J → ℝ} (hp : Minimises a b cost ε p) (hq : Minimises a b cost ε q) : p = q := by
  by_contra hpq
  have hmid : Feasible a.mass b.mass (fun i j => (1/2:ℝ)*p i j + (1/2:ℝ)*q i j) := by
    refine ⟨fun i j => add_nonneg (mul_nonneg (by norm_num) (hp.1.1 i j))
      (mul_nonneg (by norm_num) (hq.1.1 i j)), ?_, ?_⟩
    · intro i; simp [Finset.sum_add_distrib, ← Finset.mul_sum, hp.1.2.1 i, hq.1.2.1 i]; ring
    · intro j; simp [Finset.sum_add_distrib, ← Finset.mul_sum, hp.1.2.2 j, hq.1.2.2 j]; ring
  have hstrict := (objective_strictConvex a b cost ε hε ha hb).2 hp.1.1 hq.1.1 hpq
    (by norm_num : (0:ℝ)<1/2) (by norm_num : (0:ℝ)<1/2) (by norm_num : (1/2:ℝ)+1/2=1)
  have hminp := hp.2 _ hmid
  have hminq := hq.2 _ hmid
  change objective a b cost ε (fun i j => (1/2:ℝ)*p i j+(1/2:ℝ)*q i j) <
    (1/2:ℝ)*objective a b cost ε p+(1/2:ℝ)*objective a b cost ε q at hstrict
  linarith

theorem minimiser_exists_unique (a : Probability I) (b : Probability J) (cost : I → J → ℝ)
    (ε : ℝ) (hε : 0 < ε) (ha : ∀ i, 0 < a.mass i) (hb : ∀ j, 0 < b.mass j) :
    ∃! p, Minimises a b cost ε p := by
  obtain ⟨p, hp⟩ := minimiser_exists a b cost ε ha hb
  exact ⟨p, hp, fun q hq => minimiser_unique a b cost ε hε ha hb hq hp⟩

end Pudding.Transport
