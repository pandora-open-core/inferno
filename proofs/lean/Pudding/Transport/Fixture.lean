import Pudding.Transport.Additive

namespace Pudding.Transport

noncomputable def fixtureMass : Probability (Fin 2) where
  mass _ := 1/2
  nonneg _ := by norm_num
  total := by norm_num [Fin.sum_univ_two]

def fixturePotentials : Potentials (Fin 2) (Fin 2) := ⟨fun _ => 0, fun _ => 0⟩

/-- Exact zero-cost initial coupling on two equally weighted atoms. -/
theorem fixture_plan_quarter (i j : Fin 2) :
    plan fixtureMass.mass fixtureMass.mass (fun _ _ => 0) 1 fixturePotentials i j = 1/4 := by
  norm_num [plan, fixtureMass, fixturePotentials]

/-- The minimal fixture stops at its FIRST pre-update residual check. -/
theorem fixture_initial_fixed (i j : Fin 2) :
    Tf fixtureMass.mass (fun _ _ => 0) 1 fixturePotentials.g i = 0 ∧
    Tg fixtureMass.mass (fun _ _ => 0) 1 fixturePotentials.f j = 0 := by
  norm_num [Tf, Tg, fixtureMass, fixturePotentials, Fin.sum_univ_two]

end Pudding.Transport
