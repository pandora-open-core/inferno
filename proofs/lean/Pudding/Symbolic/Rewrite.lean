import Pudding.Symbolic.Denote

namespace Pudding.Symbolic
open scoped BigOperators
/-- Rule identifiers refer only to these checked real-arithmetic identities. -/
inductive RuleId where
  | addComm | squaredDistance | disjointTiles
  deriving Repr, DecidableEq

theorem add_comm_denote {m n a d} (x y : Expr m n a d) :
    denote (.add x y) = denote (.add y x) := by
  funext i k; simp [denote, add_comm]

theorem squared_distance (x y : ℝ) : (x-y)*(x-y) = x*x + y*y - 2*x*y := by ring

theorem squared_distance_tensor {m n a d} (x y : Expr m n a d) :
    denote (.mul (.sub x y) (.sub x y)) =
    fun i k => denote x i k * denote x i k + denote y i k * denote y i k -
      2 * denote x i k * denote y i k := by
  funext i k; exact squared_distance _ _

/-- Tiling requires disjointness and exact coverage; duplicate/missing tiles are excluded. -/
theorem tiled_reduction {α : Type*} [DecidableEq α] (s t : Finset α)
    (h : Disjoint s t) (f : α → ℝ) :
    ∑ x ∈ s ∪ t, f x = (∑ x ∈ s, f x) + ∑ x ∈ t, f x := by
  exact Finset.sum_union h

end Pudding.Symbolic
