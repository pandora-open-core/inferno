import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Pudding.Symbolic.Expr

namespace Pudding.Symbolic
open scoped BigOperators
noncomputable def denote {m n a d} : Expr m n a d → Tensor m n a d
  | .leaf x => x
  | .add x y => fun i k => denote x i k + denote y i k
  | .sub x y => fun i k => denote x i k - denote y i k
  | .mul x y => fun i k => denote x i k * denote y i k
  | .div x y => fun i k => denote x i k / denote y i k
  | .exp x => fun i k => Real.exp (denote x i k)
  | .log x => fun i k => Real.log (denote x i k)
  | .fromI x => fun i k => denote x i.1 k
  | .fromJ x => fun i k => denote x i.2 k
  | .fromP x => fun _ k => denote x () k
  | .sumLast x => fun i _ => ∑ k, denote x i k
  | .sumI x => fun j k => ∑ i, denote x (i,j) k
  | .sumJ x => fun i k => ∑ j, denote x (i,j) k

/-- Real.log is total in Lean; these predicates restrict the intended numeric semantics. -/
def Valid {m n a d} : Expr m n a d → Prop
  | .leaf _ => True
  | .add x y | .sub x y | .mul x y => Valid x ∧ Valid y
  | .div x y => Valid x ∧ Valid y ∧ ∀ i k, denote y i k ≠ 0
  | .log x => Valid x ∧ ∀ i k, 0 < denote x i k
  | .exp x => Valid x
  | .fromI x | .fromJ x | .fromP x => Valid x
  | .sumLast x | .sumI x | .sumJ x => Valid x

noncomputable def matDot {m n d} (x : Expr m n .I d) (y : Expr m n .J d) : Expr m n .IJ 1 :=
  .sumLast (.mul (.fromI x) (.fromJ y))
noncomputable def logSumExpJ {m n} (x : Expr m n .IJ 1) : Expr m n .I 1 :=
  .log (.sumJ (.exp x))

end Pudding.Symbolic
