import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Push

namespace Pudding.Symbolic
inductive Axis where
  | P | I | J | IJ
  deriving DecidableEq, Repr

def Index (m n : ℕ) : Axis → Type
  | .P => Unit
  | .I => Fin m
  | .J => Fin n
  | .IJ => Fin m × Fin n

abbrev Tensor (m n : ℕ) (a : Axis) (d : ℕ) := Index m n a → Fin d → ℝ
end Pudding.Symbolic
