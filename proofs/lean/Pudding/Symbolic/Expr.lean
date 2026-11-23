import Pudding.Symbolic.Shape

namespace Pudding.Symbolic
/-- Scope is typed; shape errors cannot be represented. Export validation is a separate boundary. -/
inductive Expr (m n : ℕ) : Axis → ℕ → Type where
  | leaf {a d} : Tensor m n a d → Expr m n a d
  | add {a d} : Expr m n a d → Expr m n a d → Expr m n a d
  | sub {a d} : Expr m n a d → Expr m n a d → Expr m n a d
  | mul {a d} : Expr m n a d → Expr m n a d → Expr m n a d
  | div {a d} : Expr m n a d → Expr m n a d → Expr m n a d
  | exp {a d} : Expr m n a d → Expr m n a d
  | log {a d} : Expr m n a d → Expr m n a d
  | fromI {d} : Expr m n .I d → Expr m n .IJ d
  | fromJ {d} : Expr m n .J d → Expr m n .IJ d
  | fromP {d} : Expr m n .P d → Expr m n .IJ d
  | sumLast {a d} : Expr m n a d → Expr m n a 1
  | sumI {d} : Expr m n .IJ d → Expr m n .J d
  | sumJ {d} : Expr m n .IJ d → Expr m n .I d

end Pudding.Symbolic
