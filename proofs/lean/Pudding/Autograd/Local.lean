import Mathlib.Tactic.Ring
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
namespace Pudding.Autograd
/-- Interior smooth rule; floor boundaries require a separate convention. -/
theorem product_rule (x y dx dy : ℝ) :
    HasDerivAt (fun t : ℝ => (x + t*dx)*(y + t*dy)) (dx*y + x*dy) 0 := by
  convert (((hasDerivAt_const (0 : ℝ) x).add ((hasDerivAt_id 0).mul_const dx)).mul
    ((hasDerivAt_const (0 : ℝ) y).add ((hasDerivAt_id 0).mul_const dy))) using 1 <;> simp
/-- Detach is a constant in the differentiated argument. -/
theorem detach_rule (x : ℝ) : HasDerivAt (fun _ : ℝ => x) 0 0 := hasDerivAt_const 0 x
/-- Scalar adjoint identity for the product VJP. -/
theorem product_vjp (x y dx dy cot : ℝ) :
    cot*(dx*y+x*dy) = (cot*y)*dx + (cot*x)*dy := by ring
theorem exp_rule (x : ℝ) : HasDerivAt Real.exp (Real.exp x) x := Real.hasDerivAt_exp x

theorem log_rule (x : ℝ) (hx : x ≠ 0) : HasDerivAt Real.log x⁻¹ x :=
  Real.hasDerivAt_log hx

/-- Both uses of a shared DAG node contribute to its derivative. -/
theorem shared_node_rule (f : ℝ → ℝ) (x df : ℝ) (hf : HasDerivAt f df x) :
    HasDerivAt (fun t => f t + f t) (df+df) x := hf.add hf
end Pudding.Autograd
