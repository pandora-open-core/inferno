import Mathlib.Analysis.Calculus.FDeriv.Comp
namespace Pudding.Autograd
/-- Chain rule underlying a finite smooth DAG; accumulation of shared nodes is
    covered locally by addition, not by treating the graph as a tree. -/
theorem compose_rule {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    {f : E → F} {g : F → G} {x : E}
    {df : E →L[ℝ] F} {dg : F →L[ℝ] G}
    (hf : HasFDerivAt f df x) (hg : HasFDerivAt g dg (f x)) :
    HasFDerivAt (fun y => g (f y)) (dg.comp df) x := hg.comp x hf
end Pudding.Autograd
