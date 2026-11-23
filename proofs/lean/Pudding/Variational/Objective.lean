import Mathlib.Data.Real.Basic
namespace Pudding.Variational
/-- All entries have MINIMISATION signs. gpNll is already negative GP ELBO. -/
structure Terms where
  gpNll : ℝ
  globalKl : ℝ
  localKl : ℝ
  alphaKl : ℝ
  lengthscaleKl : ℝ
  inverseWishart : ℝ
  reconKl : ℝ
  reconNll : ℝ
  latentKl : ℝ
  sinkhorn : ℝ
def loss (t : Terms) (beta lambda : ℝ) : ℝ :=
  t.gpNll + beta * (t.globalKl + t.localKl + t.alphaKl + t.lengthscaleKl +
    t.inverseWishart + t.reconKl + t.reconNll + t.latentKl) + lambda*t.sinkhorn
/-- The 134 coordinates are a carrier; potentials live on separate supports. -/
abbrev Carrier := Fin 134 → ℝ
end Pudding.Variational
