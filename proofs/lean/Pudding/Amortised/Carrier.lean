import Mathlib.Data.Real.Basic
namespace Pudding.Amortised
structure PostFusion (n : ℕ) where
  value : Fin n → ℝ
structure Compressed (n : ℕ) where
  value : Fin n → ℝ
structure Carrier (n : ℕ) where
  post : PostFusion n
  compressed : Compressed n
/-- Sharing is a semantic contract supplied by a fitted model, not width equality. -/
structure SharedCompatibility (n : ℕ) where
  compatible : (Fin n → ℝ) → Prop
structure SharedCarrier (n : ℕ) (contract : SharedCompatibility n) where
  value : Fin n → ℝ
  valid : contract.compatible value

def SharedCarrier.split {n} {c : SharedCompatibility n} (s : SharedCarrier n c) : Carrier n :=
  ⟨⟨s.value⟩, ⟨s.value⟩⟩
end Pudding.Amortised
