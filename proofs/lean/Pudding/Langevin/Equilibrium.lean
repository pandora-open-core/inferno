import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Field.Basic

namespace Pudding.Langevin
open scoped BigOperators
variable {I R : Type*} [Fintype I] [Field R]

/-- Finite atom-space equilibrium is a law fixed by a stochastic kernel. -/
def AtomStationary (mass : I → R) (kernel : I → I → R) : Prop :=
  ∀ j, ∑ i, mass i * kernel i j = mass j

def AtomDetailedBalance (mass : I → R) (kernel : I → I → R) : Prop :=
  ∀ i j, mass i * kernel i j = mass j * kernel j i

/-- Detailed balance plus row normalisation yields the equilibrium condition. -/
theorem detailed_balance_stationary (mass : I → R) (kernel : I → I → R)
    (balance : AtomDetailedBalance mass kernel)
    (rows : ∀ i, ∑ j, kernel i j = 1) : AtomStationary mass kernel := by
  intro j
  calc
    ∑ i, mass i * kernel i j = ∑ i, mass j * kernel j i := by
      apply Finset.sum_congr rfl
      intro i _
      exact balance i j
    _ = mass j * ∑ i, kernel j i := (Finset.mul_sum _ _ _).symm
    _ = mass j := by rw [rows j, mul_one]

/-- A symmetric Schrödinger scaling of a reversible reference joint law. -/
def symmetricSchrodinger (reference : I → I → R) (factor : I → R) : I → I → R :=
  fun i j => factor i * reference i j * factor j

omit [Fintype I] in
/-- Symmetric factors preserve reference reversibility. Existence of these factors
and KL optimality are separate obligations, not assumed as completed results. -/
theorem schrodinger_symmetry (reference : I → I → R) (factor : I → R)
    (symmetric : ∀ i j, reference i j = reference j i) :
    ∀ i j, symmetricSchrodinger reference factor i j =
      symmetricSchrodinger reference factor j i := by
  intro i j
  unfold symmetricSchrodinger
  rw [symmetric i j]
  simp only [mul_assoc, mul_comm, mul_left_comm]

variable [LinearOrder R] [IsStrictOrderedRing R]

/-- Conditioning an exactly balanced coupling gives stochastic rows and transports
its source law to its target law. Positivity/nonnegativity are explicit premises. -/
theorem balanced_conditional (a b : I → R) (p : I → I → R)
    (positive : ∀ i, 0 < a i) (nonnegative : ∀ i j, 0 ≤ p i j)
    (rows : ∀ i, ∑ j, p i j = a i) (cols : ∀ j, ∑ i, p i j = b j) :
    (∀ i j, 0 ≤ p i j / a i) ∧
    (∀ i, ∑ j, p i j / a i = 1) ∧
    (∀ j, ∑ i, a i * (p i j / a i) = b j) := by
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    exact div_nonneg (nonnegative i j) (le_of_lt (positive i))
  · intro i
    simp only [div_eq_mul_inv]
    rw [← Finset.sum_mul, rows i, mul_inv_cancel₀ (ne_of_gt (positive i))]
  · intro j
    convert cols j using 1
    apply Finset.sum_congr rfl
    intro i _
    have hi := ne_of_gt (positive i)
    rw [← mul_div_assoc, mul_div_cancel_left₀ _ hi]

/-- Equal endpoints yield a stationary conditional kernel. Symmetry additionally
certifies reversibility; equal marginals alone do not imply detailed balance. -/
theorem schrodinger_equilibrium (mass : I → R) (p : I → I → R)
    (positive : ∀ i, 0 < mass i)
    (nonnegative : ∀ i j, 0 ≤ p i j)
    (rows : ∀ i, ∑ j, p i j = mass i) (cols : ∀ j, ∑ i, p i j = mass j)
    (symmetric : ∀ i j, p i j = p j i) :
    AtomStationary mass (fun i j => p i j / mass i) ∧
    AtomDetailedBalance mass (fun i j => p i j / mass i) := by
  constructor
  · exact (balanced_conditional mass mass p positive nonnegative rows cols).2.2
  · intro i j
    have hi := ne_of_gt (positive i)
    have hj := ne_of_gt (positive j)
    calc
      mass i * (p i j / mass i) = p i j := by rw [← mul_div_assoc, mul_div_cancel_left₀ _ hi]
      _ = p j i := symmetric i j
      _ = mass j * (p j i / mass j) := by rw [← mul_div_assoc, mul_div_cancel_left₀ _ hj]

end Pudding.Langevin
