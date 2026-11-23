import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Pudding.Symbolic.Shape

namespace Pudding.Symbolic
/-- Typed map instructions: a finite register index cannot refer forward. -/
inductive MapOp (inputs regs : ℕ) where
  | input : Fin inputs → MapOp inputs regs
  | const : ℝ → MapOp inputs regs
  | add : Fin regs → Fin regs → MapOp inputs regs
  | sub : Fin regs → Fin regs → MapOp inputs regs
  | mul : Fin regs → Fin regs → MapOp inputs regs
  | div : Fin regs → Fin regs → MapOp inputs regs
  | exp : Fin regs → MapOp inputs regs
  | log : Fin regs → MapOp inputs regs

noncomputable def scalar {i r} (input : Fin i → ℝ) (reg : Fin r → ℝ) : MapOp i r → ℝ
  | .input j => input j
  | .const c => c
  | .add a b => reg a + reg b
  | .sub a b => reg a - reg b
  | .mul a b => reg a * reg b
  | .div a b => reg a / reg b
  | .exp a => Real.exp (reg a)
  | .log a => Real.log (reg a)

/-- Packet semantics is one map, with no cross-lane operation or reduction. -/
noncomputable def packet {i r L} (input : Fin i → Fin L → ℝ)
    (reg : Fin r → Fin L → ℝ) (op : MapOp i r) : Fin L → ℝ :=
  fun lane => scalar (fun j => input j lane) (fun j => reg j lane) op

theorem packet_lane {i r L} (input : Fin i → Fin L → ℝ)
    (reg : Fin r → Fin L → ℝ) (op : MapOp i r) (lane : Fin L) :
    packet input reg op lane = scalar (fun j => input j lane) (fun j => reg j lane) op := rfl

/-- Let-bindings share subexpressions; register zero is the newest result.
    Wire bytecode uses append order, requiring an exporter index renaming. -/
inductive Code (inputs : ℕ) : ℕ → Type where
  | ret {r} : Fin r → Code inputs r
  | bind {r} : MapOp inputs r → Code inputs (r+1) → Code inputs r

noncomputable def run {i r} (input : Fin i → ℝ) (reg : Fin r → ℝ) : Code i r → ℝ
  | .ret j => reg j
  | .bind op rest => run input (Fin.cases (scalar input reg op) reg) rest

noncomputable def runPacket {i r L} (input : Fin i → Fin L → ℝ)
    (reg : Fin r → Fin L → ℝ) : Code i r → Fin L → ℝ
  | .ret j => reg j
  | .bind op rest => runPacket input (Fin.cases (packet input reg op) reg) rest

theorem code_packet_lane {i r L} (code : Code i r) (input : Fin i → Fin L → ℝ)
    (reg : Fin r → Fin L → ℝ) (lane : Fin L) :
    runPacket input reg code lane = run (fun j => input j lane) (fun j => reg j lane) code := by
  induction code with
  | ret j => rfl
  | bind op rest ih =>
    simp only [runPacket, run]
    rw [ih]
    congr 1
    funext j
    refine Fin.cases ?_ (fun k => ?_) j <;> rfl

end Pudding.Symbolic
