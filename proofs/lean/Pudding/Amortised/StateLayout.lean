import Mathlib.Data.List.TakeDrop
import Mathlib.Data.Real.Basic
namespace Pudding.Amortised
/-- Canonical, already padded/cast row; LMC stores the executable dense matrix.
    Raw z truncation and ls broadcasting happen BEFORE this representation. -/
@[ext] structure State where
  z : List ℝ
  qa : List ℝ
  qb : List ℝ
  ls : List ℝ
  lmc : List ℝ
  pi : List ℝ

def State.pack (s : State) : List ℝ :=
  s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls ++ s.lmc ++ s.pi

def State.shaped (s : State) (L K : ℕ) : Prop :=
  s.z.length = L ∧ s.qa.length + 1 = K ∧ s.qb.length + 1 = K ∧
  s.ls.length = K ∧ s.lmc.length = K*K ∧ s.pi.length = K

def State.admissible (s : State) : Prop :=
  (∀ x ∈ s.qa ++ s.qb ++ s.ls, 0 < x) ∧
  (∀ x ∈ s.pi, 0 ≤ x) ∧ s.pi.sum = 1

/-- Unpacking uses canonical lengths, and drops the two unit padding entries. -/
def unpack (L K : ℕ) (p : List ℝ) : State :=
  ⟨p.take L, (p.drop L).take (K-1), (p.drop (L+K)).take (K-1),
   (p.drop (L+2*K)).take K, (p.drop (L+3*K)).take (K*K),
   p.drop (L+3*K+K*K)⟩

theorem packed_width (s : State) (h : s.shaped L K) :
    s.pack.length = L + 4*K + K*K := by
  rcases h with ⟨hz, ha, hb, hl, hm, hp⟩
  simp only [State.pack, List.length_append, List.length_singleton]
  omega

/-- A block-level inverse used successively at every executable offset. -/
theorem split_append (a b : List ℝ) :
    (a ++ b).take a.length = a ∧ (a ++ b).drop a.length = b := by simp

/-- The alpha padding is removed without changing any concentration. -/
theorem unpad (a : List ℝ) : (a ++ [1]).take a.length = a := by simp

private theorem slice (a b c : List ℝ) (ha : a.length = i) (hb : b.length = j) :
    ((a ++ b ++ c).drop i).take j = b := by
  subst i; subst j
  simp [List.append_assoc]

theorem unpack_pack (s : State) (h : s.shaped L K) : unpack L K s.pack = s := by
  rcases h with ⟨hz, ha, hb, hl, hm, hp⟩
  have ha' : s.qa.length = K-1 := by omega
  have hb' : s.qb.length = K-1 := by omega
  apply State.ext
  · change s.pack.take L = s.z
    have hslice :=
      (List.take_left' (l₁ := s.z) (l₂ := (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls ++ s.lmc ++ s.pi) hz)
    simpa only [State.pack, List.append_assoc] using hslice
  · change (s.pack.drop L).take (K-1) = s.qa
    have hslice :=
      slice s.z s.qa ([1] ++ (s.qb ++ [1]) ++ s.ls ++ s.lmc ++ s.pi) hz ha'
    simpa only [State.pack, List.append_assoc] using hslice
  · change (s.pack.drop (L+K)).take (K-1) = s.qb
    have hpre : (s.z ++ (s.qa ++ [1])).length = L+K := by simp; omega
    have hslice :=
      slice (s.z ++ (s.qa ++ [1])) s.qb ([1] ++ s.ls ++ s.lmc ++ s.pi) hpre hb'
    simpa only [State.pack, List.append_assoc] using hslice
  · change (s.pack.drop (L+2*K)).take K = s.ls
    have hpre : (s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1])).length = L+2*K := by simp; omega
    have hslice :=
      slice (s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1])) s.ls (s.lmc ++ s.pi) hpre hl
    simpa only [State.pack, List.append_assoc] using hslice
  · change (s.pack.drop (L+3*K)).take (K*K) = s.lmc
    have hpre : (s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls).length = L+3*K := by simp; omega
    have hslice :=
      slice (s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls) s.lmc s.pi hpre hm
    simpa only [State.pack, List.append_assoc] using hslice
  · change s.pack.drop (L+3*K+K*K) = s.pi
    have hpre : (s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls ++ s.lmc).length = L+3*K+K*K := by simp; omega
    have hslice :=
      (List.drop_left' (l₁ := s.z ++ (s.qa ++ [1]) ++ (s.qb ++ [1]) ++ s.ls ++ s.lmc) (l₂ := s.pi) hpre)

    simpa only [State.pack, List.append_assoc] using hslice
theorem admissible_roundtrip (s : State) (h : s.shaped L K) (ha : s.admissible) :
    (unpack L K s.pack).admissible := by rw [unpack_pack s h]; exact ha
end Pudding.Amortised
