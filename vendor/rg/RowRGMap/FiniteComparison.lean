import RowRGMap.AffineCocycle

namespace RowRGMap
namespace FiniteComparison

open AffineCocycle

theorem finite_comparison_bound
    (edges : List Edge) (value : ℕ → ℝ)
    (hedges : ∀ edge ∈ edges, Nonnegative edge)
    (hstep : ∀ i (hi : i < edges.length),
      value (i + 1) ≤ act (edges.get ⟨i, hi⟩) (value i)) :
    value edges.length ≤ act (block edges) (value 0) := by
  induction edges generalizing value with
  | nil => simp [block, act, identity]
  | cons edge rest inductionHypothesis =>
      have hedge : Nonnegative edge := hedges edge (by simp)
      have hrest : ∀ candidate ∈ rest, Nonnegative candidate := by
        intro candidate hcandidate
        exact hedges candidate (by simp [hcandidate])
      have hfirst : value 1 ≤ act edge (value 0) := by
        have h := hstep 0 (by simp)
        simpa using h
      have htail :
          value (rest.length + 1) ≤ act (block rest) (value 1) := by
        have hshift := inductionHypothesis (fun n => value (n + 1))
          hrest
          (by
            intro i hi
            have h := hstep (i + 1) (by simp; omega)
            simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h)
        simpa [Nat.add_comm] using hshift
      have hblock : Nonnegative (block rest) :=
        block_nonnegative hrest
      have hmono :
          act (block rest) (value 1) ≤
            act (block rest) (act edge (value 0)) := by
        simpa [act, add_comm] using
          (add_le_add_right
            (mul_le_mul_of_nonneg_left hfirst hblock.1)
            (block rest).source)
      have hbound := htail.trans hmono
      simpa [block_cons, act_compose, Nat.add_comm] using hbound

end FiniteComparison
end RowRGMap
