import RowRGMap.AffineCocycle

/-!
Manuscript correspondence: `rg:eq:scalar-face-memory`.

Checked clause and supplied hypotheses: Finite chronological scalar affine action: a zero gain in a fixed edge list makes the blocked gain zero and action independent of scalar input.

Written obligations: No assertion of complete-state memory loss. Dependence of edge coefficients on optimizer, corpus or hidden state is outside this finite scalar model.
These kernels double-check selected clauses; the written proof is independent.
-/

/-! A zero homogeneous coefficient for a fixed chronological list removes
scalar input dependence only. Optimizer and corpus states are not modeled. -/
namespace PldrTrainingDynamics.ScalarFaceMemory
open RowRGMap.AffineCocycle

theorem zero_gain_act (edge : Edge) (energy : ℝ) (h : edge.gain = 0) :
    act edge energy = edge.source := by simp [act, h]

theorem block_zero_gain (edges : List Edge)
    (h : ∃ edge ∈ edges, edge.gain = 0) : (block edges).gain = 0 := by
  induction edges with
  | nil => simp at h
  | cons first rest ih =>
      rcases h with ⟨edge, he, hz⟩
      rcases List.mem_cons.mp he with he | he
      · subst edge
        simp [hz]
      · simp [ih ⟨edge, he, hz⟩]

theorem block_zero_gain_independent (edges : List Edge) (x y : ℝ)
    (h : ∃ edge ∈ edges, edge.gain = 0) :
    act (block edges) x = act (block edges) y := by
  rw [zero_gain_act _ _ (block_zero_gain edges h),
    zero_gain_act _ _ (block_zero_gain edges h)]

end PldrTrainingDynamics.ScalarFaceMemory
