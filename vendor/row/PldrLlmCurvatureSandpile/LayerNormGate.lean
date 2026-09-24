/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact final LayerNorm gate and row-shape factorization
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace LayerNormGate

/-- The final LayerNorm affine map, written as a shared bias plus a
coordinatewise scale acting on normalized row shape. -/
def gateOutput {Row : Type*} {d : ℕ}
    (beta gamma : Fin d → ℝ) (shape : Row → Fin d → ℝ)
    (row : Row) (coordinate : Fin d) : ℝ :=
  beta coordinate + gamma coordinate * shape row coordinate

/-- Bias cancels exactly from every pair of rows. -/
theorem gate_difference {Row : Type*} {d : ℕ}
    (beta gamma : Fin d → ℝ) (shape : Row → Fin d → ℝ)
    (left right : Row) (coordinate : Fin d) :
    gateOutput beta gamma shape left coordinate
        - gateOutput beta gamma shape right coordinate
      = gamma coordinate
          * (shape left coordinate - shape right coordinate) := by
  simp [gateOutput]
  ring

/-- A coordinate of the row map is constant exactly when its gate is zero
or its normalized shape is constant. -/
theorem coordinate_constant_iff {Row : Type*} [Nonempty Row] {d : ℕ}
    (beta gamma : Fin d → ℝ) (shape : Row → Fin d → ℝ)
    (coordinate : Fin d) :
    (∀ left right,
      gateOutput beta gamma shape left coordinate
        = gateOutput beta gamma shape right coordinate)
      ↔ gamma coordinate = 0
        ∨ ∀ left right, shape left coordinate = shape right coordinate := by
  constructor
  · intro hconstant
    by_cases hgate : gamma coordinate = 0
    · exact Or.inl hgate
    · right
      intro left right
      have hproduct :
          gamma coordinate
              * (shape left coordinate - shape right coordinate) = 0 := by
        have hdifference := congrArg
          (fun value =>
            value - gateOutput beta gamma shape right coordinate)
          (hconstant left right)
        simpa [gate_difference] using hdifference
      have hshape := (mul_eq_zero.mp hproduct).resolve_left hgate
      exact sub_eq_zero.mp hshape
  · intro hcondition left right
    rcases hcondition with hgate | hshape
    · simp [gateOutput, hgate]
    · simp [gateOutput, hshape left right]

/-- Setting every final scale to zero produces the same bias row. -/
theorem zero_gate_constant {Row : Type*} {d : ℕ}
    (beta : Fin d → ℝ) (shape : Row → Fin d → ℝ) (row : Row) :
    gateOutput beta (fun _ => 0) shape row = beta := by
  funext coordinate
  simp [gateOutput]

/-- A coordinatewise row difference is bounded by the gate magnitude times
the registered normalized-shape diameter. -/
theorem coordinate_oscillation_bound {Row : Type*} {d : ℕ}
    (beta gamma : Fin d → ℝ) (shape : Row → Fin d → ℝ)
    (left right : Row) (coordinate : Fin d) {shapeDiameter : ℝ}
    (hshape :
      |shape left coordinate - shape right coordinate| ≤ shapeDiameter) :
    |gateOutput beta gamma shape left coordinate
        - gateOutput beta gamma shape right coordinate|
      ≤ |gamma coordinate| * shapeDiameter := by
  rw [gate_difference, abs_mul]
  exact mul_le_mul_of_nonneg_left hshape (abs_nonneg _)

end LayerNormGate
end PldrLlmCurvatureSandpile
