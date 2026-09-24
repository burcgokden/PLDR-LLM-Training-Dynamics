/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Full gate-sector AdamW block
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace BlockAdamWGate

/-- One coordinate of the lifted gate, first-moment, second-moment state. -/
@[ext] structure GateState where
  gate : ℝ
  firstMoment : ℝ
  secondMoment : ℝ

/-- Operation-ordered homogeneous derivative on a fixed clipping cell. -/
def smoothCellStep
    (eta decay beta₁ beta₂ firstResponse secondResponse
      firstSensitivity secondSensitivity : ℝ)
    (state : GateState) : GateState :=
  let firstNext :=
    firstResponse * state.gate + beta₁ * state.firstMoment
  let secondNext :=
    secondResponse * state.gate + beta₂ * state.secondMoment
  {
    gate :=
      (1 - eta * decay) * state.gate
        - eta * (firstSensitivity * firstNext
          + secondSensitivity * secondNext)
    firstMoment := firstNext
    secondMoment := secondNext
  }

/-- Expanding after the moment updates exposes all nine scalar blocks of the
three-by-three coordinate successor. -/
theorem full_gate_block_expansion
    (eta decay beta₁ beta₂ firstResponse secondResponse
      firstSensitivity secondSensitivity gate first second : ℝ) :
    smoothCellStep eta decay beta₁ beta₂ firstResponse secondResponse
      firstSensitivity secondSensitivity ⟨gate, first, second⟩ =
    {
      gate :=
        (1 - eta * decay
          - eta * (firstSensitivity * firstResponse
            + secondSensitivity * secondResponse)) * gate
          - eta * beta₁ * firstSensitivity * first
          - eta * beta₂ * secondSensitivity * second
      firstMoment := firstResponse * gate + beta₁ * first
      secondMoment := secondResponse * gate + beta₂ * second
    } := by
  apply GateState.ext
  all_goals simp [smoothCellStep]
  all_goals ring

/-- Decoupled weight decay contributes the direct gate multiplier
`1 - eta * decay`. -/
theorem decay_multiplier (eta decay gate : ℝ) :
    gate - eta * decay * gate = (1 - eta * decay) * gate := by
  ring

end BlockAdamWGate
end PldrLlmCurvatureSandpile
