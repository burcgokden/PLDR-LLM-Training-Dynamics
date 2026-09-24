/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Structurally inactive AdamW coordinates
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace AdamWInactiveCoordinates

/-- Position update on a fixed-batch coordinate whose gradient vanishes on a
neighborhood and whose two incoming moments are zero. -/
def inactivePosition
    (position learningRate weightDecay decayMask : ℝ) : ℝ :=
  (1 - learningRate * weightDecay * decayMask) * position

/-- The inactive position response contains only the declared decoupled
decay multiplier. No square-root denominator is evaluated. -/
theorem inactive_position_response
    (position perturbation learningRate weightDecay decayMask : ℝ) :
    inactivePosition (position + perturbation)
        learningRate weightDecay decayMask
      - inactivePosition position learningRate weightDecay decayMask =
      (1 - learningRate * weightDecay * decayMask) * perturbation := by
  unfold inactivePosition
  ring

/-- Zero incoming moments remain zero when the fixed-batch gradient is zero. -/
theorem inactive_moments_remain_zero (beta₁ beta₂ : ℝ) :
    beta₁ * 0 + (1 - beta₁) * 0 = 0 ∧
    beta₂ * 0 + (1 - beta₂) * 0 ^ 2 = 0 := by
  constructor <;> ring

end AdamWInactiveCoordinates
end PldrLlmCurvatureSandpile
