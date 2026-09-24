/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ShapeDirEnclosure

/-- Exact scalar shape secant after a supplied directional term and remainder. -/
theorem scalar_directional_shape_identity
    (gate contrast directional remainder : ℝ) :
    (gate * (contrast + directional + remainder)) ^ 2
        - (gate * contrast) ^ 2
      = (gate * (contrast + directional)) ^ 2
          - (gate * contrast) ^ 2
        + 2 * (gate * (contrast + directional)) * (gate * remainder)
        + (gate * remainder) ^ 2 := by
  ring

/-- The displayed remainder charge bounds the scalar directional shape work. -/
theorem scalar_directional_shape_enclosure
    {gate contrast directional remainder radius : ℝ}
    (hradius : 0 ≤ radius) (hremainder : |remainder| ≤ radius) :
    (gate * (contrast + directional + remainder)) ^ 2
        - (gate * contrast) ^ 2
      ≤ (gate * (contrast + directional)) ^ 2
          - (gate * contrast) ^ 2
        + 2 * |gate * (contrast + directional)| * |gate| * radius
        + |gate| ^ 2 * radius ^ 2 := by
  have hgateRemainder : |gate * remainder| ≤ |gate| * radius := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hremainder (abs_nonneg gate)
  have hcross :
      (gate * (contrast + directional)) * (gate * remainder)
        ≤ |gate * (contrast + directional)| * (|gate| * radius) := by
    calc
      (gate * (contrast + directional)) * (gate * remainder)
          ≤ |(gate * (contrast + directional)) * (gate * remainder)| :=
        le_abs_self _
      _ = |gate * (contrast + directional)| * |gate * remainder| := abs_mul _ _
      _ ≤ |gate * (contrast + directional)| * (|gate| * radius) :=
        mul_le_mul_of_nonneg_left hgateRemainder (abs_nonneg _)
  have hrightNonnegative : 0 ≤ |gate| * radius :=
    mul_nonneg (abs_nonneg gate) hradius
  have hsquareAbs :
      |gate * remainder| ^ 2 ≤ (|gate| * radius) ^ 2 :=
    (sq_le_sq₀ (abs_nonneg _) hrightNonnegative).2 hgateRemainder
  have hsquare : (gate * remainder) ^ 2 ≤ (|gate| * radius) ^ 2 := by
    calc
      (gate * remainder) ^ 2 = |gate * remainder| ^ 2 :=
        (sq_abs (gate * remainder)).symm
      _ ≤ (|gate| * radius) ^ 2 := hsquareAbs
  rw [scalar_directional_shape_identity]
  nlinarith

end ShapeDirEnclosure
end PldrLlmCurvatureSandpile

