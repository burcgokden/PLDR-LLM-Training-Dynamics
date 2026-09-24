/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PLGASignedSecant

noncomputable def positivePower (base exponent : ℝ) : ℝ :=
  Real.exp (exponent * Real.log base)

noncomputable def signedSecant
    (left right exponent : ℝ) : ℝ :=
  if left = right then
    exponent * positivePower left (exponent - 1)
  else
    (positivePower right exponent - positivePower left exponent)
      / (right - left)

/-- Exact occupied-endpoint reconstruction for every real exponent. -/
theorem signed_secant_exact
    {left right exponent : ℝ} (hne : left ≠ right) :
    positivePower right exponent - positivePower left exponent
      = signedSecant left right exponent * (right - left) := by
  simp [signedSecant, hne]
  field_simp [sub_ne_zero.mpr hne]

/-- The equal-endpoint branch is the real-power derivative by definition. -/
theorem signed_secant_diagonal
    (base exponent : ℝ) :
    signedSecant base base exponent
      = exponent * positivePower base (exponent - 1) := by
  simp [signedSecant]

/-- A supplied occupied-segment derivative maximum controls the exact secant. -/
theorem occupied_segment_bound
    {left right exponent bound : ℝ}
    (hbound : |signedSecant left right exponent| ≤ bound) :
    |positivePower right exponent - positivePower left exponent|
      ≤ bound * |right - left| := by
  by_cases hequal : left = right
  · simp [hequal]
  · rw [signed_secant_exact hequal, abs_mul]
    exact mul_le_mul_of_nonneg_right hbound (abs_nonneg _)

end PLGASignedSecant
end PldrLlmCurvatureSandpile

