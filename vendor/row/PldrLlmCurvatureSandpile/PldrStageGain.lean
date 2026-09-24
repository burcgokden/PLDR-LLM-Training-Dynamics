/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Restricted stage gains

Algebraic composition kernels for the PLGA factorized normal edge.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PldrStageGain

/-- Restricted lower gains multiply in execution order. -/
theorem restricted_conorm_comp
    {input middle output m₁ m₂ : ℝ}
    (hm₂ : 0 ≤ m₂)
    (hfirst : m₁ * input ≤ middle)
    (hsecond : m₂ * middle ≤ output) :
    (m₂ * m₁) * input ≤ output := by
  calc
    (m₂ * m₁) * input = m₂ * (m₁ * input) := by ring
    _ ≤ m₂ * middle := mul_le_mul_of_nonneg_left hfirst hm₂
    _ ≤ output := hsecond

/-- Upper gains also multiply in execution order. -/
theorem upper_gain_comp
    {input middle output L₁ L₂ : ℝ}
    (hL₂ : 0 ≤ L₂)
    (hfirst : middle ≤ L₁ * input)
    (hsecond : output ≤ L₂ * middle) :
    output ≤ (L₂ * L₁) * input := by
  calc
    output ≤ L₂ * middle := hsecond
    _ ≤ L₂ * (L₁ * input) := mul_le_mul_of_nonneg_left hfirst hL₂
    _ = (L₂ * L₁) * input := by ring

/-- A derivative modulus turns a center conorm into a tube-valid lower edge. -/
theorem tube_conorm_lower
    {center modulus radius realized : ℝ}
    (hvariation : center - modulus * radius ≤ realized) :
    max 0 (center - modulus * radius) ≤ max 0 realized := by
  exact max_le_max_left _ hvariation

end PldrStageGain
end PldrLlmCurvatureSandpile
