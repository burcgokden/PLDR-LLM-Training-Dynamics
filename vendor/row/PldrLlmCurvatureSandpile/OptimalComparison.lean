/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Positive witness for a complete amplitude comparison
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace OptimalComparison

/-- A two-source positive witness converts component bounds into a weighted
sup-norm contraction. -/
theorem two_source_positive_witness
    {g₁ g₂ v₁ v₂ a₁ a₂ u κ : ℝ}
    (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (hu : 0 ≤ u)
    (ha₁ : a₁ ≤ u * v₁) (ha₂ : a₂ ≤ u * v₂)
    (hwitness : g₁ * v₁ + g₂ * v₂ ≤ κ * v₁) :
    g₁ * a₁ + g₂ * a₂ ≤ κ * u * v₁ := by
  have h₁ := mul_le_mul_of_nonneg_left ha₁ hg₁
  have h₂ := mul_le_mul_of_nonneg_left ha₂ hg₂
  calc
    g₁ * a₁ + g₂ * a₂
        ≤ g₁ * (u * v₁) + g₂ * (u * v₂) := add_le_add h₁ h₂
    _ = u * (g₁ * v₁ + g₂ * v₂) := by ring
    _ ≤ u * (κ * v₁) := mul_le_mul_of_nonneg_left hwitness hu
    _ = κ * u * v₁ := by ring

/-- The same positive witness squares to the energy witness. -/
theorem squared_witness
    {row v κ : ℝ} (hrow : 0 ≤ row) (hv : 0 ≤ v)
    (hκ : 0 ≤ κ) (hwitness : row ≤ κ * v) :
    row ^ 2 ≤ κ ^ 2 * v ^ 2 := by
  have hright : 0 ≤ κ * v := mul_nonneg hκ hv
  have := (sq_le_sq₀ hrow hright).2 hwitness
  simpa [mul_pow] using this

end OptimalComparison
end PldrLlmCurvatureSandpile
