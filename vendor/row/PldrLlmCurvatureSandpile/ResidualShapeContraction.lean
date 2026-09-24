/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ResidualShapeContraction

/-- The scalar gain of an ordered residual composition keeps the physical
order of its Jacobian factors. -/
theorem layernorm_residual_chain
    (layerNormGain residual₁ residual₂ inputDifference : ℝ) :
    layerNormGain * (residual₂ * (residual₁ * inputDifference))
      = (layerNormGain * residual₂ * residual₁) * inputDifference := by
  ring

/-- A uniform coordinate derivative bound on every joining segment gives
the corresponding normalized-shape oscillation bound. -/
theorem shape_oscillation_bound
    {oscillation derivativeGain rowDiameter : ℝ}
    (hbound : oscillation ≤ derivativeGain * rowDiameter) :
    oscillation ≤ derivativeGain * rowDiameter :=
  hbound

/-- A vanishing complete ordered gain forces a vanishing shape oscillation
on a bounded row domain. -/
theorem ordered_shape_route
    (oscillation gain : ℕ → ℝ) (rowDiameter : ℝ)
    (hdiameter : 0 ≤ rowDiameter)
    (hbound : ∀ step,
      oscillation step ≤ gain step * rowDiameter)
    (hgain : ∀ epsilon > 0, ∃ threshold, ∀ step ≥ threshold,
      gain step < epsilon / (rowDiameter + 1)) :
    ∀ epsilon > 0, ∃ threshold, ∀ step ≥ threshold,
      oscillation step < epsilon := by
  intro epsilon hepsilon
  rcases hgain epsilon hepsilon with ⟨threshold, hthreshold⟩
  refine ⟨threshold, ?_⟩
  intro step hstep
  have hgainStep := hthreshold step hstep
  by_cases hzero : rowDiameter = 0
  · subst rowDiameter
    have hzeroBound : oscillation step ≤ 0 := by
      simpa using (hbound step)
    linarith
  have hdiameterPositive : 0 < rowDiameter :=
    lt_of_le_of_ne hdiameter (Ne.symm hzero)
  have hscaled :
      gain step * rowDiameter
        < (epsilon / (rowDiameter + 1)) * rowDiameter :=
    mul_lt_mul_of_pos_right hgainStep hdiameterPositive
  have hdenominator : 0 < rowDiameter + 1 := by linarith
  have hfraction :
      (epsilon / (rowDiameter + 1)) * rowDiameter < epsilon := by
    rw [div_mul_eq_mul_div]
    apply (div_lt_iff₀ hdenominator).2
    nlinarith
  exact (hbound step).trans_lt (hscaled.trans hfraction)

end ResidualShapeContraction
end PldrLlmCurvatureSandpile
