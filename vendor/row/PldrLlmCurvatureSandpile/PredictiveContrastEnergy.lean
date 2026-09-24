/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Cellwise predictive contrast-energy bounds

This module checks the finite weighted-sum and two-sided order argument used
after analytic or interval validation has supplied cellwise enclosures of a
pair contrast's second directional derivative.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PredictiveContrastEnergy

/-- Weighted sum of cell bounds. -/
def weightedSum {n : ℕ} (weight value : Fin n → ℝ) : ℝ :=
  ∑ index, weight index * value index

/-- Nonnegative cell weights preserve pointwise enclosure order. -/
theorem weighted_sum_mono {n : ℕ}
    (weight lower upper : Fin n → ℝ)
    (hweight : ∀ index, 0 ≤ weight index)
    (hcell : ∀ index, lower index ≤ upper index) :
    weightedSum weight lower ≤ weightedSum weight upper := by
  apply Finset.sum_le_sum
  intro index _
  exact mul_le_mul_of_nonneg_left (hcell index) (hweight index)

/-- The exact weight of a cell in the integral with kernel 1-s. -/
noncomputable def cellWeight (left right : ℝ) : ℝ :=
  (right - left) * (1 - (left + right) / 2)

/-- A subinterval of the unit segment has nonnegative contrast-energy
weight. -/
theorem cellWeight_nonnegative {left right : ℝ}
    (hordered : left ≤ right) (hright : right ≤ 1) :
    0 ≤ cellWeight left right := by
  dsimp [cellWeight]
  have hwidth : 0 ≤ right - left := sub_nonneg.mpr hordered
  have hsum : left + right ≤ 2 := by linarith
  have hkernel : 0 ≤ 1 - (left + right) / 2 := by linarith
  exact mul_nonneg hwidth hkernel

/-- Cellwise integral bounds give a source-only two-sided successor-energy
enclosure once the exact Taylor identity is supplied. -/
theorem certified_contrast_energy_step {n : ℕ}
    (sourceEnergy sourceSlope successorEnergy remainder : ℝ)
    (weight lower upper : Fin n → ℝ)
    (hidentity : successorEnergy = sourceEnergy + sourceSlope + remainder)
    (hlower : weightedSum weight lower ≤ remainder)
    (hupper : remainder ≤ weightedSum weight upper) :
    sourceEnergy + sourceSlope + weightedSum weight lower ≤ successorEnergy ∧
      successorEnergy ≤
        sourceEnergy + sourceSlope + weightedSum weight upper := by
  constructor <;> linarith

end PredictiveContrastEnergy
end PldrLlmCurvatureSandpile
