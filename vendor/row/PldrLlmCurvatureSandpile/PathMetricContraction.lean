/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Scheduled path-metric contraction
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PathMetricContraction

/-- The scalar backward metric recursion has an exact unit dissipation. -/
theorem backward_metric_identity (operator nextMetric : ℝ) :
    let metric := 1 + operator ^ 2 * nextMetric
    operator ^ 2 * nextMetric = metric - 1 := by
  simp

/-- A positive upper bound on the current scalar metric converts the unit
dissipation into a strict scheduled gain. -/
theorem backward_metric_gain
    {operator nextMetric metric upper : ℝ}
    (hidentity : metric = 1 + operator ^ 2 * nextMetric)
    (hupper : metric ≤ upper)
    (hupperPositive : 0 < upper) :
    operator ^ 2 * nextMetric
      ≤ (1 - upper⁻¹) * metric := by
  have hone : upper⁻¹ * metric ≤ 1 := by
    rw [inv_mul_le_one₀ hupperPositive]
    exact hupper
  rw [hidentity]
  field_simp
  nlinarith

/-- Nominal and perturbation gains add by the triangle inequality. -/
theorem robust_gain
    {nominal perturbation nominalGain perturbationGain : ℝ}
    (hnominal : |nominal| ≤ nominalGain)
    (hperturbation : |perturbation| ≤ perturbationGain) :
    |nominal + perturbation| ≤ nominalGain + perturbationGain := by
  exact (abs_add_le nominal perturbation).trans
    (add_le_add hnominal hperturbation)

/-- Uniform metric equivalence converts scheduled-norm contraction to a
physical Euclidean estimate. -/
theorem metric_to_physical
    {physical scheduled lower : ℝ}
    (hlower : 0 < lower)
    (henergy : lower * physical ^ 2 ≤ scheduled ^ 2)
    (hscheduled : 0 ≤ scheduled) :
    physical ≤ scheduled / Real.sqrt lower := by
  have hsqrt : 0 < Real.sqrt lower := Real.sqrt_pos.2 hlower
  apply (le_div_iff₀ hsqrt).2
  nlinarith [Real.sq_sqrt (le_of_lt hlower)]

end PathMetricContraction
end PldrLlmCurvatureSandpile
