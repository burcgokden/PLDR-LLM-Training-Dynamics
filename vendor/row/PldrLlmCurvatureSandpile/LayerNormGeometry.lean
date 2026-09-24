/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Final LayerNorm radius and gate bound kernels
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace LayerNormGeometry

/-- The squared radius of the epsilon-regularized normalized shape, written
in terms of the feature dimension and the squared centered norm. -/
noncomputable def normalizedShapeSq
    (dimension epsilon centeredSq : ℝ) : ℝ :=
  dimension * centeredSq / (dimension * epsilon + centeredSq)

/-- Positive epsilon makes the normalized-shape radius strictly smaller than
the feature dimension. -/
theorem normalized_shape_sq_lt
    {dimension epsilon centeredSq : ℝ}
    (hdimension : 0 < dimension) (hepsilon : 0 < epsilon)
    (hcentered : 0 ≤ centeredSq) :
    normalizedShapeSq dimension epsilon centeredSq < dimension := by
  unfold normalizedShapeSq
  have hdenominator : 0 < dimension * epsilon + centeredSq := by
    positivity
  apply (div_lt_iff₀ hdenominator).2
  apply mul_lt_mul_of_pos_left _ hdimension
  nlinarith [mul_pos hdimension hepsilon]

/-- The triangle inequality and a common shape-radius bound give the global
gate-controlled row-difference estimate. -/
theorem gate_diameter_bound
    {distance gateNorm leftNorm rightNorm radius : ℝ}
    (hdistance : distance ≤ gateNorm * (leftNorm + rightNorm))
    (hleft : leftNorm ≤ radius) (hright : rightNorm ≤ radius)
    (hgate : 0 ≤ gateNorm) :
    distance ≤ 2 * gateNorm * radius := by
  calc
    distance ≤ gateNorm * (leftNorm + rightNorm) := hdistance
    _ ≤ gateNorm * (radius + radius) := by
      exact mul_le_mul_of_nonneg_left (add_le_add hleft hright) hgate
    _ = 2 * gateNorm * radius := by ring

end LayerNormGeometry
end PldrLlmCurvatureSandpile
