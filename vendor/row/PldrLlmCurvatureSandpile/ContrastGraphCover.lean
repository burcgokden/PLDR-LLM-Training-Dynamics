/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Weighted path energy and physical-cover kernels
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace ContrastGraphCover

/-- Weighted scalar energy along a finite path. -/
def weightedPathEnergy {n : ℕ}
    (weight increment : Fin n → ℝ) : ℝ :=
  ∑ i, weight i * (increment i) ^ 2

/-- Cauchy--Schwarz plus a positive lower edge weight controls the squared
endpoint increment of a telescoping scalar path.  This is the coordinate
kernel of the vector graph-energy proof. -/
theorem path_difference_bound {n : ℕ}
    (weight increment : Fin n → ℝ) {minimumWeight : ℝ}
    (hminimum : 0 < minimumWeight)
    (hweight : ∀ i, minimumWeight ≤ weight i) :
    minimumWeight * (∑ i, increment i) ^ 2
      ≤ (n : ℝ) * weightedPathEnergy weight increment := by
  have hcs :
      (∑ i, increment i) ^ 2
        ≤ (n : ℝ) * ∑ i, (increment i) ^ 2 := by
    simpa using
      (sq_sum_le_card_mul_sum_sq
        (s := (Finset.univ : Finset (Fin n))) (f := increment))
  have hweighted :
      minimumWeight * ∑ i, (increment i) ^ 2
        ≤ weightedPathEnergy weight increment := by
    rw [Finset.mul_sum]
    unfold weightedPathEnergy
    exact Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_right (hweight i) (sq_nonneg _)
  calc
    minimumWeight * (∑ i, increment i) ^ 2
        ≤ minimumWeight * ((n : ℝ) * ∑ i, (increment i) ^ 2) :=
      mul_le_mul_of_nonneg_left hcs (le_of_lt hminimum)
    _ = (n : ℝ) *
        (minimumWeight * ∑ i, (increment i) ^ 2) := by ring
    _ ≤ (n : ℝ) * weightedPathEnergy weight increment :=
      mul_le_mul_of_nonneg_left hweighted (by positivity)

/-- Approximating both physical endpoints by cover vertices contributes two
Lipschitz cover radii. -/
theorem physical_cover_bound
    {physicalDistance vertexDistance lipschitz coverRadius : ℝ}
    (hleft : physicalDistance ≤
      lipschitz * coverRadius + vertexDistance
        + lipschitz * coverRadius) :
    physicalDistance ≤ vertexDistance + 2 * lipschitz * coverRadius := by
  calc
    physicalDistance ≤
        lipschitz * coverRadius + vertexDistance
          + lipschitz * coverRadius := hleft
    _ = vertexDistance + 2 * lipschitz * coverRadius := by ring

end ContrastGraphCover
end PldrLlmCurvatureSandpile
