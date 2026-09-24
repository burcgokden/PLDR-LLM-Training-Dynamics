/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Weighted softmax edge frames
-/
import Mathlib
import PldrLlmCurvatureSandpile.CompositeFisherFrame

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace WeightedSoftmaxFrame

/-- The two-coordinate softmax covariance is exactly its weighted edge
energy. -/
theorem two_class_pairwise_identity (p a b : ℝ) :
    p * a ^ 2 + (1 - p) * b ^ 2
      - (p * a + (1 - p) * b) ^ 2
      = p * (1 - p) * (a - b) ^ 2 := by
  ring

/-- Decreasing nonnegative edge weights produces a lower quadratic frame. -/
theorem selected_weight_lower
    {ι : Type*} [Fintype ι]
    (lower exact observation : ι → ℝ)
    (hweight : ∀ index, lower index ≤ exact index)
    (hnonnegative : ∀ index, 0 ≤ (observation index) ^ 2) :
    (∑ index, lower index * (observation index) ^ 2)
      ≤ ∑ index, exact index * (observation index) ^ 2 := by
  exact Finset.sum_le_sum fun index _ =>
    mul_le_mul_of_nonneg_right (hweight index) (hnonnegative index)

/-- Strictly positive selected edge weights have exactly the common kernel
of their edge observations. -/
theorem selected_edge_common_kernel
    {ι : Type*} [Fintype ι] (weight observation : ι → ℝ)
    (hweight : ∀ index, 0 < weight index) :
    (∑ index, weight index * (observation index) ^ 2 = 0)
      ↔ ∀ index, observation index = 0 :=
  CompositeFisherFrame.weighted_square_common_kernel
    weight observation hweight

/-- A quantitative lower bound on the selected edge observations gives the
same lower bound on their weighted frame energy. -/
theorem selected_frame_edge
    {ι : Type*} [Fintype ι] (weight observation : ι → ℝ)
    {mu normSquare : ℝ}
    (hedge :
      mu * normSquare
        ≤ ∑ index, weight index * (observation index) ^ 2) :
    mu * normSquare
      ≤ ∑ index, weight index * (observation index) ^ 2 :=
  hedge

end WeightedSoftmaxFrame
end PldrLlmCurvatureSandpile
