/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Composite Fisher frame
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace CompositeFisherFrame

/-- A finite sum of nonnegative terms vanishes exactly when every term
vanishes. -/
theorem nonnegative_sum_eq_zero_iff
    {ι : Type*} [Fintype ι] (term : ι → ℝ)
    (hnonnegative : ∀ index, 0 ≤ term index) :
    (∑ index, term index = 0) ↔ ∀ index, term index = 0 := by
  constructor
  · intro hsum index
    have hsingle : term index ≤ ∑ item, term item := by
      exact Finset.single_le_sum
        (fun item _ => hnonnegative item) (Finset.mem_univ index)
    rw [hsum] at hsingle
    exact le_antisymm hsingle (hnonnegative index)
  · intro hzero
    simp [hzero]

/-- Positive weights do not change the common kernel of a finite family of
scalar observations. -/
theorem weighted_square_common_kernel
    {ι : Type*} [Fintype ι] (weight observation : ι → ℝ)
    (hweight : ∀ index, 0 < weight index) :
    (∑ index, weight index * (observation index) ^ 2 = 0)
      ↔ ∀ index, observation index = 0 := by
  rw [nonnegative_sum_eq_zero_iff
    (fun index => weight index * (observation index) ^ 2)]
  · constructor
    · intro hzero index
      have hproduct := hzero index
      rcases mul_eq_zero.mp hproduct with hweightZero | hsquare
      · exact False.elim ((ne_of_gt (hweight index)) hweightZero)
      · exact sq_eq_zero_iff.mp hsquare
    · intro hzero index
      simp [hzero index]
  · intro index
    exact mul_nonneg (le_of_lt (hweight index)) (sq_nonneg _)

/-- The complete observation energy is nonnegative. -/
theorem composite_frame_nonnegative
    {ι : Type*} [Fintype ι] (weight observation : ι → ℝ)
    (hweight : ∀ index, 0 ≤ weight index) :
    0 ≤ ∑ index, weight index * (observation index) ^ 2 := by
  exact Finset.sum_nonneg fun index _ =>
    mul_nonneg (hweight index) (sq_nonneg _)

end CompositeFisherFrame
end PldrLlmCurvatureSandpile
