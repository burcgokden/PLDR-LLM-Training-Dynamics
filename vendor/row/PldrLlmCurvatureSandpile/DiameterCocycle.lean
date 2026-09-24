/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Source-derived diameter cocycle

The scalar multiplier at each step is supplied by a source-only upper
diameter certificate. Individual multipliers may exceed one. The results
below check chronological iteration and collapse when their ordered product
tends to zero.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace DiameterCocycle

open Filter

/-- Chronological product of a finite block beginning at the source. -/
def orderedProduct (multiplier : ℕ → ℝ) (start horizon : ℕ) : ℝ :=
  ∏ offset ∈ Finset.range horizon, multiplier (start + offset)

/-- A source-derived one-step diameter multiplier iterates in chronological
order without an added forcing sequence. -/
theorem diameter_product_bound
    (diameterSq multiplier : ℕ → ℝ) (start : ℕ)
    (hmultiplier : ∀ index, 0 ≤ multiplier index)
    (hstep : ∀ index,
      diameterSq (index + 1) ≤ multiplier index * diameterSq index) :
    ∀ horizon,
      diameterSq (start + horizon) ≤
        orderedProduct multiplier start horizon * diameterSq start := by
  intro horizon
  induction horizon with
  | zero => simp [orderedProduct]
  | succ horizon inductionHypothesis =>
      calc
        diameterSq (start + (horizon + 1))
            = diameterSq ((start + horizon) + 1) := by
              congr 1
        _ ≤ multiplier (start + horizon) * diameterSq (start + horizon) :=
          hstep (start + horizon)
        _ ≤ multiplier (start + horizon) *
              (orderedProduct multiplier start horizon * diameterSq start) :=
          mul_le_mul_of_nonneg_left inductionHypothesis
            (hmultiplier (start + horizon))
        _ = orderedProduct multiplier start (horizon + 1) *
              diameterSq start := by
          simp [orderedProduct, Finset.prod_range_succ]
          ring

/-- Source-derived lower multipliers iterate in the same chronological
order and provide a lower diameter-product certificate. -/
theorem diameter_product_lower_bound
    (diameterSq multiplier : ℕ → ℝ) (start : ℕ)
    (hmultiplier : ∀ index, 0 ≤ multiplier index)
    (hstep : ∀ index,
      multiplier index * diameterSq index ≤ diameterSq (index + 1)) :
    ∀ horizon,
      orderedProduct multiplier start horizon * diameterSq start ≤
        diameterSq (start + horizon) := by
  intro horizon
  induction horizon with
  | zero => simp [orderedProduct]
  | succ horizon inductionHypothesis =>
      calc
        orderedProduct multiplier start (horizon + 1) * diameterSq start
            = multiplier (start + horizon) *
              (orderedProduct multiplier start horizon * diameterSq start) := by
                simp [orderedProduct, Finset.prod_range_succ]
                ring
        _ ≤ multiplier (start + horizon) * diameterSq (start + horizon) :=
          mul_le_mul_of_nonneg_left inductionHypothesis
            (hmultiplier (start + horizon))
        _ ≤ diameterSq ((start + horizon) + 1) :=
          hstep (start + horizon)
        _ = diameterSq (start + (horizon + 1)) := by
          congr 1

/-- If the chronological product tends to zero, every nonnegative diameter
sequence below its source-scaled product also tends to zero. -/
theorem diameter_tendsto_zero
    (diameterSq multiplier : ℕ → ℝ) (start : ℕ)
    (hdiameter : ∀ index, 0 ≤ diameterSq index)
    (hbound : ∀ horizon,
      diameterSq (start + horizon) ≤
        orderedProduct multiplier start horizon * diameterSq start)
    (hproduct : Tendsto
      (fun horizon => orderedProduct multiplier start horizon)
      atTop (nhds 0)) :
    Tendsto (fun horizon => diameterSq (start + horizon)) atTop (nhds 0) := by
  apply squeeze_zero
  · intro horizon
    exact hdiameter (start + horizon)
  · exact hbound
  · simpa using hproduct.mul_const (diameterSq start)

end DiameterCocycle
end PldrLlmCurvatureSandpile
