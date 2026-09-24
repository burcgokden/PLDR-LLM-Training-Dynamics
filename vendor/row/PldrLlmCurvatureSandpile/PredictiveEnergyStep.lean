/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Two-sided predictive pair-energy step

This module checks the finite-step energy enclosure obtained from a
source-state directional derivative and a norm-bounded integral remainder.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PredictiveEnergyStep

/-- The complete upper increment used by the causal pair-energy theorem. -/
def upperIncrement {E : Type*} [SeminormedAddCommGroup E]
    [InnerProductSpace ℝ E] (source directional : E) (radius : ℝ) : ℝ :=
  2 * inner ℝ source directional + ‖directional‖ ^ 2
    + 2 * (‖source‖ + ‖directional‖) * radius + radius ^ 2

/-- The complete lower increment used by the causal re-expansion theorem. -/
def lowerIncrement {E : Type*} [SeminormedAddCommGroup E]
    [InnerProductSpace ℝ E] (source directional : E) (radius : ℝ) : ℝ :=
  2 * inner ℝ source directional + ‖directional‖ ^ 2
    - 2 * (‖source‖ + ‖directional‖) * radius

/-- A norm-bounded parameter-segment remainder yields simultaneous upper
and lower bounds on the exact squared pair-energy increment. -/
theorem two_sided_pair_energy
    {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    (source directional remainder : E) {radius : ℝ}
    (hradius : 0 ≤ radius) (hremainder : ‖remainder‖ ≤ radius) :
    lowerIncrement source directional radius ≤
        ‖source + directional + remainder‖ ^ 2 - ‖source‖ ^ 2 ∧
      ‖source + directional + remainder‖ ^ 2 - ‖source‖ ^ 2 ≤
        upperIncrement source directional radius := by
  have hnorm : ‖source + directional‖ ≤ ‖source‖ + ‖directional‖ :=
    norm_add_le source directional
  have hproduct :
      ‖source + directional‖ * ‖remainder‖ ≤
        (‖source‖ + ‖directional‖) * radius := by
    exact mul_le_mul hnorm hremainder (norm_nonneg remainder)
      (add_nonneg (norm_nonneg source) (norm_nonneg directional))
  have habs :
      |inner ℝ (source + directional) remainder| ≤
        (‖source‖ + ‖directional‖) * radius :=
    (abs_real_inner_le_norm (source + directional) remainder).trans hproduct
  have hlowerInner :
      -((‖source‖ + ‖directional‖) * radius) ≤
        inner ℝ (source + directional) remainder :=
    neg_le_of_abs_le habs
  have hupperInner :
      inner ℝ (source + directional) remainder ≤
        (‖source‖ + ‖directional‖) * radius :=
    le_trans (le_abs_self _) habs
  have hsquare : ‖remainder‖ ^ 2 ≤ radius ^ 2 :=
    (sq_le_sq₀ (norm_nonneg remainder) hradius).2 hremainder
  rw [norm_add_sq_real (source + directional) remainder]
  rw [norm_add_sq_real source directional]
  constructor
  · dsimp [lowerIncrement]
    nlinarith [sq_nonneg ‖remainder‖]
  · dsimp [upperIncrement]
    nlinarith

end PredictiveEnergyStep
end PldrLlmCurvatureSandpile
