/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Ordered path-metric contraction
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PathMetricChain

/-- Two consecutive quadratic metric inequalities retain their order and
multiply their gains. -/
theorem two_stage_quadratic_contraction
    {E0 E1 E2 q1 q2 : ℝ}
    (hfirst : E1 ≤ q1 ^ 2 * E0)
    (hsecond : E2 ≤ q2 ^ 2 * E1) :
    E2 ≤ (q2 * q1) ^ 2 * E0 := by
  calc
    E2 ≤ q2 ^ 2 * E1 := hsecond
    _ ≤ q2 ^ 2 * (q1 ^ 2 * E0) :=
      mul_le_mul_of_nonneg_left hfirst (sq_nonneg q2)
    _ = (q2 * q1) ^ 2 * E0 := by ring

/-- Chronologically ordered one-stage quadratic bounds iterate to the
product of their squared gains. -/
theorem ordered_quadratic_contraction
    (energy gain : ℕ → ℝ)
    (hstep : ∀ step,
      energy (step + 1) ≤ gain step ^ 2 * energy step) :
    ∀ horizon,
      energy horizon ≤
        (∏ step ∈ Finset.range horizon, gain step ^ 2) * energy 0 := by
  intro horizon
  induction horizon with
  | zero => simp
  | succ horizon inductionHypothesis =>
      calc
        energy (horizon + 1) ≤ gain horizon ^ 2 * energy horizon :=
          hstep horizon
        _ ≤ gain horizon ^ 2 *
            ((∏ step ∈ Finset.range horizon, gain step ^ 2) * energy 0) :=
          mul_le_mul_of_nonneg_left inductionHypothesis
            (sq_nonneg (gain horizon))
        _ = (∏ step ∈ Finset.range (horizon + 1), gain step ^ 2) *
            energy 0 := by
          rw [Finset.prod_range_succ]
          ring

end PathMetricChain
end PldrLlmCurvatureSandpile
