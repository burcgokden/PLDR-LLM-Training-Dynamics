/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Source-state dominance and chronological collapse bounds
-/
import Mathlib
import PldrLlmCurvatureSandpile.NonautonomousAttraction

namespace PldrLlmCurvatureSandpile
namespace SourceStateCollapse

/-- A pairwise source-state slack lower bound gives the displayed affine
diameter recurrence. -/
theorem state_dominance_step
    {currentDiameterSq nextDiameterSq minimumSlack alpha beta : ℝ}
    (hdiameter : nextDiameterSq ≤ currentDiameterSq - minimumSlack)
    (hdominance : alpha * currentDiameterSq - beta ≤ minimumSlack) :
    nextDiameterSq ≤ (1 - alpha) * currentDiameterSq + beta := by
  linarith

/-- Chronological source-state recurrences compose in the actual update
order, including nonzero forcing. -/
theorem ordered_forced_bound
    (diameterSq gain forcing : ℕ → ℝ) {initial : ℝ}
    (hstart : diameterSq 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep : ∀ step,
      diameterSq (step + 1)
        ≤ gain step * diameterSq step + forcing step) :
    ∀ step,
      diameterSq step ≤
        NonautonomousAttraction.orderedEnvelope gain forcing initial step := by
  exact NonautonomousAttraction.ordered_product_convolution
    diameterSq gain forcing hstart hgain hstep

/-- With zero forcing and a uniform nonnegative contraction factor, every
finite-horizon squared diameter is bounded by the corresponding power. -/
theorem homogeneous_geometric_bound
    (diameterSq : ℕ → ℝ) {initial gain : ℝ}
    (hstart : diameterSq 0 ≤ initial)
    (hgainNonnegative : 0 ≤ gain)
    (hstep : ∀ step,
      diameterSq (step + 1) ≤ gain * diameterSq step) :
    ∀ step, diameterSq step ≤ gain ^ step * initial := by
  intro step
  induction step with
  | zero => simpa using hstart
  | succ step inductionHypothesis =>
      calc
        diameterSq (step + 1) ≤ gain * diameterSq step := hstep step
        _ ≤ gain * (gain ^ step * initial) :=
          mul_le_mul_of_nonneg_left inductionHypothesis hgainNonnegative
        _ = gain ^ (step + 1) * initial := by ring

end SourceStateCollapse
end PldrLlmCurvatureSandpile
