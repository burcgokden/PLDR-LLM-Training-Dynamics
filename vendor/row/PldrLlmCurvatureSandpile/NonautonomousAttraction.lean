/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Nonautonomous invariant tube and ordered convolution
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace NonautonomousAttraction

/-- Chronological affine product-convolution envelope. -/
def orderedEnvelope (gain force : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | step + 1 =>
      gain step * orderedEnvelope gain force initial step + force step

/-- One-step affine bounds iterate in their actual chronological order. -/
theorem ordered_product_convolution
    (state gain force : ℕ → ℝ) {initial : ℝ}
    (hstart : state 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep :
      ∀ step, state (step + 1) ≤ gain step * state step + force step) :
    ∀ step, state step ≤ orderedEnvelope gain force initial step := by
  intro step
  induction step with
  | zero => simpa [orderedEnvelope] using hstart
  | succ step inductionHypothesis =>
      calc
        state (step + 1)
            ≤ gain step * state step + force step := hstep step
        _ ≤ gain step * orderedEnvelope gain force initial step
              + force step :=
          add_le_add
            (mul_le_mul_of_nonneg_left inductionHypothesis (hgain step)) le_rfl
        _ = orderedEnvelope gain force initial (step + 1) := by
          rw [orderedEnvelope]

/-- A time-dependent quadratic recurrence remains in a common invariant
tube when every registered step maps the tube radius into itself. -/
theorem nonautonomous_invariant_tube
    (state gain force : ℕ → ℝ) {quadratic radius : ℝ}
    (hstateNonnegative : ∀ step, 0 ≤ state step)
    (hgain : ∀ step, 0 ≤ gain step)
    (hquadratic : 0 ≤ quadratic)
    (hradius : 0 ≤ radius)
    (hstart : state 0 ≤ radius)
    (htube :
      ∀ step,
        gain step * radius + quadratic * radius ^ 2 + force step
          ≤ radius)
    (hstep :
      ∀ step,
        state (step + 1)
          ≤ gain step * state step
            + quadratic * (state step) ^ 2 + force step) :
    ∀ step, state step ≤ radius := by
  intro step
  induction step with
  | zero => exact hstart
  | succ step inductionHypothesis =>
      have hsquare :
          (state step) ^ 2 ≤ radius ^ 2 :=
        (sq_le_sq₀ (hstateNonnegative step) hradius).2
          inductionHypothesis
      calc
        state (step + 1)
            ≤ gain step * state step
                + quadratic * (state step) ^ 2 + force step :=
          hstep step
        _ ≤ gain step * radius
              + quadratic * radius ^ 2 + force step := by
          exact add_le_add
            (add_le_add
              (mul_le_mul_of_nonneg_left inductionHypothesis (hgain step))
              (mul_le_mul_of_nonneg_left hsquare hquadratic))
            le_rfl
        _ ≤ radius := htube step

end NonautonomousAttraction
end PldrLlmCurvatureSandpile
