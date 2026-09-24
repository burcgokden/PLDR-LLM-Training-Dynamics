/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Ordered observable capture and forced tubes
-/
import Mathlib
import PldrLlmCurvatureSandpile.NonautonomousAttraction

namespace PldrLlmCurvatureSandpile
namespace ObservableCapture

/-- Chronological scalar product in physical update order. -/
def chronologicalProduct (gain : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | step + 1 => gain step * chronologicalProduct gain step

/-- With zero forcing, the exact recurrence is the chronological product
times the initial observable. -/
theorem zero_force_exact_product
    (state gain : ℕ → ℝ)
    (hstep : ∀ step, state (step + 1) = gain step * state step) :
    ∀ step, state step = chronologicalProduct gain step * state 0 := by
  intro step
  induction step with
  | zero => simp [chronologicalProduct]
  | succ step inductionHypothesis =>
      rw [hstep step, inductionHypothesis, chronologicalProduct]
      ring

/-- The product-convolution envelope used for measured observable capture is
the nonautonomous ordered envelope. -/
theorem ordered_capture_envelope
    (state gain force : ℕ → ℝ) {initial : ℝ}
    (hstart : state 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep :
      ∀ step, state (step + 1) ≤ gain step * state step + force step) :
    ∀ step,
      state step
        ≤ NonautonomousAttraction.orderedEnvelope gain force initial step :=
  NonautonomousAttraction.ordered_product_convolution
    state gain force hstart hgain hstep

/-- Constant nonzero forcing has the familiar affine fixed point whenever
the gain differs from one. -/
theorem forced_fixed_point
    (gain force : ℝ) (hgain : gain ≠ 1) :
    gain * (force / (1 - gain)) + force = force / (1 - gain) := by
  field_simp [sub_ne_zero.mpr hgain]
  ring

end ObservableCapture
end PldrLlmCurvatureSandpile
