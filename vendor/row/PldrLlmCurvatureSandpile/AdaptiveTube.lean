/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Variable-radius nonlinear tubes
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace AdaptiveTube

/-- A quadratic term inside a radius becomes a time-local affine gain. -/
theorem quadratic_to_effective_gain
    {state radius gain quadratic : ℝ}
    (hstate : 0 ≤ state) (hradius : state ≤ radius)
    (hquadratic : 0 ≤ quadratic) :
    gain * state + quadratic * state ^ 2
      ≤ (gain + quadratic * radius) * state := by
  have hproduct : state ^ 2 ≤ radius * state := by
    nlinarith
  nlinarith [mul_le_mul_of_nonneg_left hproduct hquadratic]

/-- Time-dependent radii are forward invariant when every realized
quadratic image fits in the next registered radius. -/
theorem variable_radius_invariant
    (state gain quadratic force radius : ℕ → ℝ)
    (hstateNonnegative : ∀ step, 0 ≤ state step)
    (hgain : ∀ step, 0 ≤ gain step)
    (hquadratic : ∀ step, 0 ≤ quadratic step)
    (hradius : ∀ step, 0 ≤ radius step)
    (hstart : state 0 ≤ radius 0)
    (himage :
      ∀ step,
        gain step * radius step
          + quadratic step * (radius step) ^ 2 + force step
          ≤ radius (step + 1))
    (hstep :
      ∀ step,
        state (step + 1)
          ≤ gain step * state step
            + quadratic step * (state step) ^ 2 + force step) :
    ∀ step, state step ≤ radius step := by
  intro step
  induction step with
  | zero => exact hstart
  | succ step inductionHypothesis =>
      have hsquare :
          (state step) ^ 2 ≤ (radius step) ^ 2 :=
        (sq_le_sq₀ (hstateNonnegative step) (hradius step)).2
          inductionHypothesis
      calc
        state (step + 1)
            ≤ gain step * state step
                + quadratic step * (state step) ^ 2 + force step :=
          hstep step
        _ ≤ gain step * radius step
              + quadratic step * (radius step) ^ 2 + force step := by
          exact add_le_add
            (add_le_add
              (mul_le_mul_of_nonneg_left inductionHypothesis (hgain step))
              (mul_le_mul_of_nonneg_left hsquare (hquadratic step)))
            le_rfl
        _ ≤ radius (step + 1) := himage step

/-- Inside a registered radius, the nonlinear one-step comparison is bounded
by the effective affine gain plus the same force. -/
theorem variable_radius_affine_step
    {current next gain quadratic force radius : ℝ}
    (hcurrent : 0 ≤ current) (hradius : current ≤ radius)
    (hquadratic : 0 ≤ quadratic)
    (hnext :
      next ≤ gain * current + quadratic * current ^ 2 + force) :
    next ≤ (gain + quadratic * radius) * current + force := by
  have heffective :=
    quadratic_to_effective_gain (gain := gain)
      hcurrent hradius hquadratic
  linarith

end AdaptiveTube
end PldrLlmCurvatureSandpile
