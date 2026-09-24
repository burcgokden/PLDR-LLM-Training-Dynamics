/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Comprehensive forced block-normal comparison
-/
import Mathlib
import PldrLlmCurvatureSandpile.AdaptiveTube
import PldrLlmCurvatureSandpile.NonautonomousAttraction
import PldrLlmCurvatureSandpile.PathMetricContraction

namespace PldrLlmCurvatureSandpile
namespace ComprehensiveRowMapCollapse

/-- Inside a registered nonlinear tube, the exact quadratic block recurrence
is bounded by the chronological affine product-convolution with effective
gain `gain + quadratic * radius`. -/
theorem comprehensive_block_envelope
    (state gain quadratic force radius : ℕ → ℝ) {initial : ℝ}
    (hstateNonnegative : ∀ block, 0 ≤ state block)
    (hgain : ∀ block, 0 ≤ gain block)
    (hquadratic : ∀ block, 0 ≤ quadratic block)
    (hradiusNonnegative : ∀ block, 0 ≤ radius block)
    (hinTube : ∀ block, state block ≤ radius block)
    (hstart : state 0 ≤ initial)
    (hstep : ∀ block,
      state (block + 1) ≤ gain block * state block
        + quadratic block * (state block) ^ 2 + force block) :
    ∀ block, state block ≤
      NonautonomousAttraction.orderedEnvelope
        (fun index => gain index + quadratic index * radius index)
        force initial block := by
  apply NonautonomousAttraction.ordered_product_convolution
  · exact hstart
  · intro block
    exact add_nonneg (hgain block)
      (mul_nonneg (hquadratic block) (hradiusNonnegative block))
  · intro block
    exact AdaptiveTube.variable_radius_affine_step
      (hstateNonnegative block) (hinTube block) (hquadratic block)
      (hstep block)

/-- A uniform force budget below the missing contraction margin gives a
forward-invariant physical radius. -/
theorem forced_invariant_radius
    (state force : ℕ → ℝ) {gain forceBound radius : ℝ}
    (hstateNonnegative : ∀ block, 0 ≤ state block)
    (hgainNonnegative : 0 ≤ gain)
    (hradiusNonnegative : 0 ≤ radius)
    (hforce : ∀ block, force block ≤ forceBound)
    (hbudget : forceBound ≤ (1 - gain) * radius)
    (hstart : state 0 ≤ radius)
    (hstep : ∀ block,
      state (block + 1) ≤ gain * state block + force block) :
    ∀ block, state block ≤ radius := by
  apply NonautonomousAttraction.nonautonomous_invariant_tube
      state (fun _ => gain) force
      (quadratic := 0) (radius := radius)
  · exact hstateNonnegative
  · intro
    exact hgainNonnegative
  · norm_num
  · exact hradiusNonnegative
  · exact hstart
  · intro block
    have hforceBlock := hforce block
    nlinarith
  · intro block
    simpa using hstep block

/-- Uniform metric equivalence transfers a scheduled normal-envelope bound
to the physical row-map quotient. -/
theorem normal_envelope_to_physical
    {physical scheduled envelope lower : ℝ}
    (hlower : 0 < lower)
    (henergy : lower * physical ^ 2 ≤ scheduled ^ 2)
    (hscheduled : 0 ≤ scheduled)
    (henvelope : scheduled ≤ envelope) :
    physical ≤ envelope / Real.sqrt lower := by
  exact (PathMetricContraction.metric_to_physical hlower henergy
    hscheduled).trans
      (div_le_div_of_nonneg_right henvelope (Real.sqrt_nonneg lower))

end ComprehensiveRowMapCollapse
end PldrLlmCurvatureSandpile
