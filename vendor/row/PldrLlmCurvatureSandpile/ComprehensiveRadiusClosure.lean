/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Forward radius closure for the local robustness theorem
-/
import Mathlib
import PldrLlmCurvatureSandpile.ComprehensiveRowMapCollapse

namespace PldrLlmCurvatureSandpile
namespace ComprehensiveRowMapCollapse

/-- The forward radius recursion derives tube membership rather than assuming
it. This closes the hypothesis needed by the local affine comparison. -/
theorem comprehensive_block_envelope_from_C3
    (state gain quadratic force radius : ℕ → ℝ) {initial : ℝ}
    (hstateNonnegative : ∀ block, 0 ≤ state block)
    (hgain : ∀ block, 0 ≤ gain block)
    (hquadratic : ∀ block, 0 ≤ quadratic block)
    (hradiusNonnegative : ∀ block, 0 ≤ radius block)
    (hstateStart : state 0 ≤ radius 0)
    (henvelopeStart : state 0 ≤ initial)
    (hC3 : ∀ block,
      gain block * radius block
        + quadratic block * (radius block) ^ 2 + force block
        ≤ radius (block + 1))
    (hstep : ∀ block,
      state (block + 1) ≤ gain block * state block
        + quadratic block * (state block) ^ 2 + force block) :
    (∀ block, state block ≤ radius block) ∧
    (∀ block, state block ≤
      NonautonomousAttraction.orderedEnvelope
        (fun index => gain index + quadratic index * radius index)
        force initial block) := by
  have hinTube : ∀ block, state block ≤ radius block := by
    intro block
    induction block with
    | zero => exact hstateStart
    | succ block inductionHypothesis =>
        have hsquare : (state block) ^ 2 ≤ (radius block) ^ 2 :=
          (sq_le_sq₀ (hstateNonnegative block)
            (hradiusNonnegative block)).2 inductionHypothesis
        calc
          state (block + 1) ≤
              gain block * state block
                + quadratic block * (state block) ^ 2 + force block :=
            hstep block
          _ ≤ gain block * radius block
                + quadratic block * (radius block) ^ 2 + force block := by
            exact add_le_add
              (add_le_add
                (mul_le_mul_of_nonneg_left inductionHypothesis (hgain block))
                (mul_le_mul_of_nonneg_left hsquare (hquadratic block)))
              le_rfl
          _ ≤ radius (block + 1) := hC3 block
  constructor
  · exact hinTube
  · exact comprehensive_block_envelope
      state gain quadratic force radius
      hstateNonnegative hgain hquadratic hradiusNonnegative hinTube
      henvelopeStart hstep

end ComprehensiveRowMapCollapse
end PldrLlmCurvatureSandpile
