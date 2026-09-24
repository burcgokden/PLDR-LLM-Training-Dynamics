/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Program-bound energy step

Directional-derivative and enclosure kernels for one owned successor edge.
-/
import Mathlib
import PldrLlmCurvatureSandpile.RowJacobianEnergy

namespace PldrLlmCurvatureSandpile
namespace ProgramEnergyStep

/-- A JVP plus an integral remainder has the expected norm enclosure. -/
theorem jvp_remainder_energy_bound
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (jvp remainder : E) {remainderBound : ℝ}
    (hremainder : ‖remainder‖ ≤ remainderBound) :
    ‖jvp + remainder‖ ≤ ‖jvp‖ + remainderBound := by
  exact (norm_add_le jvp remainder).trans
    (add_le_add le_rfl hremainder)

/-- An outward enclosure of the complete program-owned energy increment
transfers directly to the exact successor state. -/
theorem enclosed_program_edge_contracts {n : ℕ}
    (weight sourceState increment : Fin n → ℝ)
    {q source upperIncrement : ℝ}
    (henclosure :
      2 * ∑ i, weight i * sourceState i * increment i +
        RowJacobianEnergy.energy weight increment ≤ upperIncrement)
    (hbudget : upperIncrement ≤
      -(1 - q ^ 2) * RowJacobianEnergy.energy weight sourceState + source) :
    RowJacobianEnergy.energy weight (sourceState + increment) ≤
      q ^ 2 * RowJacobianEnergy.energy weight sourceState + source := by
  apply RowJacobianEnergy.negative_work_contracts
  exact henclosure.trans hbudget

end ProgramEnergyStep
end PldrLlmCurvatureSandpile
