/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# PLGA row-constant defect obstruction
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PLGARowConstantDefect

/-- At a row-constant upstream input, a response bound forces the response
term to vanish, so the downstream quotient is exactly the structural defect. -/
theorem constant_input_output_is_defect
    {E : Type*} [NormedAddCommGroup E]
    (output response defect upstream : E) (kappa : ℝ)
    (hupstream : upstream = 0)
    (houtput : output = response + defect)
    (hresponse : ‖response‖ ≤ kappa * ‖upstream‖) :
    output = defect := by
  have hresponseZero : ‖response‖ = 0 := by
    apply le_antisymm
    · simpa [hupstream] using hresponse
    · exact norm_nonneg response
  have : response = 0 := norm_eq_zero.mp hresponseZero
  simp [houtput, this]

/-- A nonzero row-constant-input defect rules out automatic preservation of
the upstream row-constant face. -/
theorem nonzero_defect_obstructs_row_preservation
    {E : Type*} [NormedAddCommGroup E]
    (output response defect upstream : E) (kappa : ℝ)
    (hupstream : upstream = 0)
    (houtput : output = response + defect)
    (hresponse : ‖response‖ ≤ kappa * ‖upstream‖)
    (hdefect : defect ≠ 0) :
    output ≠ 0 := by
  rw [constant_input_output_is_defect output response defect upstream kappa
    hupstream houtput hresponse]
  exact hdefect

/-- The quotient-defect triangle bound remains valid without invariance. -/
theorem quotient_defect_bound
    {E : Type*} [SeminormedAddCommGroup E]
    (output response defect upstream : E) (kappa : ℝ)
    (houtput : output = response + defect)
    (hresponse : ‖response‖ ≤ kappa * ‖upstream‖) :
    ‖output‖ ≤ kappa * ‖upstream‖ + ‖defect‖ := by
  rw [houtput]
  exact (norm_add_le response defect).trans (add_le_add hresponse le_rfl)

end PLGARowConstantDefect
end PldrLlmCurvatureSandpile
