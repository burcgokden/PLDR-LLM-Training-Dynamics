/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Nonautomatic PLGA quotient transfer
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PLGAQuotientTransfer

/-- Separate the response to upstream contrast from the row-constant-input
force. The response coefficient is an assumed or independently bounded
quantity, not a ratio defined by this result. -/
theorem quotient_transfer_step
    {E : Type*} [SeminormedAddCommGroup E]
    (output response force upstream : E) (kappa : ℝ)
    (houtput : output = response + force)
    (hresponse : ‖response‖ ≤ kappa * ‖upstream‖) :
    ‖output‖ ≤ kappa * ‖upstream‖ + ‖force‖ := by
  rw [houtput]
  exact (norm_add_le response force).trans
    (add_le_add hresponse le_rfl)

/-- A strict frozen response cap gives one-step attenuation when the
row-constant-input force vanishes. -/
theorem strict_attenuation_step
    {E : Type*} [SeminormedAddCommGroup E]
    (output response upstream : E) (kappa : ℝ)
    (houtput : output = response)
    (hresponse : ‖response‖ ≤ kappa * ‖upstream‖) :
    ‖output‖ ≤ kappa * ‖upstream‖ := by
  simpa [houtput] using hresponse

end PLGAQuotientTransfer
end PldrLlmCurvatureSandpile
