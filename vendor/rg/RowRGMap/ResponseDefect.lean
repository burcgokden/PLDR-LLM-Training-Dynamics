import Mathlib

/-!
# Response plus invariance defect

An abstract normed-space kernel for downstream PLGA transfer.  It makes the
constant-input defect an explicit source coordinate.
-/

namespace RowRGMap
namespace ResponseDefect

variable {V : Type*} [SeminormedAddCommGroup V]

theorem response_plus_defect_bound
    (output response defect : V) (inputNorm kappa : ℝ)
    (hdecomposition : output = response + defect)
    (hresponse : ‖response‖ ≤ kappa * inputNorm) :
    ‖output‖ ≤ kappa * inputNorm + ‖defect‖ := by
  calc
    ‖output‖ = ‖response + defect‖ := by rw [hdecomposition]
    _ ≤ ‖response‖ + ‖defect‖ := norm_add_le _ _
    _ ≤ kappa * inputNorm + ‖defect‖ :=
      add_le_add hresponse (le_refl _)

theorem nonzero_defect_obstructs_zero_output
    (output defect : V) (hface : output = defect)
    (hdefect : defect ≠ 0) : output ≠ 0 := by
  simpa [hface] using hdefect

end ResponseDefect
end RowRGMap
