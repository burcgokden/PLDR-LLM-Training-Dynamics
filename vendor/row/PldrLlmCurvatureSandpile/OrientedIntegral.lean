/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Oriented vector enclosure
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace OrientedIntegral

/-- Keeping the vector center until after summation preserves orientation;
only the residual radius is added outside its norm. -/
theorem oriented_center_enclosure
    {E : Type*} [SeminormedAddCommGroup E]
    (actual center error : E) {radius : ℝ}
    (hactual : actual = center + error)
    (herror : ‖error‖ ≤ radius) :
    ‖actual‖ ≤ ‖center‖ + radius := by
  rw [hactual]
  calc
    ‖center + error‖ ≤ ‖center‖ + ‖error‖ := norm_add_le center error
    _ ≤ ‖center‖ + radius := by gcongr

/-- The oriented norm of two centers is no larger than their stagewise
triangle sum. -/
theorem oriented_not_larger_than_stagewise
    {E : Type*} [SeminormedAddCommGroup E] (first second : E) :
    ‖first + second‖ ≤ ‖first‖ + ‖second‖ :=
  norm_add_le first second

end OrientedIntegral
end PldrLlmCurvatureSandpile
