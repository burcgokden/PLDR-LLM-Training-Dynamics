/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Physical chordal row covers

This module checks the geometric step that transfers a registered pairwise
segment cover to nearby physical rows without filling the ambient coordinate
box.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ChordalRowCover

/-- If both endpoints move by at most epsilon, every point on their joining
segment moves by at most epsilon. -/
theorem segment_endpoint_perturbation
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (x y a b : E) {s epsilon : ℝ}
    (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (hxa : ‖x - a‖ ≤ epsilon)
    (hyb : ‖y - b‖ ≤ epsilon) :
    ‖((1 - s) • x + s • y) - ((1 - s) • a + s • b)‖ ≤ epsilon := by
  have hone : 0 ≤ 1 - s := sub_nonneg.mpr hs1
  have hdecomp :
      ((1 - s) • x + s • y) - ((1 - s) • a + s • b)
        = (1 - s) • (x - a) + s • (y - b) := by
    module
  calc
    ‖((1 - s) • x + s • y) - ((1 - s) • a + s • b)‖
        = ‖(1 - s) • (x - a) + s • (y - b)‖ := by rw [hdecomp]
    _ ≤ ‖(1 - s) • (x - a)‖ + ‖s • (y - b)‖ := norm_add_le _ _
    _ = (1 - s) * ‖x - a‖ + s * ‖y - b‖ := by
      simp [norm_smul, Real.norm_eq_abs, abs_of_nonneg hone,
        abs_of_nonneg hs0]
    _ ≤ (1 - s) * epsilon + s * epsilon :=
      add_le_add
        (mul_le_mul_of_nonneg_left hxa hone)
        (mul_le_mul_of_nonneg_left hyb hs0)
    _ = epsilon := by ring

/-- A registered chord net of radius h and an endpoint net of radius epsilon
give a physical chord net of radius h plus epsilon. -/
theorem chordal_net_radius
    {registeredDistance endpointDistance h epsilon : ℝ}
    (hregistered : registeredDistance ≤ h)
    (hendpoint : endpointDistance ≤ epsilon) :
    registeredDistance + endpointDistance ≤ h + epsilon :=
  add_le_add hregistered hendpoint

/-- The derivative-Lipschitz transfer uses the complete chordal radius. -/
theorem chordal_derivative_transfer
    {direct grid L h epsilon : ℝ}
    (hbound : direct ≤ grid + L * (h + epsilon)) :
    direct ≤ grid + L * h + L * epsilon := by
  calc
    direct ≤ grid + L * (h + epsilon) := hbound
    _ = grid + L * h + L * epsilon := by ring

end ChordalRowCover
end PldrLlmCurvatureSandpile
