import Mathlib

/-!
# Scalar radial and tangential energy kernel

The full manuscript statement is a finite-dimensional orthogonal
decomposition.  This module checks the scalar algebra after orthogonality has
removed the cross term.
-/

namespace RowRGMap
namespace RadialEnergy

def gain (alpha tangentCharge : ℝ) : ℝ :=
  (1 - alpha) ^ 2 + tangentCharge

theorem gain_nonnegative {alpha tangentCharge : ℝ}
    (htangent : 0 ≤ tangentCharge) :
    0 ≤ gain alpha tangentCharge := by
  exact add_nonneg (sq_nonneg (1 - alpha)) htangent

theorem relative_increment (alpha tangentCharge : ℝ) :
    gain alpha tangentCharge - 1 =
      -2 * alpha + alpha ^ 2 + tangentCharge := by
  simp [gain]
  ring

theorem strict_closing_iff {alpha tangentCharge : ℝ}
    (htangent : 0 ≤ tangentCharge) :
    gain alpha tangentCharge < 1 ↔
      0 < alpha ∧ alpha < 2 ∧
      tangentCharge < alpha * (2 - alpha) := by
  constructor
  · intro hgain
    have hmargin : tangentCharge < alpha * (2 - alpha) := by
      simp [gain] at hgain
      nlinarith
    have hpositive : 0 < alpha * (2 - alpha) :=
      lt_of_le_of_lt htangent hmargin
    constructor
    · nlinarith
    constructor
    · nlinarith
    · exact hmargin
  · rintro ⟨halphaPositive, halphaTwo, hmargin⟩
    simp [gain]
    nlinarith

end RadialEnergy
end RowRGMap
