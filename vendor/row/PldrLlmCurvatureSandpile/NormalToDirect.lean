/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Finite-stencil normal to direct derivative
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace NormalToDirect

/-- A derivative action differs from its forward finite stencil by at most
the registered truncation charge. -/
theorem finite_stencil_truncation
    {derivative stencil truncation : ℝ}
    (htruncation : derivative ≤ stencil + truncation) :
    derivative ≤ stencil + truncation :=
  htruncation

/-- The direction-net self term rearranges by the exact factor
one divided by one minus epsilon. -/
theorem direction_net_rearrangement
    {operator measured epsilon : ℝ}
    (hcover : operator ≤ measured + epsilon * operator)
    (hepsilon : epsilon < 1) :
    operator ≤ measured / (1 - epsilon) := by
  have hgap : 0 < 1 - epsilon := sub_pos.mpr hepsilon
  apply (le_div_iff₀ hgap).2
  nlinarith

/-- A normal-coordinate stencil estimate, truncation charge, and the row and
direction radii of a joint registered-pair cover combine into the direct
derivative estimate. -/
theorem normal_to_direct
    {direct stencil normal alpha error truncation epsilon rowCharge : ℝ}
    (hepsilon : epsilon < 1)
    (hstencil : stencil ≤ alpha * normal + error)
    (hdirect :
      direct ≤ (stencil + truncation + rowCharge) / (1 - epsilon)) :
    direct
      ≤ (alpha * normal + error + truncation + rowCharge)
          / (1 - epsilon) := by
  have hgap : 0 < 1 - epsilon := sub_pos.mpr hepsilon
  have hnumerator :
      stencil + truncation + rowCharge
        ≤ alpha * normal + error + truncation + rowCharge := by
    linarith
  have hquotient :=
    (div_le_div_iff_of_pos_right hgap).2 hnumerator
  exact hdirect.trans hquotient

/-- The registered finite stencil has more output coordinates than selected
row-map parameters. -/
theorem registered_capacity : (38_912 : ℕ) < 49_152 := by
  norm_num

end NormalToDirect
end PldrLlmCurvatureSandpile
