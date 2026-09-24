import Mathlib

/-! Selected real Hilbert and scalar algebra. Probability-space realization,
centering, integrability and all asymptotic rates are proved in the manuscript. -/
namespace ModelRG

theorem readout_squared_norm_error_bound
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v e : E) :
    |‖v + e‖ ^ 2 - ‖v‖ ^ 2| ≤ 2 * ‖v‖ * ‖e‖ + ‖e‖ ^ 2 := by
  have h := abs_real_inner_le_norm v e
  obtain ⟨hl, hu⟩ := abs_le.mp h
  rw [norm_add_sq_real]
  apply abs_le.mpr
  constructor <;> nlinarith [sq_nonneg ‖e‖]

theorem readout_relative_scale_budget (s d r : ℝ)
    (hs : 0 ≤ s) (hd : 0 ≤ d) (hr : 0 ≤ r) (herr : d ≤ r * s) :
    2 * s * d + d ^ 2 ≤ (2 * r + r ^ 2) * s ^ 2 := by
  have hfirst := mul_le_mul_of_nonneg_left herr hs
  have hsecond : d ^ 2 ≤ (r * s) ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr herr) (show 0 ≤ r*s+d by positivity)]
  nlinarith

theorem readout_combined_law_budget (a b c D s d : ℝ)
    (hlaw : |a - b| ≤ D)
    (hreadout : |b - c| ≤ 2 * s * d + d ^ 2) :
    |a - c| ≤ D + 2 * s * d + d ^ 2 := by
  calc
    |a - c| = |(a - b) + (b - c)| := by congr 1; ring
    _ ≤ |a - b| + |b - c| := abs_add_le _ _
    _ ≤ D + (2 * s * d + d ^ 2) := add_le_add hlaw hreadout
    _ = D + 2 * s * d + d ^ 2 := by ring

end ModelRG

