import Mathlib

/-!
Manuscript correspondence: `prop:absolute-relative`.

Checked clause and supplied hypotheses: Scalar reconstruction with nonzero denominator; upper amplitude bound with nonnegative contrast and positive total; lower positive amplitude floor for the reverse inequality.

Written obligations: Matrix projection geometry, convergence interpretation and matrix counterexamples remain written obligations.
These kernels double-check selected clauses; the written proof is independent.
-/

namespace PldrTrainingDynamics

/-- Absolute and normalized energies require an explicit positive scale. -/
noncomputable def rowFraction (contrast total : ℝ) : ℝ := contrast / total

theorem rowFraction_reconstruct (contrast total : ℝ) (h : total ≠ 0) :
    rowFraction contrast total * total = contrast := by
  exact div_mul_cancel₀ contrast h

theorem contrast_le_scaled_fraction (contrast total cap : ℝ)
    (ht : 0 < total) (hc : 0 ≤ contrast) (hcap : total ≤ cap) :
    contrast ≤ rowFraction contrast total * cap := by
  calc
    contrast = rowFraction contrast total * total :=
      (rowFraction_reconstruct contrast total (ne_of_gt ht)).symm
    _ ≤ rowFraction contrast total * cap :=
      mul_le_mul_of_nonneg_left hcap (div_nonneg hc (le_of_lt ht))

theorem fraction_le_scaled_contrast (contrast total floor : ℝ)
    (hf : 0 < floor) (hc : 0 ≤ contrast) (ht : floor ≤ total) :
    rowFraction contrast total ≤ contrast / floor := by
  exact div_le_div_of_nonneg_left hc hf ht

end PldrTrainingDynamics
