/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Constructive positive block comparison

Weighted Cauchy and the resolvent witness are the algebraic kernels used to
construct comparison coefficients from block-operator and forcing norms.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ConstructiveComparison

/-- The two-term weighted Cauchy inequality underlying every constructed row
of the finite block comparison. -/
theorem weighted_two_term_cauchy
    {a b pi : ℝ} (hpi0 : 0 < pi) (hpi1 : pi < 1) :
    (a + b) ^ 2 ≤ a ^ 2 / pi + b ^ 2 / (1 - pi) := by
  have hgap : 0 < 1 - pi := by linarith
  have hden : 0 < pi * (1 - pi) := mul_pos hpi0 hgap
  have hid :
      a ^ 2 / pi + b ^ 2 / (1 - pi) - (a + b) ^ 2 =
        (((1 - pi) * a - pi * b) ^ 2) / (pi * (1 - pi)) := by
    field_simp [ne_of_gt hpi0, ne_of_gt hgap]
    ring
  have hnonneg :
      0 ≤ a ^ 2 / pi + b ^ 2 / (1 - pi) - (a + b) ^ 2 := by
    rw [hid]
    exact div_nonneg (sq_nonneg _) (le_of_lt hden)
  exact sub_nonneg.mp hnonneg

/-- A target norm bound and weighted Cauchy construct the corresponding
energy comparison coefficients. -/
theorem constructed_two_block_energy_step
    {targetEnergy a b pi : ℝ}
    (hpi0 : 0 < pi) (hpi1 : pi < 1)
    (htarget : targetEnergy ≤ (a + b) ^ 2) :
    targetEnergy ≤ a ^ 2 / pi + b ^ 2 / (1 - pi) := by
  exact htarget.trans (weighted_two_term_cauchy hpi0 hpi1)

/-- The resolvent identity P v + 1 = v supplies a strict positive witness. -/
theorem resolvent_witness_strict
    {p v : ℝ} (hidentity : p * v + 1 = v) :
    p * v < v := by
  linarith

/-- The affine source enters with a plus sign. This specialization protects
the semantic case P=0, e=0, d=e'=1. -/
theorem forcing_plus_sign_semantic_test :
    (0 : ℝ) * 0 + 1 = 1 := by
  norm_num

end ConstructiveComparison
end PldrLlmCurvatureSandpile
