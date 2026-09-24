/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Cross-entropy on the logit quotient

Finite-sum kernels for removing the common-logit direction and for the
probability-floor curvature bound on centered representatives.
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace LogitQuotient

/-- Subtract the uniform-coordinate mean. -/
noncomputable def project {n : ℕ} [NeZero n] (w : Fin n → ℝ) : Fin n → ℝ :=
  fun i => w i - (∑ j, w j) / n

/-- The projected vector has zero coordinate sum. -/
theorem project_sum_zero {n : ℕ} [NeZero n] (w : Fin n → ℝ) :
    ∑ i, project w i = 0 := by
  have hn : (n : ℝ) ≠ 0 := by
    exact_mod_cast (NeZero.ne n)
  simp only [project, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- Projection fixes an already zero-sum vector. -/
theorem project_eq_self {n : ℕ} [NeZero n] (w : Fin n → ℝ)
    (hzero : ∑ i, w i = 0) :
    project w = w := by
  funext i
  simp [project, hzero]

/-- The categorical covariance annihilates the common-logit direction. -/
theorem common_shift_covariance_zero {n : ℕ} (p : Fin n → ℝ)
    (hprob : ∑ i, p i = 1) :
    (∑ i, p i * (1 : ℝ) ^ 2) - (∑ i, p i * (1 : ℝ)) ^ 2 = 0 := by
  simp [hprob]

/-- A coordinate probability floor gives the pmin lower edge on every
centered representative. -/
theorem probability_floor_centered_curvature {n : ℕ}
    (p v : Fin n → ℝ) (pmin : ℝ)
    (hfloor : ∀ i, pmin ≤ p i) :
    pmin * ∑ i, v i ^ 2 ≤ ∑ i, p i * v i ^ 2 := by
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i _ =>
    mul_le_mul_of_nonneg_right (hfloor i) (sq_nonneg _)

end LogitQuotient
end PldrLlmCurvatureSandpile
