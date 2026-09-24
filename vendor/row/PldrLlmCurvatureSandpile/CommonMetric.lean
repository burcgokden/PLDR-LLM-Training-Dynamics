/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Common metric for an affine operator interval

The endpoint argument is stated after any positive-metric conjugation, so
the ambient norm below is the common metric norm.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace CommonMetric

/-- The norm gain of an affine combination is bounded by the endpoint gains. -/
theorem endpoint_common_gain
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (A₀ A₁ : E →L[ℝ] E) {q t : ℝ}
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (h₀ : ∀ x, ‖A₀ x‖ ≤ q * ‖x‖)
    (h₁ : ∀ x, ‖A₁ x‖ ≤ q * ‖x‖) :
    ∀ x, ‖((1 - t) • A₀ + t • A₁) x‖ ≤ q * ‖x‖ := by
  intro x
  have hgap : 0 ≤ 1 - t := sub_nonneg.mpr ht1
  calc
    ‖((1 - t) • A₀ + t • A₁) x‖
        = ‖(1 - t) • A₀ x + t • A₁ x‖ := by rfl
    _ ≤ ‖(1 - t) • A₀ x‖ + ‖t • A₁ x‖ := norm_add_le _ _
    _ = (1 - t) * ‖A₀ x‖ + t * ‖A₁ x‖ := by
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
        abs_of_nonneg hgap, abs_of_nonneg ht0]
    _ ≤ (1 - t) * (q * ‖x‖) + t * (q * ‖x‖) :=
      add_le_add
        (mul_le_mul_of_nonneg_left (h₀ x) hgap)
        (mul_le_mul_of_nonneg_left (h₁ x) ht0)
    _ = q * ‖x‖ := by ring

/-- A common gain composes even when consecutive operators differ. -/
theorem changing_operator_two_step
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (A B : E →L[ℝ] E) {q : ℝ} (hq : 0 ≤ q)
    (hA : ∀ x, ‖A x‖ ≤ q * ‖x‖)
    (hB : ∀ x, ‖B x‖ ≤ q * ‖x‖) :
    ∀ x, ‖B (A x)‖ ≤ q ^ 2 * ‖x‖ := by
  intro x
  calc
    ‖B (A x)‖ ≤ q * ‖A x‖ := hB (A x)
    _ ≤ q * (q * ‖x‖) := mul_le_mul_of_nonneg_left (hA x) hq
    _ = q ^ 2 * ‖x‖ := by ring

end CommonMetric
end PldrLlmCurvatureSandpile
