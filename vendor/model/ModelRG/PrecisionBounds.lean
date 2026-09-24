import Mathlib

/-! Finite-norm cores for the measured pulse error proposition.
The analytic differentiability and native error hypotheses are not proved here. -/
namespace ModelRG

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem odd_error_bound (ep em : E) (ε : ℝ)
    (hp : ‖ep‖ ≤ ε) (hm : ‖em‖ ≤ ε) :
    ‖(1 / 2 : ℝ) • (ep - em)‖ ≤ ε := by
  rw [norm_smul]
  norm_num
  calc
    (1 / 2 : ℝ) * ‖ep - em‖ ≤ (1 / 2 : ℝ) * (‖ep‖ + ‖em‖) :=
      mul_le_mul_of_nonneg_left (norm_sub_le ep em) (by norm_num)
    _ ≤ ε := by linarith

theorem halving_arithmetic_bound (r ef eh : E) (a ε : ℝ)
    (hr : ‖r‖ ≤ a) (hf : ‖ef‖ ≤ ε) (hh : ‖eh‖ ≤ ε) :
    ‖r + ef - (2 : ℝ) • eh‖ ≤ a + 3 * ε := by
  have ht : ‖(2 : ℝ) • eh‖ = 2 * ‖eh‖ := by
    rw [norm_smul]
    norm_num
  calc
    ‖r + ef - (2 : ℝ) • eh‖ ≤ ‖r + ef‖ + ‖(2 : ℝ) • eh‖ := norm_sub_le _ _
    _ ≤ (‖r‖ + ‖ef‖) + 2 * ‖eh‖ := by
      rw [ht]
      exact add_le_add (norm_add_le r ef) (le_refl _)
    _ ≤ a + 3 * ε := by linarith

theorem even_error_bound (ep em e0 : E) (ε : ℝ)
    (hp : ‖ep‖ ≤ ε) (hm : ‖em‖ ≤ ε) (h0 : ‖e0‖ ≤ ε) :
    ‖(1 / 2 : ℝ) • (ep + em) - e0‖ ≤ 2 * ε := by
  have hs : ‖(1 / 2 : ℝ) • (ep + em)‖ ≤ ε := by
    have hn : ‖-em‖ ≤ ε := by simpa using hm
    simpa using odd_error_bound ep (-em) ε hp hn
  calc
    _ ≤ ‖(1 / 2 : ℝ) • (ep + em)‖ + ‖e0‖ := norm_sub_le _ _
    _ ≤ 2 * ε := by linarith

end ModelRG
