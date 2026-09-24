import Mathlib

/-! Selected finite arithmetic in the standalone inference proofs.
The native graph induction, probability laws and analytic logit bounds
are proved separately in the manuscript. -/
noncomputable section
open scoped BigOperators
namespace ModelRG

theorem score_margin_stability
    (correct alternative changedCorrect changedAlternative error : ℝ)
    (hc : |changedCorrect - correct| ≤ error)
    (ha : |changedAlternative - alternative| ≤ error) :
    |(changedCorrect - changedAlternative) - (correct - alternative)| ≤ 2 * error := by
  obtain ⟨hc₁, hc₂⟩ := abs_le.mp hc
  obtain ⟨ha₁, ha₂⟩ := abs_le.mp ha
  apply abs_le.mpr
  constructor <;> linarith

theorem score_margin_positive
    (correct alternative changedCorrect changedAlternative error : ℝ)
    (hc : |changedCorrect - correct| ≤ error)
    (ha : |changedAlternative - alternative| ≤ error)
    (margin : 2 * error < correct - alternative) :
    changedAlternative < changedCorrect := by
  have bound := score_margin_stability correct alternative changedCorrect changedAlternative error hc ha
  obtain ⟨lower, upper⟩ := abs_le.mp bound
  linarith

theorem weighted_prefix_loss_bound
    {ι : Type*} [Fintype ι] (w a b e : ι → ℝ)
    (hw : ∀ i, 0 ≤ w i) (h : ∀ i, |a i - b i| ≤ 2 * e i) :
    |∑ i, w i * (a i - b i)| ≤ 2 * ∑ i, w i * e i := by
  calc
    |∑ i, w i * (a i - b i)| ≤ ∑ i, |w i * (a i - b i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, w i * |a i - b i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (hw i)]
    _ ≤ ∑ i, w i * (2 * e i) := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (h i) (hw i)
    _ = 2 * ∑ i, w i * e i := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring

end ModelRG
