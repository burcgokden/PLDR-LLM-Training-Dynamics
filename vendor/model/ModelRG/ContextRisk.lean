import Mathlib

/-!
Selected finite algebra for declared context weights and a squared-risk tail.
The independent written proof supplies the interpretation as empirical variance.
This file makes no probabilistic independence, population, or floating-point claim.
-/
namespace ModelRG.ContextRisk
open scoped BigOperators

/-- The context law may be nonuniform; only native denominators must be nonzero. -/
theorem weighted_ratio {ι : Type*} [Fintype ι]
    (mass native residual : ι → ℝ) (hnative : ∀ i, 0 < native i) :
    (∑ i, mass i * residual i) / (∑ i, mass i * native i) =
      ∑ i, ((mass i * native i) / (∑ j, mass j * native j)) *
        (residual i / native i) := by
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  have hi : native i ≠ 0 := ne_of_gt (hnative i)
  by_cases hs : (∑ j, mass j * native j) = 0
  · simp [hs]
  · field_simp

/-- A finite Markov bound. The argument is squared risk, threshold is tau squared. -/
theorem weighted_tail {ι : Type*} [Fintype ι]
    (weight risk : ι → ℝ) (threshold : ℝ)
    (hw : ∀ i, 0 ≤ weight i) (hr : ∀ i, 0 ≤ risk i)
    (hs : ∑ i, weight i = 1) (ht : 0 < threshold) :
    (∑ i ∈ Finset.univ.filter (fun i => threshold < risk i), weight i) ≤
      min 1 ((∑ i, weight i * risk i) / threshold) := by
  classical
  let s := Finset.univ.filter (fun i => threshold < risk i)
  have hmass : (∑ i ∈ s, weight i) ≤ ∑ i, weight i := by
    apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
    intro i _ _
    exact hw i
  have hbound : threshold * (∑ i ∈ s, weight i) ≤
      ∑ i, weight i * risk i := by
    calc
      threshold * (∑ i ∈ s, weight i) = ∑ i ∈ s, threshold * weight i := by
        rw [Finset.mul_sum]
      _ ≤ ∑ i ∈ s, weight i * risk i := by
        apply Finset.sum_le_sum
        intro i hi
        have hri : threshold < risk i := (Finset.mem_filter.mp hi).2
        nlinarith [hw i]
      _ ≤ ∑ i, weight i * risk i := by
        apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
        intro i _ _
        exact mul_nonneg (hw i) (hr i)
  apply le_min
  · simpa [hs] using hmass
  · apply (le_div_iff₀ ht).2
    simpa [mul_comm] using hbound

/-- Equal-context variance means yield the same aggregate ratio as sums. -/
theorem means_ratio (native residual count : ℝ) (hc : count ≠ 0) :
    (residual / count) / (native / count) = residual / native := by
  by_cases hn : native = 0
  · simp [hn]
  · field_simp

end ModelRG.ContextRisk

#print axioms ModelRG.ContextRisk.weighted_ratio
#print axioms ModelRG.ContextRisk.weighted_tail
#print axioms ModelRG.ContextRisk.means_ratio
