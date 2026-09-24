/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Nonautonomous source multiplier cocycle

Source-derived multipliers may exceed one. Their ordered product, not a
per-step sign condition, controls chronological row-map collapse.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SourceMultiplierCocycle

open Filter

/-- Ordered product of source multipliers on a chronological block. -/
def orderedSourceProduct (multiplier : ℕ → ℝ) (start horizon : ℕ) : ℝ :=
  ∏ offset ∈ Finset.range horizon, multiplier (start + offset)

/-- One-step source upper multipliers compose in execution order, including
steps with multiplier greater than one. -/
theorem ordered_source_product
    (diameterSq multiplier : ℕ → ℝ) (start : ℕ)
    (hmultiplier : ∀ index, 0 ≤ multiplier index)
    (hstep : ∀ index,
      diameterSq (index + 1) ≤ multiplier index * diameterSq index) :
    ∀ horizon,
      diameterSq (start + horizon) ≤
        orderedSourceProduct multiplier start horizon * diameterSq start := by
  intro horizon
  induction horizon with
  | zero => simp [orderedSourceProduct]
  | succ horizon inductionHypothesis =>
      calc
        diameterSq (start + (horizon + 1)) =
            diameterSq ((start + horizon) + 1) := by congr 1
        _ ≤ multiplier (start + horizon) * diameterSq (start + horizon) :=
          hstep (start + horizon)
        _ ≤ multiplier (start + horizon) *
              (orderedSourceProduct multiplier start horizon *
                diameterSq start) :=
          mul_le_mul_of_nonneg_left inductionHypothesis
            (hmultiplier (start + horizon))
        _ = orderedSourceProduct multiplier start (horizon + 1) *
              diameterSq start := by
          simp [orderedSourceProduct, Finset.prod_range_succ]
          ring

/-- A vanishing ordered source product forces every nonnegative diameter
sequence below that product to collapse. -/
theorem source_product_collapse
    (diameterSq multiplier : ℕ → ℝ) (start : ℕ)
    (hdiameter : ∀ index, 0 ≤ diameterSq index)
    (hmultiplier : ∀ index, 0 ≤ multiplier index)
    (hstep : ∀ index,
      diameterSq (index + 1) ≤ multiplier index * diameterSq index)
    (hproduct : Tendsto
      (fun horizon => orderedSourceProduct multiplier start horizon)
      atTop (nhds 0)) :
    Tendsto (fun horizon => diameterSq (start + horizon)) atTop (nhds 0) := by
  apply squeeze_zero
  · exact fun horizon => hdiameter (start + horizon)
  · exact ordered_source_product diameterSq multiplier start hmultiplier hstep
  · simpa using hproduct.mul_const (diameterSq start)

/-- Divergent cumulative restoring coefficients force collapse through the
standard exponential envelope. -/
theorem summable_restoring_collapse
    (diameterSq restoring : ℕ → ℝ) (start : ℕ)
    (hdiameter : ∀ index, 0 ≤ diameterSq index)
    (hrestoring : ∀ index, 0 ≤ restoring index ∧ restoring index ≤ 1)
    (hstep : ∀ index,
      diameterSq (index + 1) ≤
        (1 - restoring index) * diameterSq index)
    (hdivergent : Tendsto
      (fun horizon => ∑ offset ∈ Finset.range horizon,
        restoring (start + offset)) atTop atTop) :
    Tendsto (fun horizon => diameterSq (start + horizon)) atTop (nhds 0) := by
  let multiplier : ℕ → ℝ := fun index => 1 - restoring index
  have hmultiplier : ∀ index, 0 ≤ multiplier index := by
    intro index
    dsimp [multiplier]
    linarith [(hrestoring index).2]
  have hproductEnvelope : ∀ horizon,
      orderedSourceProduct multiplier start horizon ≤
        Real.exp (-(∑ offset ∈ Finset.range horizon,
          restoring (start + offset))) := by
    intro horizon
    dsimp [orderedSourceProduct, multiplier]
    calc
      (∏ offset ∈ Finset.range horizon,
          (1 - restoring (start + offset))) ≤
          ∏ offset ∈ Finset.range horizon,
            Real.exp (-restoring (start + offset)) := by
        apply Finset.prod_le_prod
        · intro offset hoffset
          exact hmultiplier (start + offset)
        · intro offset hoffset
          exact Real.one_sub_le_exp_neg _
      _ = Real.exp (-(∑ offset ∈ Finset.range horizon,
          restoring (start + offset))) := by
        rw [← Real.exp_sum]
        congr 1
        simp
  have hexponential : Tendsto
      (fun horizon => Real.exp (-(∑ offset ∈ Finset.range horizon,
        restoring (start + offset)))) atTop (nhds 0) := by
    exact Real.tendsto_exp_atBot.comp
      (tendsto_neg_atTop_atBot.comp hdivergent)
  have hproductNonnegative : ∀ horizon,
      0 ≤ orderedSourceProduct multiplier start horizon := by
    intro horizon
    exact Finset.prod_nonneg fun offset hoffset => hmultiplier (start + offset)
  have hproduct : Tendsto
      (fun horizon => orderedSourceProduct multiplier start horizon)
      atTop (nhds 0) := by
    exact squeeze_zero hproductNonnegative hproductEnvelope hexponential
  exact source_product_collapse diameterSq multiplier start hdiameter
    hmultiplier (by simpa [multiplier] using hstep) hproduct

end SourceMultiplierCocycle
end PldrLlmCurvatureSandpile
