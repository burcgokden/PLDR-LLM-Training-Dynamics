/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Absolute reopening budgets
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ReopeningBudget

open scoped BigOperators
open Filter

/-- Positive energy variation on one executed edge. -/
def reopening (energy : ℕ → ℝ) (step : ℕ) : ℝ :=
  max (energy (step + 1) - energy step) 0

theorem increment_le_reopening (energy : ℕ → ℝ) (step : ℕ) :
    energy (step + 1) - energy step ≤ reopening energy step := by
  exact le_max_left _ _

/-- Every step is bounded by its absolute positive-variation charge. -/
theorem one_step_reopening_envelope (energy : ℕ → ℝ) (step : ℕ) :
    energy (step + 1) ≤ energy step + reopening energy step := by
  have h := increment_le_reopening energy step
  linarith

/-- An affine energy upper bound gives an absolute positive-variation bound
without dividing by the source energy. -/
theorem affine_reopening_bound
    (source endpoint gain charge : ℝ)
    (hstep : endpoint ≤ gain * source + charge) :
    max (endpoint - source) 0 ≤
      max (source * (gain - 1) + charge) 0 := by
  apply max_le_max
  · nlinarith
  · exact le_rfl

/-- Substituting a work-minus-charge representation of the affine gain
produces the source-ledger reopening bound. -/
theorem work_gram_reopening_bound
    (source endpoint work gramCharge nativeCharge : ℝ)
    (hstep :
      endpoint ≤ (1 - work + gramCharge) * source + nativeCharge) :
    max (endpoint - source) 0 ≤
      max (source * (-work + gramCharge) + nativeCharge) 0 := by
  apply max_le_max
  · nlinarith
  · exact le_rfl

/-- Positive variations accumulate into a finite pathwise envelope.  The
bound remains defined across zero energy and arbitrary relative gains. -/
theorem finite_reopening_envelope
    (energy : ℕ → ℝ) (start : ℕ) :
    ∀ horizon,
      energy (start + horizon) ≤ energy start +
        ∑ offset ∈ Finset.range horizon,
          reopening energy (start + offset) := by
  intro horizon
  induction horizon with
  | zero => simp
  | succ horizon inductionHypothesis =>
      calc
        energy (start + (horizon + 1)) =
            energy ((start + horizon) + 1) := by congr 1
        _ ≤ energy (start + horizon) +
              reopening energy (start + horizon) :=
          one_step_reopening_envelope energy (start + horizon)
        _ ≤ (energy start +
              ∑ offset ∈ Finset.range horizon,
                reopening energy (start + offset)) +
              reopening energy (start + horizon) :=
          add_le_add inductionHypothesis le_rfl
        _ = energy start +
              ∑ offset ∈ Finset.range (horizon + 1),
                reopening energy (start + offset) := by
          rw [Finset.sum_range_succ]
          ring

/-- If a nonnegative block maximum is bounded by vanishing anchor energy plus
vanishing reopening budget, then the complete block maximum vanishes. -/
theorem block_reopening_collapse
    (blockMaximum anchor budget : ℕ → ℝ)
    (hnonnegative : ∀ block, 0 ≤ blockMaximum block)
    (hbound : ∀ block,
      blockMaximum block ≤ anchor block + budget block)
    (henvelope : Tendsto
      (fun block => anchor block + budget block) atTop (nhds 0)) :
    Tendsto blockMaximum atTop (nhds 0) := by
  exact squeeze_zero hnonnegative hbound henvelope

/-- Summable positive reopening together with arbitrarily late small energy
forces the full nonnegative energy sequence to vanish.  The small-energy
hypothesis is the direct real-valued content of liminf zero and remains
meaningful even for sequences that are not bounded above. -/
theorem summable_reopening_collapse
    (energy : ℕ → ℝ)
    (henergy : ∀ step, 0 ≤ energy step)
    (hsmall : ∀ ε > 0, ∀ start, ∃ step ≥ start, energy step < ε)
    (hsummable : Summable (reopening energy)) :
    Tendsto energy atTop (nhds 0) := by
  apply tendsto_order.2
  constructor
  · intro lower hlower
    exact Filter.Eventually.of_forall fun step =>
      hlower.trans_le (henergy step)
  · intro ε hε
    let partialSum : ℕ → ℝ := fun horizon =>
      ∑ step ∈ Finset.range horizon, reopening energy step
    have hcauchy : CauchySeq partialSum :=
      hsummable.hasSum.tendsto_sum_nat.cauchySeq
    obtain ⟨tailStart, htailStart⟩ :=
      (Metric.cauchySeq_iff.1 hcauchy) (ε / 2) (half_pos hε)
    obtain ⟨anchor, hanchorStart, hanchorEnergy⟩ :=
      hsmall (ε / 2) (half_pos hε) tailStart
    apply eventually_atTop.2
    refine ⟨anchor, ?_⟩
    intro endpoint hanchorEndpoint
    have hpartialMonotone :
        partialSum anchor ≤ partialSum endpoint := by
      unfold partialSum
      exact Finset.sum_le_sum_of_subset_of_nonneg
        (Finset.range_mono hanchorEndpoint)
        (fun step _ _ => le_max_right _ _)
    have htailDistance :=
      htailStart anchor hanchorStart endpoint
        (hanchorStart.trans hanchorEndpoint)
    have htail :
        (∑ step ∈ Finset.Ico anchor endpoint,
          reopening energy step) < ε / 2 := by
      rw [Finset.sum_Ico_eq_sub _ hanchorEndpoint]
      rw [Real.dist_eq,
        abs_of_nonpos (sub_nonpos.mpr hpartialMonotone)] at htailDistance
      simpa [partialSum, neg_sub] using htailDistance
    have henvelope :=
      finite_reopening_envelope energy anchor (endpoint - anchor)
    have hendpoint : anchor + (endpoint - anchor) = endpoint :=
      Nat.add_sub_of_le hanchorEndpoint
    rw [hendpoint] at henvelope
    rw [← Finset.sum_Ico_eq_sum_range
      (reopening energy) anchor endpoint] at henvelope
    linarith

end ReopeningBudget
end PldrLlmCurvatureSandpile
