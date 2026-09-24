/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Causal score quotient and PLGA visibility witness
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace CausalScoreQuotient

/-- Number of independent causal score differences in one batch and head
stack, after removing one common shift from each nonempty causal row. -/
def causalQuotientDimension (batch heads context : ℕ) : ℕ :=
  batch * heads * (context * (context - 1) / 2)

/-- The predecessor counts in causal rows form the triangular number. -/
theorem causal_predecessor_count (context : ℕ) :
    ∑ position ∈ Finset.range context, position
      = context * (context - 1) / 2 := by
  simpa using Finset.sum_range_id context

/-- Expanding the batch and head multiplicities gives the declared causal
quotient dimension. -/
theorem causal_score_quotient_count (batch heads context : ℕ) :
    causalQuotientDimension batch heads context
      = batch * heads * (context * (context - 1) / 2) := by
  rfl

/-- A causal score difference exposes exactly the key-difference witness
used by the PLGA visibility argument. -/
theorem plga_score_difference
    (query metric key keyReference : ℝ) :
    query * metric * key - query * metric * keyReference
      = query * metric * (key - keyReference) := by
  ring

end CausalScoreQuotient
end PldrLlmCurvatureSandpile
