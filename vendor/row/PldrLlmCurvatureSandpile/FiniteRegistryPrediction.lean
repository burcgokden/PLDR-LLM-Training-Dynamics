/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Predictive finite-registry maximum

Unlike an assumed attainment hypothesis, `finiteMaximum` is the actual
maximum of a function over a nonempty `Finset` of registered pairs.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace FiniteRegistryPrediction

variable {Pair : Type*}

/-- The actual maximum of a real-valued quantity on a nonempty finite pair
registry. -/
noncomputable def finiteMaximum (pairs : Finset Pair) (value : Pair → ℝ)
    (hpairs : pairs.Nonempty) : ℝ :=
  (pairs.image value).max' (hpairs.image value)

/-- Every registered value is bounded by the actual finite maximum. -/
theorem le_finiteMaximum (pairs : Finset Pair) (value : Pair → ℝ)
    (hpairs : pairs.Nonempty) {pair : Pair} (hpair : pair ∈ pairs) :
    value pair ≤ finiteMaximum pairs value hpairs := by
  apply Finset.le_max'
  exact Finset.mem_image.mpr ⟨pair, hpair, rfl⟩

/-- A pointwise bound passes through the actual finite maximum. -/
theorem finiteMaximum_mono (pairs : Finset Pair) (left right : Pair → ℝ)
    (hpairs : pairs.Nonempty)
    (hbound : ∀ pair ∈ pairs, left pair ≤ right pair) :
    finiteMaximum pairs left hpairs ≤ finiteMaximum pairs right hpairs := by
  apply (Finset.max'_le_iff (pairs.image left) (hpairs.image left)).2
  intro value hvalue
  rcases Finset.mem_image.mp hvalue with ⟨pair, hpair, rfl⟩
  exact (hbound pair hpair).trans
    (le_finiteMaximum pairs right hpairs hpair)

/-- Pairwise causal upper enclosures give the predictive successor-diameter
upper bound by taking one finite maximum. -/
theorem predictive_diameter_upper
    (pairs : Finset Pair) (sourceEnergy upperIncrement successorEnergy :
      Pair → ℝ) (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      successorEnergy pair ≤ sourceEnergy pair + upperIncrement pair) :
    finiteMaximum pairs successorEnergy hpairs ≤
      finiteMaximum pairs
        (fun pair => sourceEnergy pair + upperIncrement pair) hpairs := by
  exact finiteMaximum_mono pairs successorEnergy
    (fun pair => sourceEnergy pair + upperIncrement pair) hpairs henclosure

/-- Pairwise causal lower enclosures give the predictive successor-diameter
lower bound by taking the actual finite maximum. -/
theorem predictive_diameter_lower
    (pairs : Finset Pair) (sourceEnergy lowerIncrement successorEnergy :
      Pair → ℝ) (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      sourceEnergy pair + lowerIncrement pair ≤ successorEnergy pair) :
    finiteMaximum pairs
        (fun pair => sourceEnergy pair + lowerIncrement pair) hpairs ≤
      finiteMaximum pairs successorEnergy hpairs := by
  exact finiteMaximum_mono pairs
    (fun pair => sourceEnergy pair + lowerIncrement pair)
    successorEnergy hpairs henclosure

/-- Simultaneous pairwise lower and upper enclosures give the complete
two-sided finite-registry successor-diameter enclosure. -/
theorem predictive_diameter_two_sided
    (pairs : Finset Pair)
    (sourceEnergy lowerIncrement upperIncrement successorEnergy : Pair → ℝ)
    (hpairs : pairs.Nonempty)
    (hlower : ∀ pair ∈ pairs,
      sourceEnergy pair + lowerIncrement pair ≤ successorEnergy pair)
    (hupper : ∀ pair ∈ pairs,
      successorEnergy pair ≤ sourceEnergy pair + upperIncrement pair) :
    finiteMaximum pairs
        (fun pair => sourceEnergy pair + lowerIncrement pair) hpairs ≤
        finiteMaximum pairs successorEnergy hpairs ∧
      finiteMaximum pairs successorEnergy hpairs ≤
        finiteMaximum pairs
          (fun pair => sourceEnergy pair + upperIncrement pair) hpairs := by
  exact ⟨predictive_diameter_lower pairs sourceEnergy lowerIncrement
    successorEnergy hpairs hlower,
    predictive_diameter_upper pairs sourceEnergy upperIncrement
      successorEnergy hpairs hupper⟩

/-- A strict source-state upper bound predicts strict registry contraction
before any successor registered value is opened. -/
theorem predictive_strict_contraction
    (pairs : Finset Pair) (sourceEnergy upperIncrement successorEnergy :
      Pair → ℝ) (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      successorEnergy pair ≤ sourceEnergy pair + upperIncrement pair)
    (hstrict : finiteMaximum pairs
        (fun pair => sourceEnergy pair + upperIncrement pair) hpairs <
      finiteMaximum pairs sourceEnergy hpairs) :
    finiteMaximum pairs successorEnergy hpairs <
      finiteMaximum pairs sourceEnergy hpairs := by
  exact (predictive_diameter_upper pairs sourceEnergy upperIncrement
    successorEnergy hpairs henclosure).trans_lt hstrict

/-- A lower enclosure for one named pair predicts re-expansion even when
that pair is not a source maximizer. -/
theorem predictive_one_pair_expansion
    (pairs : Finset Pair) (sourceEnergy lowerIncrement successorEnergy :
      Pair → ℝ) (hpairs : pairs.Nonempty) {pair : Pair}
    (hpair : pair ∈ pairs)
    (henclosure : sourceEnergy pair + lowerIncrement pair ≤
      successorEnergy pair)
    (hstrict : finiteMaximum pairs sourceEnergy hpairs <
      sourceEnergy pair + lowerIncrement pair) :
    finiteMaximum pairs sourceEnergy hpairs <
      finiteMaximum pairs successorEnergy hpairs := by
  exact hstrict.trans_le (henclosure.trans
    (le_finiteMaximum pairs successorEnergy hpairs hpair))

end FiniteRegistryPrediction
end PldrLlmCurvatureSandpile
