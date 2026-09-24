/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Source-locked pair-energy prediction

This module checks the finite-registry reduction from source pair energies,
signed first-order work, a validated remainder bound, and an arithmetic charge
to a successor-diameter bound. It does not construct the numerical bounds.
-/
import Mathlib
import PldrLlmCurvatureSandpile.FiniteRegistryPrediction

namespace PldrLlmCurvatureSandpile
namespace SourcePredictiveEnergy

open FiniteRegistryPrediction

variable {Pair : Type*}

/-- Source-computable upper energy for one registered row pair. -/
def sourceUpper (energy work remainder charge : Pair → ℝ) (pair : Pair) : ℝ :=
  energy pair + work pair + remainder pair + charge pair

/-- Restoring margin relative to the source squared diameter. -/
def sourceMargin (diameterSq : ℝ)
    (energy work remainder charge : Pair → ℝ) (pair : Pair) : ℝ :=
  diameterSq - sourceUpper energy work remainder charge pair

/-- Pointwise source-locked energy enclosures pass through the exact finite
maximum over the registered pairs. -/
theorem pair_upper_to_diameter_upper
    (pairs : Finset Pair) (energy work remainder charge successor : Pair → ℝ)
    (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      successor pair ≤ sourceUpper energy work remainder charge pair) :
    finiteMaximum pairs successor hpairs ≤
      finiteMaximum pairs (sourceUpper energy work remainder charge) hpairs := by
  exact finiteMaximum_mono pairs successor
    (sourceUpper energy work remainder charge) hpairs henclosure

/-- A uniform lower bound on every source restoring margin gives a diameter
decrement even when the maximizing pair changes. -/
theorem source_restoring_step
    (pairs : Finset Pair) (energy work remainder charge successor : Pair → ℝ)
    (diameterSq minimumMargin : ℝ) (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      successor pair ≤ sourceUpper energy work remainder charge pair)
    (hmargin : ∀ pair ∈ pairs,
      minimumMargin ≤ sourceMargin diameterSq energy work remainder charge pair) :
    finiteMaximum pairs successor hpairs ≤ diameterSq - minimumMargin := by
  apply (Finset.max'_le_iff (pairs.image successor)
    (hpairs.image successor)).2
  intro value hvalue
  rcases Finset.mem_image.mp hvalue with ⟨pair, hpair, rfl⟩
  have hupper := henclosure pair hpair
  have hrestore := hmargin pair hpair
  simp only [sourceMargin, sourceUpper] at hrestore hupper ⊢
  linarith

/-- If every source restoring margin dominates a fraction of the actual
source diameter, the next registered diameter satisfies the corresponding
strict source-state contraction bound. -/
theorem strict_source_contraction
    (pairs : Finset Pair) (energy work remainder charge successor : Pair → ℝ)
    (diameterSq contraction : ℝ) (hpairs : pairs.Nonempty)
    (henclosure : ∀ pair ∈ pairs,
      successor pair ≤ sourceUpper energy work remainder charge pair)
    (hmargin : ∀ pair ∈ pairs,
      contraction * diameterSq ≤
        sourceMargin diameterSq energy work remainder charge pair) :
    finiteMaximum pairs successor hpairs ≤
      (1 - contraction) * diameterSq := by
  have hstep := source_restoring_step pairs energy work remainder charge
    successor diameterSq (contraction * diameterSq) hpairs henclosure hmargin
  linarith

end SourcePredictiveEnergy
end PldrLlmCurvatureSandpile
