/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import PldrLlmCurvatureSandpile.FiniteRegistryPrediction

namespace PldrLlmCurvatureSandpile
namespace DiameterSlack

open Finset
open FiniteRegistryPrediction

def pairDecrement {Pair : Type*}
    (current next : Pair → ℝ) (pair : Pair) : ℝ :=
  current pair - next pair

def diameterSlack {Pair : Type*}
    (diameterSq : ℝ) (current next : Pair → ℝ) (pair : Pair) : ℝ :=
  diameterSq - current pair + pairDecrement current next pair

/-- The successor energy of each pair is current diameter squared minus its
exact maximum-level slack. -/
theorem diameter_slack_identity {Pair : Type*}
    (diameterSq : ℝ) (current next : Pair → ℝ) (pair : Pair) :
    next pair = diameterSq - diameterSlack diameterSq current next pair := by
  simp [diameterSlack, pairDecrement]

/-- Pairwise lower decrement bounds give a successor-diameter bound without
requiring each pair to contract relative to its own current value. -/
theorem lower_slack_diameter_step {Pair : Type*}
    (current next lowerDecrement : Pair → ℝ)
    (diameterSq nextDiameterSq minimumLowerSlack : ℝ)
    (hlower : ∀ pair,
      lowerDecrement pair ≤ current pair - next pair)
    (hminimum : ∀ pair,
      minimumLowerSlack
        ≤ diameterSq - current pair + lowerDecrement pair)
    (hattained : ∃ pair, nextDiameterSq = next pair) :
    nextDiameterSq ≤ diameterSq - minimumLowerSlack := by
  rcases hattained with ⟨pair, rfl⟩
  have hslack :
      minimumLowerSlack
        ≤ diameterSq - current pair + (current pair - next pair) := by
    linarith [hminimum pair, hlower pair]
  linarith

/-- A lower slack proportional to the current squared diameter, up to
forcing, yields the displayed affine diameter recurrence. -/
theorem diameter_affine_blocks
    {currentDiameterSq nextDiameterSq minimumLowerSlack
      contraction forcing : ℝ}
    (hstep :
      nextDiameterSq ≤ currentDiameterSq - minimumLowerSlack)
    (hdrift :
      contraction * currentDiameterSq - forcing
        ≤ minimumLowerSlack) :
    nextDiameterSq
      ≤ (1 - contraction) * currentDiameterSq + forcing := by
  linarith

/-- The actual minimum of a real-valued quantity over a nonempty finite
registry. -/
noncomputable def finiteMinimum {Pair : Type*}
    (pairs : Finset Pair) (value : Pair → ℝ) (hpairs : pairs.Nonempty) : ℝ :=
  (pairs.image value).min' (hpairs.image value)

/-- The successor maximum is exactly the source diameter squared minus the
minimum realized maximum-level slack. -/
theorem finite_min_slack_identity {Pair : Type*}
    (pairs : Finset Pair) (current next : Pair → ℝ)
    (diameterSq : ℝ) (hpairs : pairs.Nonempty) :
    finiteMaximum pairs next hpairs =
      diameterSq - finiteMinimum pairs
        (diameterSlack diameterSq current next) hpairs := by
  let maximum := finiteMaximum pairs next hpairs
  let minimum := finiteMinimum pairs
    (diameterSlack diameterSq current next) hpairs
  have hmaxMem : maximum ∈ pairs.image next := by
    exact Finset.max'_mem (pairs.image next) (hpairs.image next)
  have hminMem : minimum ∈
      pairs.image (diameterSlack diameterSq current next) := by
    exact Finset.min'_mem
      (pairs.image (diameterSlack diameterSq current next))
      (hpairs.image (diameterSlack diameterSq current next))
  rcases Finset.mem_image.mp hmaxMem with ⟨maxPair, hmaxPair, hmaxValue⟩
  rcases Finset.mem_image.mp hminMem with ⟨minPair, hminPair, hminValue⟩
  have hminLe : minimum ≤
      diameterSlack diameterSq current next maxPair := by
    exact Finset.min'_le _ _
      (Finset.mem_image.mpr ⟨maxPair, hmaxPair, rfl⟩)
  have hnextLe : next minPair ≤ maximum := by
    exact Finset.le_max' _ _
      (Finset.mem_image.mpr ⟨minPair, hminPair, rfl⟩)
  have hmaxIdentity :
      next maxPair =
        diameterSq - diameterSlack diameterSq current next maxPair :=
    diameter_slack_identity diameterSq current next maxPair
  have hminIdentity :
      next minPair =
        diameterSq - diameterSlack diameterSq current next minPair :=
    diameter_slack_identity diameterSq current next minPair
  change maximum = diameterSq - minimum
  apply le_antisymm
  · rw [← hmaxValue, hmaxIdentity]
    linarith
  · rw [hminValue] at hminIdentity
    rw [← hminIdentity]
    exact hnextLe

/-- Both endpoint diameters are actual maxima on the same nonempty finite
registry. This specialization removes the free source-diameter parameter from
the finite slack identity. -/
theorem finite_diameter_slack_identity {Pair : Type*}
    (pairs : Finset Pair) (current next : Pair → ℝ)
    (hpairs : pairs.Nonempty) :
    finiteMaximum pairs next hpairs =
      finiteMaximum pairs current hpairs - finiteMinimum pairs
        (diameterSlack (finiteMaximum pairs current hpairs) current next)
        hpairs := by
  exact finite_min_slack_identity pairs current next
    (finiteMaximum pairs current hpairs) hpairs

/-- Strict contraction of squared diameter is equivalent to a positive
minimum slack. -/
theorem strict_contraction_iff
    {sourceDiameterSq nextDiameterSq minimumSlack : ℝ}
    (hidentity : nextDiameterSq = sourceDiameterSq - minimumSlack) :
    nextDiameterSq < sourceDiameterSq ↔ 0 < minimumSlack := by
  constructor <;> intro hypothesis <;> linarith

end DiameterSlack
end PldrLlmCurvatureSandpile
