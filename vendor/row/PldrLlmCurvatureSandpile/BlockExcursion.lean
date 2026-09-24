/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact finite-block excursions
-/
import Mathlib
import PldrLlmCurvatureSandpile.FiniteRegistryPrediction
import PldrLlmCurvatureSandpile.ReopeningBudget

namespace PldrLlmCurvatureSandpile
namespace BlockExcursion

open Filter
open scoped BigOperators

/-- The offsets in a block containing its anchor and endpoint. -/
def blockOffsets (horizon : ℕ) : Finset ℕ :=
  Finset.range (horizon + 1)

theorem block_offsets_nonempty (horizon : ℕ) :
    (blockOffsets horizon).Nonempty := by
  exact ⟨0, by simp [blockOffsets]⟩

/-- The exact largest nonnegative energy rise above the block anchor. -/
noncomputable def blockExcursion
    (energy : ℕ → ℝ) (start horizon : ℕ) : ℝ :=
  FiniteRegistryPrediction.finiteMaximum (blockOffsets horizon)
    (fun offset => max (energy (start + offset) - energy start) 0)
    (block_offsets_nonempty horizon)

/-- The attained maximum energy on the same finite block. -/
noncomputable def blockMaximum
    (energy : ℕ → ℝ) (start horizon : ℕ) : ℝ :=
  FiniteRegistryPrediction.finiteMaximum (blockOffsets horizon)
    (fun offset => energy (start + offset))
    (block_offsets_nonempty horizon)

/-- The exact positive variation charged on the edges of a block. -/
def blockPositiveVariation
    (energy : ℕ → ℝ) (start horizon : ℕ) : ℝ :=
  ∑ offset ∈ Finset.range horizon,
    ReopeningBudget.reopening energy (start + offset)

/-- A number bounds the excursion exactly when it bounds every realized
rise in the finite block. -/
theorem block_excursion_le_iff
    (energy : ℕ → ℝ) (start horizon : ℕ) (bound : ℝ) :
    blockExcursion energy start horizon ≤ bound ↔
      ∀ offset ∈ blockOffsets horizon,
        max (energy (start + offset) - energy start) 0 ≤ bound := by
  constructor
  · intro hbound offset hoffset
    exact (FiniteRegistryPrediction.le_finiteMaximum
      (blockOffsets horizon)
      (fun index => max (energy (start + index) - energy start) 0)
      (block_offsets_nonempty horizon) hoffset).trans hbound
  · intro hpointwise
    unfold blockExcursion FiniteRegistryPrediction.finiteMaximum
    apply (Finset.max'_le_iff _ _).2
    intro value hvalue
    rcases Finset.mem_image.mp hvalue with ⟨offset, hoffset, rfl⟩
    exact hpointwise offset hoffset

/-- The exact block excursion is nonnegative. -/
theorem block_excursion_nonnegative
    (energy : ℕ → ℝ) (start horizon : ℕ) :
    0 ≤ blockExcursion energy start horizon := by
  have hzero : 0 ∈ blockOffsets horizon := by simp [blockOffsets]
  have hmaximum := FiniteRegistryPrediction.le_finiteMaximum
    (blockOffsets horizon)
    (fun index => max (energy (start + index) - energy start) 0)
    (block_offsets_nonempty horizon) hzero
  simpa [blockExcursion] using hmaximum

/-- The excursion is zero exactly when no state in the block exceeds its
anchor. -/
theorem block_excursion_eq_zero_iff
    (energy : ℕ → ℝ) (start horizon : ℕ) :
    blockExcursion energy start horizon = 0 ↔
      ∀ offset ∈ blockOffsets horizon,
        energy (start + offset) ≤ energy start := by
  constructor
  · intro hzero offset hoffset
    have hmaximum :
        max (energy (start + offset) - energy start) 0 ≤ 0 :=
      (block_excursion_le_iff energy start horizon 0).1
        hzero.le offset hoffset
    have := (max_le_iff.mp hmaximum).1
    linarith
  · intro hnoRise
    apply le_antisymm
    · apply (block_excursion_le_iff energy start horizon 0).2
      intro offset hoffset
      apply max_le
      · linarith [hnoRise offset hoffset]
      · exact le_rfl
    · exact block_excursion_nonnegative energy start horizon

/-- Every state in a block is below the anchor plus its exact excursion. -/
theorem energy_le_anchor_add_excursion
    (energy : ℕ → ℝ) (start horizon offset : ℕ)
    (hoffset : offset ∈ blockOffsets horizon) :
    energy (start + offset) ≤
      energy start + blockExcursion energy start horizon := by
  have hrise : energy (start + offset) - energy start ≤
      max (energy (start + offset) - energy start) 0 :=
    le_max_left _ _
  have hmaximum : max (energy (start + offset) - energy start) 0 ≤
      blockExcursion energy start horizon :=
    FiniteRegistryPrediction.le_finiteMaximum
      (blockOffsets horizon)
      (fun index => max (energy (start + index) - energy start) 0)
      (block_offsets_nonempty horizon) hoffset
  linarith

/-- The anchor is one of the values entering the attained block maximum. -/
theorem anchor_le_block_maximum
    (energy : ℕ → ℝ) (start horizon : ℕ) :
    energy start ≤ blockMaximum energy start horizon := by
  have hzero : 0 ∈ blockOffsets horizon := by simp [blockOffsets]
  simpa [blockMaximum] using
    (FiniteRegistryPrediction.le_finiteMaximum
      (blockOffsets horizon)
      (fun offset => energy (start + offset))
      (block_offsets_nonempty horizon) hzero)

/-- The attained block maximum is exactly anchor energy plus excursion. -/
theorem block_excursion_eq_maximum
    (energy : ℕ → ℝ) (start horizon : ℕ) :
    blockMaximum energy start horizon =
      energy start + blockExcursion energy start horizon := by
  apply le_antisymm
  · unfold blockMaximum FiniteRegistryPrediction.finiteMaximum
    apply (Finset.max'_le_iff _ _).2
    intro value hvalue
    rcases Finset.mem_image.mp hvalue with ⟨offset, hoffset, rfl⟩
    exact energy_le_anchor_add_excursion
      energy start horizon offset hoffset
  · have hanchor := anchor_le_block_maximum energy start horizon
    have hexcursion :
        blockExcursion energy start horizon ≤
          blockMaximum energy start horizon - energy start := by
      apply (block_excursion_le_iff energy start horizon
        (blockMaximum energy start horizon - energy start)).2
      intro offset hoffset
      have hvalue : energy (start + offset) ≤
          blockMaximum energy start horizon :=
        FiniteRegistryPrediction.le_finiteMaximum
          (blockOffsets horizon)
          (fun index => energy (start + index))
          (block_offsets_nonempty horizon) hoffset
      apply max_le
      · linarith
      · linarith
    linarith

/-- Vanishing exact block maxima are equivalent to vanishing anchors and
excursions.  This is the finite-block kernel of the full orbitwise criterion. -/
theorem block_excursion_collapse_iff
    (energy : ℕ → ℝ) (start horizon : ℕ → ℕ)
    (henergy : ∀ block, 0 ≤ energy (start block)) :
    Tendsto
      (fun block => blockMaximum energy (start block)
        (horizon block)) atTop (nhds 0) ↔
      Tendsto (fun block => energy (start block))
        atTop (nhds 0) ∧
      Tendsto
        (fun block => blockExcursion energy (start block)
          (horizon block)) atTop (nhds 0) := by
  constructor
  · intro hmaximum
    constructor
    · apply squeeze_zero
      · exact henergy
      · intro block
        exact anchor_le_block_maximum energy
          (start block) (horizon block)
      · exact hmaximum
    · apply squeeze_zero
      · intro block
        exact block_excursion_nonnegative energy
          (start block) (horizon block)
      · intro block
        show blockExcursion energy (start block) (horizon block) ≤
          blockMaximum energy (start block) (horizon block)
        have hequality := block_excursion_eq_maximum energy
          (start block) (horizon block)
        linarith [henergy block]
      · exact hmaximum
  · rintro ⟨hanchor, hexcursion⟩
    simpa [block_excursion_eq_maximum] using hanchor.add hexcursion

/-- Exact excursion is bounded by the complete positive edge variation. -/
theorem block_excursion_le_positive_variation
    (energy : ℕ → ℝ) (start horizon : ℕ) :
    blockExcursion energy start horizon ≤
      blockPositiveVariation energy start horizon := by
  apply (block_excursion_le_iff energy start horizon
    (blockPositiveVariation energy start horizon)).2
  intro offset hoffset
  have hoffsetLe : offset ≤ horizon :=
    Nat.lt_succ_iff.mp (Finset.mem_range.mp hoffset)
  have hpath := ReopeningBudget.finite_reopening_envelope
    energy start offset
  have hsumLe :
      (∑ edge ∈ Finset.range offset,
        ReopeningBudget.reopening energy (start + edge)) ≤
      blockPositiveVariation energy start horizon := by
    unfold blockPositiveVariation
    exact Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.range_mono hoffsetLe)
      (fun edge _ _ => le_max_right _ _)
  have hrise : energy (start + offset) - energy start ≤
      blockPositiveVariation energy start horizon := by
    linarith
  have hvariationNonnegative :
      0 ≤ blockPositiveVariation energy start horizon := by
    unfold blockPositiveVariation
    exact Finset.sum_nonneg fun edge _ => le_max_right _ _
  exact max_le hrise hvariationNonnegative

/-- Vanishing anchor-plus-positive-variation bounds force the exact block
envelopes to vanish.  Unlike an assumed maximum bound, the comparison here
is derived from the realized finite block. -/
theorem block_envelope_collapse
    (energy : ℕ → ℝ) (start horizon : ℕ → ℕ)
    (henergy : ∀ block, 0 ≤ energy (start block))
    (henvelope : Tendsto
      (fun block => energy (start block) +
        blockPositiveVariation energy (start block) (horizon block))
      atTop (nhds 0)) :
    Tendsto
      (fun block => energy (start block) +
        blockExcursion energy (start block) (horizon block))
      atTop (nhds 0) := by
  apply squeeze_zero
  · intro block
    exact add_nonneg (henergy block)
      (block_excursion_nonnegative energy (start block) (horizon block))
  · intro block
    exact add_le_add le_rfl
      (block_excursion_le_positive_variation
        energy (start block) (horizon block))
  · exact henvelope

end BlockExcursion
end PldrLlmCurvatureSandpile
