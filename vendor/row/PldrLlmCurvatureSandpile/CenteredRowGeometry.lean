/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Centered finite row geometry

The scalar identity is applied coordinatewise to the physical row map. The two
inequality kernels are the algebraic steps converting centered energy and a
finite pairwise diameter.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace CenteredRowGeometry

open scoped BigOperators

/-- Scalar centered energy on a finite row registry. -/
def centeredEnergy {Row : Type*} [Fintype Row]
    (value : Row → ℝ) (center : ℝ) : ℝ :=
  ∑ row, (value row - center) ^ 2

/-- Ordered scalar pair energy. Each unordered pair occurs twice. -/
def orderedPairEnergy {Row : Type*} [Fintype Row]
    (value : Row → ℝ) : ℝ :=
  ∑ left, ∑ right, (value left - value right) ^ 2

/-- If the supplied center has zero total residual, ordered pair energy equals
twice the row count times centered energy. Applying this coordinatewise gives
the Frobenius pairwise-variance identity. -/
theorem pairwise_variance_identity
    {Row : Type*} [Fintype Row] (value : Row → ℝ) (center : ℝ)
    (hcenter : ∑ row, (value row - center) = 0) :
    orderedPairEnergy value =
      2 * (Fintype.card Row : ℝ) * centeredEnergy value center := by
  classical
  let residual : Row → ℝ := fun row => value row - center
  have hzero : ∑ row, residual row = 0 := hcenter
  unfold orderedPairEnergy centeredEnergy
  simp_rw [show ∀ left right,
    value left - value right =
      (value left - center) - (value right - center) by
        intro left right
        ring]
  change (∑ left, ∑ right, (residual left - residual right) ^ 2) =
    2 * (Fintype.card Row : ℝ) * ∑ row, residual row ^ 2
  simp_rw [sub_sq]
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  have hcross :
      (∑ left, ∑ right, 2 * residual left * residual right) = 0 := by
    apply Finset.sum_eq_zero
    intro left hleft
    rw [← Finset.mul_sum, hzero, mul_zero]
  rw [hcross]
  simp
  rw [← Finset.mul_sum]
  ring

/-- Two centered row contributions contained in the total energy control their
pairwise squared difference by twice that energy. -/
theorem diameter_le_centered_energy
    {left right energy : ℝ}
    (hbudget : left ^ 2 + right ^ 2 ≤ energy) :
    (left - right) ^ 2 ≤ 2 * energy := by
  nlinarith [sq_nonneg (left + right)]

/-- Summing a uniform squared-diameter bound over all distinct ordered row
pairs produces exactly `card Row * (card Row - 1)` copies of the bound. The
diagonal terms vanish and therefore require no diameter hypothesis. -/
theorem ordered_pair_energy_le_diameter
    {Row : Type*} [Fintype Row] (value : Row → ℝ) {diameterSq : ℝ}
    (hdiameter : ∀ left right, left ≠ right →
      (value left - value right) ^ 2 ≤ diameterSq) :
    orderedPairEnergy value ≤
      (Fintype.card Row : ℝ)
        * ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq := by
  classical
  unfold orderedPairEnergy
  have hinner : ∀ left : Row,
      (∑ right, (value left - value right) ^ 2) ≤
        ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq := by
    intro left
    calc
      (∑ right, (value left - value right) ^ 2) =
          ∑ right ∈ Finset.univ.erase left,
            (value left - value right) ^ 2 := by
        rw [← Finset.sum_erase_add Finset.univ
          (fun right => (value left - value right) ^ 2)
          (Finset.mem_univ left)]
        simp
      _ ≤ ∑ _right ∈ Finset.univ.erase left, diameterSq := by
        apply Finset.sum_le_sum
        intro right hright
        exact hdiameter left right (Finset.ne_of_mem_erase hright).symm
      _ = ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq := by
        simp [Finset.card_erase_of_mem]
  calc
    (∑ left, ∑ right, (value left - value right) ^ 2) ≤
        ∑ _left : Row,
          ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq :=
      Finset.sum_le_sum fun left _ => hinner left
    _ = (Fintype.card Row : ℝ)
          * ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq := by
      simp
      ring

/-- The upper half of the finite centered-energy/diameter equivalence. Unlike
the earlier scalar cancellation lemma below, this theorem performs the
ordered-pair count internally. -/
theorem centered_energy_le_finite_diameter
    {Row : Type*} [Fintype Row] [Nonempty Row]
    (value : Row → ℝ) (center diameterSq : ℝ)
    (hcenter : ∑ row, (value row - center) = 0)
    (hdiameter : ∀ left right, left ≠ right →
      (value left - value right) ^ 2 ≤ diameterSq) :
    2 * centeredEnergy value center ≤
      ((Fintype.card Row - 1 : ℕ) : ℝ) * diameterSq := by
  have hpair := ordered_pair_energy_le_diameter value hdiameter
  rw [pairwise_variance_identity value center hcenter] at hpair
  have hcard : 0 < (Fintype.card Row : ℝ) := by
    exact_mod_cast Fintype.card_pos
  nlinarith

/-- Scalar cancellation used after an external ordered-pair sum has already
been established. `centered_energy_le_finite_diameter` is the self-contained
finite-row statement. -/
theorem centered_energy_le_diameter
    {rowCount centered diameterSq : ℝ}
    (hrowCount : 0 < rowCount)
    (hsum : rowCount * centered ≤
      rowCount * (rowCount - 1) * diameterSq / 2) :
    2 * centered ≤ (rowCount - 1) * diameterSq := by
  have hscaled : 2 * rowCount * centered ≤
      rowCount * ((rowCount - 1) * diameterSq) := by
    nlinarith
  nlinarith

end CenteredRowGeometry
end PldrLlmCurvatureSandpile
