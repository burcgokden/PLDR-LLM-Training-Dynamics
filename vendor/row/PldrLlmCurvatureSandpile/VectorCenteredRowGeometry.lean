/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Direct finite-vector centered row geometry
-/
import Mathlib
import PldrLlmCurvatureSandpile.CenteredRowGeometry

namespace PldrLlmCurvatureSandpile
namespace VectorCenteredRowGeometry

open Filter
open scoped BigOperators

/-- Frobenius centered energy, defined directly for finite row vectors. -/
def vectorCenteredEnergy {Row : Type*} [Fintype Row] {d : ℕ}
    (value : Row → Fin d → ℝ) (center : Fin d → ℝ) : ℝ :=
  ∑ coordinate,
    CenteredRowGeometry.centeredEnergy
      (fun row => value row coordinate) (center coordinate)

/-- Squared Euclidean distance between two finite row vectors. -/
def vectorPairDistanceSq {Row : Type*} [Fintype Row] {d : ℕ}
    (value : Row → Fin d → ℝ) (left right : Row) : ℝ :=
  ∑ coordinate, (value left coordinate - value right coordinate) ^ 2

/-- Ordered vector pair energy. Each unordered vector pair occurs twice. -/
def vectorOrderedPairEnergy {Row : Type*} [Fintype Row] {d : ℕ}
    (value : Row → Fin d → ℝ) : ℝ :=
  ∑ coordinate,
    CenteredRowGeometry.orderedPairEnergy
      (fun row => value row coordinate)

/-- The pairwise-variance identity in the same finite vector norm used by the
physical row map. This is not an assembly of coordinatewise maxima. -/
theorem vector_pairwise_variance_identity
    {Row : Type*} [Fintype Row] {d : ℕ}
    (value : Row → Fin d → ℝ) (center : Fin d → ℝ)
    (hcenter : ∀ coordinate,
      ∑ row, (value row coordinate - center coordinate) = 0) :
    vectorOrderedPairEnergy value =
      2 * (Fintype.card Row : ℝ) * vectorCenteredEnergy value center := by
  classical
  unfold vectorOrderedPairEnergy vectorCenteredEnergy
  calc
    (∑ coordinate,
        CenteredRowGeometry.orderedPairEnergy
          (fun row => value row coordinate)) =
        ∑ coordinate,
          2 * (Fintype.card Row : ℝ) *
            CenteredRowGeometry.centeredEnergy
              (fun row => value row coordinate) (center coordinate) := by
      apply Finset.sum_congr rfl
      intro coordinate _
      exact CenteredRowGeometry.pairwise_variance_identity
        (fun row => value row coordinate) (center coordinate)
        (hcenter coordinate)
    _ = 2 * (Fintype.card Row : ℝ) *
        ∑ coordinate,
          CenteredRowGeometry.centeredEnergy
            (fun row => value row coordinate) (center coordinate) := by
      rw [Finset.mul_sum]

/-- Every distinct physical row pair is controlled by twice the total
centered Frobenius energy. -/
theorem vector_pair_distance_le_twice_centered_energy
    {Row : Type*} [Fintype Row] {d : ℕ}
    (value : Row → Fin d → ℝ) (center : Fin d → ℝ)
    (left right : Row) (hne : left ≠ right) :
    vectorPairDistanceSq value left right ≤
      2 * vectorCenteredEnergy value center := by
  classical
  unfold vectorPairDistanceSq vectorCenteredEnergy
  calc
    (∑ coordinate, (value left coordinate - value right coordinate) ^ 2) ≤
        ∑ coordinate, 2 * CenteredRowGeometry.centeredEnergy
          (fun row => value row coordinate) (center coordinate) := by
      apply Finset.sum_le_sum
      intro coordinate _
      have hpair :
          (value left coordinate - center coordinate) ^ 2 +
              (value right coordinate - center coordinate) ^ 2 ≤
            CenteredRowGeometry.centeredEnergy
              (fun row => value row coordinate) (center coordinate) := by
        unfold CenteredRowGeometry.centeredEnergy
        calc
          (value left coordinate - center coordinate) ^ 2 +
                (value right coordinate - center coordinate) ^ 2 =
              ∑ row ∈ ({left, right} : Finset Row),
                (value row coordinate - center coordinate) ^ 2 := by
            rw [Finset.sum_pair hne]
          _ ≤ ∑ row : Row,
                (value row coordinate - center coordinate) ^ 2 := by
            exact Finset.sum_le_sum_of_subset_of_nonneg
              (by simp) (fun row _ _ => sq_nonneg _)
      have hcenteredDifference :
          value left coordinate - value right coordinate =
            (value left coordinate - center coordinate) -
              (value right coordinate - center coordinate) := by ring
      rw [hcenteredDifference]
      exact CenteredRowGeometry.diameter_le_centered_energy hpair
    _ = 2 * ∑ coordinate,
          CenteredRowGeometry.centeredEnergy
            (fun row => value row coordinate) (center coordinate) := by
      rw [Finset.mul_sum]

/-- Collapse of centered Frobenius energy forces every named distinct row
pair to collapse, without a fixed maximizing pair. -/
theorem vector_pair_collapse_of_centered_energy
    {Row : Type*} [Fintype Row] {d : ℕ}
    (value : ℕ → Row → Fin d → ℝ) (center : ℕ → Fin d → ℝ)
    (left right : Row) (hne : left ≠ right)
    (henergy : Tendsto
      (fun step => vectorCenteredEnergy (value step) (center step))
      atTop (nhds 0)) :
    Tendsto (fun step => vectorPairDistanceSq (value step) left right)
      atTop (nhds 0) := by
  apply squeeze_zero
  · intro step
    unfold vectorPairDistanceSq
    exact Finset.sum_nonneg fun coordinate _ => sq_nonneg _
  · intro step
    exact vector_pair_distance_le_twice_centered_energy
      (value step) (center step) left right hne
  · simpa using henergy.const_mul 2

end VectorCenteredRowGeometry
end PldrLlmCurvatureSandpile
