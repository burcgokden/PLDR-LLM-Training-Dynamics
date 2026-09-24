import Mathlib

/-!
Finite scalar algebra used in the stand-alone consuming-population proof.
The permutation law, vector covariance and physical-clock limit are not
formalized here. Coordinate projections give the written vector interpretation.
-/
namespace ModelRG.FinitePopulation
open scoped BigOperators

theorem centered_offdiagonal {ι : Type*} [Fintype ι] [DecidableEq ι]
    (u : ι → ℝ) (hcenter : ∑ i, u i = 0) :
    (∑ i, ∑ j ∈ Finset.univ.erase i, u i * u j) = -(∑ i, (u i)^2) := by
  have row (i : ι) : (∑ j ∈ Finset.univ.erase i, u i * u j) = -(u i)^2 := by
    have h := Finset.sum_erase_add (s := Finset.univ)
      (f := fun j => u i * u j) (Finset.mem_univ i)
    have total : (∑ j, u i * u j) = 0 := by
      rw [← Finset.mul_sum, hcenter, mul_zero]
    rw [total] at h
    nlinarith
  simp_rw [row]
  exact Finset.sum_neg_distrib (fun i : ι => (u i)^2)

theorem batch_covariance_collect (M B diagonal offdiagonal : ℝ)
    (hB : B ≠ 0) (hM : M ≠ 1) :
    (M-B) / (B*(M-1)) * diagonal - offdiagonal / (M-1) =
      ((M/B) * diagonal - (diagonal+offdiagonal)) / (M-1) := by
  have h : M-1 ≠ 0 := sub_ne_zero.mpr hM
  field_simp
  ring

end ModelRG.FinitePopulation
