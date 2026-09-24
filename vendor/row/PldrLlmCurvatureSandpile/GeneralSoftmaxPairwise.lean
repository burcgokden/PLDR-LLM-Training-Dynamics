/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# General-vocabulary softmax pairwise energy
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace GeneralSoftmaxPairwise

/-- Covariance energy of one vocabulary-space perturbation. -/
def covarianceEnergy {V : Type*} [Fintype V]
    (probability value : V → ℝ) : ℝ :=
  (∑ token, probability token * (value token) ^ 2)
    - (∑ token, probability token * value token) ^ 2

/-- Ordered pair form of the same covariance energy. -/
noncomputable def orderedPairEnergy {V : Type*} [Fintype V]
    (probability value : V → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ left, ∑ right,
    probability left * probability right
      * (value left - value right) ^ 2

theorem double_sum_product {V : Type*} [Fintype V]
    (left right : V → ℝ) :
    (∑ i, ∑ j, left i * right j)
      = (∑ i, left i) * (∑ j, right j) := by
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum]

/-- The softmax covariance identity holds for every finite vocabulary, not
only a two-class reduction. -/
theorem general_vocabulary_pairwise_identity
    {V : Type*} [Fintype V] (probability value : V → ℝ)
    (hmass : ∑ token, probability token = 1) :
    covarianceEnergy probability value
      = orderedPairEnergy probability value := by
  have hleft :
      (∑ i, ∑ j,
        probability i * probability j * (value i) ^ 2)
        = (∑ i, probability i * (value i) ^ 2)
          * (∑ j, probability j) := by
    calc
      _ = ∑ i, ∑ j,
          (probability i * (value i) ^ 2) * probability j := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = _ := double_sum_product
        (fun i => probability i * (value i) ^ 2) probability
  have hright :
      (∑ i, ∑ j,
        probability i * probability j * (value j) ^ 2)
        = (∑ i, probability i)
          * (∑ j, probability j * (value j) ^ 2) := by
    calc
      _ = ∑ i, ∑ j,
          probability i * (probability j * (value j) ^ 2) := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = _ := double_sum_product probability
        (fun j => probability j * (value j) ^ 2)
  have hcross :
      (∑ i, ∑ j,
        probability i * probability j * (2 * value i * value j))
        = 2 * (∑ i, probability i * value i)
          * (∑ j, probability j * value j) := by
    calc
      _ = ∑ i, ∑ j,
          (2 * probability i * value i)
            * (probability j * value j) := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = (∑ i, 2 * probability i * value i)
          * (∑ j, probability j * value j) :=
        double_sum_product
          (fun i => 2 * probability i * value i)
          (fun j => probability j * value j)
      _ = _ := by
        congr 1
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro i _
        ring
  simp only [covarianceEnergy, orderedPairEnergy]
  simp_rw [sub_sq]
  simp only [mul_add, mul_sub, Finset.sum_add_distrib,
    Finset.sum_sub_distrib]
  rw [hleft, hcross, hright, hmass]
  ring

/-- Nonnegative probabilities make the pairwise form manifestly
nonnegative. -/
theorem ordered_pair_nonnegative {V : Type*} [Fintype V]
    (probability value : V → ℝ)
    (hprobability : ∀ token, 0 ≤ probability token) :
    0 ≤ orderedPairEnergy probability value := by
  unfold orderedPairEnergy
  apply mul_nonneg (by norm_num)
  apply Finset.sum_nonneg
  intro left _
  apply Finset.sum_nonneg
  intro right _
  exact mul_nonneg
    (mul_nonneg (hprobability left) (hprobability right)) (sq_nonneg _)

end GeneralSoftmaxPairwise
end PldrLlmCurvatureSandpile
