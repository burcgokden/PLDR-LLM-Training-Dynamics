import ModelRG.ReplicaStatistics

/- Finite real-arithmetic normalization and degeneracy for complete replicas. -/
noncomputable section
namespace ModelRG.ReplicaBootstrap
open scoped BigOperators
variable {ι : Type*} [Fintype ι]

theorem count_weights_sum (c : ι → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hc : ∑ i, c i = s) :
    (∑ i, c i / s) = 1 := by
  rw [← Finset.sum_div, hc]
  exact div_self hs

theorem count_pair_variance (c x : ι → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hs1 : s - 1 ≠ 0) (hc : ∑ i, c i = s) :
    (∑ i, ∑ j, c i * c j * (x i - x j)^2) / (2 * s * (s - 1)) =
      s / (s - 1) * ModelRG.fisherEnergy (fun i => c i / s) x := by
  have h := ModelRG.ReplicaStatistics.weighted_pair_variance
    (fun i => c i / s) x (count_weights_sum c s hs hc)
  have point (i j : ι) : (c i / s) * (c j / s) * (x i - x j)^2 =
      c i * c j * (x i - x j)^2 / (s * s) := by ring
  simp_rw [point, ← Finset.sum_div] at h
  rw [← h]
  field_simp

theorem count_pair_nonnegative (c x : ι → ℝ)
    (hc : ∀ i, 0 ≤ c i) :
    0 ≤ ∑ i, ∑ j, c i * c j * (x i - x j)^2 := by
  exact Finset.sum_nonneg (fun i _ => Finset.sum_nonneg
    (fun j _ => mul_nonneg (mul_nonneg (hc i) (hc j)) (sq_nonneg _)))

theorem single_support_pair_zero (c x : ι → ℝ) (k : ι)
    (hc : ∀ i, i ≠ k → c i = 0) :
    (∑ i, ∑ j, c i * c j * (x i - x j)^2) = 0 := by
  classical
  apply Finset.sum_eq_zero
  intro i _
  apply Finset.sum_eq_zero
  intro j _
  by_cases hi : i = k
  · by_cases hj : j = k
    · subst i
      subst j
      simp
    · simp [hc j hj]
  · simp [hc i hi]

theorem count_vector_variance {d : ℕ} (c : ι → ℝ) (x : ι → Fin d → ℝ)
    (s : ℝ) (hs : s ≠ 0) (hs1 : s - 1 ≠ 0) (hc : ∑ i, c i = s) :
    (∑ k, (∑ i, ∑ j, c i * c j * (x i k - x j k)^2) /
      (2 * s * (s - 1))) / (d : ℝ) =
      (s / (s - 1) * ∑ k, ModelRG.fisherEnergy (fun i => c i / s) (fun i => x i k)) / (d : ℝ) := by
  simp_rw [count_pair_variance c (fun i => x i _) s hs hs1 hc]
  rw [Finset.mul_sum]

end ModelRG.ReplicaBootstrap
