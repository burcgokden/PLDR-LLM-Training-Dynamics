import ModelRG.Covariance
import ModelRG.NoiseCovariance

/-! Finite sample covariance and signed fluctuation transfer. Probability,
independence, sampling, and asymptotic limits are established in the manuscript. -/
noncomputable section
open scoped BigOperators
namespace ModelRG

def sampleMeanR {n : ℕ} (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n

def sampleCovR {n : ℕ} (x y : Fin n → ℝ) : ℝ :=
  (∑ i, (x i - sampleMeanR x) * (y i - sampleMeanR y)) / (n - 1 : ℝ)

private theorem sample_mean_sum {n m : ℕ} (d : Fin n → Fin m → ℝ) :
    sampleMeanR (fun i => ∑ j, d i j) = ∑ j, sampleMeanR (fun i => d i j) := by
  simp only [sampleMeanR, ← Finset.sum_div]
  rw [Finset.sum_comm]

private theorem sample_mean_shift {n : ℕ} (x : Fin n → ℝ) (a : ℝ)
    (hn : (n : ℝ) ≠ 0) : sampleMeanR (fun i => x i - a) = sampleMeanR x - a := by
  simp only [sampleMeanR, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, sub_div]
  field_simp

/-- Unbiased sample covariance of a sum, including all cross terms. -/
theorem sample_covariance_sum {n m : ℕ} (d : Fin n → Fin m → ℝ) :
    (∑ j, ∑ k, sampleCovR (fun i => d i j) (fun i => d i k)) =
      sampleCovR (fun i => ∑ j, d i j) (fun i => ∑ j, d i j) := by
  calc
    _ = (∑ j, ∑ k, ∑ i,
        (d i j - sampleMeanR (fun u => d u j)) *
        (d i k - sampleMeanR (fun u => d u k))) / (n - 1 : ℝ) := by
      simp only [sampleCovR, Finset.sum_div]
    _ = (∑ i, (∑ j, (d i j - sampleMeanR (fun u => d u j)))^2) /
        (n - 1 : ℝ) := by rw [covariance_block_sum]
    _ = _ := by
      simp only [sampleCovR, sample_mean_sum, Finset.sum_sub_distrib, pow_two]

/-- A common incoming value cancels from the centered endpoint sample. -/
theorem endpoint_sample_covariance {n m : ℕ} (d : Fin n → Fin m → ℝ)
    (endpoint : Fin n → ℝ) (incoming : ℝ) (hn : 1 < n)
    (hpath : ∀ i, ∑ j, d i j = endpoint i - incoming) :
    (∑ j, ∑ k, sampleCovR (fun i => d i j) (fun i => d i k)) =
      sampleCovR endpoint endpoint := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt (lt_trans Nat.zero_lt_one hn))
  rw [sample_covariance_sum]
  have hfun : (fun i => ∑ j, d i j) = (fun i => endpoint i - incoming) := funext hpath
  rw [hfun]
  simp only [sampleCovR, sample_mean_shift endpoint incoming hn0]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- The four measured intervals telescope pathwise without stochastic assumptions. -/
theorem four_increment_endpoint (r0 r1 r4 r16 r64 : ℝ) :
    (r1-r0)+(r4-r1)+(r16-r4)+(r64-r16) = r64-r0 := by ring

/-- Signed covariance cancellation, for centered variables interpreted as moments. -/
theorem signed_second_moment_difference {ι : Type*} [Fintype ι]
    (w x e : ι → ℝ) :
    (∑ i, w i * (x i + e i)^2) - (∑ i, w i * (x i)^2) =
      2 * (∑ i, w i * x i * e i) + (∑ i, w i * (e i)^2) := by
  have h := weighted_second_moment_transport w x e 1
  simp only [one_mul, one_pow, mul_one] at h
  linarith

/-- Reflection preserves centered second moments despite a nonzero error. -/
theorem reflection_preserves_second_moment {ι : Type*} [Fintype ι]
    (w x : ι → ℝ) :
    (∑ i, w i * (x i + (-2 * x i))^2) = (∑ i, w i * (x i)^2) := by
  apply Finset.sum_congr rfl
  intro i _
  ring

end ModelRG
#print axioms ModelRG.sample_covariance_sum
#print axioms ModelRG.endpoint_sample_covariance
#print axioms ModelRG.four_increment_endpoint
#print axioms ModelRG.signed_second_moment_difference
#print axioms ModelRG.reflection_preserves_second_moment
