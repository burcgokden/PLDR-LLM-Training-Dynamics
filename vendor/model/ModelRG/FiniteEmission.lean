import ModelRG.Response

/- Finite algebra for predictive metric transport. The manuscript supplies the exponential
   density-ratio and integral-Taylor proofs independently. -/
namespace ModelRG.FiniteEmission
open scoped BigOperators
variable {ι : Type*} [Fintype ι]

theorem centered_square (p v : ι → ℝ) (c : ℝ) (hp : ∑ i, p i = 1) :
    (∑ i, p i * (v i - c)^2) =
      fisherEnergy p v + (weightedMean p v - c)^2 := by
  rw [fisher_variance p v hp]
  have hi (i : ι) : p i * (v i - c)^2 =
      p i * v i^2 - 2*c*(p i*v i) + p i*c^2 := by ring
  simp_rw [hi]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.sum_mul, hp, one_mul]
  change (∑ i, p i * v i^2) - 2*c*weightedMean p v + c^2 = _
  ring

theorem fisher_comparison (p q v : ι → ℝ) (a b : ℝ)
    (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1)
    (ha : 0 ≤ a)
    (hl : ∀ i, a*p i ≤ q i) (hu : ∀ i, q i ≤ b*p i) :
    a*fisherEnergy p v ≤ fisherEnergy q v ∧
      fisherEnergy q v ≤ b*fisherEnergy p v := by
  have low := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
      mul_le_mul_of_nonneg_right (hl i) (sq_nonneg (v i - weightedMean q v)))
  have high := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
      mul_le_mul_of_nonneg_right (hu i) (sq_nonneg (v i - weightedMean p v)))
  simp_rw [mul_assoc] at low high
  rw [← Finset.mul_sum] at low high
  rw [centered_square p v (weightedMean q v) hp] at low
  rw [centered_square q v (weightedMean p v) hq] at high
  change a*(fisherEnergy p v + (weightedMean p v - weightedMean q v)^2) ≤ fisherEnergy q v at low
  change fisherEnergy q v + (weightedMean q v - weightedMean p v)^2 ≤ b*fisherEnergy p v at high
  constructor
  · nlinarith [mul_nonneg ha (sq_nonneg (weightedMean p v - weightedMean q v))]
  · nlinarith [sq_nonneg (weightedMean q v - weightedMean p v)]

theorem signed_bilinear_contrast {E : Type*} [AddCommGroup E] [Module ℝ E]
    (B0 B1 : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (d e : E)
    (hsym : B0 d e = B0 e d) :
    (B1 (d+e) (d+e) - B0 d d)/2 =
      B0 e d + B0 e e/2 + (B1 (d+e) (d+e)-B0 (d+e) (d+e))/2 := by
  simp only [map_add, LinearMap.add_apply]
  rw [hsym]
  ring

end ModelRG.FiniteEmission

