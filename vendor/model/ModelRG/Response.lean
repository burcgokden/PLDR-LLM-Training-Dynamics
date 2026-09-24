import Mathlib

namespace ModelRG

open scoped BigOperators

variable {ι : Type*} [Fintype ι]

def weightedMean (p v : ι → ℝ) : ℝ := ∑ i, p i * v i
def fisherEnergy (p v : ι → ℝ) : ℝ :=
  ∑ i, p i * (v i - weightedMean p v)^2

theorem fisher_nonnegative (p v : ι → ℝ) (hp : ∀ i, 0 ≤ p i) :
    0 ≤ fisherEnergy p v := by
  exact Finset.sum_nonneg (fun i _ => mul_nonneg (hp i) (sq_nonneg _))

theorem weightedMean_shift (p v : ι → ℝ) (c : ℝ) (hp : ∑ i, p i = 1) :
    weightedMean p (fun i => v i + c) = weightedMean p v + c := by
  simp only [weightedMean, mul_add, Finset.sum_add_distrib]
  rw [← Finset.sum_mul, hp, one_mul]

theorem fisher_gauge_invariant (p v : ι → ℝ) (c : ℝ) (hp : ∑ i, p i = 1) :
    fisherEnergy p (fun i => v i + c) = fisherEnergy p v := by
  unfold fisherEnergy
  rw [weightedMean_shift p v c hp]
  congr 1
  ext i
  ring

theorem fisher_variance (p v : ι → ℝ) (hp : ∑ i, p i = 1) :
    fisherEnergy p v = (∑ i, p i * v i^2) - (weightedMean p v)^2 := by
  unfold fisherEnergy
  have hi (i : ι) : p i * (v i - weightedMean p v)^2 =
      p i * v i^2 - 2 * weightedMean p v * (p i * v i) +
      p i * (weightedMean p v)^2 := by ring
  simp_rw [hi]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.sum_mul, hp, one_mul]
  change (∑ i, p i * v i^2) - 2 * weightedMean p v * weightedMean p v +
      weightedMean p v^2 = _
  ring

/-- Schur completion of the square, the scalar algebraic core. -/
theorem schur_complete_square (a b c x y : ℝ) (hc : c ≠ 0) :
    a*x^2+2*b*x*y+c*y^2 = (a-b^2/c)*x^2 + c*(y+b*x/c)^2 := by
  field_simp
  ring

theorem zero_variance_not_divergence (variance n : ℝ) (h : variance = 0) :
    n * variance = 0 := by simp [h]

end ModelRG
