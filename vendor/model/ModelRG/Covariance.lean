import Mathlib

namespace ModelRG
open scoped BigOperators

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Exact covariance-block numerator, valid with arbitrary cross-position dependence. -/
theorem covariance_block_sum (x : ι → κ → ℝ) :
    (∑ i, (∑ j, x i j)^2) = ∑ j, ∑ k, ∑ i, x i j * x i k := by
  simp_rw [pow_two, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [Finset.sum_comm]

/-- Correlated offsets can contribute at order b rather than being averaged away. -/
theorem constant_sector_susceptibility (b within between : ℝ) (hb : b ≠ 0) :
    b * (between + within / b) = b * between + within := by
  field_simp

/-- Rectangular weighted second-moment perturbation, including both ordered
mixed terms. Covariance applications supply centering and probability weights. -/
theorem covariance_add_perturbation {d k s : ℕ}
    (A : Matrix (Fin k) (Fin d) ℝ)
    (X : Matrix (Fin d) (Fin s) ℝ)
    (E : Matrix (Fin k) (Fin s) ℝ)
    (W : Matrix (Fin s) (Fin s) ℝ) :
    (A * X + E) * W * (A * X + E).transpose =
      A * (X * W * X.transpose) * A.transpose +
      A * (X * W * E.transpose) +
      (E * W * X.transpose) * A.transpose +
      E * W * E.transpose := by
  simp only [Matrix.transpose_add, Matrix.transpose_mul, Matrix.add_mul,
    Matrix.mul_add, Matrix.mul_assoc]
  abel

end ModelRG
