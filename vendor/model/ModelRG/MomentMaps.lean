import Mathlib

namespace ModelRG

/-- Rectangular observation maps compose without discarding covariance sectors. -/
theorem covariance_congruence_comp {d q r : ℕ}
    (A : Matrix (Fin r) (Fin q) ℝ) (B : Matrix (Fin q) (Fin d) ℝ)
    (C : Matrix (Fin d) (Fin d) ℝ) :
    A * (B * C * B.transpose) * A.transpose =
      (A * B) * C * (A * B).transpose := by
  simp only [Matrix.transpose_mul, Matrix.mul_assoc]

/-- Exact linear images compose; successive coordinate enclosures need not. -/
theorem linear_set_image_comp {d q r : ℕ}
    (A : Matrix (Fin r) (Fin q) ℝ) (B : Matrix (Fin q) (Fin d) ℝ)
    (T : Set (Fin d → ℝ)) :
    A.mulVec '' (B.mulVec '' T) = (A * B).mulVec '' T := by
  rw [Set.image_image]
  congr 1
  funext x
  exact Matrix.mulVec_mulVec x A B

end ModelRG
