/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Scaled chronological products
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ScaledChronologicalProduct

/-- One endpoint-scaled linear successor. -/
def scaledStep {n : ℕ}
    (scaleNext operator scaleInverse : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  scaleNext * operator * scaleInverse

/-- Adjacent endpoint scalings cancel in a two-step chronological product.
-/
theorem scaled_two_step_conjugacy {n : ℕ}
    (scaleZeroInverse scaleOne scaleOneInverse scaleTwo
      operatorZero operatorOne : Matrix (Fin n) (Fin n) ℝ)
    (hcancel : scaleOneInverse * scaleOne = 1) :
    scaledStep scaleTwo operatorOne scaleOneInverse
        * scaledStep scaleOne operatorZero scaleZeroInverse
      = scaleTwo * (operatorOne * operatorZero) * scaleZeroInverse := by
  simp only [scaledStep]
  calc
    (scaleTwo * operatorOne * scaleOneInverse)
          * (scaleOne * operatorZero * scaleZeroInverse)
        = scaleTwo * operatorOne * (scaleOneInverse * scaleOne)
            * operatorZero * scaleZeroInverse := by
          noncomm_ring
    _ = scaleTwo * (operatorOne * operatorZero) * scaleZeroInverse := by
      rw [hcancel]
      simp [Matrix.mul_assoc]

end ScaledChronologicalProduct
end PldrLlmCurvatureSandpile
