/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PLGAScoreTransfer

/-- Exact three-factor telescoping with rectangular query and key row counts. -/
theorem rectangular_three_factor
    {queryRows keyRows width : ℕ}
    (queryZero queryOne : Matrix (Fin queryRows) (Fin width) ℝ)
    (generatorZero generatorOne : Matrix (Fin width) (Fin width) ℝ)
    (keyZero keyOne : Matrix (Fin keyRows) (Fin width) ℝ) :
    queryOne * generatorOne * keyOne.transpose
        - queryZero * generatorZero * keyZero.transpose
      = (queryOne - queryZero) * generatorZero * keyZero.transpose
        + queryOne * (generatorOne - generatorZero) * keyZero.transpose
        + queryOne * generatorOne * (keyOne - keyZero).transpose := by
  rw [Matrix.transpose_sub]
  simp only [Matrix.sub_mul, Matrix.mul_sub]
  abel

/-- Fixed query and key leave only the curvature transfer term. -/
theorem fixed_query_key
    {queryRows keyRows width : ℕ}
    (query : Matrix (Fin queryRows) (Fin width) ℝ)
    (generatorZero generatorOne : Matrix (Fin width) (Fin width) ℝ)
    (key : Matrix (Fin keyRows) (Fin width) ℝ) :
    query * generatorOne * key.transpose
        - query * generatorZero * key.transpose
      = query * (generatorOne - generatorZero) * key.transpose := by
  simp only [Matrix.mul_sub, Matrix.sub_mul]

end PLGAScoreTransfer
end PldrLlmCurvatureSandpile

