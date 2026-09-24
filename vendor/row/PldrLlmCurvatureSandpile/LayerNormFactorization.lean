/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib
import PldrLlmCurvatureSandpile.LayerNormGeometry

namespace PldrLlmCurvatureSandpile
namespace LayerNormFactorization

/-- Coordinate form of the final LayerNorm row output. -/
def rowOutput {d : ℕ}
    (bias gate shape : Fin d → ℝ) : Fin d → ℝ :=
  fun coordinate => bias coordinate + gate coordinate * shape coordinate

/-- The shared bias cancels and the gate multiplies the exact shape contrast. -/
theorem row_contrast {d : ℕ}
    (bias gate left right : Fin d → ℝ) (coordinate : Fin d) :
    rowOutput bias gate left coordinate - rowOutput bias gate right coordinate
      = gate coordinate * (left coordinate - right coordinate) := by
  simp [rowOutput]
  ring

/-- A zero gate makes the corresponding output coordinate row independent. -/
theorem zero_gate_row_independent {d : ℕ}
    (bias gate left right : Fin d → ℝ) (coordinate : Fin d)
    (hgate : gate coordinate = 0) :
    rowOutput bias gate left coordinate = rowOutput bias gate right coordinate := by
  simp [rowOutput, hgate]

/-- The strict regularized radius kernel used by the written factorization. -/
theorem strict_regularized_radius
    {dimension epsilon centeredSq : ℝ}
    (hdimension : 0 < dimension) (hepsilon : 0 < epsilon)
    (hcentered : 0 ≤ centeredSq) :
    LayerNormGeometry.normalizedShapeSq dimension epsilon centeredSq
      < dimension :=
  LayerNormGeometry.normalized_shape_sq_lt
    hdimension hepsilon hcentered

end LayerNormFactorization
end PldrLlmCurvatureSandpile

