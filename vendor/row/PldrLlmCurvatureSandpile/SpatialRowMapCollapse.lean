/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Spatial two-channel row-map collapse

This module checks the scalar inequality and limit kernels behind the final
LayerNorm-gain and normalized-shape channels. Runtime construction of the
nonlinear segment chords is outside this module.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SpatialRowMapCollapse

open Filter

/-- An operator chord bound and a final coordinatewise gain bound compose to
control the physical row-map diameter. -/
theorem finite_chord_diameter_bound
    {diameter gate shapeCoefficient inputDiameter : ℝ}
    (hdiameter : diameter ≤ gate * (shapeCoefficient * inputDiameter)) :
    diameter ≤ gate * shapeCoefficient * inputDiameter := by
  simpa [mul_assoc] using hdiameter

/-- The smaller of the universal LayerNorm range bound and the occupied
shape-chord bound remains a valid physical diameter bound. -/
theorem gate_shape_product_bound
    {diameter gate shapeBound chordBound : ℝ}
    (hrange : diameter ≤ gate * shapeBound)
    (hchord : diameter ≤ gate * chordBound) :
    diameter ≤ min (gate * shapeBound) (gate * chordBound) := by
  exact le_min hrange hchord

/-- If the nonnegative gate-shape-input product tends to zero and bounds the
diameter, the physical row-map diameter tends to zero. -/
theorem two_channel_collapse
    (diameter gate shapeCoefficient inputDiameter : ℕ → ℝ)
    (hdiameter : ∀ index, 0 ≤ diameter index)
    (hbound : ∀ index,
      diameter index ≤
        gate index * shapeCoefficient index * inputDiameter index)
    (hproduct : Tendsto
      (fun index => gate index * shapeCoefficient index * inputDiameter index)
      atTop (nhds 0)) :
    Tendsto diameter atTop (nhds 0) := by
  exact squeeze_zero hdiameter hbound hproduct

end SpatialRowMapCollapse
end PldrLlmCurvatureSandpile
