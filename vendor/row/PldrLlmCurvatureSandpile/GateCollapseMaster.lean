/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Gate, shape, and mixed routes to row-map collapse
-/
import Mathlib
import PldrLlmCurvatureSandpile.LayerNormGate

namespace PldrLlmCurvatureSandpile
namespace GateCollapseMaster

/-- The architecture-owned gate route: a bounded normalized-shape
difference converts a small gate into a small row difference. -/
theorem gate_route
    {gate shapeDifference shapeBound tolerance : ℝ}
    (hgate : |gate| ≤ tolerance / shapeBound)
    (hshape : |shapeDifference| ≤ shapeBound)
    (hshapePositive : 0 < shapeBound) :
    |gate * shapeDifference| ≤ tolerance := by
  have hquotient : 0 ≤ tolerance / shapeBound :=
    (abs_nonneg gate).trans hgate
  rw [abs_mul]
  calc
    |gate| * |shapeDifference|
        ≤ (tolerance / shapeBound) * shapeBound :=
      mul_le_mul hgate hshape (abs_nonneg _) hquotient
    _ = tolerance := by field_simp

/-- The shape route collapses a coordinate while the corresponding gate is
only required to remain bounded. -/
theorem shape_route
    {gate shapeDifference gateBound tolerance : ℝ}
    (hgate : |gate| ≤ gateBound)
    (hshape : |shapeDifference| ≤ tolerance / gateBound)
    (hgatePositive : 0 < gateBound) :
    |gate * shapeDifference| ≤ tolerance := by
  rw [abs_mul]
  calc
    |gate| * |shapeDifference|
        ≤ gateBound * (tolerance / gateBound) :=
      mul_le_mul hgate hshape (abs_nonneg _) (le_of_lt hgatePositive)
    _ = tolerance := by field_simp

/-- Gate and shape increments split the successor row difference into the
old term and three mixed charges. -/
theorem mixed_route_identity
    (gate gateIncrement shapeDifference shapeIncrement : ℝ) :
    (gate + gateIncrement) * (shapeDifference + shapeIncrement)
      = gate * shapeDifference
        + gateIncrement * shapeDifference
        + gate * shapeIncrement
        + gateIncrement * shapeIncrement := by
  ring

/-- Exact zero of either the gate or the normalized-shape contrast is a
complete coordinatewise collapse mechanism. -/
theorem exact_coordinate_collapse
    (gate shapeDifference : ℝ)
    (hroute : gate = 0 ∨ shapeDifference = 0) :
    gate * shapeDifference = 0 := by
  rcases hroute with hgate | hshape
  · simp [hgate]
  · simp [hshape]

end GateCollapseMaster
end PldrLlmCurvatureSandpile
