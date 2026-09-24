/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace CoordinateOscillation

/-- The coordinate carrier of row dependence is the product of gate
magnitude and normalized-shape oscillation. -/
def carrier {d : ℕ} (gate oscillation : Fin d → ℝ) (coordinate : Fin d) : ℝ :=
  |gate coordinate| * oscillation coordinate

/-- A pair attaining a coordinate oscillation transfers its coordinate
difference directly to the Euclidean diameter bound. -/
theorem coordinate_oscillation_le_diameter
    {coordinateOscillation diameter attainedDifference : ℝ}
    (hattained : coordinateOscillation = |attainedDifference|)
    (hcoordinate : |attainedDifference| ≤ diameter) :
    coordinateOscillation ≤ diameter := by
  simpa [hattained] using hcoordinate

/-- If every pair distance is bounded by the l2 carrier envelope and one
pair attains the diameter, then the diameter obeys the same envelope. -/
theorem diameter_le_l2_oscillation
    {Pair : Type*} {d : ℕ}
    (distance : Pair → ℝ) (z : Fin d → ℝ)
    (diameter : ℝ)
    (hpair : ∀ pair, distance pair ≤ Real.sqrt (∑ j, (z j) ^ 2))
    (hattained : ∃ pair, diameter = distance pair) :
    diameter ≤ Real.sqrt (∑ j, (z j) ^ 2) := by
  rcases hattained with ⟨pair, rfl⟩
  exact hpair pair

/-- On a finite coordinate set, vanishing of every gate-shape carrier is
equivalent to the exhaustive gate-or-shape alternative. -/
theorem finite_registry_collapse_iff {d : ℕ}
    (gate oscillation : Fin d → ℝ) :
    (∀ j, gate j * oscillation j = 0) ↔
      ∀ j, gate j = 0 ∨ oscillation j = 0 := by
  constructor
  · intro h j
    exact mul_eq_zero.mp (h j)
  · intro h j
    exact mul_eq_zero.mpr (h j)

/-- Expanding separate nonnegative gate and shape recurrences gives the
explicit mixed carrier recurrence used by the master theorem. -/
theorem carrier_product_expansion
    (gate shape gateGain shapeGain gateForce shapeForce : ℝ) :
    (gateGain * gate + gateForce) * (shapeGain * shape + shapeForce)
      = gateGain * shapeGain * (gate * shape)
        + gateGain * gate * shapeForce
        + shapeGain * shape * gateForce
        + gateForce * shapeForce := by
  ring

end CoordinateOscillation
end PldrLlmCurvatureSandpile
