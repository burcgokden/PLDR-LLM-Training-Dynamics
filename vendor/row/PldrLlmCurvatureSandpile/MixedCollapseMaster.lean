/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Assembly kernels for mixed row-map collapse
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace MixedCollapseMaster

/-- Coordinatewise nonnegative energy bounds sum to the total energy bound.
-/
theorem coordinate_decay_energy {d : ℕ}
    (energy bound : Fin d → ℝ)
    (hcoordinate : ∀ j, energy j ≤ bound j) :
    ∑ j, energy j ≤ ∑ j, bound j := by
  exact Finset.sum_le_sum fun j _ => hcoordinate j

/-- The squared connected-graph estimate converts a sufficiently small graph
energy into a diameter target. -/
theorem graph_energy_collapse
    {minimumWeight diameter graphDiameter energy tolerance : ℝ}
    (hminimum : 0 < minimumWeight)
    (htolerance : 0 ≤ tolerance)
    (hgraph : minimumWeight * diameter ^ 2 ≤ graphDiameter * energy)
    (hsmall : graphDiameter * energy
      ≤ minimumWeight * tolerance ^ 2) :
    diameter ≤ tolerance := by
  have hsquares : diameter ^ 2 ≤ tolerance ^ 2 :=
    le_of_mul_le_mul_left (hgraph.trans hsmall) hminimum
  nlinarith

/-- Gate and shape logarithmic decrements add before exponentiation. -/
theorem mixed_route_decay
    (initial gateDecrement shapeDecrement : ℝ) :
    initial * Real.exp (-gateDecrement) * Real.exp (-shapeDecrement)
      = initial * Real.exp (-(gateDecrement + shapeDecrement)) := by
  rw [mul_assoc, ← Real.exp_add]
  congr 2
  ring

end MixedCollapseMaster
end PldrLlmCurvatureSandpile
