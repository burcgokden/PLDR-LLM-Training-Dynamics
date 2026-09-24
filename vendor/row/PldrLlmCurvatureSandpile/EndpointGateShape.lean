/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Endpoint gate-shape algebra
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace EndpointGateShape

/-- Adding and subtracting the fixed-source-shape endpoint gives the exact
gate-minus-shape decrement. -/
theorem gate_shape_decrement
    (sourceEnergy fixedEndpointEnergy endpointEnergy : ℝ) :
    sourceEnergy - endpointEnergy =
      (sourceEnergy - fixedEndpointEnergy)
      - (endpointEnergy - fixedEndpointEnergy) := by
  ring

/-- The scalar polarization bound underlying the vector shape-work bound. -/
theorem shape_work_upper
    (fixedComponent increment radius : ℝ)
    (hradius : |increment| ≤ radius) (hradiusNonnegative : 0 ≤ radius) :
    (fixedComponent + increment) ^ 2 - fixedComponent ^ 2
      ≤ 2 * |fixedComponent| * radius + radius ^ 2 := by
  have hproduct : fixedComponent * increment ≤ |fixedComponent| * radius := by
    have habsProduct : |fixedComponent * increment| ≤ |fixedComponent| * radius := by
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left hradius (abs_nonneg fixedComponent)
    exact (le_abs_self (fixedComponent * increment)).trans habsProduct
  have hsquare : increment ^ 2 ≤ radius ^ 2 := by
    have bound := (sq_le_sq₀ (abs_nonneg increment) hradiusNonnegative).2 hradius
    simpa only [sq_abs] using bound
  nlinarith

/-- The chronological gate balance is exact when the executed endpoint is
written as a decayed gate minus its finite innovation. -/
theorem executed_gate_balance
    (sourceGate decayedGate innovation shapeWeight : ℝ) :
    shapeWeight * (sourceGate ^ 2 - (decayedGate - innovation) ^ 2) =
      shapeWeight * (sourceGate ^ 2 - decayedGate ^ 2)
      + 2 * shapeWeight * decayedGate * innovation
      - shapeWeight * innovation ^ 2 := by
  ring

end EndpointGateShape
end PldrLlmCurvatureSandpile
