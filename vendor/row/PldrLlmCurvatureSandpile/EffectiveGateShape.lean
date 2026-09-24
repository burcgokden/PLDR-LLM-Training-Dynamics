/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Effective gate-shape factorization
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace EffectiveGateShape

open scoped BigOperators

/-- Total normalized-shape contrast energy. -/
def shapeEnergy {d : ℕ} (shape : Fin d → ℝ) : ℝ :=
  ∑ coordinate, shape coordinate

/-- Physical energy after the final diagonal LayerNorm gate. -/
def physicalEnergy {d : ℕ}
    (shape gate : Fin d → ℝ) : ℝ :=
  ∑ coordinate, shape coordinate * gate coordinate ^ 2

/-- A coordinatewise lower squared-gate bound transfers to total energy. -/
theorem gate_lower_energy_bound {d : ℕ}
    (shape gate : Fin d → ℝ) (lower : ℝ)
    (hshape : ∀ coordinate, 0 ≤ shape coordinate)
    (hgate : ∀ coordinate, lower ≤ gate coordinate ^ 2) :
    lower * shapeEnergy shape ≤ physicalEnergy shape gate := by
  unfold shapeEnergy physicalEnergy
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun coordinate _ => by
    simpa [mul_comm] using
      mul_le_mul_of_nonneg_right (hgate coordinate) (hshape coordinate)

/-- A coordinatewise upper squared-gate bound transfers to total energy. -/
theorem gate_upper_energy_bound {d : ℕ}
    (shape gate : Fin d → ℝ) (upper : ℝ)
    (hshape : ∀ coordinate, 0 ≤ shape coordinate)
    (hgate : ∀ coordinate, gate coordinate ^ 2 ≤ upper) :
    physicalEnergy shape gate ≤ upper * shapeEnergy shape := by
  unfold shapeEnergy physicalEnergy
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun coordinate _ => by
    simpa [mul_comm] using
      mul_le_mul_of_nonneg_left (hgate coordinate) (hshape coordinate)

/-- Positive endpoint energy gain splits exactly into normalized-shape gain,
the endpoint gate acting on source occupancy, and occupancy alignment. -/
theorem endpoint_three_channel_gain
    {shapeSource shapeEndpoint effectiveSource gateOnlyEndpoint
      effectiveEndpoint : ℝ}
    (hshapeSource : shapeSource ≠ 0)
    (heffectiveSource : effectiveSource ≠ 0)
    (hgateOnlyEndpoint : gateOnlyEndpoint ≠ 0) :
    (shapeEndpoint * effectiveEndpoint)
          / (shapeSource * effectiveSource) =
      (shapeEndpoint / shapeSource)
        * (gateOnlyEndpoint / effectiveSource)
        * (effectiveEndpoint / gateOnlyEndpoint) := by
  field_simp [hshapeSource, heffectiveSource, hgateOnlyEndpoint]

end EffectiveGateShape
end PldrLlmCurvatureSandpile
