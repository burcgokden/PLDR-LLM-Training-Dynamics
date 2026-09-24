/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact AdamW gate-energy dissipation
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace AdamWGateDissipation

/-- Fixed-shape diagonal gate energy. -/
def gateEnergy {d : ℕ} (shapeMetric gate : Fin d → ℝ) : ℝ :=
  ∑ j, shapeMetric j * (gate j) ^ 2

/-- The exact finite fixed-shape energy change under `gate - eta * direction`.
-/
theorem gate_energy_step {d : ℕ}
    (shapeMetric gate direction : Fin d → ℝ) (eta : ℝ) :
    gateEnergy shapeMetric (fun j => gate j - eta * direction j)
        - gateEnergy shapeMetric gate
      = -2 * eta * ∑ j, shapeMetric j * gate j * direction j
        + eta ^ 2 * ∑ j, shapeMetric j * (direction j) ^ 2 := by
  classical
  rw [gateEnergy, gateEnergy, ← Finset.sum_sub_distrib]
  calc
    ∑ j, (shapeMetric j * (gate j - eta * direction j) ^ 2
          - shapeMetric j * gate j ^ 2)
        = ∑ j, (-2 * eta * (shapeMetric j * gate j * direction j)
          + eta ^ 2 * (shapeMetric j * direction j ^ 2)) := by
            apply Finset.sum_congr rfl
            intro j _
            ring
    _ = -2 * eta * ∑ j, shapeMetric j * gate j * direction j
        + eta ^ 2 * ∑ j, shapeMetric j * direction j ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]

/-- Decoupled decay and the adaptive loss-memory direction form the exact
coordinate successor used in the gate theorem. -/
theorem decay_loss_step
    (gate adaptive eta decay : ℝ) :
    gate - eta * (decay * gate + adaptive)
      = (1 - eta * decay) * gate - eta * adaptive := by
  ring

/-- If the adaptive direction is smaller than a fixed fraction of the direct
decay direction, one gate coordinate contracts in absolute value. -/
theorem decay_dominated_step
    {gate adaptive eta decay rho : ℝ}
    (heta : 0 ≤ eta)
    (hstep : eta * decay ≤ 1)
    (hadaptive : |adaptive| ≤ rho * decay * |gate|) :
    |(1 - eta * decay) * gate - eta * adaptive|
      ≤ (1 - (1 - rho) * eta * decay) * |gate| := by
  have hmultiplier : 0 ≤ 1 - eta * decay := sub_nonneg.mpr hstep
  calc
    |(1 - eta * decay) * gate - eta * adaptive|
        ≤ |(1 - eta * decay) * gate| + |eta * adaptive| :=
      abs_sub _ _
    _ = (1 - eta * decay) * |gate| + eta * |adaptive| := by
      rw [abs_mul, abs_mul, abs_of_nonneg hmultiplier, abs_of_nonneg heta]
    _ ≤ (1 - eta * decay) * |gate|
        + eta * (rho * decay * |gate|) := by
      exact add_le_add (le_refl _)
        (mul_le_mul_of_nonneg_left hadaptive heta)
    _ = (1 - (1 - rho) * eta * decay) * |gate| := by ring

end AdamWGateDissipation
end PldrLlmCurvatureSandpile
