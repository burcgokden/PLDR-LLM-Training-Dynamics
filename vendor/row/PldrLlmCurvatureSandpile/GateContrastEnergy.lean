/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact gate-shape contrast energy
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace GateContrastEnergy

/-- Diagonal row-shape energy in the final LayerNorm gate coordinates. -/
def contrastEnergy {d : ℕ} (gate shapeMetric : Fin d → ℝ) : ℝ :=
  ∑ coordinate, shapeMetric coordinate * (gate coordinate) ^ 2

/-- The gate contribution to one exact energy increment. -/
def gateWork {d : ℕ}
    (gate gateIncrement shapeMetric : Fin d → ℝ) : ℝ :=
  ∑ coordinate,
    shapeMetric coordinate
      * (2 * gate coordinate * gateIncrement coordinate
        + (gateIncrement coordinate) ^ 2)

/-- The shape contribution is evaluated at the successor gate. -/
def shapeWork {d : ℕ}
    (gateNext shapeMetric shapeMetricNext : Fin d → ℝ) : ℝ :=
  ∑ coordinate,
    (shapeMetricNext coordinate - shapeMetric coordinate)
      * (gateNext coordinate) ^ 2

/-- Gate work plus shape work is the exact finite increment, with no
linearization remainder. -/
theorem exact_gate_shape_increment {d : ℕ}
    (gate gateIncrement shapeMetric shapeMetricNext : Fin d → ℝ) :
    contrastEnergy (fun coordinate =>
        gate coordinate + gateIncrement coordinate) shapeMetricNext
      - contrastEnergy gate shapeMetric
      = gateWork gate gateIncrement shapeMetric
        + shapeWork
          (fun coordinate => gate coordinate + gateIncrement coordinate)
          shapeMetric shapeMetricNext := by
  classical
  simp only [contrastEnergy, gateWork, shapeWork,
    ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- A weighted square construction makes every diagonal shape coefficient
nonnegative. -/
theorem shape_metric_nonnegative {Pair : Type*} [Fintype Pair]
    {d : ℕ} (weight : Pair → ℝ) (shapeDifference : Pair → Fin d → ℝ)
    (hweight : ∀ pair, 0 ≤ weight pair) (coordinate : Fin d) :
    0 ≤ ∑ pair,
      weight pair * (shapeDifference pair coordinate) ^ 2 := by
  exact Finset.sum_nonneg fun pair _ =>
    mul_nonneg (hweight pair) (sq_nonneg _)

/-- A zero gate has zero contrast energy for every row-shape metric. -/
theorem zero_gate_zero_energy {d : ℕ} (shapeMetric : Fin d → ℝ) :
    contrastEnergy (fun _ => 0) shapeMetric = 0 := by
  simp [contrastEnergy]

end GateContrastEnergy
end PldrLlmCurvatureSandpile
