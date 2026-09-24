/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Gate, shape, and mixed logarithmic routes
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace LogRouteCollapse

/-- Coordinate contrast energy. -/
def coordinateEnergy (gate shapeMetric : ℝ) : ℝ :=
  shapeMetric * gate ^ 2

/-- For nonzero factors, the coordinate energy ratio is exactly the product
of the squared-gate ratio and the shape-metric ratio. -/
theorem coordinate_ratio_product
    {gate gateNext shapeMetric shapeMetricNext : ℝ}
    (hgate : gate ≠ 0) (hshape : shapeMetric ≠ 0) :
    coordinateEnergy gateNext shapeMetricNext
        / coordinateEnergy gate shapeMetric
      = (gateNext ^ 2 / gate ^ 2)
        * (shapeMetricNext / shapeMetric) := by
  unfold coordinateEnergy
  field_simp

/-- Per-step realized logarithmic decrement. -/
noncomputable def logDecrement (before after : ℝ) : ℝ :=
  Real.log before - Real.log after

/-- Realized logarithmic decrements telescope for every time-varying path.
Positivity is needed only when the result is interpreted as a log ratio. -/
theorem log_route_telescopes (energy : ℕ → ℝ) :
    ∀ n, (∑ t ∈ Finset.range n,
      logDecrement (energy t) (energy (t + 1)))
        = Real.log (energy 0) - Real.log (energy n) := by
  intro n
  induction n with
  | zero => simp
  | succ n inductionHypothesis =>
      rw [Finset.sum_range_succ, inductionHypothesis]
      unfold logDecrement
      ring

/-- Exact zero of either factor annihilates the coordinate energy. -/
theorem zero_factor_energy
    (gate shapeMetric : ℝ) (hzero : gate = 0 ∨ shapeMetric = 0) :
    coordinateEnergy gate shapeMetric = 0 := by
  rcases hzero with hgate | hshape
  · simp [coordinateEnergy, hgate]
  · simp [coordinateEnergy, hshape]

end LogRouteCollapse
end PldrLlmCurvatureSandpile
