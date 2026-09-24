/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib
import PldrLlmCurvatureSandpile.NonautonomousAttraction
import PldrLlmCurvatureSandpile.GateShapeFactorization

namespace PldrLlmCurvatureSandpile
namespace AdamWGateForcing

open Filter

/-- Two chronological AdamW gate steps have the first nontrivial Duhamel
form, fixing the product and forcing order. -/
theorem adamw_gate_duhamel
    (gate multiplier₀ multiplier₁ innovation₀ innovation₁ : ℝ) :
    multiplier₁ * (multiplier₀ * gate - innovation₀) - innovation₁
      = (multiplier₁ * multiplier₀) * gate
        - (multiplier₁ * innovation₀ + innovation₁) := by
  ring

/-- One stable gate step propagates the homogeneous magnitude and charges
the adaptive innovation additively. -/
theorem gate_forcing_step
    {gate next multiplier innovation contraction : ℝ}
    (hnext : next = multiplier * gate - innovation)
    (hmultiplier : |multiplier| ≤ contraction) :
    |next| ≤ contraction * |gate| + |innovation| := by
  rw [hnext]
  calc
    |multiplier * gate - innovation|
        ≤ |multiplier * gate| + |innovation| := abs_sub _ _
    _ = |multiplier| * |gate| + |innovation| := by rw [abs_mul]
    _ ≤ contraction * |gate| + |innovation| := by
      have hproduct :
          |multiplier| * |gate| ≤ contraction * |gate| :=
        mul_le_mul_of_nonneg_right hmultiplier (abs_nonneg gate)
      linarith

/-- A stable gate update preserves any radius that absorbs the bounded
innovation. -/
theorem persistent_innovation_gate_tube
    {gate next multiplier innovation contraction force radius : ℝ}
    (hnext : next = multiplier * gate - innovation)
    (hmultiplier : |multiplier| ≤ contraction)
    (hcontraction : 0 ≤ contraction)
    (hgate : |gate| ≤ radius)
    (hforce : |innovation| ≤ force)
    (hradius : contraction * radius + force ≤ radius) :
    |next| ≤ radius := by
  calc
    |next| ≤ contraction * |gate| + |innovation| :=
      gate_forcing_step hnext hmultiplier
    _ ≤ contraction * radius + force :=
      add_le_add
        (mul_le_mul_of_nonneg_left hgate hcontraction) hforce
    _ ≤ radius := hradius

/-- If both the inherited stable part and the innovation are below halves
of a target tolerance, the next gate lies below that target. -/
theorem vanishing_innovation_gate_collapse
    {gate next multiplier innovation tolerance : ℝ}
    (hnext : next = multiplier * gate - innovation)
    (hhomogeneous : |multiplier| * |gate| < tolerance / 2)
    (hinnovation : |innovation| < tolerance / 2) :
    |next| < tolerance := by
  rw [hnext]
  calc
    |multiplier * gate - innovation|
        ≤ |multiplier * gate| + |innovation| := abs_sub _ _
    _ = |multiplier| * |gate| + |innovation| := by rw [abs_mul]
    _ < tolerance := by linarith

/-- An Adam denominator floor converts a first-moment bound into an adaptive
direction bound. -/
theorem vanishing_gradient_direction
    {moment denominator epsilon bound : ℝ}
    (hepsilon : 0 < epsilon)
    (hdenominator : epsilon ≤ denominator)
    (hmoment : |moment| ≤ epsilon * bound)
    (hbound : 0 ≤ bound) :
    |moment / denominator| ≤ bound := by
  have hdenominatorPositive : 0 < denominator :=
    lt_of_lt_of_le hepsilon hdenominator
  rw [abs_div, abs_of_pos hdenominatorPositive]
  apply (div_le_iff₀ hdenominatorPositive).2
  calc
    |moment| ≤ epsilon * bound := hmoment
    _ ≤ denominator * bound :=
      mul_le_mul_of_nonneg_right hdenominator hbound
    _ = bound * denominator := by ring

/-- The complete finite implemented AdamW gate recurrence is the chronological
product-convolution with signed adaptive innovation. -/
theorem adamw_gate_duhamel_finite
    (gate multiplier innovation : ℕ → ℝ)
    (hstep : ∀ step,
      gate (step + 1) = multiplier step * gate step - innovation step) :
    ∀ horizon, gate horizon =
      NonautonomousAttraction.orderedEnvelope multiplier
        (fun step => -innovation step) (gate 0) horizon := by
  intro horizon
  induction horizon with
  | zero => simp [NonautonomousAttraction.orderedEnvelope]
  | succ horizon inductionHypothesis =>
      rw [hstep horizon, inductionHypothesis]
      simp [NonautonomousAttraction.orderedEnvelope, sub_eq_add_neg]

/-- Taking absolute values gives the homogeneous magnitude plus the absolute
innovation convolution in the same chronological order. -/
theorem adamw_gate_absolute_convolution
    (gate multiplier innovation : ℕ → ℝ)
    (hstep : ∀ step,
      gate (step + 1) = multiplier step * gate step - innovation step) :
    ∀ horizon, |gate horizon| ≤
      NonautonomousAttraction.orderedEnvelope
        (fun step => |multiplier step|)
        (fun step => |innovation step|) |gate 0| horizon := by
  apply NonautonomousAttraction.ordered_product_convolution
  · exact le_rfl
  · exact fun step => abs_nonneg (multiplier step)
  · intro step
    rw [hstep step]
    calc
      |multiplier step * gate step - innovation step|
          ≤ |multiplier step * gate step| + |innovation step| :=
        abs_sub _ _
      _ = |multiplier step| * |gate step| + |innovation step| := by
        rw [abs_mul]

/-- Vanishing of the full absolute innovation convolution, including the
homogeneous term, forces the implemented scalar gate coordinate to vanish. -/
theorem adamw_gate_collapse_of_vanishing_convolution
    (gate multiplier innovation : ℕ → ℝ)
    (hstep : ∀ step,
      gate (step + 1) = multiplier step * gate step - innovation step)
    (henvelope : Tendsto
      (NonautonomousAttraction.orderedEnvelope
        (fun step => |multiplier step|)
        (fun step => |innovation step|) |gate 0|)
      atTop (nhds 0)) :
    Tendsto gate atTop (nhds 0) := by
  rw [tendsto_zero_iff_abs_tendsto_zero]
  apply squeeze_zero
  · exact fun step => abs_nonneg (gate step)
  · exact adamw_gate_absolute_convolution gate multiplier innovation hstep
  · exact henvelope

/-- The same exact criterion applies coordinatewise to every finite
LayerNorm gate. -/
theorem adamw_finite_gate_coordinate_collapse {d : ℕ}
    (gate multiplier innovation : ℕ → Fin d → ℝ)
    (hstep : ∀ step coordinate,
      gate (step + 1) coordinate =
        multiplier step coordinate * gate step coordinate -
          innovation step coordinate)
    (henvelope : ∀ coordinate, Tendsto
      (NonautonomousAttraction.orderedEnvelope
        (fun step => |multiplier step coordinate|)
        (fun step => |innovation step coordinate|) |gate 0 coordinate|)
      atTop (nhds 0)) :
    ∀ coordinate, Tendsto (fun step => gate step coordinate)
      atTop (nhds 0) := by
  intro coordinate
  exact adamw_gate_collapse_of_vanishing_convolution
    (fun step => gate step coordinate)
    (fun step => multiplier step coordinate)
    (fun step => innovation step coordinate)
    (fun step => hstep step coordinate)
    (henvelope coordinate)

/-- The implemented coordinate recurrence, its complete absolute convolution,
and a uniform normalized-shape bound together force the finite mixed row
energy to vanish. -/
theorem adamw_gate_shape_energy_collapse {d : ℕ}
    (gate multiplier innovation shapeNorm : ℕ → Fin d → ℝ)
    (shapeBound : ℝ)
    (hstep : ∀ step coordinate,
      gate (step + 1) coordinate =
        multiplier step coordinate * gate step coordinate -
          innovation step coordinate)
    (henvelope : ∀ coordinate, Tendsto
      (NonautonomousAttraction.orderedEnvelope
        (fun step => |multiplier step coordinate|)
        (fun step => |innovation step coordinate|) |gate 0 coordinate|)
      atTop (nhds 0))
    (hshapeBound : 0 ≤ shapeBound)
    (hshape : ∀ step coordinate,
      |shapeNorm step coordinate| ≤ shapeBound) :
    Tendsto
      (fun step => GateShapeFactorization.gateShapeEnergy
        (gate step) (shapeNorm step))
      atTop (nhds 0) := by
  apply GateShapeFactorization.gate_shape_energy_collapse_of_gate_collapse
    gate shapeNorm shapeBound hshapeBound hshape
  exact adamw_finite_gate_coordinate_collapse
    gate multiplier innovation hstep henvelope

end AdamWGateForcing
end PldrLlmCurvatureSandpile
