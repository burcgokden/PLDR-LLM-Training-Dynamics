/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Program-energy renormalization map

Algebraic kernel for ordered affine blocking and positive endpoint gauges.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ProgramEnergyRG

/-- A scalar positive-comparison edge, written as `x ↦ gain * x + source`. -/
@[ext] structure AffineEdge where
  gain : ℝ
  source : ℝ

/-- Temporal composition. The earlier edge is the second argument. -/
def compose (later earlier : AffineEdge) : AffineEdge where
  gain := later.gain * earlier.gain
  source := later.gain * earlier.source + later.source

/-- The neutral affine comparison. -/
def identity : AffineEdge where
  gain := 1
  source := 0

@[simp] theorem compose_gain (later earlier : AffineEdge) :
    (compose later earlier).gain = later.gain * earlier.gain := rfl

@[simp] theorem compose_source (later earlier : AffineEdge) :
    (compose later earlier).source =
      later.gain * earlier.source + later.source := rfl

/-- Ordered affine blocking is associative. -/
theorem compose_assoc (third second first : AffineEdge) :
    compose (compose third second) first =
      compose third (compose second first) := by
  apply AffineEdge.ext <;> simp [compose] <;> ring

@[simp] theorem identity_law (edge : AffineEdge) :
    compose identity edge = edge := by
  cases edge
  simp [compose, identity]

@[simp] theorem identity_r_law (edge : AffineEdge) :
    compose edge identity = edge := by
  cases edge
  simp [compose, identity]

/-- Both coefficients of a positive affine comparison are nonnegative. -/
def Nonnegative (edge : AffineEdge) : Prop :=
  0 ≤ edge.gain ∧ 0 ≤ edge.source

/-- Positive comparison edges are closed under ordered blocking. -/
theorem compose_nonnegative {later earlier : AffineEdge}
    (hlater : Nonnegative later) (hearlier : Nonnegative earlier) :
    Nonnegative (compose later earlier) := by
  constructor
  · exact mul_nonneg hlater.1 hearlier.1
  · exact add_nonneg (mul_nonneg hlater.1 hearlier.2) hlater.2

/-- Change positive scalar energy units at the two endpoints. -/
noncomputable def normalize (edge : AffineEdge)
    (sourceGauge targetGauge : ℝ) :
    AffineEdge where
  gain := edge.gain * sourceGauge / targetGauge
  source := edge.source / targetGauge

/-- Endpoint normalization commutes with blocking when the shared gauge is
nonzero. -/
theorem normalize_compose (later earlier : AffineEdge)
    (sourceGauge sharedGauge targetGauge : ℝ)
    (hshared : sharedGauge ≠ 0) (htarget : targetGauge ≠ 0) :
    compose
        (normalize later sharedGauge targetGauge)
        (normalize earlier sourceGauge sharedGauge) =
      normalize (compose later earlier) sourceGauge targetGauge := by
  ext <;> simp [normalize, compose]
  <;> field_simp [hshared, htarget]

/-- Two aligned binary blocks equal the corresponding ordered four-edge
block. -/
theorem aligned_four_edge_block (fourth third second first : AffineEdge) :
    compose (compose fourth third) (compose second first) =
      compose fourth (compose third (compose second first)) := by
  exact compose_assoc fourth third (compose second first)

/-- A stationary forced affine edge fixes its geometric-source value whenever
the denominator is nonzero. -/
theorem stationary_forced_fixed_point (gain source : ℝ)
    (hdenominator : 1 - gain ≠ 0) :
    gain * (source / (1 - gain)) + source = source / (1 - gain) := by
  field_simp [hdenominator]
  ring

end ProgramEnergyRG
end PldrLlmCurvatureSandpile
