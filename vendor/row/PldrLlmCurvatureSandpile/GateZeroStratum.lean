/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Gate-zero AdamW stratum
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace GateZeroStratum

/-- With a nonzero learning rate and denominator, a zero gate remains zero
after the adaptive write exactly when the updated first moment is zero. -/
theorem zero_gate_successor_iff_zero_moment
    (learningRate denominator updatedMoment : ℝ)
    (hrate : learningRate ≠ 0)
    (hdenominator : denominator ≠ 0) :
    -learningRate * (updatedMoment / denominator) = 0
      ↔ updatedMoment = 0 := by
  constructor
  · intro hzero
    have hquotient : updatedMoment / denominator = 0 :=
      (mul_eq_zero.mp hzero).resolve_left (neg_ne_zero.mpr hrate)
    exact (div_eq_zero_iff.mp hquotient).resolve_right hdenominator
  · intro hmoment
    simp [hmoment]

/-- On the gate-zero, first-moment-zero face, invariance is equivalent to a
zero clipped gate gradient whenever the first-moment write is active. -/
theorem zero_gate_zero_moment_invariant_iff
    (beta learningRate denominator clippedGradient : ℝ)
    (hbeta : beta ≠ 1)
    (hrate : learningRate ≠ 0)
    (hdenominator : denominator ≠ 0) :
    -learningRate
        * ((beta * 0 + (1 - beta) * clippedGradient) / denominator)
        = 0
      ↔ clippedGradient = 0 := by
  rw [
    zero_gate_successor_iff_zero_moment
      learningRate denominator
      (beta * 0 + (1 - beta) * clippedGradient)
      hrate hdenominator,
  ]
  simp only [mul_zero, zero_add]
  constructor
  · intro hproduct
    have hcoefficient : 1 - beta ≠ 0 := sub_ne_zero.mpr (Ne.symm hbeta)
    exact (mul_eq_zero.mp hproduct).resolve_left hcoefficient
  · intro hgradient
    simp [hgradient]

end GateZeroStratum
end PldrLlmCurvatureSandpile
