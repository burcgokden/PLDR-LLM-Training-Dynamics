/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ChronologicalMargin

def executedMargin
    (decay alignment shapeGain finiteCharge shapeCharge numericalCharge : ℝ) :
    ℝ :=
  decay + alignment + shapeGain
    - finiteCharge - shapeCharge - numericalCharge

def lowerCertificate
    (decay alignmentLower shapeGainLower finiteCharge
      shapeChargeUpper numericalChargeUpper : ℝ) : ℝ :=
  decay + alignmentLower + shapeGainLower
    - finiteCharge - shapeChargeUpper - numericalChargeUpper

/-- Lower signed alignments and upper nonnegative charges give a genuine lower
certificate for the executed margin. -/
theorem lower_certificate_le_executed
    {decay alignment alignmentLower shapeGain shapeGainLower finiteCharge
      shapeCharge shapeChargeUpper numericalCharge numericalChargeUpper : ℝ}
    (halignment : alignmentLower ≤ alignment)
    (hshapeGain : shapeGainLower ≤ shapeGain)
    (hshapeCharge : shapeCharge ≤ shapeChargeUpper)
    (hnumerical : numericalCharge ≤ numericalChargeUpper) :
    lowerCertificate decay alignmentLower shapeGainLower finiteCharge
        shapeChargeUpper numericalChargeUpper
      ≤ executedMargin decay alignment shapeGain finiteCharge
        shapeCharge numericalCharge := by
  unfold lowerCertificate executedMargin
  linarith

/-- A positive lower certificate forces one-step pair contraction. -/
theorem positive_certificate_contracts
    {energy energyNext certificate : ℝ}
    (hbalance : energyNext - energy ≤ -certificate)
    (hpositive : 0 < certificate) :
    energyNext < energy := by
  linarith

/-- Cauchy-type tail charging preserves the scale and sign of the executed
optimizer displacement. -/
theorem signed_tail_lower
    {eta state tail : ℝ} (heta : 0 ≤ eta) :
    -2 * eta * |state| * |tail| ≤ 2 * eta * state * tail := by
  have hproduct : -(|state| * |tail|) ≤ state * tail := by
    simpa [abs_mul] using (neg_abs_le (state * tail))
  nlinarith

/-- Clipping a negative affine coefficient at zero keeps a valid upper
recursion when the state is nonnegative. -/
theorem clipped_affine_coefficient
    {next state coefficient forcing : ℝ}
    (hnext : next ≤ coefficient * state + forcing)
    (hstate : 0 ≤ state) :
    next ≤ max 0 coefficient * state + forcing := by
  by_cases hcoefficient : 0 ≤ coefficient
  · rw [max_eq_right hcoefficient]
    exact hnext
  · have hmax : max 0 coefficient = 0 := max_eq_left (le_of_not_ge hcoefficient)
    rw [hmax]
    have hproduct : coefficient * state ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (le_of_not_ge hcoefficient) hstate
    linarith

end ChronologicalMargin
end PldrLlmCurvatureSandpile

