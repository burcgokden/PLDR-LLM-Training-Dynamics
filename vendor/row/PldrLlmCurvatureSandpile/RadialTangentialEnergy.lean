/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact radial--tangential observable energy

Finite real-arithmetic kernels for the radial normal form, source
recomposition, paired contrast, positive excursions, and zero-face restarts.
The results make no optimizer contraction assumption.
-/
import Mathlib
import PldrLlmCurvatureSandpile.DirectObservableEnergy
import PldrLlmCurvatureSandpile.NonautonomousAttraction
import PldrLlmCurvatureSandpile.ObservableCapture

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace RadialTangentialEnergy

open DirectObservableEnergy

noncomputable section

/-- Radial coefficient of a realized increment at a nonzero observable state. -/
def radialCoefficient {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) : ℝ :=
  -pairing state increment / energy state

/-- Residual after removing the source-radial component of an increment. -/
def tangentResidual {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) : Coordinate → ℝ :=
  fun coordinate =>
    increment coordinate + radialCoefficient state increment * state coordinate

/-- Normalized tangent charge. It is used only when the source energy is nonzero. -/
def tangentCharge {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) : ℝ :=
  energy (tangentResidual state increment) / energy state

/-- Exact squared gain associated with the realized radial and tangent parts. -/
def gainSquared {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) : ℝ :=
  (1 - radialCoefficient state increment) ^ 2
    + tangentCharge state increment

lemma pairing_add_right
    {Coordinate : Type*} [Fintype Coordinate]
    (state left right : Coordinate → ℝ) :
    pairing state (fun coordinate => left coordinate + right coordinate)
      = pairing state left + pairing state right := by
  classical
  simp [pairing, mul_add, Finset.sum_add_distrib]

lemma pairing_add_left
    {Coordinate : Type*} [Fintype Coordinate]
    (left right value : Coordinate → ℝ) :
    pairing (fun coordinate => left coordinate + right coordinate) value
      = pairing left value + pairing right value := by
  classical
  simp [pairing, add_mul, Finset.sum_add_distrib]

lemma pairing_comm
    {Coordinate : Type*} [Fintype Coordinate]
    (left right : Coordinate → ℝ) :
    pairing left right = pairing right left := by
  classical
  unfold pairing
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

lemma pairing_scale_right
    {Coordinate : Type*} [Fintype Coordinate]
    (state value : Coordinate → ℝ) (scale : ℝ) :
    pairing state (fun coordinate => scale * value coordinate)
      = scale * pairing state value := by
  classical
  unfold pairing
  change (∑ coordinate, state coordinate * (scale * value coordinate))
    = scale * ∑ coordinate, state coordinate * value coordinate
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

lemma pairing_scale_left
    {Coordinate : Type*} [Fintype Coordinate]
    (state value : Coordinate → ℝ) (scale : ℝ) :
    pairing (fun coordinate => scale * state coordinate) value
      = scale * pairing state value := by
  classical
  unfold pairing
  change (∑ coordinate, (scale * state coordinate) * value coordinate)
    = scale * ∑ coordinate, state coordinate * value coordinate
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

lemma pairing_self_eq_energy
    {Coordinate : Type*} [Fintype Coordinate]
    (state : Coordinate → ℝ) : pairing state state = energy state := by
  classical
  simp [pairing, energy, pow_two]

lemma energy_scale
    {Coordinate : Type*} [Fintype Coordinate]
    (state : Coordinate → ℝ) (scale : ℝ) :
    energy (fun coordinate => scale * state coordinate)
      = scale ^ 2 * energy state := by
  classical
  simp [energy, mul_pow, Finset.mul_sum]

/-- The tangent residual is exactly orthogonal to every nonzero source state. -/
theorem tangent_residual_orthogonal
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    pairing state (tangentResidual state increment) = 0 := by
  change pairing state (fun coordinate =>
    increment coordinate
      + radialCoefficient state increment * state coordinate) = 0
  rw [pairing_add_right, pairing_scale_right,
    pairing_self_eq_energy, radialCoefficient]
  field_simp [henergy]
  ring

/-- The realized increment is the sum of its radial and tangent parts. -/
theorem radial_tangent_split
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    ∀ coordinate,
      increment coordinate
        = -radialCoefficient state increment * state coordinate
          + tangentResidual state increment coordinate := by
  intro coordinate
  simp [tangentResidual]

/-- Unnormalized exact radial--tangential energy identity. -/
theorem exact_radial_tangential_energy
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    energy (fun coordinate => state coordinate + increment coordinate)
      = (1 - radialCoefficient state increment) ^ 2 * energy state
        + energy (tangentResidual state increment) := by
  let scale : ℝ := 1 - radialCoefficient state increment
  let tangent := tangentResidual state increment
  have hpoint :
      (fun coordinate => state coordinate + increment coordinate)
        = (fun coordinate => scale * state coordinate + tangent coordinate) := by
    funext coordinate
    simp [scale, tangent, tangentResidual]
    ring
  have horthogonal : pairing state tangent = 0 := by
    simpa [tangent] using
      tangent_residual_orthogonal state increment henergy
  have hscaledPair :
      pairing (fun coordinate => scale * state coordinate) tangent = 0 := by
    rw [pairing_scale_left, horthogonal, mul_zero]
  have hledger := exact_work_charge
    (fun coordinate => scale * state coordinate) tangent
  rw [hpoint]
  calc
    energy (fun coordinate => scale * state coordinate + tangent coordinate)
        = energy (fun coordinate => scale * state coordinate)
            + 2 * pairing (fun coordinate => scale * state coordinate) tangent
            + energy tangent := by
          linarith [hledger]
    _ = (1 - radialCoefficient state increment) ^ 2 * energy state
          + energy (tangentResidual state increment) := by
      rw [energy_scale, hscaledPair]
      simp [scale, tangent]

/-- Exact normalized radial--tangential gain. -/
theorem exact_radial_tangential_gain
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    energy (fun coordinate => state coordinate + increment coordinate)
        / energy state
      = gainSquared state increment := by
  rw [exact_radial_tangential_energy state increment henergy]
  unfold gainSquared tangentCharge
  field_simp [henergy]

/-- Scalar closing criterion behind the radial normal form. -/
theorem scalar_closing_criterion
    (alpha tauSquared : ℝ) (htau : 0 ≤ tauSquared) :
    (1 - alpha) ^ 2 + tauSquared < 1 ↔
      0 < alpha ∧ alpha < 2 ∧ tauSquared < alpha * (2 - alpha) := by
  constructor
  · intro hclose
    have hmargin : tauSquared < alpha * (2 - alpha) := by
      nlinarith
    have hpositive : 0 < alpha * (2 - alpha) :=
      lt_of_le_of_lt htau hmargin
    constructor
    · nlinarith
    constructor
    · nlinarith
    · exact hmargin
  · rintro ⟨halpha, halphaTwo, hmargin⟩
    nlinarith

/-- Strict energy closing is equivalent to a nonovershooting radial step and
strictly submargin tangent charge. -/
theorem strict_energy_decrease_iff
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ)
    (henergy : 0 < energy state) :
    energy (fun coordinate => state coordinate + increment coordinate)
          < energy state ↔
      0 < radialCoefficient state increment
        ∧ radialCoefficient state increment < 2
        ∧ tangentCharge state increment
          < radialCoefficient state increment
              * (2 - radialCoefficient state increment) := by
  have hnonzero : energy state ≠ 0 := ne_of_gt henergy
  have htangentEnergy : 0 ≤ energy (tangentResidual state increment) := by
    unfold energy
    exact Finset.sum_nonneg fun coordinate _ => sq_nonneg _
  have htangent : 0 ≤ tangentCharge state increment := by
    unfold tangentCharge
    exact div_nonneg htangentEnergy (le_of_lt henergy)
  rw [← scalar_closing_criterion _ _ htangent]
  have hgain := exact_radial_tangential_gain state increment hnonzero
  unfold gainSquared at hgain
  rw [← hgain]
  exact (div_lt_one henergy).symm

/-- Radial coefficient commutes with a finite sum of increment sources. -/
theorem radial_coefficient_sum
    {Coordinate Source : Type*} [Fintype Coordinate] [Fintype Source]
    (state : Coordinate → ℝ) (component : Source → Coordinate → ℝ) :
    radialCoefficient state
        (fun coordinate => ∑ source, component source coordinate)
      = ∑ source, radialCoefficient state (component source) := by
  classical
  unfold radialCoefficient pairing
  rw [show (∑ coordinate, state coordinate * ∑ source, component source coordinate)
      = ∑ source, ∑ coordinate, state coordinate * component source coordinate by
        simp_rw [Finset.mul_sum]
        exact Finset.sum_comm]
  simp only [div_eq_mul_inv]
  rw [← Finset.sum_mul, ← Finset.sum_neg_distrib]

/-- Tangent residual also commutes with the same finite source sum. -/
theorem tangent_residual_sum
    {Coordinate Source : Type*} [Fintype Coordinate] [Fintype Source]
    (state : Coordinate → ℝ) (component : Source → Coordinate → ℝ) :
    tangentResidual state
        (fun coordinate => ∑ source, component source coordinate)
      = fun coordinate => ∑ source, tangentResidual state (component source) coordinate := by
  classical
  funext coordinate
  rw [tangentResidual, radial_coefficient_sum]
  simp [tangentResidual, Finset.sum_add_distrib, Finset.sum_mul]

/-- Four tangent sources retain the complete diagonal and cross-Gram charge. -/
theorem exact_four_tangent_gram
    {Coordinate : Type*} [Fintype Coordinate]
    (shape gate interaction defect : Coordinate → ℝ) :
    energy (fun coordinate =>
        shape coordinate + gate coordinate + interaction coordinate
          + defect coordinate)
      = energy shape + energy gate + energy interaction + energy defect
        + 2 * (pairing shape gate + pairing shape interaction
          + pairing shape defect + pairing gate interaction
          + pairing gate defect + pairing interaction defect) := by
  have hledger := exact_four_component_energy_ledger
    (fun _coordinate => (0 : ℝ)) shape gate interaction defect
  simpa [energy, pairing] using hledger

/-- Scalar normalization of the paired radial--tangential contrast. The
vector theorem supplies the unnormalized identity recorded in `hcontrast`. -/
theorem paired_radial_tangential_normalize
    (energyValue contrast alpha beta tangentInteraction tangentChargeValue : ℝ)
    (henergy : energyValue ≠ 0)
    (hcontrast :
      contrast
        = (-2 * (1 - alpha) * beta + beta ^ 2) * energyValue
          + tangentInteraction + tangentChargeValue) :
    contrast / energyValue
      = -2 * (1 - alpha) * beta + beta ^ 2
        + (tangentInteraction + tangentChargeValue) / energyValue := by
  rw [hcontrast]
  field_simp [henergy]
  ring

/-- The squared norm of a realized increment is its radial charge plus its
orthogonal tangent charge. -/
theorem increment_energy_radial_tangent
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    energy increment
      = (radialCoefficient state increment) ^ 2 * energy state
        + energy (tangentResidual state increment) := by
  let alpha := radialCoefficient state increment
  let tangent := tangentResidual state increment
  have hsplit : increment =
      fun coordinate => -alpha * state coordinate + tangent coordinate := by
    funext coordinate
    simpa [alpha, tangent] using
      radial_tangent_split state increment coordinate
  have horthogonal : pairing state tangent = 0 := by
    simpa [tangent] using
      tangent_residual_orthogonal state increment henergy
  have hscaled :
      pairing (fun coordinate => -alpha * state coordinate) tangent = 0 := by
    rw [pairing_scale_left, horthogonal, mul_zero]
  have hledger := exact_work_charge
    (fun coordinate => -alpha * state coordinate) tangent
  have hradial :
      energy (fun coordinate => -alpha * state coordinate)
        = alpha ^ 2 * energy state := by
    rw [energy_scale]
    ring
  calc
    energy increment =
        energy (fun coordinate =>
          -alpha * state coordinate + tangent coordinate) :=
      congrArg energy hsplit
    _ = alpha ^ 2 * energy state + energy tangent := by
      linarith
    _ = (radialCoefficient state increment) ^ 2 * energy state
          + energy (tangentResidual state increment) := by
      simp [alpha, tangent]

/-- Pairing two realized increments retains their radial product and the full
tangent interaction. -/
theorem pairing_radial_tangent_increments
    {Coordinate : Type*} [Fintype Coordinate]
    (state left right : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    pairing left right
      = radialCoefficient state left * radialCoefficient state right
          * energy state
        + pairing (tangentResidual state left)
            (tangentResidual state right) := by
  let alpha := radialCoefficient state left
  let beta := radialCoefficient state right
  let tangentLeft := tangentResidual state left
  let tangentRight := tangentResidual state right
  have hleft : left =
      fun coordinate => -alpha * state coordinate + tangentLeft coordinate := by
    funext coordinate
    simpa [alpha, tangentLeft] using
      radial_tangent_split state left coordinate
  have hright : right =
      fun coordinate => -beta * state coordinate + tangentRight coordinate := by
    funext coordinate
    simpa [beta, tangentRight] using
      radial_tangent_split state right coordinate
  have hleftOrthogonal : pairing state tangentLeft = 0 := by
    simpa [tangentLeft] using
      tangent_residual_orthogonal state left henergy
  have hrightOrthogonal : pairing state tangentRight = 0 := by
    simpa [tangentRight] using
      tangent_residual_orthogonal state right henergy
  have hleftOrthogonalReverse : pairing tangentLeft state = 0 := by
    rw [pairing_comm tangentLeft state, hleftOrthogonal]
  calc
    pairing left right =
        pairing
          (fun coordinate => -alpha * state coordinate + tangentLeft coordinate)
          (fun coordinate =>
            -beta * state coordinate + tangentRight coordinate) :=
      congrArg₂ pairing hleft hright
    _ = alpha * beta * energy state
          + pairing tangentLeft tangentRight := by
      rw [pairing_add_left, pairing_add_right, pairing_add_right,
        pairing_scale_left, pairing_scale_right, pairing_scale_left,
        pairing_scale_right, pairing_self_eq_energy,
        hrightOrthogonal, hleftOrthogonalReverse]
      ring
    _ = radialCoefficient state left * radialCoefficient state right
          * energy state
        + pairing (tangentResidual state left)
            (tangentResidual state right) := by
      simp [alpha, beta, tangentLeft, tangentRight]

/-- Full finite-vector radial and tangential paired contrast. Unlike the
scalar normalization lemma, this statement starts from the source, natural
increment, and paired increment themselves. -/
theorem exact_paired_radial_tangential_vector
    {Coordinate : Type*} [Fintype Coordinate]
    (state natural paired : Coordinate → ℝ)
    (henergy : energy state ≠ 0) :
    energy (fun coordinate =>
        state coordinate + (natural coordinate + paired coordinate))
        - energy (fun coordinate => state coordinate + natural coordinate)
      = (-2 * (1 - radialCoefficient state natural)
            * radialCoefficient state paired
          + (radialCoefficient state paired) ^ 2) * energy state
        + 2 * pairing
            (tangentResidual state natural)
            (tangentResidual state paired)
        + energy (tangentResidual state paired) := by
  have hcontrast := exact_paired_energy_contrast state natural
    (fun coordinate => natural coordinate + paired coordinate)
  have hcontrastSimplified :
      energy (fun coordinate =>
          state coordinate + (natural coordinate + paired coordinate))
          - energy (fun coordinate => state coordinate + natural coordinate)
        = 2 * pairing state paired
          + 2 * pairing natural paired + energy paired := by
    simpa using hcontrast
  have hpairedOrthogonal :=
    tangent_residual_orthogonal state paired henergy
  have hstatePaired :
      pairing state paired
        = -radialCoefficient state paired * energy state := by
    change pairing state (fun coordinate =>
      paired coordinate
        + radialCoefficient state paired * state coordinate) = 0
      at hpairedOrthogonal
    rw [pairing_add_right, pairing_scale_right,
      pairing_self_eq_energy] at hpairedOrthogonal
    linarith
  calc
    energy (fun coordinate =>
        state coordinate + (natural coordinate + paired coordinate))
        - energy (fun coordinate => state coordinate + natural coordinate)
      = 2 * pairing state paired
          + 2 * pairing natural paired + energy paired :=
      hcontrastSimplified
    _ = (-2 * (1 - radialCoefficient state natural)
            * radialCoefficient state paired
          + (radialCoefficient state paired) ^ 2) * energy state
        + 2 * pairing
            (tangentResidual state natural)
            (tangentResidual state paired)
        + energy (tangentResidual state paired) := by
      rw [hstatePaired,
        pairing_radial_tangent_increments state natural paired henergy,
        increment_energy_radial_tangent state paired henergy]
      ring

/-- Exact positive-excursion product for a chronological gain recurrence. -/
theorem exact_positive_excursion_product
    (energySequence gain : ℕ → ℝ)
    (hstep : ∀ step,
      energySequence (step + 1) = gain step * energySequence step) :
    ∀ step,
      energySequence step
        = ObservableCapture.chronologicalProduct gain step
          * energySequence 0 :=
  ObservableCapture.zero_force_exact_product energySequence gain hstep

/-- Gain used by the full affine energy cocycle; division occurs only off the
zero face. -/
def faceGain (source endpoint : ℝ) : ℝ :=
  if source = 0 then 0 else endpoint / source

/-- Explicit reopening injection at the zero face. -/
def faceReopening (source endpoint : ℝ) : ℝ :=
  if source = 0 then endpoint else 0

/-- Every scalar energy edge has an exact radial-face affine form. -/
theorem exact_face_affine_step (source endpoint : ℝ) :
    endpoint = faceGain source endpoint * source
      + faceReopening source endpoint := by
  by_cases hsource : source = 0
  · simp [faceGain, faceReopening, hsource]
  · simp [faceGain, faceReopening, hsource]

/-- The complete energy sequence, including zero-face restarts, is exactly the
chronological affine product-convolution. -/
theorem exact_face_affine_cocycle
    (energySequence : ℕ → ℝ)
    (gain reopening : ℕ → ℝ)
    (hstep : ∀ step,
      energySequence (step + 1)
        = gain step * energySequence step + reopening step) :
    ∀ step,
      energySequence step
        = NonautonomousAttraction.orderedEnvelope
            gain reopening (energySequence 0) step := by
  intro step
  induction step with
  | zero => simp [NonautonomousAttraction.orderedEnvelope]
  | succ step inductionHypothesis =>
      rw [hstep step, inductionHypothesis,
        NonautonomousAttraction.orderedEnvelope]

/-- Specializing the affine cocycle to the canonical face gain and reopening
keeps every zero-face transition explicit. -/
theorem canonical_face_affine_cocycle
    (energySequence : ℕ → ℝ) :
    ∀ step,
      energySequence step
        = NonautonomousAttraction.orderedEnvelope
            (fun t => faceGain (energySequence t) (energySequence (t + 1)))
            (fun t => faceReopening
              (energySequence t) (energySequence (t + 1)))
            (energySequence 0) step := by
  apply exact_face_affine_cocycle
  intro step
  exact exact_face_affine_step
    (energySequence step) (energySequence (step + 1))

end
end RadialTangentialEnergy
end PldrLlmCurvatureSandpile
