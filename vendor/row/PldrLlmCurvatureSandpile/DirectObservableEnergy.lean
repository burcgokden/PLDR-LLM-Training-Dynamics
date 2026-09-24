/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact direct observable-energy ledgers

Finite real-arithmetic kernels for the physical row-quotient energy.  The
index type may encode row, column, context, layer, and head coordinates.
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace DirectObservableEnergy

/-- Squared Frobenius energy after flattening any finite observable tensor. -/
def energy {Coordinate : Type*} [Fintype Coordinate]
    (state : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, (state coordinate) ^ 2

/-- Frobenius pairing after the same finite flattening. -/
def pairing {Coordinate : Type*} [Fintype Coordinate]
    (left right : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, left coordinate * right coordinate

/-- Every finite observable increment has an exact work-charge balance. -/
theorem exact_work_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    energy (fun coordinate => state coordinate + increment coordinate)
        - energy state
      = 2 * pairing state increment + energy increment := by
  classical
  simp only [energy, pairing, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- The old-shape, old-gate contribution to a product secant. -/
def shapeSecant (shapeSource shapeEndpoint gateSource : ℝ) : ℝ :=
  (shapeEndpoint - shapeSource) * gateSource

/-- The old-shape, gate-increment contribution to a product secant. -/
def gateSecant (shapeSource gateSource gateEndpoint : ℝ) : ℝ :=
  shapeSource * (gateEndpoint - gateSource)

/-- The bilinear interaction in a product secant. -/
def interactionSecant
    (shapeSource shapeEndpoint gateSource gateEndpoint : ℝ) : ℝ :=
  (shapeEndpoint - shapeSource) * (gateEndpoint - gateSource)

/-- Exact two-factor secant with no omitted product interaction. -/
theorem exact_gate_shape_secant
    (shapeSource shapeEndpoint gateSource gateEndpoint : ℝ) :
    shapeEndpoint * gateEndpoint - shapeSource * gateSource
      = shapeSecant shapeSource shapeEndpoint gateSource
        + gateSecant shapeSource gateSource gateEndpoint
        + interactionSecant
            shapeSource shapeEndpoint gateSource gateEndpoint := by
  unfold shapeSecant gateSecant interactionSecant
  ring

/-- Expanding an increment into shape, gate, and interaction parts gives the
complete energy ledger, including every cross-Gram term. -/
theorem exact_three_component_energy_ledger
    {Coordinate : Type*} [Fintype Coordinate]
    (state shape gate interaction : Coordinate → ℝ) :
    energy (fun coordinate =>
          state coordinate +
            (shape coordinate + gate coordinate + interaction coordinate))
        - energy state
      = 2 * (pairing state shape + pairing state gate
          + pairing state interaction)
        + energy shape + energy gate + energy interaction
        + 2 * (pairing shape gate + pairing shape interaction
          + pairing gate interaction) := by
  classical
  simp only [energy, pairing, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- Endpoint change of the native-versus-factorized implementation defect. -/
def implementationDefectSecant
    (nativeSource nativeEndpoint factorizedSource factorizedEndpoint : ℝ) : ℝ :=
  (nativeEndpoint - factorizedEndpoint) -
    (nativeSource - factorizedSource)

/-- The native product secant is the three algebraic factor sources plus the
endpoint change of the implementation defect. -/
theorem exact_precision_resolved_gate_shape_secant
    (nativeSource nativeEndpoint shapeSource shapeEndpoint
      gateSource gateEndpoint : ℝ) :
    nativeEndpoint - nativeSource
      = shapeSecant shapeSource shapeEndpoint gateSource
        + gateSecant shapeSource gateSource gateEndpoint
        + interactionSecant
            shapeSource shapeEndpoint gateSource gateEndpoint
        + implementationDefectSecant nativeSource nativeEndpoint
            (shapeSource * gateSource) (shapeEndpoint * gateEndpoint) := by
  unfold shapeSecant gateSecant interactionSecant implementationDefectSecant
  ring

/-- Four exact increment sources give the complete work and cross-Gram
energy ledger, including the implementation-defect source. -/
theorem exact_four_component_energy_ledger
    {Coordinate : Type*} [Fintype Coordinate]
    (state shape gate interaction defect : Coordinate → ℝ) :
    energy (fun coordinate =>
          state coordinate + (shape coordinate + gate coordinate
            + interaction coordinate + defect coordinate))
        - energy state
      = 2 * (pairing state shape + pairing state gate
          + pairing state interaction + pairing state defect)
        + energy shape + energy gate + energy interaction + energy defect
        + 2 * (pairing shape gate + pairing shape interaction
          + pairing shape defect + pairing gate interaction
          + pairing gate defect + pairing interaction defect) := by
  classical
  simp only [energy, pairing, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- The energy contrast between two increments from the same source state is
state conditional.  It contains source projection, interaction with the
natural increment, and a nonnegative paired charge. -/
theorem exact_paired_energy_contrast
    {Coordinate : Type*} [Fintype Coordinate]
    (state natural control : Coordinate → ℝ) :
    energy (fun coordinate => state coordinate + control coordinate)
        - energy (fun coordinate => state coordinate + natural coordinate)
      = 2 * pairing state (fun coordinate =>
            control coordinate - natural coordinate)
        + 2 * pairing natural (fun coordinate =>
            control coordinate - natural coordinate)
        + energy (fun coordinate =>
            control coordinate - natural coordinate) := by
  classical
  simp only [energy, pairing, ← Finset.sum_sub_distrib,
    ← Finset.sum_add_distrib, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- At the row-constant face, the endpoint energy is exactly the quadratic
charge of the physical increment.  The linear work term vanishes there. -/
theorem face_endpoint_energy
    {Coordinate : Type*} [Fintype Coordinate]
    (increment : Coordinate → ℝ) :
    energy (fun coordinate => (0 : ℝ) + increment coordinate)
      = energy increment := by
  simp [energy]

/-- A finite row-constant face is preserved exactly when every physical
increment coordinate vanishes. -/
theorem face_preserved_iff
    {Coordinate : Type*} [Fintype Coordinate]
    (increment : Coordinate → ℝ) :
    energy increment = 0 ↔ ∀ coordinate, increment coordinate = 0 := by
  classical
  constructor
  · intro henergy coordinate
    have hcoordinate : (increment coordinate) ^ 2 ≤ energy increment := by
      unfold energy
      exact Finset.single_le_sum
        (fun index _ => sq_nonneg (increment index))
        (Finset.mem_univ coordinate)
    have hsquare : (increment coordinate) ^ 2 = 0 := by
      apply le_antisymm
      · simpa [henergy] using hcoordinate
      · exact sq_nonneg _
    exact (sq_eq_zero_iff).mp hsquare
  · intro hzero
    simp [energy, hzero]

end DirectObservableEnergy
end PldrLlmCurvatureSandpile
