/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib
import PldrLlmCurvatureSandpile.ObservableEnergyDissipation

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace ExecutedAdamWEnergy

def energy {Coordinate : Type*} [Fintype Coordinate]
    (metric state : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, metric coordinate * state coordinate ^ 2

def pairing {Coordinate : Type*} [Fintype Coordinate]
    (metric left right : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, metric coordinate * left coordinate * right coordinate

def decayDissipation {Coordinate : Type*} [Fintype Coordinate]
    (metric gate decayMultiplier : Coordinate → ℝ) : ℝ :=
  energy metric gate -
    energy metric (fun coordinate => decayMultiplier coordinate * gate coordinate)

/-- Exact fixed-metric energy balance for the executed AdamW gate step. -/
theorem executed_adamw_balance
    {Coordinate : Type*} [Fintype Coordinate]
    (metric gate direction decayMultiplier : Coordinate → ℝ) (eta : ℝ) :
    energy metric
          (fun coordinate =>
            decayMultiplier coordinate * gate coordinate
              - eta * direction coordinate)
        - energy metric gate
      = -decayDissipation metric gate decayMultiplier
        - 2 * eta * pairing metric
            (fun coordinate => decayMultiplier coordinate * gate coordinate)
            direction
        + eta ^ 2 * energy metric direction := by
  have hstep :=
    ObservableEnergyDissipation.moving_metric_adamw_identity
      metric metric
      (fun coordinate => decayMultiplier coordinate * gate coordinate)
      direction eta
  have hstep' :
      energy metric
          (fun coordinate =>
            decayMultiplier coordinate * gate coordinate
              - eta * direction coordinate)
        - energy metric
          (fun coordinate => decayMultiplier coordinate * gate coordinate)
        = -2 * eta * pairing metric
            (fun coordinate => decayMultiplier coordinate * gate coordinate)
            direction
          + eta ^ 2 * energy metric direction := by
    simpa [energy, pairing,
      ObservableEnergyDissipation.diagonalEnergy,
      ObservableEnergyDissipation.diagonalPairing,
      ObservableEnergyDissipation.metricShapeWork] using hstep
  unfold decayDissipation
  linarith

/-- A coordinatewise multiplier in the unit interval dissipates energy in a
nonnegative diagonal metric. -/
theorem decay_dissipation_nonnegative
    {Coordinate : Type*} [Fintype Coordinate]
    (metric gate decayMultiplier : Coordinate → ℝ)
    (hmetric : ∀ coordinate, 0 ≤ metric coordinate)
    (hmultiplier : ∀ coordinate,
      -1 ≤ decayMultiplier coordinate ∧ decayMultiplier coordinate ≤ 1) :
    0 ≤ decayDissipation metric gate decayMultiplier := by
  classical
  unfold decayDissipation energy
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_nonneg
  intro coordinate _
  have hsquare : decayMultiplier coordinate ^ 2 ≤ 1 := by
    rcases hmultiplier coordinate with ⟨hlower, hupper⟩
    nlinarith [sq_nonneg (decayMultiplier coordinate - 1),
      sq_nonneg (decayMultiplier coordinate + 1)]
  have hgate : 0 ≤ gate coordinate ^ 2 := sq_nonneg _
  nlinarith [mul_nonneg (hmetric coordinate) hgate]

end ExecutedAdamWEnergy
end PldrLlmCurvatureSandpile

