/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact finite-increment observable balance
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace FiniteIncrementObservableBalance

variable {E Y : Type*}

/-- Base-route chronological transport of the initial signed displacement. -/
def homogeneous [AddCommGroup E]
    (operator : ℕ → E →+ E) (initial : E) : ℕ → E
  | 0 => initial
  | step + 1 => operator step (homogeneous operator initial step)

/-- Chronological accumulation of the exact finite-state correction. -/
def accumulatedCorrection [AddCommGroup E]
    (operator : ℕ → E →+ E) (correction : ℕ → E) : ℕ → E
  | 0 => 0
  | step + 1 =>
      operator step (accumulatedCorrection operator correction step)
        + correction step

/-- The finite state difference equals base-route homogeneous transport plus
the chronologically accumulated one-step correction. -/
theorem finite_increment_state_unroll
    [AddCommGroup E]
    (operator : ℕ → E →+ E)
    (stateDifference correction : ℕ → E)
    (initial : E)
    (hinitial : stateDifference 0 = initial)
    (hstep : ∀ step,
      stateDifference (step + 1) =
        operator step (stateDifference step) + correction step) :
    ∀ step,
      stateDifference step =
        homogeneous operator initial step
          + accumulatedCorrection operator correction step := by
  intro step
  induction step with
  | zero =>
      simp [homogeneous, accumulatedCorrection, hinitial]
  | succ step ih =>
      rw [hstep step, ih]
      simp only [homogeneous, accumulatedCorrection, map_add]
      abel

/-- At a fixed horizon, the signed observable response is exactly the sum of
the observed homogeneous state, observed state correction, and the residual
observation correction. -/
theorem exact_finite_increment_observable_balance
    [AddCommGroup E] [AddCommGroup Y]
    (observe : E →+ Y)
    (stateDifference homogeneousState stateCorrection : E)
    (observedResponse : Y)
    (hstate :
      stateDifference = homogeneousState + stateCorrection) :
    observedResponse =
      observe homogeneousState
        + observe stateCorrection
        + (observedResponse - observe stateDifference) := by
  rw [hstate, map_add]
  abel


end FiniteIncrementObservableBalance
end PldrLlmCurvatureSandpile
