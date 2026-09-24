/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Linear kernels and nonlinear observation zero fibers
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ObservationZeroFiber

open Filter

/-- A constant nonzero path in a normed group cannot converge to zero. -/
theorem constant_nonzero_not_tendsto_zero
    {E : Type*} [NormedAddCommGroup E]
    (state : E) (hstate : state ≠ 0) :
    ¬ Tendsto (fun _ : ℕ => state) atTop (nhds 0) := by
  intro hzero
  have hstateLimit :
      Tendsto (fun _ : ℕ => state) atTop (nhds state) :=
    tendsto_const_nhds
  have hunique : (0 : E) = state :=
    tendsto_nhds_unique hzero hstateLimit
  exact hstate hunique.symm

/-- A persistent nonzero vector in the common kernel of linear observations
gives identically zero observations without complete-state convergence. -/
theorem linear_common_kernel_obstruction
    {E Y : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    (observe : ℕ → E →L[ℝ] Y)
    (state : E) (hstate : state ≠ 0)
    (hzero : ∀ step, observe step state = 0) :
    Tendsto (fun step => observe step state) atTop (nhds 0)
      ∧ ¬ Tendsto (fun _ : ℕ => state) atTop (nhds 0) := by
  constructor
  · have heq :
        (fun step => observe step state) = (fun _ : ℕ => (0 : Y)) :=
      funext hzero
    rw [heq]
    exact tendsto_const_nhds
  · exact constant_nonzero_not_tendsto_zero state hstate

/-- For nonlinear observations, the corresponding constant-path obstruction
requires membership in the actual common zero fiber. -/
theorem nonlinear_common_zero_fiber_obstruction
    {E Y : Type*}
    [NormedAddCommGroup E] [NormedAddCommGroup Y]
    (observe : ℕ → E → Y)
    (state : E) (hstate : state ≠ 0)
    (hzero : ∀ step, observe step state = 0) :
    Tendsto (fun step => observe step state) atTop (nhds 0)
      ∧ ¬ Tendsto (fun _ : ℕ => state) atTop (nhds 0) := by
  constructor
  · have heq :
        (fun step => observe step state) = (fun _ : ℕ => (0 : Y)) :=
      funext hzero
    rw [heq]
    exact tendsto_const_nhds
  · exact constant_nonzero_not_tendsto_zero state hstate

/-- The scalar square is the explicit example separating a derivative kernel
at zero from the nonlinear zero fiber. -/
def squareObservation (state : ℝ) : ℝ := state ^ 2

theorem square_observation_zero_derivative :
    deriv squareObservation 0 = 0 := by
  change deriv (fun state : ℝ => state ^ 2) 0 = 0
  rw [deriv_pow_field]
  norm_num

theorem square_observation_nonzero_at_one :
    squareObservation 1 ≠ 0 := by
  norm_num [squareObservation]

/-- Zero derivative at the origin does not place the nonzero state one in the
nonlinear zero fiber. -/
theorem derivative_kernel_does_not_imply_zero_fiber :
    deriv squareObservation 0 = 0 ∧ squareObservation 1 ≠ 0 := by
  exact ⟨square_observation_zero_derivative,
    square_observation_nonzero_at_one⟩

end ObservationZeroFiber
end PldrLlmCurvatureSandpile
