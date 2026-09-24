/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Full three-block AdamW successor
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace LiftedAdamWMatrix

/-- A scalar coordinate of the lifted position, first moment, and second
moment state. -/
@[ext] structure LiftedState where
  position : ℝ
  firstMoment : ℝ
  secondMoment : ℝ

/-- The homogeneous scalar block of the full AdamW derivative. -/
def fullLinearStep
    (eta decay beta₁ beta₂ dm dv mPosition vPosition : ℝ)
    (state : LiftedState) : LiftedState :=
  let mNext := mPosition * state.position + beta₁ * state.firstMoment
  let vNext := vPosition * state.position + beta₂ * state.secondMoment
  {
    position :=
      (1 - decay) * state.position - eta * (dm * mNext + dv * vNext)
    firstMoment := mNext
    secondMoment := vNext
  }

/-- Expanding the moment-first update gives all three block columns of the
full successor matrix. -/
theorem full_adamw_block_expansion
    (eta decay beta₁ beta₂ dm dv mPosition vPosition n m v : ℝ) :
    fullLinearStep eta decay beta₁ beta₂ dm dv mPosition vPosition
      ⟨n, m, v⟩ =
    {
      position :=
        (1 - decay - eta * (dm * mPosition + dv * vPosition)) * n
          - eta * beta₁ * dm * m
          - eta * beta₂ * dv * v
      firstMoment := mPosition * n + beta₁ * m
      secondMoment := vPosition * n + beta₂ * v
    } := by
  apply LiftedState.ext
  all_goals simp [fullLinearStep]
  all_goals ring

/-- A frozen second moment is a special case obtained only by setting both
its position response and incoming second-moment column to zero. -/
theorem frozen_second_moment_specialization
    (eta decay beta₁ dm mPosition n m : ℝ) :
    (fullLinearStep eta decay beta₁ 0 dm 0 mPosition 0
      ⟨n, m, 0⟩).secondMoment = 0 := by
  simp [fullLinearStep]

/-- The same additive forcing cancels in the difference of two affine
responses; this is a ring identity. -/
theorem state_independent_intervention
    (linear forcing state perturbation : ℝ) :
    (linear * (state + perturbation) + forcing) -
      (linear * state + forcing) = linear * perturbation := by
  ring

end LiftedAdamWMatrix
end PldrLlmCurvatureSandpile
