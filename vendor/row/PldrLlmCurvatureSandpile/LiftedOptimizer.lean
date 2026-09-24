/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Operation-ordered lifted optimizer recurrence

This module checks the scalar eigenmode of the row-Jacobian and first-moment
recurrence. It keeps the implemented order: the moment is updated first,
then the bias-corrected parameter displacement transports the row Jacobian.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace LiftedOptimizer

/-- Homogeneous lifted step for one normal mode. -/
def liftedHomogeneous (alpha beta lam : ℝ) (state : ℝ × ℝ) : ℝ × ℝ :=
  let z := state.1
  let m := state.2
  ((1 - alpha * (1 - beta) * lam) * z - alpha * beta * m,
   (1 - beta) * lam * z + beta * m)

/-- Exact algebraic recurrence with activity forcing and named moment and
transport defects. -/
theorem lifted_optimizer_recurrence
    (alpha beta lam z m forcing momentDefect transportDefect : ℝ) :
    let mNext := beta * m + (1 - beta) * (lam * z + forcing)
      + momentDefect
    let zNext := z - alpha * mNext + transportDefect
    (zNext, mNext) =
      let homogeneous := liftedHomogeneous alpha beta lam (z, m)
      (homogeneous.1 - alpha * (1 - beta) * forcing
          - alpha * momentDefect + transportDefect,
       homogeneous.2 + (1 - beta) * forcing + momentDefect) := by
  apply Prod.ext <;> simp [liftedHomogeneous] <;> ring

/-- The exact bias-corrected coefficient used by the transported step. -/
noncomputable def biasCorrectedCoefficient (eta beta : ℝ) (step : ℕ) : ℝ :=
  eta / (1 - beta ^ step)

/-- Substituting the bias-corrected coefficient preserves the recurrence. -/
theorem lifted_optimizer_bias_corrected
    (eta beta lam z m forcing momentDefect transportDefect : ℝ)
    (step : ℕ) :
    let alpha := biasCorrectedCoefficient eta beta step
    let mNext := beta * m + (1 - beta) * (lam * z + forcing)
      + momentDefect
    let zNext := z - alpha * mNext + transportDefect
    (zNext, mNext) =
      let homogeneous := liftedHomogeneous alpha beta lam (z, m)
      (homogeneous.1 - alpha * (1 - beta) * forcing
          - alpha * momentDefect + transportDefect,
       homogeneous.2 + (1 - beta) * forcing + momentDefect) := by
  exact lifted_optimizer_recurrence
    (biasCorrectedCoefficient eta beta step) beta lam z m forcing
    momentDefect transportDefect

/-- Trace of the scalar-mode homogeneous matrix. -/
def liftedTrace (alpha beta lam : ℝ) : ℝ :=
  1 + beta - alpha * (1 - beta) * lam

/-- Determinant of the scalar-mode homogeneous matrix. -/
theorem lifted_determinant (alpha beta lam : ℝ) :
    (1 - alpha * (1 - beta) * lam) * beta
      - (-alpha * beta) * ((1 - beta) * lam) = beta := by
  ring

/-- The three strict scalar Jury margins. -/
theorem lifted_jury_margins
    {alpha beta lam : ℝ} (halpha : 0 < alpha)
    (hbeta1 : beta < 1) (hlam : 0 < lam)
    (hupper : alpha * (1 - beta) * lam < 2 * (1 + beta)) :
    0 < 1 - beta ∧
    0 < 1 - liftedTrace alpha beta lam + beta ∧
    0 < 1 + liftedTrace alpha beta lam + beta := by
  constructor
  · linarith
  constructor
  · dsimp [liftedTrace]
    have hb : 0 < 1 - beta := by linarith
    have hproduct : 0 < alpha * (1 - beta) * lam :=
      mul_pos (mul_pos halpha hb) hlam
    linarith
  · dsimp [liftedTrace]
    linarith

/-- Trace of the scalar lifted matrix including a scalar decoupled-decay
operator. -/
def liftedTraceWithDecay (alpha beta lam decay : ℝ) : ℝ :=
  1 - decay - alpha * (1 - beta) * lam + beta

/-- The determinant remains independent of the normal eigenvalue when the
scalar decoupled-decay term is retained. -/
theorem lifted_determinant_with_decay
    (alpha beta lam decay : ℝ) :
    (1 - decay - alpha * (1 - beta) * lam) * beta
      - (-alpha * beta) * ((1 - beta) * lam)
      = beta * (1 - decay) := by
  ring

/-- The three strict Jury margins for the complete scalar mode, including
decoupled decay, match the written two-sided spectral condition. -/
theorem lifted_jury_margins_with_decay
    {alpha beta lam decay : ℝ}
    (halpha : 0 < alpha)
    (hbeta0 : 0 ≤ beta) (hbeta1 : beta < 1)
    (hlam : 0 < lam)
    (hdecay0 : 0 ≤ decay) (hdecay1 : decay < 1)
    (hupper :
      alpha * (1 - beta) * lam + decay * (1 + beta)
        < 2 * (1 + beta)) :
    0 < 1 - beta * (1 - decay) ∧
    0 < 1 - liftedTraceWithDecay alpha beta lam decay
      + beta * (1 - decay) ∧
    0 < 1 + liftedTraceWithDecay alpha beta lam decay
      + beta * (1 - decay) := by
  have hbetaGap : 0 < 1 - beta := by linarith
  have hdecayGap : 0 < 1 - decay := by linarith
  have hdetNonneg : 0 ≤ beta * (1 - decay) :=
    mul_nonneg hbeta0 (le_of_lt hdecayGap)
  have hdetLt : beta * (1 - decay) < 1 := by
    calc
      beta * (1 - decay) ≤ beta * 1 := by
        exact mul_le_mul_of_nonneg_left (by linarith) hbeta0
      _ = beta := by ring
      _ < 1 := hbeta1
  have hnormal : 0 < alpha * (1 - beta) * lam :=
    mul_pos (mul_pos halpha hbetaGap) hlam
  have hdecayTerm : 0 ≤ decay * (1 - beta) :=
    mul_nonneg hdecay0 (le_of_lt hbetaGap)
  constructor
  · nlinarith [hdetNonneg, hdetLt]
  constructor
  · dsimp [liftedTraceWithDecay]
    nlinarith [hnormal, hdecayTerm]
  · dsimp [liftedTraceWithDecay]
    nlinarith

/-- A homogeneous contraction and a triangle bound give the one-step
forced recurrence used by the residual convolution. -/
theorem lifted_forced_norm_step
    {linear forcing defect q current next : ℝ}
    (hlinear : linear ≤ q * current)
    (hnext : next ≤ linear + forcing + defect) :
    next ≤ q * current + (forcing + defect) := by
  linarith

end LiftedOptimizer
end PldrLlmCurvatureSandpile
