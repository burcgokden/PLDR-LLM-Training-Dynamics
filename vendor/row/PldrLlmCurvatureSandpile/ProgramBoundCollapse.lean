/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Program-bound collapse assembly

Finite, frozen, and controlled-tail conclusions assembled from an owned
energy successor recurrence.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ProgramBoundCollapse

/-- A finite sequence of owned affine energy edges propagates only through
its registered terminal index. -/
theorem finite_program_energy_collapse
    (energy envelope q source : ℕ → ℝ) (N : ℕ)
    (hentry : energy 0 ≤ envelope 0)
    (henvelope : ∀ t, t < N →
      envelope (t + 1) = q t * envelope t + source t)
    (hq : ∀ t, 0 ≤ q t)
    (hstep : ∀ t, t < N →
      energy (t + 1) ≤ q t * energy t + source t) :
    ∀ t, t ≤ N → energy t ≤ envelope t := by
  intro t ht
  induction t with
  | zero => exact hentry
  | succ t ih =>
      calc
        energy (t + 1) ≤ q t * energy t + source t :=
          hstep t (Nat.lt_of_succ_le ht)
        _ ≤ q t * envelope t + source t :=
          add_le_add
            (mul_le_mul_of_nonneg_left (ih (Nat.le_of_succ_le ht)) (hq t))
            le_rfl
        _ = envelope (t + 1) := (henvelope t (Nat.lt_of_succ_le ht)).symm

/-- Repeated evaluation of a bitwise-frozen checkpoint preserves its energy
bound by identity. -/
theorem frozen_energy_persistence
    {State : Type*} (energy : State → ℝ) (checkpoint : State)
    {bound : ℝ} (hbound : energy checkpoint ≤ bound) :
    ∀ _evaluation : ℕ, energy checkpoint ≤ bound := by
  intro _
  exact hbound

/-- A separately quantified all-future owned successor recurrence propagates
the controlled tail. -/
theorem controlled_tail_energy_collapse
    (energy envelope q source : ℕ → ℝ)
    (hentry : energy 0 ≤ envelope 0)
    (henvelope : ∀ t, envelope (t + 1) = q t * envelope t + source t)
    (hq : ∀ t, 0 ≤ q t)
    (hstep : ∀ t, energy (t + 1) ≤ q t * energy t + source t) :
    ∀ t, energy t ≤ envelope t := by
  intro t
  induction t with
  | zero => exact hentry
  | succ t ih =>
      calc
        energy (t + 1) ≤ q t * energy t + source t := hstep t
        _ ≤ q t * envelope t + source t :=
          add_le_add (mul_le_mul_of_nonneg_left ih (hq t)) le_rfl
        _ = envelope (t + 1) := (henvelope t).symm

end ProgramBoundCollapse
end PldrLlmCurvatureSandpile
