/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Stacked row-Jacobian norm conversion

Finite-dimensional check of the operator-output to Frobenius-output factor
used when a d-column row Jacobian is vectorized.
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace StackedJacobian

/-- If each of the d column energies is at most L², their vectorized
Frobenius energy is at most d L². -/
theorem column_energy_sum_le
    {d m : ℕ} (M : Matrix (Fin m) (Fin d) ℝ) (L : ℝ)
    (hcol : ∀ j, ∑ i, (M i j) ^ 2 ≤ L ^ 2) :
    ∑ j, ∑ i, (M i j) ^ 2 ≤ (d : ℝ) * L ^ 2 := by
  calc
    ∑ j, ∑ i, (M i j) ^ 2 ≤ ∑ _j : Fin d, L ^ 2 :=
      Finset.sum_le_sum fun j _ => hcol j
    _ = (d : ℝ) * L ^ 2 := by simp

/-- The scalar multiple of the identity attains the d factor exactly. -/
theorem scalar_identity_energy (d : ℕ) (L : ℝ) :
    ∑ j : Fin d, ∑ i : Fin d,
      ((L • (1 : Matrix (Fin d) (Fin d) ℝ)) i j) ^ 2
      = (d : ℝ) * L ^ 2 := by
  simp [Matrix.one_apply, apply_ite]

end StackedJacobian
end PldrLlmCurvatureSandpile
