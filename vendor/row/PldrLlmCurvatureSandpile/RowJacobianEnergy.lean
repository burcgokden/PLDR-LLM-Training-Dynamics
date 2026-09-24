/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Complete row-Jacobian energy

Exact finite-dimensional quadratic identities for the program-bound energy
formulation of row-map collapse.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace RowJacobianEnergy

/-- Diagonal quadratic energy on a finite complete coordinate registry. -/
def energy {n : ℕ} (weight z : Fin n → ℝ) : ℝ :=
  ∑ i, weight i * z i ^ 2

/-- Polarization gives the exact energy increment, coordinate by coordinate. -/
theorem energy_increment_identity {n : ℕ}
    (weight z dz : Fin n → ℝ) :
    energy weight (z + dz) - energy weight z =
      2 * ∑ i, weight i * z i * dz i + energy weight dz := by
  simp only [energy, Pi.add_apply]
  rw [← Finset.sum_sub_distrib]
  calc
    ∑ i, (weight i * (z i + dz i) ^ 2 - weight i * z i ^ 2) =
        ∑ i, (2 * (weight i * z i * dz i) + weight i * dz i ^ 2) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = 2 * ∑ i, weight i * z i * dz i + ∑ i, weight i * dz i ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.mul_sum]

/-- A source-complete negative-work inequality implies the affine energy
step without an invertibility or rank hypothesis. -/
theorem negative_work_contracts {n : ℕ}
    (weight z dz : Fin n → ℝ) {q source : ℝ}
    (hwork :
      2 * ∑ i, weight i * z i * dz i + energy weight dz ≤
        -(1 - q ^ 2) * energy weight z + source) :
    energy weight (z + dz) ≤ q ^ 2 * energy weight z + source := by
  have hid := energy_increment_identity weight z dz
  calc
    energy weight (z + dz) =
        energy weight z +
          (2 * ∑ i, weight i * z i * dz i + energy weight dz) := by
      linarith
    _ ≤ energy weight z +
          (-(1 - q ^ 2) * energy weight z + source) :=
      add_le_add le_rfl hwork
    _ = q ^ 2 * energy weight z + source := by ring

/-- When the target metric differs from the source metric, its exact
conversion charge is a separate visible summand. -/
theorem changing_metric_charge {n : ℕ}
    (sourceWeight targetWeight z dz : Fin n → ℝ) :
    energy targetWeight (z + dz) - energy sourceWeight z =
      (energy sourceWeight (z + dz) - energy sourceWeight z) +
      ∑ i, (targetWeight i - sourceWeight i) * (z i + dz i) ^ 2 := by
  unfold energy
  rw [← Finset.sum_sub_distrib]
  rw [← Finset.sum_sub_distrib]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [Pi.add_apply]
  ring

end RowJacobianEnergy
end PldrLlmCurvatureSandpile
