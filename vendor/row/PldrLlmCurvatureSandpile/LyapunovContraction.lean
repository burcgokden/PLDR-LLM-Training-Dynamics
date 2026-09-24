/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Quadratic Lyapunov contraction

This module checks the passage from a quadratic-energy inequality to the
associated energy-norm contraction. The interval or rational certificate
that establishes the inequality is explicit input data.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace LyapunovContraction

/-- Quadratic energy of a two-coordinate lifted scalar mode. -/
def quadraticEnergy (h11 h12 h22 z m : ℝ) : ℝ :=
  h11 * z ^ 2 + 2 * h12 * z * m + h22 * m ^ 2

/-- A positive definite two-by-two symmetric certificate has positive
quadratic energy away from the origin. -/
theorem quadraticEnergy_pos
    {h11 h12 h22 z m : ℝ} (hh11 : 0 < h11)
    (hdet : h12 ^ 2 < h11 * h22) (hne : z ≠ 0 ∨ m ≠ 0) :
    0 < quadraticEnergy h11 h12 h22 z m := by
  have hh22 : 0 < h22 := by
    have hs : 0 ≤ h12 ^ 2 := sq_nonneg h12
    have hp : 0 < h11 * h22 := lt_of_le_of_lt hs hdet
    exact pos_of_mul_pos_right hp hh11.le
  by_cases hz : z = 0
  · subst z
    simp [quadraticEnergy]
    exact mul_pos hh22 (sq_pos_of_ne_zero (hne.resolve_left (by simp)))
  · have hid : quadraticEnergy h11 h12 h22 z m =
        h11 * (z + h12 / h11 * m) ^ 2
          + (h22 - h12 ^ 2 / h11) * m ^ 2 := by
        rw [quadraticEnergy]
        field_simp [ne_of_gt hh11]
        ring
    have hschur : 0 < h22 - h12 ^ 2 / h11 := by
      rw [sub_pos, div_lt_iff₀ hh11]
      simpa [mul_comm] using hdet
    rw [hid]
    by_cases hm : m = 0
    · subst m
      simp
      exact mul_pos hh11 (sq_pos_of_ne_zero hz)
    · exact add_pos_of_nonneg_of_pos
        (mul_nonneg hh11.le (sq_nonneg _))
        (mul_pos hschur (sq_pos_of_ne_zero hm))

/-- The scalar core of `A^* H A <= q^2 H`: taking square roots yields
contraction in the energy norm. -/
theorem lyapunov_energy_step
    {energyBefore energyAfter q : ℝ}
    (hq : 0 ≤ q) (hbefore : 0 ≤ energyBefore)
    (hcontract : energyAfter ≤ q ^ 2 * energyBefore) :
    Real.sqrt energyAfter ≤ q * Real.sqrt energyBefore := by
  rw [Real.sqrt_le_iff]
  constructor
  · positivity
  · calc
      energyAfter ≤ q ^ 2 * energyBefore := hcontract
      _ = (q * Real.sqrt energyBefore) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt hbefore]

/-- Adding a residual after the homogeneous energy contraction gives the
forced norm recurrence consumed by the convolution theorem. -/
theorem lyapunov_forced_step
    {energyBefore energyLinear q residual nextNorm : ℝ}
    (hq : 0 ≤ q) (hbefore : 0 ≤ energyBefore)
    (hcontract : energyLinear ≤ q ^ 2 * energyBefore)
    (htriangle : nextNorm ≤ Real.sqrt energyLinear + residual) :
    nextNorm ≤ q * Real.sqrt energyBefore + residual := by
  have hstep := lyapunov_energy_step hq hbefore hcontract
  linarith

end LyapunovContraction
end PldrLlmCurvatureSandpile
