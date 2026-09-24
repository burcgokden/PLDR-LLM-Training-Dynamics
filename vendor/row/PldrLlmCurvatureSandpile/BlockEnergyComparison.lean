/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Complete block-energy comparison

Finite all-index reduction, ordered affine paths, and a weighted positive
witness for the layer and optimizer-memory energy vector.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace BlockEnergyComparison

/-- Every source block in a complete nonnegative comparison contributes to
the weighted affine target bound. -/
theorem complete_block_energy_step {n : ℕ}
    (P : Fin n → Fin n → ℝ) (e v d : Fin n → ℝ)
    {B κ D : ℝ}
    (hB : 0 ≤ B)
    (hP : ∀ i j, 0 ≤ P i j)
    (he : ∀ i, e i ≤ B * v i)
    (hPv : ∀ i, ∑ j, P i j * v j ≤ κ * v i)
    (hd : ∀ i, d i ≤ D * v i) :
    ∀ i, (∑ j, P i j * e j) + d i ≤ (κ * B + D) * v i := by
  intro i
  have hlinear : ∑ j, P i j * e j ≤ (κ * B) * v i := by
    calc
      ∑ j, P i j * e j ≤ ∑ j, P i j * (B * v j) := by
        exact Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (he j) (hP i j)
      _ = B * ∑ j, P i j * v j := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ ≤ B * (κ * v i) := mul_le_mul_of_nonneg_left (hPv i) hB
      _ = (κ * B) * v i := by ring
  calc
    (∑ j, P i j * e j) + d i
        ≤ (κ * B) * v i + D * v i := add_le_add hlinear (hd i)
    _ = (κ * B + D) * v i := by ring

/-- Two nonnegative affine energy steps compose in temporal order. -/
theorem ordered_energy_path
    {e0 e1 e2 p0 p1 d0 d1 : ℝ}
    (hp1 : 0 ≤ p1)
    (h01 : e1 ≤ p0 * e0 + d0)
    (h12 : e2 ≤ p1 * e1 + d1) :
    e2 ≤ (p1 * p0) * e0 + p1 * d0 + d1 := by
  calc
    e2 ≤ p1 * e1 + d1 := h12
    _ ≤ p1 * (p0 * e0 + d0) + d1 :=
      add_le_add (mul_le_mul_of_nonneg_left h01 hp1) le_rfl
    _ = (p1 * p0) * e0 + p1 * d0 + d1 := by ring

/-- One scalar step of the persistent-floor comparison. -/
theorem weighted_energy_tail
    {current next κ persistent transient : ℝ}
    (hstep : next ≤ κ * current + persistent + transient) :
    next - persistent ≤ κ * current + transient := by
  linarith

end BlockEnergyComparison
end PldrLlmCurvatureSandpile
