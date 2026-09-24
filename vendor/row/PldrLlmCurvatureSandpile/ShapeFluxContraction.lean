/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Normalized-shape flux contraction
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace ShapeFluxContraction

def normSq {n : ℕ} (value : Fin n → ℝ) : ℝ :=
  ∑ i, (value i) ^ 2

def dot {n : ℕ} (left right : Fin n → ℝ) : ℝ :=
  ∑ i, left i * right i

/-- The exact squared-norm increment for a first-order shape flux plus its
finite remainder. -/
theorem shape_norm_increment {n : ℕ}
    (shape flux remainder : Fin n → ℝ) :
    normSq (fun i => shape i + flux i + remainder i) - normSq shape
      = 2 * dot shape flux + normSq flux
        + 2 * dot (fun i => shape i + flux i) remainder
        + normSq remainder := by
  classical
  rw [normSq, normSq, dot, normSq, dot, normSq,
    ← Finset.sum_sub_distrib]
  calc
    ∑ i, ((shape i + flux i + remainder i) ^ 2 - shape i ^ 2)
        = ∑ i, (2 * (shape i * flux i) + flux i ^ 2
          + 2 * ((shape i + flux i) * remainder i)
          + remainder i ^ 2) := by
            apply Finset.sum_congr rfl
            intro i _
            ring
    _ = 2 * ∑ i, shape i * flux i + ∑ i, flux i ^ 2
        + 2 * ∑ i, (shape i + flux i) * remainder i
        + ∑ i, remainder i ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
        Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]

/-- A negative flux-and-remainder budget gives one-step shape contraction.
-/
theorem shape_flux_contracts {n : ℕ}
    (shape flux remainder : Fin n → ℝ) {kappa : ℝ}
    (hcriterion :
      2 * dot shape flux + normSq flux
        + 2 * dot (fun i => shape i + flux i) remainder
        + normSq remainder ≤ -kappa * normSq shape) :
    normSq (fun i => shape i + flux i + remainder i)
      ≤ (1 - kappa) * normSq shape := by
  have hincrement := shape_norm_increment shape flux remainder
  nlinarith

end ShapeFluxContraction
end PldrLlmCurvatureSandpile
