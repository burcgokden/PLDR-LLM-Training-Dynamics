/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Constant-map normal geometry

Algebraic kernel for the equivalence between complete row-Jacobian energy and
squared realizable normal distance.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace NormalGeometry

/-- Two-sided normal-chart norm bounds imply the squared energy bounds. -/
theorem squared_normal_energy_equivalence
    {normalNorm stackNorm lower upper : ℝ}
    (hnormal : 0 ≤ normalNorm) (hstack : 0 ≤ stackNorm)
    (hlower : 0 ≤ lower) (hupper : 0 ≤ upper)
    (hlo : lower * normalNorm ≤ stackNorm)
    (hhi : stackNorm ≤ upper * normalNorm) :
    lower ^ 2 * normalNorm ^ 2 ≤ stackNorm ^ 2 ∧
      stackNorm ^ 2 ≤ upper ^ 2 * normalNorm ^ 2 := by
  constructor
  · have hsq :
        (lower * normalNorm) ^ 2 ≤ stackNorm ^ 2 :=
      (sq_le_sq₀ (mul_nonneg hlower hnormal) hstack).2 hlo
    simpa [mul_pow] using hsq
  · have hsq :
        stackNorm ^ 2 ≤ (upper * normalNorm) ^ 2 :=
      (sq_le_sq₀ hstack (mul_nonneg hupper hnormal)).2 hhi
    simpa [mul_pow] using hsq

/-- A positive lower chart edge makes zero stack energy equivalent to zero
normal distance. -/
theorem zero_energy_forces_zero_normal
    {normalNorm stackNorm lower : ℝ}
    (hnormal : 0 ≤ normalNorm)
    (hlower : 0 < lower)
    (hlo : lower * normalNorm ≤ stackNorm)
    (hzero : stackNorm ^ 2 = 0) :
    normalNorm = 0 := by
  have hs : stackNorm = 0 := sq_eq_zero_iff.mp hzero
  rw [hs] at hlo
  nlinarith

/-- The finite-radius lower edge obtained from derivative variation in a
normal chart. -/
theorem chart_lower_edge
    {sigma variation radius : ℝ}
    (hsmall : variation * radius / 2 < sigma) :
    0 < sigma - variation * radius / 2 := by
  linarith

end NormalGeometry
end PldrLlmCurvatureSandpile
