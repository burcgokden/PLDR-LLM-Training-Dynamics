/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Nonlinear attracting tube

Scalar kernels for the invariant basin, forced fixed point, and the zero-force
comparison used by the comprehensive theorem.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace NonlinearAttraction

def step (κ C F u : ℝ) : ℝ := κ * u + C * u ^ 2 + F

/-- The fixed-point polynomial factors through its two roots. -/
theorem root_factorization
    {κ C F rMinus rPlus u : ℝ}
    (hsum : C * (rMinus + rPlus) = 1 - κ)
    (hprod : C * rMinus * rPlus = F) :
    step κ C F u - u = C * (u - rMinus) * (u - rPlus) := by
  have hκeq : κ = 1 - C * (rMinus + rPlus) := by
    linarith
  rw [hκeq, ← hprod]
  dsimp [step]
  ring

/-- An increasing nonlinear step with a nonnegative fixed upper root leaves
the interval below that root invariant. -/
theorem invariant_upper_root
    {κ C F r u : ℝ}
    (hκ : 0 ≤ κ) (hC : 0 ≤ C) (hF : 0 ≤ F)
    (hr : 0 ≤ r) (hu0 : 0 ≤ u) (hur : u ≤ r)
    (hroot : step κ C F r = r) :
    0 ≤ step κ C F u ∧ step κ C F u ≤ r := by
  constructor
  · dsimp [step]
    positivity
  · have hu_sq : u ^ 2 ≤ r ^ 2 := (sq_le_sq₀ hu0 hr).2 hur
    calc
      step κ C F u ≤ step κ C F r := by
        dsimp [step]
        gcongr
      _ = r := hroot

/-- Inside a strict zero-force basin, the nonlinear recurrence is dominated
by one geometric sequence. -/
theorem zero_force_geometric
    (u : ℕ → ℝ) {κ C b : ℝ}
    (hκ : 0 ≤ κ) (hC : 0 ≤ C) (hb : 0 ≤ b)
    (hubound : ∀ n, 0 ≤ u n ∧ u n ≤ b)
    (hstep : ∀ n, u (n + 1) ≤ κ * u n + C * (u n) ^ 2) :
    ∀ n, u n ≤ (κ + C * b) ^ n * u 0 := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
      have hquad : (u n) ^ 2 ≤ b * u n := by
        nlinarith [(hubound n).1, (hubound n).2]
      calc
        u (n + 1) ≤ κ * u n + C * (u n) ^ 2 := hstep n
        _ ≤ κ * u n + C * (b * u n) := by
          gcongr
        _ = (κ + C * b) * u n := by ring
        _ ≤ (κ + C * b) * ((κ + C * b) ^ n * u 0) := by
          apply mul_le_mul_of_nonneg_left ih
          positivity
        _ = (κ + C * b) ^ (n + 1) * u 0 := by ring

end NonlinearAttraction
end PldrLlmCurvatureSandpile
