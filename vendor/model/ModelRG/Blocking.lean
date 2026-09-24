import Mathlib

/-! Algebraic core of the combined dataset and observable renormalization.
The probability pushforward/CLT arguments are separate prose proofs. -/

noncomputable section

namespace ModelRG

/-- Log generating-function blocking with an arbitrary multiplicative scale. -/
def cgfRG (b a : ℝ) (W : ℝ → ℝ) (j : ℝ) : ℝ := b * W (j / a)

theorem cgfRG_compose (b c a d : ℝ) (W : ℝ → ℝ) (j : ℝ) :
    cgfRG c d (cgfRG b a W) j = cgfRG (b*c) (a*d) W j := by
  simp only [cgfRG, div_div]
  rw [mul_comm d a]
  ring

theorem cgfRG_identity (W : ℝ → ℝ) (j : ℝ) : cgfRG 1 1 W j = W j := by
  simp [cgfRG]

/-- A cumulant of order n has multiplier b/a^n. -/
def cumulantRG (b a : ℝ) (n : ℕ) (k : ℝ) : ℝ := b / a^n * k

theorem cumulantRG_compose (b c a d : ℝ) (n : ℕ) (k : ℝ) :
    cumulantRG c d n (cumulantRG b a n k) =
      cumulantRG (b*c) (a*d) n k := by
  simp only [cumulantRG, mul_pow, div_eq_mul_inv, mul_inv_rev]
  ring

theorem gaussian_covariance_fixed (b a k : ℝ) (ha : a ≠ 0) (hb : b = a^2) :
    cumulantRG b a 2 k = k := by
  rw [cumulantRG, hb, div_self (pow_ne_zero 2 ha), one_mul]

theorem independent_variance_block (n variance : ℝ) (hn : n ≠ 0) :
    n * (variance / n^2) = variance / n := by
  field_simp

/-- A regular independent-source susceptibility is invariant under blocking. -/
theorem susceptibility_independent (n variance : ℝ) (hn : n ≠ 0) :
    n * (variance / n) = variance := by
  field_simp

end ModelRG
