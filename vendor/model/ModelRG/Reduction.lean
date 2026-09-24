import Mathlib

noncomputable section
namespace ModelRG

/-- Ordered finite-kernel defect propagated through a temporal block. -/
def closureDefect {n m : Type*} [Fintype n] [Fintype m]
    [DecidableEq n] [DecidableEq m]
    (K : Matrix n n ℝ) (P : Matrix n m ℝ) (Q : Matrix m m ℝ) :
    ℕ → Matrix n m ℝ
  | 0 => 0
  | k+1 => K * closureDefect K P Q k + (K * P - P * Q) * Q ^ k

/-- The exact finite-matrix error underlying the analytic closure estimate.
No Markov positivity, total variation, or Wasserstein bound is asserted here. -/
theorem closureDefect_eq {n m : Type*} [Fintype n] [Fintype m]
    [DecidableEq n] [DecidableEq m]
    (K : Matrix n n ℝ) (P : Matrix n m ℝ) (Q : Matrix m m ℝ) (k : ℕ) :
    closureDefect K P Q k = K ^ k * P - P * Q ^ k := by
  induction k with
  | zero => simp [closureDefect]
  | succ k ih =>
    simp only [closureDefect, ih, pow_succ', Matrix.mul_sub, Matrix.sub_mul,
      Matrix.mul_assoc]
    abel

/-- Rectangular observation maps commute with a weighted block sum. -/
theorem linear_block_compose {E F G : Type*} [AddCommGroup E] [Module ℝ E]
    [AddCommGroup F] [Module ℝ F] [AddCommGroup G] [Module ℝ G]
    {ι : Type*} [Fintype ι] (C : E →ₗ[ℝ] F) (D : F →ₗ[ℝ] G)
    (a b : ℝ) (x : ι → E) :
    a • D (b • C (∑ i, x i)) = (a*b) • (D.comp C) (∑ i, x i) := by
  simp [smul_smul]

/-- Any zero-sum vector contrast cancels a common field. -/
theorem contrast_shared_cancel {ι E : Type*} [Fintype ι]
    [AddCommGroup E] [Module ℝ E] (d : ι → ℝ) (x : ι → E) (u : E)
    (hd : ∑ i, d i = 0) :
    (∑ i, d i • (u + x i)) = ∑ i, d i • x i := by
  simp_rw [smul_add]
  rw [Finset.sum_add_distrib, ← Finset.sum_smul, hd, zero_smul, zero_add]

/-- The categorical cubic includes the signed logit-acceleration cross term. -/
theorem cubic_response_cross {ι : Type*} [Fintype ι]
    (p v w : ι → ℝ) (mv mw : ℝ) :
    (∑ i, p i * ((v i-mv)^3 + 3*(v i-mv)*(w i-mw))) =
      (∑ i, p i * (v i-mv)^3) + 3 * ∑ i, p i*(v i-mv)*(w i-mw) := by
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- A finite telescoping error bound, with nonnegative stability. -/
theorem constant_stability_bound (e : ℕ → ℝ) (L d : ℝ)
    (hL : 0 ≤ L) (hzero : e 0 = 0)
    (hstep : ∀ n, e (n+1) ≤ L * e n + d) (n : ℕ) :
    e n ≤ d * ∑ j ∈ Finset.range n, L^j := by
  induction n with
  | zero => simp [hzero]
  | succ n ih =>
    calc
      e (n+1) ≤ L * e n + d := hstep n
      _ ≤ L * (d * ∑ j ∈ Finset.range n, L^j) + d :=
        by linarith [mul_le_mul_of_nonneg_left ih hL]
      _ = d * ∑ j ∈ Finset.range (n+1), L^j := by
        rw [Finset.sum_range_succ']
        simp only [pow_zero, pow_succ', ← Finset.mul_sum]
        ring

end ModelRG
