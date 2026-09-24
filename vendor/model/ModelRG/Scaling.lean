import Mathlib

/-! The finite recursive error bound used before taking a joint size-time limit.
The probabilistic coupling and asymptotic hypotheses are stated in the paper. -/
noncomputable section
namespace ModelRG

theorem accumulated_error_bound
    (e : ℕ → ℝ) (rho delta : ℝ)
    (hrho : 0 ≤ rho) (hrho1 : rho < 1)
    (hstep : ∀ n, e (n + 1) ≤ rho * e n + delta) (n : ℕ) :
    e n ≤ rho ^ n * e 0 + delta * (1 - rho ^ n) / (1 - rho) := by
  have hd : 0 < 1 - rho := by linarith
  have hne : 1 - rho ≠ 0 := ne_of_gt hd
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      e (n + 1) ≤ rho * e n + delta := hstep n
      _ ≤ rho * (rho ^ n * e 0 + delta * (1 - rho ^ n) / (1 - rho)) + delta :=
        by linarith [mul_le_mul_of_nonneg_left ih hrho]
      _ = rho ^ (n + 1) * e 0 + delta * (1 - rho ^ (n + 1)) / (1 - rho) := by
        rw [pow_succ]
        field_simp [hne]
        ring

/-- An unforced parameter coordinate has exactly the prescribed decay clock. -/
theorem autonomous_decay
    (w : ℕ → ℝ) (q : ℝ)
    (hstep : ∀ n, w (n + 1) = q * w n) (n : ℕ) :
    w n = q ^ n * w 0 := by
  induction n with
  | zero => simp
  | succ n ih => rw [hstep, ih, pow_succ]; ring

/-- Chronological error propagation under an inhomogeneous, possibly expanding drive.
This is the finite recurrence in the law-averaged closure proposition. -/
theorem nonstationary_error_bound
    (e d L : ℕ → ℝ) (hL : ∀ n, 0 ≤ L n)
    (hstep : ∀ n, e (n + 1) ≤ L n * e n + d n) (m : ℕ) :
    e m ≤ e 0 * (∏ k ∈ Finset.Ico 0 m, L k) +
      ∑ j ∈ Finset.Ico 0 m, d j * ∏ k ∈ Finset.Ico (j + 1) m, L k := by
  exact discrete_gronwall_prod_general (fun n _ => hstep n) (fun n _ => hL n)
    (Nat.zero_le m)

end ModelRG
