import Mathlib

noncomputable section

namespace ModelRG

/-- Finite transfer kernels, including source-weighted kernels, regroup exactly. -/
theorem temporal_kernel_assoc {n : Type*} [Fintype n] [DecidableEq n]
    (A B C : Matrix n n ℝ) : (A * B) * C = A * (B * C) := by
  exact Matrix.mul_assoc A B C

/-- An exact projection intertwining survives every finite temporal block. -/
theorem kernel_intertwining_powers {n m : Type*} [Fintype n] [Fintype m]
    [DecidableEq n] [DecidableEq m]
    (K : Matrix n n ℝ) (P : Matrix n m ℝ) (Q : Matrix m m ℝ)
    (h : K * P = P * Q) (k : ℕ) : K ^ k * P = P * Q ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [pow_succ, pow_succ, Matrix.mul_assoc, h, ← Matrix.mul_assoc, ih,
      Matrix.mul_assoc]

/-- Scalar moment algebra for the within-state/between-state covariance split.
Probability normalization and the conditional-expectation identification are
stated and proved separately in the manuscript. -/
theorem training_covariance_decomposition {ι : Type*} [Fintype ι]
    (w second mean : ι → ℝ) (grand : ℝ) :
    (∑ i, w i * (second i - mean i ^ 2)) +
      ((∑ i, w i * mean i ^ 2) - grand ^ 2) =
      (∑ i, w i * second i) - grand ^ 2 := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  ring

def trainingMixtureRG (b : ℝ) (within between : ℝ) : ℝ × ℝ :=
  (within, b * between)

/-- Shared-state document blocking preserves the conditional covariance and
rescales the training-state mean covariance. -/
theorem training_mixture_compose (b c within between : ℝ) :
    trainingMixtureRG c (trainingMixtureRG b within between).1
      (trainingMixtureRG b within between).2 =
      trainingMixtureRG (b*c) within between := by
  simp only [trainingMixtureRG, Prod.mk.injEq, true_and]
  ring

theorem feedback_mean_fixed (a d : ℝ) (ha : a ≠ 0) :
    (1-a) * (d/a) + d = d/a := by
  field_simp
  ring

theorem feedback_variance_fixed (a σ : ℝ) (ha : a ≠ 0) (hb : 2-a ≠ 0) :
    (1-a)^2 * (σ^2/(a*(2-a))) + σ^2 = σ^2/(a*(2-a)) := by
  field_simp
  ring

end ModelRG
