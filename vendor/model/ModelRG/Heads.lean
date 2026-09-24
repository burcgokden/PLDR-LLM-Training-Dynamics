import Mathlib

/-! Selected algebra for conditional head covariance and shape-aware variance.
The probability laws, exchangeability, conditioning, and limit arguments are
proved in the manuscript. These declarations do not assume those hypotheses
have been established for trained models. -/

noncomputable section

namespace ModelRG

def headSusceptibility (n diagonal cross : ℝ) : ℝ :=
  diagonal + (n - 1) * cross

theorem head_susceptibility_count (n diagonal cross : ℝ) (hn : n ≠ 0) :
    (n * diagonal + n * (n - 1) * cross) / n =
      headSusceptibility n diagonal cross := by
  unfold headSusceptibility
  field_simp

theorem head_susceptibility_upper (n diagonal cross : ℝ)
    (hn : 1 ≤ n) (hc : cross ≤ diagonal) :
    headSusceptibility n diagonal cross ≤ n * diagonal := by
  unfold headSusceptibility
  nlinarith [mul_nonneg (sub_nonneg.mpr hn) (sub_nonneg.mpr hc)]

def headCovarianceRG (b scale diagonal cross : ℝ) : ℝ × ℝ :=
  (b / scale ^ 2 * (diagonal + (b - 1) * cross),
   b ^ 2 / scale ^ 2 * cross)

theorem head_covariance_compose (a b s t diagonal cross : ℝ)
    (hs : s ≠ 0) (ht : t ≠ 0) :
    headCovarianceRG a t (headCovarianceRG b s diagonal cross).1
      (headCovarianceRG b s diagonal cross).2 =
      headCovarianceRG (a * b) (s * t) diagonal cross := by
  apply Prod.ext <;> simp only [headCovarianceRG, Prod.fst, Prod.snd] <;>
    field_simp <;> ring

theorem head_covariance_identity (diagonal cross : ℝ) :
    headCovarianceRG 1 1 diagonal cross = (diagonal, cross) := by
  simp [headCovarianceRG]

theorem collective_stationary_variance (lambda innovation : ℝ)
    (h : 1 - lambda ^ 2 ≠ 0) :
    lambda ^ 2 * (innovation / (1 - lambda ^ 2)) + innovation =
      innovation / (1 - lambda ^ 2) := by
  field_simp
  ring

theorem shape_aware_linear_variance (input output refInput refOutput : ℝ)
    (hi : input ≠ 0) (hs : input + output ≠ 0)
    (hr : refInput + refOutput ≠ 0) :
    (2 / (input + output)) *
      (refInput * (input + output) / (input * (refInput + refOutput))) =
      (2 * refInput / (refInput + refOutput)) / input := by
  field_simp

/-- A decayed coordinate remains inside its force-dominated envelope. -/
theorem contractive_decay_bound (eta decay force bound theta ratio : ℝ)
    (heta : 0 ≤ eta) (hdecay : 0 ≤ 1 - eta * decay)
    (htheta : |theta| ≤ bound) (hratio : |ratio| ≤ force)
    (hforce : force ≤ decay * bound) :
    |(1 - eta * decay) * theta - eta * ratio| ≤ bound := by
  calc
    |(1 - eta * decay) * theta - eta * ratio| ≤
        |(1 - eta * decay) * theta| + |eta * ratio| := abs_sub _ _
    _ = (1 - eta * decay) * |theta| + eta * |ratio| := by
      rw [abs_mul, abs_mul, abs_of_nonneg hdecay, abs_of_nonneg heta]
    _ ≤ (1 - eta * decay) * bound + eta * force :=
      add_le_add (mul_le_mul_of_nonneg_left htheta hdecay)
        (mul_le_mul_of_nonneg_left hratio heta)
    _ ≤ bound := by
      nlinarith [mul_le_mul_of_nonneg_left hforce heta]

/-- The only fixed mean for a contracting bias-power coordinate is zero. -/
theorem bias_boundary_fixed (beta mean : ℝ)
    (hbeta : beta < 1) (hfixed : mean = beta * mean) : mean = 0 := by
  have hnonzero : 1 - beta ≠ 0 := ne_of_gt (sub_pos.mpr hbeta)
  have hproduct : (1 - beta) * mean = 0 := by nlinarith [hfixed]
  exact (mul_eq_zero.mp hproduct).resolve_left hnonzero

end ModelRG
