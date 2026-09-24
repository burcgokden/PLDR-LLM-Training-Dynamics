import Mathlib

/- Selected finite simplex algebra for the spatial covariance decomposition.
   The probability interpretation and empirical estimator are separate. -/
namespace ModelRG
open scoped BigOperators

theorem finite_color_centering
    (n : ℕ) (q : ℝ) (hq : (n : ℝ) = q) (hq0 : q ≠ 0)
    (joint left right : Fin n → ℝ)
    (hl : ∑ a, left a = 1) (hr : ∑ a, right a = 1) :
    (∑ a, (joint a - left a * right a)) +
      (∑ a, (left a - 1 / q) * (right a - 1 / q)) =
      (∑ a, joint a) - 1 / q := by
  rw [← Finset.sum_add_distrib]
  have point : ∀ a, (joint a - left a * right a) +
      (left a - 1 / q) * (right a - 1 / q) =
      joint a - left a / q - right a / q + 1 / q ^ 2 := by
    intro a
    ring
  simp_rw [point]
  simp only [div_eq_mul_inv, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.sum_mul, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, hl, hr, hq]
  field_simp [hq0]
  ring

theorem normalized_spatial_decomposition
    (n : ℕ) (q : ℝ) (hq : (n : ℝ) = q) (hq0 : q ≠ 0)
    (joint left right : Fin n → ℝ)
    (hl : ∑ a, left a = 1) (hr : ∑ a, right a = 1) :
    (q * (∑ a, joint a) - 1) / (q - 1) =
      q / (q - 1) * (∑ a, (joint a - left a * right a)) +
      q / (q - 1) * (∑ a, (left a - 1 / q) * (right a - 1 / q)) := by
  rw [← mul_add, finite_color_centering n q hq hq0 joint left right hl hr]
  by_cases hq1 : q - 1 = 0
  · simp [hq1]
  · field_simp

theorem moment_ratio_signed_error (a b c d : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
    d / c^2 - b / a^2 =
      (a^2 * (d-b) - b * (c-a) * (c+a)) / (a^2 * c^2) := by
  field_simp
  <;> ring

theorem moment_ratio_error_bound (a b c d : ℝ)
    (ha : 0 < a) (hc : 0 < c) (hb : 0 ≤ b) :
    |d / c^2 - b / a^2| ≤
      |d-b| / c^2 + b * |c-a| * (c+a) / (a^2 * c^2) := by
  have hid : d / c^2 - b / a^2 =
      (d-b) / c^2 - b * (c-a) * (c+a) / (a^2 * c^2) := by
    field_simp
    <;> ring
  have hca : 0 < c+a := add_pos hc ha
  rw [hid]
  calc
    _ ≤ |(d-b) / c^2| + |b * (c-a) * (c+a) / (a^2 * c^2)| := by
      simpa only [sub_zero, zero_sub, abs_neg] using
        (abs_sub_le ((d-b) / c^2) 0 (b * (c-a) * (c+a) / (a^2 * c^2)))
    _ = _ := by
      simp only [abs_div, abs_mul, abs_of_nonneg hb, abs_of_pos hca,
        abs_of_nonneg (sq_nonneg a), abs_of_nonneg (sq_nonneg c)]

theorem finite_thermal_contrast_error
    (up um vp vm ep em h : ℝ) (hh : 0 < h)
    (hu : |up - vp| ≤ ep) (hm : |um - vm| ≤ em) :
    |(up - um) / (2*h) - (vp - vm) / (2*h)| ≤ (ep + em) / (2*h) := by
  have hd : 0 < 2*h := by positivity
  have hn : |(up-vp) - (um-vm)| ≤ ep + em := by
    obtain ⟨hu1, hu2⟩ := abs_le.mp hu
    obtain ⟨hm1, hm2⟩ := abs_le.mp hm
    exact abs_le.mpr ⟨by linarith, by linarith⟩
  calc
    _ = |(up-vp) - (um-vm)| / (2*h) := by
      rw [← sub_div, abs_div, abs_of_pos hd]
      have hid : (up-um) - (vp-vm) = (up-vp) - (um-vm) := by ring
      rw [hid]
    _ ≤ (ep+em) / (2*h) := (div_le_div_iff_of_pos_right hd).mpr hn

end ModelRG
