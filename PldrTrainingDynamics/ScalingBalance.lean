import Mathlib

/-!
Manuscript correspondence: `rg:prop:universality-taxonomy`, `rg:eq:log-balance-identity`, `rg:eq:hurst-crossover`, `rg:eq:bounded-factor-crossover`.

Checked clause and supplied hypotheses: Positive scalar log-balance identity; bounded log-ratio squeeze; real-sequence reciprocal limit under diverging crossover, exact balance and subpower along the sequence; positive real-power inequalities with explicit constants; stable-index arithmetic.

Written obligations: No probability theorem, stable-attraction criterion, slowly-varying-to-subpower theorem, asymptotic inverse, empirical dependence law, or full probabilistic tail example. Sequence hypotheses are stated pointwise after any needed finite tail restriction. Power bounds use the equivalent quotient form (a*r/delta)^(1/(1-H)).
These kernels double-check selected clauses; the written proof is independent.
-/

/-! Scalar crossover kernels. Positivity, subpower ratios, and the balance are
hypotheses. No stable limit theorem or native dependence law is inferred. -/
namespace PldrTrainingDynamics.ScalingBalance
open Filter Topology

theorem log_balance_identity (δ ξ L ρ H : ℝ)
    (hδ : 0 < δ) (hξ : 0 < ξ) (hL : 0 < L) (hρ : 0 < ρ)
    (balance : δ * ξ ^ (1 - H) = ρ * L) :
    (1 - H) * Real.log ξ = Real.log (1 / δ) + Real.log L + Real.log ρ := by
  have h := congrArg Real.log balance
  rw [Real.log_mul hδ.ne' (Real.rpow_pos_of_pos hξ _).ne',
    Real.log_mul hρ.ne' hL.ne', Real.log_rpow hξ] at h
  rw [one_div, Real.log_inv]
  linarith

/-- Bounded positive balance ratios contribute no logarithmic exponent. -/
theorem bounded_log_ratio (ξ ρ : ℕ → ℝ) (r R : ℝ) (hr : 0 < r)
    (hρ : ∀ n, r ≤ ρ n ∧ ρ n ≤ R) (hξ : Tendsto ξ atTop atTop) :
    Tendsto (fun n => Real.log (ρ n) / Real.log (ξ n)) atTop (𝓝 0) := by
  have hlog := Real.tendsto_log_atTop.comp hξ
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le'
    ((tendsto_const_nhds (x := Real.log r)).div_atTop hlog)
    ((tendsto_const_nhds (x := Real.log R)).div_atTop hlog) ?_ ?_
  · filter_upwards [hlog.eventually (eventually_gt_atTop 0)] with n hn
    exact div_le_div_of_nonneg_right (Real.log_le_log hr (hρ n).1) hn.le
  · filter_upwards [hlog.eventually (eventually_gt_atTop 0)] with n hn
    exact div_le_div_of_nonneg_right
      (Real.log_le_log (lt_of_lt_of_le hr (hρ n).1) (hρ n).2) hn.le

/-- Sequence form of the logarithmic exponent; subpower is required along the
chosen diverging crossover. The bounded ratio lemma supplies the last limit. -/
theorem log_exponent_of_subpower (δ ξ L ρ : ℕ → ℝ) (H : ℝ) (hH : H < 1)
    (hδ : ∀ n, 0 < δ n) (hξpos : ∀ n, 0 < ξ n)
    (hL : ∀ n, 0 < L n) (hρ : ∀ n, 0 < ρ n)
    (hξ : Tendsto ξ atTop atTop)
    (balance : ∀ n, δ n * ξ n ^ (1 - H) = ρ n * L n)
    (subpower : Tendsto (fun n => Real.log (L n) / Real.log (ξ n)) atTop (𝓝 0))
    (ratio : Tendsto (fun n => Real.log (ρ n) / Real.log (ξ n)) atTop (𝓝 0)) :
    Tendsto (fun n => Real.log (ξ n) / Real.log (1 / δ n))
      atTop (𝓝 (1 / (1 - H))) := by
  have limit : Tendsto (fun n => (1 - H) - Real.log (L n) / Real.log (ξ n) -
      Real.log (ρ n) / Real.log (ξ n)) atTop (𝓝 (1 - H)) := by
    simpa using (tendsto_const_nhds.sub subpower).sub ratio
  have reciprocal : Tendsto (fun n => Real.log (1 / δ n) / Real.log (ξ n))
      atTop (𝓝 (1 - H)) := by
    apply limit.congr'
    filter_upwards [(Real.tendsto_log_atTop.comp hξ).eventually
      (eventually_gt_atTop 0)] with n hn
    have hb := log_balance_identity (δ n) (ξ n) (L n) (ρ n) H
      (hδ n) (hξpos n) (hL n) (hρ n) (balance n)
    have hn0 : Real.log (ξ n) ≠ 0 := ne_of_gt hn
    field_simp [hn0]
    nlinarith
  have hi := reciprocal.inv₀ (ne_of_gt (sub_pos.mpr hH))
  simpa [one_div, inv_div] using hi

/-- The full scalar logarithmic claim with explicit two-sided balance bounds.
All inequalities may be applied after discarding a finite sequence prefix. -/
theorem log_exponent_of_bounded_balance (δ ξ L ρ : ℕ → ℝ) (H r R : ℝ)
    (hH : H < 1) (hr : 0 < r)
    (hδ : ∀ n, 0 < δ n) (hξpos : ∀ n, 0 < ξ n) (hL : ∀ n, 0 < L n)
    (hρ : ∀ n, r ≤ ρ n ∧ ρ n ≤ R) (hξ : Tendsto ξ atTop atTop)
    (balance : ∀ n, δ n * ξ n ^ (1 - H) = ρ n * L n)
    (subpower : Tendsto (fun n => Real.log (L n) / Real.log (ξ n)) atTop (𝓝 0)) :
    Tendsto (fun n => Real.log (ξ n) / Real.log (1 / δ n))
      atTop (𝓝 (1 / (1 - H))) :=
  log_exponent_of_subpower δ ξ L ρ H hH hδ hξpos hL
    (fun n => lt_of_lt_of_le hr (hρ n).1) hξ balance subpower
    (bounded_log_ratio ξ ρ r R hr hρ hξ)

/-- Finite positive-power bounds, with constants exactly as in the book. -/
theorem bounded_factor_power_bounds (δ ξ L ρ H a A r R : ℝ)
    (hδ : 0 < δ) (hξ : 0 < ξ) (hH : H < 1)
    (ha : 0 < a) (hr : 0 < r)
    (hL : a ≤ L ∧ L ≤ A) (hρ : r ≤ ρ ∧ ρ ≤ R)
    (balance : δ * ξ ^ (1 - H) = ρ * L) :
    (a * r / δ) ^ (1 / (1 - H)) ≤ ξ ∧
      ξ ≤ (A * R / δ) ^ (1 / (1 - H)) := by
  have hLpos : 0 < L := lt_of_lt_of_le ha hL.1
  have hρpos : 0 < ρ := lt_of_lt_of_le hr hρ.1
  have hpower : 0 < 1 - H := sub_pos.mpr hH
  have lower : a * r / δ ≤ ξ ^ (1 - H) := by
    apply (div_le_iff₀ hδ).mpr
    have h := mul_le_mul hL.1 hρ.1 hr.le hLpos.le
    nlinarith [balance]
  have upper : ξ ^ (1 - H) ≤ A * R / δ := by
    apply (le_div_iff₀ hδ).mpr
    have h := mul_le_mul hL.2 hρ.2 hρpos.le (le_trans hLpos.le hL.2)
    nlinarith [balance]
  have cancel : (ξ ^ (1 - H)) ^ (1 / (1 - H)) = ξ := by
    rw [← Real.rpow_mul hξ.le, mul_one_div_cancel hpower.ne', Real.rpow_one]
  constructor
  · have h := Real.rpow_le_rpow (div_nonneg (mul_nonneg ha.le hr.le) hδ.le)
      lower (one_div_pos.mpr hpower).le
    rw [cancel] at h
    exact h
  · have h := Real.rpow_le_rpow (Real.rpow_pos_of_pos hξ _).le
      upper (one_div_pos.mpr hpower).le
    rw [cancel] at h
    exact h

theorem stable_exponent_identity (α : ℝ) (hα : 1 < α) :
    1 / (1 - 1 / α) = α / (α - 1) := by
  have hα0 : α ≠ 0 := ne_of_gt (lt_trans zero_lt_one hα)
  have hden : α - 1 ≠ 0 := ne_of_gt (sub_pos.mpr hα)
  field_simp [hα0, hden]

theorem stable_three_halves :
    (1 : ℝ) / (1 - 1 / (3 / 2)) = 3 ∧
      (1 : ℝ) / ((3 / 2) - 1) = 2 := by norm_num

end PldrTrainingDynamics.ScalingBalance
