import Mathlib

/-! Observation error and finite-sample components. Analytic probability limits,
matrix error certification, and the entropy Taylor integral remain in the text. -/
noncomputable section
namespace ModelRG
open scoped BigOperators

theorem norm_sq_perturbation {E : Type*} [NormedAddCommGroup E]
    (x e : E) :
    |‖x + e‖ ^ 2 - ‖x‖ ^ 2| ≤ 2 * ‖x‖ * ‖e‖ + ‖e‖ ^ 2 := by
  have hx := norm_nonneg x
  have he := norm_nonneg e
  have hxe := norm_nonneg (x + e)
  have hu := norm_add_le x e
  have hl : ‖x‖ ≤ ‖x + e‖ + ‖e‖ := by
    simpa only [add_sub_cancel_right] using norm_sub_le (x + e) e
  have pu := mul_nonneg (sub_nonneg.mpr hu)
    (by positivity : 0 ≤ ‖x‖ + ‖e‖ + ‖x + e‖)
  have pl := mul_nonneg (sub_nonneg.mpr hl)
    (by positivity : 0 ≤ ‖x + e‖ + ‖e‖ + ‖x‖)
  rw [abs_le]
  constructor <;> nlinarith

theorem centered_energy_identity {ι E : Type*} [Fintype ι]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : ι → E) (mu : E)
    (hmean : ∑ i, x i = (Fintype.card ι : ℝ) • mu) :
    (∑ i, ‖x i - mu‖ ^ 2) =
      (∑ i, ‖x i‖ ^ 2) - (Fintype.card ι : ℝ) * ‖mu‖ ^ 2 := by
  simp_rw [norm_sub_sq_real]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      ← sum_inner, hmean, real_inner_smul_left, real_inner_self_eq_norm_sq]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  ring

theorem centered_energy_contraction {ι E : Type*} [Fintype ι]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : ι → E) (mu : E)
    (hmean : ∑ i, x i = (Fintype.card ι : ℝ) • mu) :
    (∑ i, ‖x i - mu‖ ^ 2) ≤ ∑ i, ‖x i‖ ^ 2 := by
  rw [centered_energy_identity x mu hmean]
  exact sub_le_self _ (by positivity)

/-- The unbiased sample factor, before the norm-to-susceptibility identification. -/
theorem sample_norm_transfer {E : Type*} [NormedAddCommGroup E]
    (x e : E) (N s : ℝ) (hN : 0 ≤ N) (hs : 1 < s) :
    |N / (s-1) * ‖x+e‖ ^ 2 - N / (s-1) * ‖x‖ ^ 2| ≤
      N / (s-1) * (2 * ‖x‖ * ‖e‖ + ‖e‖ ^ 2) := by
  have hc : 0 ≤ N / (s-1) := div_nonneg hN (by linarith)
  rw [← mul_sub, abs_mul, abs_of_nonneg hc]
  exact mul_le_mul_of_nonneg_left (norm_sq_perturbation x e) hc

theorem sample_uncentered_bound {ι E : Type*} [Fintype ι]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x : ι → E) (mu : E) (N : ℝ) (hN : 0 ≤ N)
    (hs : 1 < (Fintype.card ι : ℝ))
    (hmean : ∑ i, x i = (Fintype.card ι : ℝ) • mu) :
    N / ((Fintype.card ι : ℝ)-1) * (∑ i, ‖x i - mu‖ ^ 2) ≤
      N * (Fintype.card ι : ℝ) / ((Fintype.card ι : ℝ)-1) *
        ((∑ i, ‖x i‖ ^ 2) / (Fintype.card ι : ℝ)) := by
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by linarith
  calc
    _ ≤ N / ((Fintype.card ι : ℝ)-1) * (∑ i, ‖x i‖ ^ 2) :=
      mul_le_mul_of_nonneg_left (centered_energy_contraction x mu hmean)
        (div_nonneg hN (by linarith))
    _ = _ := by field_simp

/-- A zero-sum tangent vector has the sharp simplex norm constant. -/
theorem simplex_tangent_l1 {ι : Type*} [Fintype ι]
    (v : ι → ℝ) (hv : ∑ i, v i = 0) :
    2 * (∑ i, (v i)^2) ≤ (∑ i, |v i|)^2 := by
  classical
  have hi (i : ι) : 2 * |v i| ≤ ∑ j, |v j| := by
    have hsum := Finset.sum_erase_add (s := Finset.univ) v (Finset.mem_univ i)
    have habs := Finset.sum_erase_add (s := Finset.univ) (fun j => |v j|) (Finset.mem_univ i)
    have heq : ∑ j ∈ Finset.univ.erase i, v j = -v i := by linarith
    have hbound := Finset.abs_sum_le_sum_abs (s := Finset.univ.erase i) (f := v)
    rw [heq, abs_neg] at hbound
    linarith
  have hterm (i : ι) : 2 * (v i)^2 ≤ |v i| * (∑ j, |v j|) := by
    have hh := mul_le_mul_of_nonneg_left (hi i) (abs_nonneg (v i))
    nlinarith [sq_abs (v i)]
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hterm i)
  rw [← Finset.mul_sum, ← Finset.sum_mul] at h
  nlinarith

theorem simplex_tangent_hessian {ι : Type*} [Fintype ι]
    (p v : ι → ℝ) (hp : ∀ i, 0 < p i) (hp1 : ∑ i, p i = 1)
    (hv : ∑ i, v i = 0) :
    2 * (∑ i, (v i)^2) ≤ ∑ i, (v i)^2 / p i := by
  have h := Finset.sq_sum_div_le_sum_sq_div Finset.univ (fun i => |v i|)
    (fun i _ => hp i)
  simp only [hp1, div_one, sq_abs] at h
  exact (simplex_tangent_l1 v hv).trans h

/-- Finite weighted covariance expansion underlying inference visibility. -/
theorem visibility_energy {ι : Type*} [Fintype ι]
    (w r e : ι → ℝ) (a : ℝ) :
    (∑ i, w i * (a*r i+e i)^2) =
      a^2 * (∑ i, w i * (r i)^2) +
      2*a * (∑ i, w i * r i * e i) + (∑ i, w i * (e i)^2) := by
  simp_rw [show ∀ i, w i * (a*r i+e i)^2 =
    a^2 * (w i*(r i)^2) + 2*a*(w i*r i*e i) + w i*(e i)^2 by intro i; ring]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]

end ModelRG
