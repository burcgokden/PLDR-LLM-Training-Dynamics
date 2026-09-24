import Mathlib

/-! Finite weighted projection algebra under explicit orthogonality.
Projection existence, normal equations, statistical interpretation and limiting
claims are established separately in the stand-alone manuscript proof. -/
open scoped BigOperators
namespace ModelRG

theorem weighted_residual_split {ι : Type*} [Fintype ι]
    (w y q z : ι → ℝ)
    (ho : ∑ i, w i * (y i-q i) * (q i-z i) = 0) :
    (∑ i, w i*(y i-z i)^2) =
      (∑ i, w i*(y i-q i)^2) + (∑ i, w i*(q i-z i)^2) := by
  have h : ∀ i, w i*(y i-z i)^2 =
      w i*(y i-q i)^2 + w i*(q i-z i)^2 +
      2*(w i*(y i-q i)*(q i-z i)) := by intro i; ring
  simp only [h, Finset.sum_add_distrib, ← Finset.mul_sum, ho, mul_zero, add_zero]

theorem weighted_projection_lower_bound {ι : Type*} [Fintype ι]
    (w y q z : ι → ℝ) (hw : ∀ i, 0 ≤ w i)
    (ho : ∑ i, w i * (y i-q i) * (q i-z i) = 0) :
    (∑ i, w i*(y i-q i)^2) ≤ (∑ i, w i*(y i-z i)^2) := by
  rw [weighted_residual_split w y q z ho]
  have h : 0 ≤ ∑ i, w i*(q i-z i)^2 :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hw i) (sq_nonneg _))
  linarith

end ModelRG
