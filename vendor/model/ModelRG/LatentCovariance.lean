import Mathlib

/-!
Finite algebra for heterogeneous document-latent covariance and context risk.
Registered selected components of the independent written proofs.
Covariance interpretation, square integrability, conditional probability and
independence-to-zero-cross-covariance are supplied by stand-alone written proofs.
The first identity applies coordinatewise to a matrix covariance kernel.
-/
namespace ModelRG.LatentCovariance
open scoped BigOperators

theorem kernel_sum (b : ℕ) (between : ℝ) (residual : Fin b → ℝ) :
    (∑ i : Fin b, ∑ j : Fin b,
      (between + if i = j then residual i else 0)) =
      (b : ℝ)^2 * between + ∑ i : Fin b, residual i := by
  classical
  simp [Finset.sum_add_distrib, pow_two, mul_assoc]

theorem heterogeneous_susceptibility (b : ℕ) (hb : 0 < b)
    (between : ℝ) (residual : Fin b → ℝ) (kernel : Fin b → Fin b → ℝ)
    (hkernel : ∀ i j, kernel i j = between + if i = j then residual i else 0) :
    (∑ i : Fin b, ∑ j : Fin b, kernel i j) / (b : ℝ) =
      (b : ℝ) * between + (∑ i : Fin b, residual i) / (b : ℝ) := by
  have hb0 : (b : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hb
  simp_rw [hkernel]
  rw [kernel_sum]
  field_simp

theorem common_covariance_specialization (b : ℕ) (hb : 0 < b)
    (between within : ℝ) :
    (b : ℝ) * between + (∑ _ : Fin b, within) / (b : ℝ) =
      (b : ℝ) * between + within := by
  have hb0 : (b : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hb
  simp [hb0]

theorem context_weighted_ratio {ι : Type*} [Fintype ι]
    (native residual : ι → ℝ) (hpositive : ∀ i, 0 < native i) :
    (∑ i, residual i) / (∑ i, native i) =
      ∑ i, (native i / (∑ j, native j)) * (residual i / native i) := by
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  have hi : native i ≠ 0 := ne_of_gt (hpositive i)
  by_cases hs : (∑ j, native j) = 0
  · simp [hs]
  · field_simp

end ModelRG.LatentCovariance

