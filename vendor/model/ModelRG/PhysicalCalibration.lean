import Mathlib

/-! Selected finite algebra for physical calibration. The manuscript separately
proves the KL, differentiability, norm, and limiting statements. -/
namespace ModelRG

theorem physical_relative_bracket (truth estimate epsilon : ℝ)
    (ht : 0 < truth) (he : epsilon < 1)
    (herr : |estimate - truth| ≤ epsilon * truth) :
    (1-epsilon)*truth ≤ estimate ∧ estimate ≤ (1+epsilon)*truth ∧ 0 < estimate := by
  obtain ⟨hl, hu⟩ := abs_le.mp herr
  have hp : 0 < (1-epsilon)*truth := mul_pos (by linarith) ht
  constructor
  · nlinarith
  constructor
  · nlinarith
  · nlinarith

theorem physical_visible_eigenvector
    {V W : Type*} [AddCommGroup V] [Module ℝ V]
    [AddCommGroup W] [Module ℝ W]
    (A : V →ₗ[ℝ] V) (B : W →ₗ[ℝ] W) (C : V →ₗ[ℝ] W)
    (v : V) (lam : ℝ)
    (hcomm : ∀ x, B (C x) = C (A x))
    (heigen : A v = lam • v) (hvisible : C v ≠ 0) :
    C v ≠ 0 ∧ B (C v) = lam • C v := by
  constructor
  · exact hvisible
  · rw [hcomm v, heigen, map_smul]

end ModelRG
