import Mathlib

/-! Selected exact algebra for low-rank adaptation. Native floating-point
export equivalence and statistical performance are checked separately. -/
namespace ModelRG

variable {V U K : Type*} [AddCommGroup V] [Module ℝ V]
  [AddCommGroup U] [Module ℝ U] [AddCommGroup K] [Module ℝ K]

theorem lowrank_emission (W : V →ₗ[ℝ] U) (A : V →ₗ[ℝ] K)
    (B : K →ₗ[ℝ] U) (x : V) :
    (W + B.comp A) x = W x + B (A x) := by
  rfl

theorem factor_gradient_displacement (a b eta g : ℝ) :
    (a - eta * b * g) * (b - eta * a * g) - a * b =
      -eta * g * (a ^ 2 + b ^ 2) + eta ^ 2 * g ^ 2 * (a * b) := by
  ring

theorem factor_gauge_emission (a b c : ℝ) (hc : c ≠ 0) :
    (c * a) * (b / c) = a * b := by
  field_simp

end ModelRG
