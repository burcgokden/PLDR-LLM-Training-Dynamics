import Mathlib

/-! Finite pulse and optimizer transport. Covariance applications supply
centering and moment hypotheses. Native arithmetic and probability limits
are not assertions of these algebraic statements. -/

namespace ModelRG

theorem pulse_reconstruct_plus (x p m : ℝ) :
    x + ((p + m) / 2 - x) + (p - m) / 2 = p := by ring

theorem pulse_reconstruct_minus (x p m : ℝ) :
    x + ((p + m) / 2 - x) - (p - m) / 2 = m := by ring

theorem pulse_linear_transport {E F : Type*}
    [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]
    (B : E →ₗ[ℝ] F) (x p m : E) :
    B ((1 / 2 : ℝ) • (p + m) - x) =
      (1 / 2 : ℝ) • (B p + B m) - B x ∧
    B ((1 / 2 : ℝ) • (p - m)) = (1 / 2 : ℝ) • (B p - B m) := by
  constructor <;> simp

theorem pulse_three_sector_bilinear
    (x e o y f p s : ℝ) :
    (x + e + s*o) * (y + f + s*p) =
      x*y + e*f + s^2*o*p + x*f + e*y +
      s*(x*p + o*y + e*p + o*f) := by ring

theorem adam_finite_first_moment (theta eta decay beta m g delta a d : ℝ) :
    ((1-eta*decay)*theta - eta*(beta*(m+delta)+(1-beta)*g)/(a*d)) -
      ((1-eta*decay)*theta - eta*(beta*m+(1-beta)*g)/(a*d)) =
      -eta*beta*delta/(a*d) := by
  simp only [div_eq_mul_inv]
  ring

theorem adam_finite_denominator (theta eta decay m a d0 d1 : ℝ) :
    ((1-eta*decay)*theta - eta*m/(a*d1)) -
      ((1-eta*decay)*theta - eta*m/(a*d0)) =
      eta*m/a*(1/d0-1/d1) := by
  simp only [div_eq_mul_inv, mul_inv_rev, one_mul]
  ring

theorem adam_second_multiplicative (beta v g factor : ℝ) :
    (beta*(factor*v)+(1-beta)*g^2) - (beta*v+(1-beta)*g^2) =
      beta*v*(factor-1) := by ring

end ModelRG
