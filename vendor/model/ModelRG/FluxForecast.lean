import Mathlib

/-! Selected rational algebra for finite-flux errors and energy-coordinate
composition. Analytic inequalities, closure, limiting distributions and native
numerics retain their independent written arguments. -/
namespace ModelRG.FluxForecast

theorem error_identity (a b c ah bh ch : ℝ)
    (hc : 1 + c ≠ 0) (hch : 1 + ch ≠ 0) :
    (a+b)/(1+c) - (ah+bh)/(1+ch) =
      ((a-ah)+(b-bh) - (c-ch)*((ah+bh)/(1+ch)))/(1+c) := by
  field_simp
  ring

/-- Ordered energy increments telescope with the incoming energy ratio. -/
theorem energy_composition (E0 E1 E2 C0 C1 C2 : ℝ)
    (h0 : E0 ≠ 0) (h1 : E1 ≠ 0) :
    ((C1-C0)/E0 + (E1/E0)*((C2-C1)/E1), (E1/E0)*(E2/E1)) =
      ((C2-C0)/E0, E2/E0) := by
  apply Prod.ext <;> dsimp <;> field_simp <;> ring

/-- Associativity preserves the chronological order of affine energy maps. -/
theorem energy_associativity (q1 q2 q3 r1 r2 r3 : ℝ) :
    ((q1+r1*q2)+(r1*r2)*q3, (r1*r2)*r3) =
      (q1+r1*(q2+r2*q3), r1*(r2*r3)) := by
  apply Prod.ext <;> dsimp <;> ring

/-- Row fractions transform by the same finite energy map. -/
theorem row_action (u q1 q2 r1 r2 : ℝ)
    (h1 : r1 ≠ 0) (h2 : r2 ≠ 0) :
    ((u+q1)/r1+q2)/r2 = (u+(q1+r1*q2))/(r1*r2) := by
  field_simp
  ring

end ModelRG.FluxForecast
