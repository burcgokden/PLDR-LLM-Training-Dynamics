/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# True-loss curvature after a signed second-jet charge
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace FullLossCurvature

/-- A relative signed residual charge preserves a positive part of a Fisher
lower edge. -/
theorem relative_curvature_edge
    {fisher residual mu chi epsilon normSquare : ℝ}
    (hchi1 : chi ≤ 1)
    (hfisher : mu * normSquare ≤ fisher)
    (hresidual :
      -chi * fisher - epsilon * normSquare ≤ residual) :
    ((1 - chi) * mu - epsilon) * normSquare
      ≤ fisher + residual := by
  have hfactor : 0 ≤ 1 - chi := by linarith
  have hscaled :
      (1 - chi) * (mu * normSquare) ≤ (1 - chi) * fisher :=
    mul_le_mul_of_nonneg_left hfisher hfactor
  nlinarith

/-- Exact conditional balance removes both signed contractions when the
conditional residual is zero. -/
theorem exact_balance_zero
    (firstJet secondJet : ℝ) :
    0 * firstJet = 0 ∧ 0 * secondJet = 0 := by
  simp

/-- Scalar Cauchy bounds a balanced normal force by the product of the
residual and Jacobian envelopes. -/
theorem scalar_force_bound
    {residual jacobian residualBound jacobianBound : ℝ}
    (hresidual : |residual| ≤ residualBound)
    (hjacobian : |jacobian| ≤ jacobianBound)
    (hr : 0 ≤ residualBound) :
    |residual * jacobian| ≤ residualBound * jacobianBound := by
  rw [abs_mul]
  exact mul_le_mul hresidual hjacobian (abs_nonneg _) hr

end FullLossCurvature
end PldrLlmCurvatureSandpile
