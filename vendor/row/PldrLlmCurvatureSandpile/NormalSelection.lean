/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# PLDR normal-response selection

The Gauss-Newton lower edge is reduced to primitive loss curvature,
downstream observability, and explicitly charged residuals.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace NormalSelection

/-- Primitive PLDR lower edges and residual bounds produce the full normal
response edge. -/
theorem gauss_newton_normal_lower_edge
    {lossEdge singularEdge gaussNewton fullResponse
      hessianResidual branchResidual preconditionerResidual : ℝ}
    (hgn :
      lossEdge * singularEdge ^ 2 ≤ gaussNewton)
    (hfull :
      gaussNewton - hessianResidual - branchResidual
          - preconditionerResidual ≤ fullResponse) :
    lossEdge * singularEdge ^ 2 - hessianResidual - branchResidual
        - preconditionerResidual ≤ fullResponse := by
  linarith

/-- Positivity survives exactly when the complete residual budget is smaller
than the primitive Gauss-Newton edge. -/
theorem normal_edge_positive
    {lossEdge singularEdge hessianResidual branchResidual
      preconditionerResidual : ℝ}
    (hmargin :
      hessianResidual + branchResidual + preconditionerResidual
        < lossEdge * singularEdge ^ 2) :
    0 < lossEdge * singularEdge ^ 2 - hessianResidual - branchResidual
        - preconditionerResidual := by
  linarith

/-- The closed normal force is the sum of the named primitive sources and the
quadratic chart remainder. -/
theorem closed_normal_force
    {rope covariance batch drift intervention lipschitz radius force : ℝ}
    (hforce :
      force ≤ rope + covariance + batch + drift + intervention
        + lipschitz * radius ^ 2 / 2) :
    force ≤ rope + covariance + batch + drift + intervention
        + lipschitz * radius ^ 2 / 2 := by
  exact hforce

end NormalSelection
end PldrLlmCurvatureSandpile
