/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Implemented positive-base PLGA power

The implemented iSwiGLU base is strictly positive after addition of the
registered adjustment. Consequently its arbitrary real power is finite and
admits exact two-coordinate endpoint telescopes, including zero crossings.
-/
import Mathlib
import PldrLlmCurvatureSandpile.PLGADividedDifference

namespace PldrLlmCurvatureSandpile
namespace PLGAImplementedPower

open PLGADividedDifference

/-- Logistic sigmoid in the implemented PLGA scalar kernel. -/
noncomputable def sigmoid (z : ℝ) : ℝ := (1 + Real.exp (-z))⁻¹

/-- Strictly positive base used by the learned real power. -/
noncomputable def implementedBase (epsilon z : ℝ) : ℝ :=
  z ^ 2 * sigmoid z + epsilon

/-- The actual positive-base PLGA power. -/
noncomputable def implementedPower (epsilon z exponent : ℝ) : ℝ :=
  implementedBase epsilon z ^ exponent

/-- The positive adjustment is a lower bound on the implemented base. -/
theorem plga_base_lower (epsilon z : ℝ) :
    epsilon ≤ implementedBase epsilon z := by
  have hsigmoid : 0 ≤ sigmoid z := by
    dsimp [sigmoid]
    positivity
  dsimp [implementedBase]
  nlinarith [sq_nonneg z]

/-- Every implemented PLGA power is positive for every real exponent. -/
theorem implemented_power_positive {epsilon : ℝ} (hepsilon : 0 < epsilon)
    (z exponent : ℝ) : 0 < implementedPower epsilon z exponent := by
  apply Real.rpow_pos_of_pos
  exact hepsilon.trans_le (plga_base_lower epsilon z)

/-- The actual derivative of the implemented positive-base power with
respect to preactivation. This is used on the diagonal of its secant. -/
noncomputable def implementedZDerivative
    (epsilon exponent z : ℝ) : ℝ :=
  deriv (fun input => implementedPower epsilon input exponent) z

/-- The actual derivative of the implemented positive-base power with
respect to its learned exponent. -/
noncomputable def implementedPDerivative
    (epsilon z exponent : ℝ) : ℝ :=
  deriv (fun power => implementedPower epsilon z power) exponent

/-- Divided difference of the implemented power in preactivation. -/
noncomputable def implementedZSecant
    (epsilon z0 z1 exponent : ℝ) : ℝ :=
  dividedDifference (fun z => implementedPower epsilon z exponent)
    (implementedZDerivative epsilon exponent) z0 z1

/-- Divided difference of the implemented power in exponent. -/
noncomputable def implementedPSecant
    (epsilon p0 p1 z : ℝ) : ℝ :=
  dividedDifference (fun p => implementedPower epsilon z p)
    (implementedPDerivative epsilon z) p0 p1

/-- On a zero preactivation increment, the implemented secant is the true
preactivation derivative rather than an arbitrary placeholder. -/
theorem implemented_z_secant_diagonal (epsilon z exponent : ℝ) :
    implementedZSecant epsilon z z exponent =
      implementedZDerivative epsilon exponent z := by
  exact divided_difference_diagonal _ _ _

/-- On a zero exponent increment, the implemented secant is the true
exponent derivative. -/
theorem implemented_p_secant_diagonal (epsilon z exponent : ℝ) :
    implementedPSecant epsilon exponent exponent z =
      implementedPDerivative epsilon z exponent := by
  exact divided_difference_diagonal _ _ _

/-- The implemented two-variable power has an exact multiplied-contribution
telescope in either crossing or noncrossing configurations. -/
theorem implemented_two_variable_telescope
    (epsilon z0 z1 p0 p1 : ℝ) :
    implementedPower epsilon z1 p1 - implementedPower epsilon z0 p0 =
      implementedZSecant epsilon z0 z1 p1 * (z1 - z0) +
      implementedPSecant epsilon p0 p1 z0 * (p1 - p0) := by
  have hz := divided_difference_exact
    (fun z => implementedPower epsilon z p1)
      (implementedZDerivative epsilon p1) z0 z1
  have hp := divided_difference_exact
    (fun p => implementedPower epsilon z0 p)
      (implementedPDerivative epsilon z0) p0 p1
  simp only [implementedZSecant, implementedPSecant]
  linarith

/-- The opposite coordinate order is also exact. Together with
`implemented_two_variable_telescope`, this makes the finite endpoint route
explicit rather than treating a raw secant as a coordinate-free gain. -/
theorem implemented_two_variable_telescope_exponent_last
    (epsilon z0 z1 p0 p1 : ℝ) :
    implementedPower epsilon z1 p1 - implementedPower epsilon z0 p0 =
      implementedPSecant epsilon p0 p1 z1 * (p1 - p0) +
      implementedZSecant epsilon z0 z1 p0 * (z1 - z0) := by
  have hp := divided_difference_exact
    (fun p => implementedPower epsilon z1 p)
      (implementedPDerivative epsilon z1) p0 p1
  have hz := divided_difference_exact
    (fun z => implementedPower epsilon z p0)
      (implementedZDerivative epsilon p0) z0 z1
  simp only [implementedZSecant, implementedPSecant]
  linarith

end PLGAImplementedPower
end PldrLlmCurvatureSandpile
