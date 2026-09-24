/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Two-variable crossing-safe PLGA telescope
-/
import Mathlib
import PldrLlmCurvatureSandpile.PLGADividedDifference

namespace PldrLlmCurvatureSandpile
namespace PLGATwoVariableSecant

open PLGADividedDifference

/-- Positive-base real power used by the implemented PLGA map. -/
noncomputable def positiveBasePower (base exponent : ℝ) : ℝ :=
  Real.exp (exponent * Real.log base)

theorem positive_base_power_pos (base exponent : ℝ) :
    0 < positiveBasePower base exponent := by
  exact Real.exp_pos _

/-- Divided difference in the preactivation coordinate. -/
noncomputable def zSecant
    (function zDerivative : ℝ → ℝ → ℝ)
    (z0 z1 exponent : ℝ) : ℝ :=
  dividedDifference
    (fun z => function z exponent)
    (fun z => zDerivative z exponent) z0 z1

/-- Divided difference in the exponent coordinate. -/
noncomputable def pSecant
    (function pDerivative : ℝ → ℝ → ℝ)
    (p0 p1 preactivation : ℝ) : ℝ :=
  dividedDifference
    (fun p => function preactivation p)
    (fun p => pDerivative preactivation p) p0 p1

/-- Changing the base coordinate first and the exponent coordinate second is
an exact endpoint telescope, including both diagonal branches. -/
theorem base_exponent_telescope
    (function zDerivative pDerivative : ℝ → ℝ → ℝ)
    (z0 z1 p0 p1 : ℝ) :
    function z1 p1 - function z0 p0 =
      zSecant function zDerivative z0 z1 p1 * (z1 - z0)
      + pSecant function pDerivative p0 p1 z0 * (p1 - p0) := by
  have hz := divided_difference_exact
    (fun z => function z p1) (fun z => zDerivative z p1) z0 z1
  have hp := divided_difference_exact
    (fun p => function z0 p) (fun p => pDerivative z0 p) p0 p1
  simp only [zSecant, pSecant]
  linarith

/-- The opposite coordinate order is also an exact endpoint telescope. -/
theorem exponent_base_telescope
    (function zDerivative pDerivative : ℝ → ℝ → ℝ)
    (z0 z1 p0 p1 : ℝ) :
    function z1 p1 - function z0 p0 =
      pSecant function pDerivative p0 p1 z1 * (p1 - p0)
      + zSecant function zDerivative z0 z1 p0 * (z1 - z0) := by
  have hp := divided_difference_exact
    (fun p => function z1 p) (fun p => pDerivative z1 p) p0 p1
  have hz := divided_difference_exact
    (fun z => function z p0) (fun z => zDerivative z p0) z0 z1
  simp only [zSecant, pSecant]
  linarith

/-- A zero crossing of the preactivation does not alter the exact telescope;
only the positive-base definition of the powered map is needed. -/
theorem crossing_safe
    (function zDerivative pDerivative : ℝ → ℝ → ℝ)
    (z0 z1 p0 p1 : ℝ) :
    function z1 p1 - function z0 p0 =
      zSecant function zDerivative z0 z1 p1 * (z1 - z0)
      + pSecant function pDerivative p0 p1 z0 * (p1 - p0) := by
  exact base_exponent_telescope
    function zDerivative pDerivative z0 z1 p0 p1

end PLGATwoVariableSecant
end PldrLlmCurvatureSandpile
