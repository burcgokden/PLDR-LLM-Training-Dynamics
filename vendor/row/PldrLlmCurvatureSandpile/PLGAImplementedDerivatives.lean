/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Closed-form derivatives of the implemented PLGA power
-/
import Mathlib
import PldrLlmCurvatureSandpile.PLGAImplementedPower

namespace PldrLlmCurvatureSandpile
namespace PLGAImplementedPower

/-- The implementation-specific sigmoid agrees definitionally with the
standard real sigmoid and has its usual derivative. -/
theorem hasDerivAt_sigmoid (z : ℝ) :
    HasDerivAt sigmoid (sigmoid z * (1 - sigmoid z)) z := by
  have hfun : sigmoid = Real.sigmoid := by
    funext input
    rfl
  rw [hfun]
  exact Real.hasDerivAt_sigmoid z

/-- Closed derivative of the strictly positive PLGA base. -/
theorem hasDerivAt_implementedBase (epsilon z : ℝ) :
    HasDerivAt (implementedBase epsilon)
      (2 * z * sigmoid z + z ^ 2 * sigmoid z * (1 - sigmoid z)) z := by
  convert ((hasDerivAt_pow 2 z).mul (hasDerivAt_sigmoid z)).add_const epsilon using 1
  · exact AddCommGroup.ext rfl
  · exact Module.ext rfl
  · funext input
    rfl
  · ring

/-- The implemented power is differentiable in preactivation for every real
exponent because its registered base is strictly positive. -/
theorem hasDerivAt_implementedPower_z
    {epsilon : ℝ} (hepsilon : 0 < epsilon) (z exponent : ℝ) :
    HasDerivAt (fun input => implementedPower epsilon input exponent)
      ((2 * z * sigmoid z + z ^ 2 * sigmoid z * (1 - sigmoid z))
        * exponent * implementedBase epsilon z ^ (exponent - 1)) z := by
  have hbase : 0 < implementedBase epsilon z :=
    hepsilon.trans_le (plga_base_lower epsilon z)
  simpa only [implementedPower] using
    (hasDerivAt_implementedBase epsilon z).rpow_const
      (p := exponent) (Or.inl hbase.ne')

/-- The implemented power is differentiable in its learned exponent. -/
theorem hasDerivAt_implementedPower_exponent
    {epsilon : ℝ} (hepsilon : 0 < epsilon) (z exponent : ℝ) :
    HasDerivAt (fun power => implementedPower epsilon z power)
      (implementedPower epsilon z exponent * Real.log (implementedBase epsilon z))
      exponent := by
  have hbase : 0 < implementedBase epsilon z :=
    hepsilon.trans_le (plga_base_lower epsilon z)
  simpa only [implementedPower, id_eq, one_mul, mul_one, mul_comm] using
    (hasDerivAt_id exponent).const_rpow hbase

/-- The derivative used on the preactivation secant diagonal has a closed
formula on the implemented positive-base domain. -/
theorem implemented_z_derivative_closed
    {epsilon : ℝ} (hepsilon : 0 < epsilon) (z exponent : ℝ) :
    implementedZDerivative epsilon exponent z =
      (2 * z * sigmoid z + z ^ 2 * sigmoid z * (1 - sigmoid z))
        * exponent * implementedBase epsilon z ^ (exponent - 1) := by
  exact (hasDerivAt_implementedPower_z hepsilon z exponent).deriv

/-- The derivative used on the exponent secant diagonal has a closed formula
on the implemented positive-base domain. -/
theorem implemented_p_derivative_closed
    {epsilon : ℝ} (hepsilon : 0 < epsilon) (z exponent : ℝ) :
    implementedPDerivative epsilon z exponent =
      implementedPower epsilon z exponent * Real.log (implementedBase epsilon z) := by
  exact (hasDerivAt_implementedPower_exponent hepsilon z exponent).deriv

end PLGAImplementedPower
end PldrLlmCurvatureSandpile
