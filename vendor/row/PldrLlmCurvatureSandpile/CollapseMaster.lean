/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Algebraic master collapse bound
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace CollapseMaster

/-- Substitution of a transverse envelope into the explicit normal-to-direct
bridge gives the finite-horizon master bound. -/
theorem finite_master_bound
    {direct coefficient state envelope resolution : ℝ}
    (hcoefficient : 0 ≤ coefficient)
    (hstate : state ≤ envelope)
    (hdirect : direct ≤ coefficient * state + resolution) :
    direct ≤ coefficient * envelope + resolution := by
  have hscaled :
      coefficient * state ≤ coefficient * envelope :=
    mul_le_mul_of_nonneg_left hstate hcoefficient
  linarith

/-- Vanishing transverse and resolution terms force a nonnegative direct
seminorm to vanish. -/
theorem zero_terms_force_collapse
    {direct coefficient state resolution : ℝ}
    (hdirectNonnegative : 0 ≤ direct)
    (hstate : state = 0) (hresolution : resolution = 0)
    (hdirect : direct ≤ coefficient * state + resolution) :
    direct = 0 := by
  subst state
  subst resolution
  simp at hdirect
  exact le_antisymm hdirect hdirectNonnegative

/-- A stationary transverse radius and resolution charge give the stated
near-collapse radius. -/
theorem forced_radius_bound
    {direct coefficient state forcedRadius resolution : ℝ}
    (hcoefficient : 0 ≤ coefficient)
    (hstate : state ≤ forcedRadius)
    (hdirect : direct ≤ coefficient * state + resolution) :
    direct ≤ coefficient * forcedRadius + resolution :=
  finite_master_bound hcoefficient hstate hdirect

end CollapseMaster
end PldrLlmCurvatureSandpile
