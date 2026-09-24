/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact PLGA divided-difference transfer
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PLGADividedDifference

/-- Endpoint divided difference with the supplied derivative on the
diagonal. -/
noncomputable def dividedDifference
    (function derivative : ℝ → ℝ) (left right : ℝ) : ℝ :=
  if left = right then derivative left
  else (function right - function left) / (right - left)

/-- The divided difference reconstructs every endpoint difference exactly. -/
theorem divided_difference_exact
    (function derivative : ℝ → ℝ) (left right : ℝ) :
    function right - function left
      = dividedDifference function derivative left right * (right - left) := by
  by_cases hequal : left = right
  · subst right
    simp [dividedDifference]
  · simp [dividedDifference, hequal]
    field_simp [sub_ne_zero.mpr hequal]

/-- On the diagonal, the chosen derivative is returned by definition. -/
theorem divided_difference_diagonal
    (function derivative : ℝ → ℝ) (point : ℝ) :
    dividedDifference function derivative point point = derivative point := by
  simp [dividedDifference]

/-- Composing two coordinatewise PLGA nonlinearities multiplies their exact
occupied-interval secants. -/
theorem divided_difference_chain
    (outer outerDerivative inner innerDerivative : ℝ → ℝ)
    (left right : ℝ) :
    outer (inner right) - outer (inner left)
      = dividedDifference outer outerDerivative (inner left) (inner right)
        * dividedDifference inner innerDerivative left right
        * (right - left) := by
  rw [divided_difference_exact outer outerDerivative]
  rw [divided_difference_exact inner innerDerivative]
  ring

/-- A following affine PLGA coupling multiplies the secant transfer by its
coupling coefficient; its bias cancels. -/
theorem affine_secant_transfer
    (coupling bias : ℝ) (function derivative : ℝ → ℝ)
    (left right : ℝ) :
    (coupling * function right + bias)
        - (coupling * function left + bias)
      = coupling * dividedDifference function derivative left right
        * (right - left) := by
  calc
    coupling * function right + bias - (coupling * function left + bias)
        = coupling * (function right - function left) := by ring
    _ = coupling * dividedDifference function derivative left right
          * (right - left) := by
      calc
        coupling * (function right - function left)
            = coupling * (dividedDifference function derivative left right
              * (right - left)) :=
          congrArg (fun value => coupling * value)
            (divided_difference_exact function derivative left right)
        _ = _ := by ring

end PLGADividedDifference
end PldrLlmCurvatureSandpile
