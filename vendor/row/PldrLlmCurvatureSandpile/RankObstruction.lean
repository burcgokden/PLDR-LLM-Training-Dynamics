/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Ambient rank obstruction

The complete row-Jacobian stack need not be a parameter chart. This module
checks the finite-dimensional obstruction behind that design constraint.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace RankObstruction

/-- An endomorphism less than one operator-norm unit from the identity is
surjective. This is the Neumann-series step used by the rank obstruction. -/
theorem near_identity_surjective
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E]
    (T : E →L[ℝ] E)
    (hnear : ‖(1 : E →L[ℝ] E) - T‖ < 1) :
    Function.Surjective T := by
  have hunit : IsUnit (1 - ((1 : E →L[ℝ] E) - T)) :=
    isUnit_one_sub_of_norm_lt_one hnear
  have hT : IsUnit T := by
    simpa only [sub_sub_cancel] using hunit
  exact (ContinuousLinearMap.isUnit_iff_bijective.mp hT).2

/-- If B composed with R is within one operator-norm unit of the identity on
the observable space, then that space cannot have larger finite dimension
than parameter space. -/
theorem ambient_right_inverse_finrank_le
    {Parameter Observable : Type*}
    [NormedAddCommGroup Parameter] [NormedSpace ℝ Parameter]
    [NormedAddCommGroup Observable] [NormedSpace ℝ Observable]
    [CompleteSpace Observable]
    [FiniteDimensional ℝ Parameter] [FiniteDimensional ℝ Observable]
    (B : Parameter →L[ℝ] Observable)
    (R : Observable →L[ℝ] Parameter)
    (hnear : ‖(1 : Observable →L[ℝ] Observable) - B.comp R‖ < 1) :
    Module.finrank ℝ Observable ≤ Module.finrank ℝ Parameter := by
  have hBR : Function.Surjective (B.comp R) :=
    near_identity_surjective (B.comp R) hnear
  have hB : Function.Surjective B := by
    intro y
    obtain ⟨x, hx⟩ := hBR y
    exact ⟨R x, hx⟩
  exact B.toLinearMap.finrank_le_finrank_of_surjective hB

end RankObstruction
end PldrLlmCurvatureSandpile
