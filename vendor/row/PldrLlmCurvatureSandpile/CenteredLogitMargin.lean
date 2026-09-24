/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib
import PldrLlmCurvatureSandpile.ApplicationMargin

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace CenteredLogitMargin

noncomputable def center {Vocabulary : Type*} [Fintype Vocabulary]
    (logit : Vocabulary → ℝ) : Vocabulary → ℝ :=
  fun token =>
    logit token
      - (∑ other, logit other) / (Fintype.card Vocabulary : ℝ)

/-- Centering preserves every pairwise logit difference exactly. -/
theorem centered_pair_difference
    {Vocabulary : Type*} [Fintype Vocabulary]
    (logit : Vocabulary → ℝ) (left right : Vocabulary) :
    center logit left - center logit right = logit left - logit right := by
  simp [center]

/-- The sharp paired perturbation condition preserves the selected winner. -/
theorem sharp_centered_margin
    {winner other winnerChange otherChange margin normError : ℝ}
    (hmargin : other + margin ≤ winner)
    (hpair :
      |winnerChange - otherChange| ≤ Real.sqrt 2 * normError)
    (hsharp : Real.sqrt 2 * normError < margin) :
    other + otherChange < winner + winnerChange :=
  ApplicationMargin.sharp_sqrt_two_margin hmargin hpair hsharp

/-- The two-coordinate witness with entries one and minus one attains the
square-root-two conversion: its pair difference is two and its Euclidean
norm is square root two. -/
theorem sqrt_two_attained :
    |(1 : ℝ) - (-1)|
      = Real.sqrt 2 * Real.sqrt ((1 : ℝ) ^ 2 + (-1 : ℝ) ^ 2) := by
  have hsqrt : Real.sqrt 2 * Real.sqrt 2 = (2 : ℝ) := by
    rw [Real.mul_self_sqrt]
    norm_num
  norm_num [hsqrt]

end CenteredLogitMargin
end PldrLlmCurvatureSandpile

