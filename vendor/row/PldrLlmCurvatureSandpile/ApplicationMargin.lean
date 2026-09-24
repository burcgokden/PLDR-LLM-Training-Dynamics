/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Same-source logit margin
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ApplicationMargin

/-- A reference winner remains strictly above one competitor when their
paired perturbation difference is smaller than the reference margin. -/
theorem paired_logit_margin
    {winner other winnerChange otherChange margin error : ℝ}
    (hmargin : other + margin ≤ winner)
    (hpair : |winnerChange - otherChange| ≤ error)
    (herror : error < margin) :
    other + otherChange < winner + winnerChange := by
  rw [abs_le] at hpair
  linarith

/-- The sharp square-root-two Euclidean conversion is sufficient for the
same paired margin conclusion. -/
theorem sharp_sqrt_two_margin
    {winner other winnerChange otherChange margin normError : ℝ}
    (hmargin : other + margin ≤ winner)
    (hpair :
      |winnerChange - otherChange| ≤ Real.sqrt 2 * normError)
    (hsharp : Real.sqrt 2 * normError < margin) :
    other + otherChange < winner + winnerChange := by
  exact paired_logit_margin hmargin hpair hsharp

end ApplicationMargin
end PldrLlmCurvatureSandpile
