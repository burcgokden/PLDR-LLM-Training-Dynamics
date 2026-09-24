/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Three-source permutation-symmetric attribution
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace FiniteSourceShapley

/-- Shapley contribution of source A for three declared macro sources. -/
noncomputable def contributionA
    (f0 fa fb fc fab fac fbc fabc : ℝ) : ℝ :=
  (fa - f0) / 3 + ((fab - fb) + (fac - fc)) / 6
    + (fabc - fbc) / 3

/-- Shapley contribution of source B for three declared macro sources. -/
noncomputable def contributionB
    (f0 fa fb fc fab fac fbc fabc : ℝ) : ℝ :=
  (fb - f0) / 3 + ((fab - fa) + (fbc - fc)) / 6
    + (fabc - fac) / 3

/-- Shapley contribution of source C for three declared macro sources. -/
noncomputable def contributionC
    (f0 fa fb fc fab fac fbc fabc : ℝ) : ℝ :=
  (fc - f0) / 3 + ((fac - fa) + (fbc - fb)) / 6
    + (fabc - fab) / 3

/-- Efficiency: the three symmetric contributions reconstruct the complete
endpoint difference exactly. -/
theorem three_source_efficiency
    (f0 fa fb fc fab fac fbc fabc : ℝ) :
    contributionA f0 fa fb fc fab fac fbc fabc
      + contributionB f0 fa fb fc fab fac fbc fabc
      + contributionC f0 fa fb fc fab fac fbc fabc = fabc - f0 := by
  unfold contributionA contributionB contributionC
  ring

end FiniteSourceShapley
end PldrLlmCurvatureSandpile
