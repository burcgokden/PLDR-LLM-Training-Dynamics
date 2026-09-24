/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PairwiseDiameterCollapse

def pairEnvelope {Pair : Type*}
    (factor forcing : ℕ → Pair → ℝ) (initial : Pair → ℝ) :
    ℕ → Pair → ℝ
  | 0 => initial
  | step + 1 =>
      fun pair =>
        factor step pair * pairEnvelope factor forcing initial step pair
          + forcing step pair

/-- Each pair is iterated in chronological order with its own coefficient. -/
theorem pairwise_envelope_comparison
    {Pair : Type*}
    (energy factor forcing : ℕ → Pair → ℝ)
    (hfactor : ∀ step pair, 0 ≤ factor step pair)
    (hstep : ∀ step pair,
      energy (step + 1) pair
        ≤ factor step pair * energy step pair + forcing step pair) :
    ∀ step pair,
      energy step pair
        ≤ pairEnvelope factor forcing (energy 0) step pair := by
  intro step
  induction step with
  | zero =>
      intro pair
      simp [pairEnvelope]
  | succ step inductionHypothesis =>
      intro pair
      rw [pairEnvelope]
      exact (hstep step pair).trans <|
        add_le_add
          (mul_le_mul_of_nonneg_left
            (inductionHypothesis pair) (hfactor step pair))
          le_rfl

/-- A common upper bound on all pair envelopes is a squared-diameter bound. -/
theorem all_pairs_to_diameter
    {Pair : Type*}
    (distanceSq envelope : Pair → ℝ) (diameterSq bound : ℝ)
    (hdistance : ∀ pair, distanceSq pair ≤ envelope pair)
    (henvelope : ∀ pair, envelope pair ≤ bound)
    (hdiameter : ∃ pair, diameterSq = distanceSq pair) :
    diameterSq ≤ bound := by
  rcases hdiameter with ⟨pair, rfl⟩
  exact (hdistance pair).trans (henvelope pair)

/-- Uniform vanishing of pair envelopes forces every registered pair to
vanish, in the explicit epsilon form used by the master theorem. -/
theorem uniform_pair_collapse
    {Pair : Type*}
    (distanceSq envelope : ℕ → Pair → ℝ)
    (hdistance : ∀ step pair, distanceSq step pair ≤ envelope step pair)
    (henvelope : ∀ epsilon > 0, ∃ threshold, ∀ step ≥ threshold, ∀ pair,
      envelope step pair < epsilon) :
    ∀ epsilon > 0, ∃ threshold, ∀ step ≥ threshold, ∀ pair,
      distanceSq step pair < epsilon := by
  intro epsilon hepsilon
  rcases henvelope epsilon hepsilon with ⟨threshold, hthreshold⟩
  exact ⟨threshold, fun step hstep pair =>
    (hdistance step pair).trans_lt (hthreshold step hstep pair)⟩

end PairwiseDiameterCollapse
end PldrLlmCurvatureSandpile

