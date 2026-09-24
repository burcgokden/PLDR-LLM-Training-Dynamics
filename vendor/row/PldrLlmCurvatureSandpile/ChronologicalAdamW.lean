/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ChronologicalAdamW

/-- Newest-first finite first-moment unrolling with a stored tail state. -/
def chronologicalMoment {d : ℕ} (beta : ℝ) :
    List (Fin d → ℝ) → (Fin d → ℝ) → Fin d → ℝ
  | [], tail => tail
  | gradient :: history, tail =>
      fun coordinate =>
        (1 - beta) * gradient coordinate
          + beta * chronologicalMoment beta history tail coordinate

/-- The contribution of the explicitly retained gradient history. -/
def historyContribution {d : ℕ} (beta : ℝ) :
    List (Fin d → ℝ) → Fin d → ℝ
  | [] => fun _ => 0
  | gradient :: history =>
      fun coordinate =>
        (1 - beta) * gradient coordinate
          + beta * historyContribution beta history coordinate

/-- Exact split into signed retained history and the geometrically weighted
stored tail. -/
theorem chronological_moment_split {d : ℕ}
    (beta : ℝ) (history : List (Fin d → ℝ)) (tail : Fin d → ℝ)
    (coordinate : Fin d) :
    chronologicalMoment beta history tail coordinate
      = historyContribution beta history coordinate
        + beta ^ history.length * tail coordinate := by
  induction history with
  | nil =>
      simp [chronologicalMoment, historyContribution]
  | cons gradient history inductionHypothesis =>
      simp only [chronologicalMoment, historyContribution, List.length_cons]
      rw [inductionHypothesis, pow_succ]
      ring

/-- Applying the same diagonal preconditioner and bias correction preserves
the signed chronological split exactly. -/
theorem common_preconditioner_split {d : ℕ}
    (beta biasCorrection : ℝ)
    (history : List (Fin d → ℝ)) (tail preconditioner : Fin d → ℝ)
    (coordinate : Fin d) :
    preconditioner coordinate
        * (chronologicalMoment beta history tail coordinate / biasCorrection)
      = preconditioner coordinate
          * (historyContribution beta history coordinate / biasCorrection)
        + preconditioner coordinate
          * (beta ^ history.length * tail coordinate / biasCorrection) := by
  rw [chronological_moment_split]
  ring

/-- The first update has the intended newest-gradient indexing. -/
theorem first_update_index {d : ℕ}
    (beta : ℝ) (gradient tail : Fin d → ℝ) (coordinate : Fin d) :
    chronologicalMoment beta [gradient] tail coordinate
      = (1 - beta) * gradient coordinate + beta * tail coordinate := by
  simp [chronologicalMoment]

end ChronologicalAdamW
end PldrLlmCurvatureSandpile

