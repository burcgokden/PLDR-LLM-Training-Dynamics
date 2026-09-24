/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Cumulative direct-energy telescope

The cumulative work-charge expression is exactly the energy sequence when
the one-step ledger holds.  Its convergence is therefore a restatement, not
an independent dynamical closure hypothesis.
-/
import Mathlib

open Filter
open scoped BigOperators Topology

namespace PldrLlmCurvatureSandpile
namespace DirectEnergyCumulative

/-- Finite cumulative work-charge ledger from a declared initial energy. -/
def cumulativeLedger
    (initial : ℝ) (work charge : ℕ → ℝ) (n : ℕ) : ℝ :=
  initial +
    (Finset.range n).sum (fun t => -work t + charge t)

/-- The finite cumulative ledger telescopes exactly to the endpoint energy. -/
theorem cumulative_ledger_eq_energy
    (energy work charge : ℕ → ℝ)
    (step : ∀ t, energy (t + 1) - energy t = -work t + charge t)
    (n : ℕ) :
    cumulativeLedger (energy 0) work charge n = energy n := by
  induction n with
  | zero => simp [cumulativeLedger]
  | succ n inductionHypothesis =>
      rw [cumulativeLedger, Finset.sum_range_succ]
      have previous : energy 0 + (Finset.range n).sum
          (fun t => -work t + charge t) = energy n := by
        simpa [cumulativeLedger] using inductionHypothesis
      rw [← add_assoc, previous]
      linarith [step n]

/-- Under the one-step identity, cumulative-ledger convergence to zero is
literally energy convergence to zero. -/
theorem cumulative_ledger_tendsto_zero_iff
    (energy work charge : ℕ → ℝ)
    (step : ∀ t, energy (t + 1) - energy t = -work t + charge t) :
    Tendsto (cumulativeLedger (energy 0) work charge) atTop (𝓝 0)
      ↔ Tendsto energy atTop (𝓝 0) := by
  have equality : cumulativeLedger (energy 0) work charge = energy := by
    funext n
    exact cumulative_ledger_eq_energy energy work charge step n
  rw [equality]

end DirectEnergyCumulative
end PldrLlmCurvatureSandpile
