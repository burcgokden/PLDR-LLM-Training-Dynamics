import Mathlib

/-!
Manuscript correspondence: `row:cor:linear-stochastic-forcing`, `model:cor:row-path-error`.

Checked clause and supplied hypotheses: Explicit cancellation recurrence; diag(1,0) noise-killing recurrence and surviving first direction; centered two-sign moments and positive radial error budget.

Written obligations: The operator-norm envelope and analytic directional derivative in these counterexamples have stand-alone written proofs; a native probabilistic limit is not claimed.
These kernels double-check selected clauses; the written proof is independent.
-/

namespace PldrTrainingDynamics.CancellationExamples

def cancelled (n : ℕ) : ℝ := if n = 0 then 1 else 0
def bias (n : ℕ) : ℝ := if n = 0 then -1 else 0

theorem cancellation_recurrence (n : ℕ) : cancelled (n+1) = cancelled n + bias n := by
  by_cases h : n = 0 <;> simp [cancelled, bias, h]

theorem cancellation_components (n : ℕ) (h : 1 ≤ n) :
    cancelled n = (1 : ℝ) + (-1) ∧ cancelled n = 0 := by
  have hn : n ≠ 0 := by omega
  simp [cancelled, hn]

/-- The map diag(1,0), with noise along e₂, kills that noise in one transport. -/
def project (x : ℝ × ℝ) : ℝ × ℝ := (x.1, 0)
def killedPath (ε : ℝ) (n : ℕ) : ℝ × ℝ := if n = 1 then (0, ε) else (0, 0)
def noise (ε : ℝ) (n : ℕ) : ℝ × ℝ := if n = 0 then (0, ε) else (0, 0)

theorem killed_noise_recurrence (ε : ℝ) (n : ℕ) :
    killedPath ε (n+1) = project (killedPath ε n) + noise ε n := by
  by_cases hn : n = 0
  · subst n; simp [killedPath, project, noise]
  · have hn1 : n+1 ≠ 1 := by omega
    by_cases h : n = 1 <;> simp [killedPath, project, noise, hn, hn1, h]

theorem killed_noise_tail (ε : ℝ) (n : ℕ) (h : 2 ≤ n) : killedPath ε n = (0,0) := by
  have hn : n ≠ 1 := by omega
  simp [killedPath, hn]

theorem centered_sign_variance : ((1 : ℝ) + (-1))/2 = 0 ∧
    ((1 : ℝ)^2 + (-1)^2)/2 = 1 := by norm_num

theorem retained_unit_direction : project (1,0) = (1,0) := rfl

theorem radial_budget_positive :
    (3*(1/2 : ℝ)^2 + (1/2 : ℝ)^3)/(1-(1/2 : ℝ))^2 = 7/2 ∧ (0 : ℝ) < 7/2 := by
  norm_num

end PldrTrainingDynamics.CancellationExamples
