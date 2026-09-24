import Mathlib

/-!
Manuscript correspondence: `row:cor:linear-stochastic-forcing`.

Checked clause and supplied hypotheses: Nonnegative real sequences: sum tends to zero iff both terms do; vanishing upper envelope suffices; separate scalar mean decay suffices.

Written obligations: The exact stochastic decomposition is a separate hypothesis; vector mean convergence and native attraction are not inferred.
These kernels double-check selected clauses; the written proof is independent.
-/

/-! Sequence limits from an assumed exact nonnegative decomposition. These
kernels do not establish a stochastic decomposition or native attraction. -/
namespace PldrTrainingDynamics.LimitCriteria
open Filter Topology

theorem nonnegative_sum_iff (m v : ℕ → ℝ) (hm : ∀ n, 0 ≤ m n) (hv : ∀ n, 0 ≤ v n) :
    Tendsto (fun n => m n + v n) atTop (𝓝 0) ↔
      Tendsto m atTop (𝓝 0) ∧ Tendsto v atTop (𝓝 0) := by
  constructor
  · intro h
    constructor
    · exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds h hm
        (fun n => le_add_of_nonneg_right (hv n))
    · exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds h hv
        (fun n => le_add_of_nonneg_left (hm n))
  · rintro ⟨h₁, h₂⟩
    simpa using h₁.add h₂

theorem envelope_suffices (v b : ℕ → ℝ) (hv : ∀ n, 0 ≤ v n)
    (hb : ∀ n, v n ≤ b n) (h : Tendsto b atTop (𝓝 0)) :
    Tendsto v atTop (𝓝 0) :=
  tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds h hv hb

theorem separate_decay_suffices (a b : ℕ → ℝ)
    (ha : Tendsto a atTop (𝓝 0)) (hb : Tendsto b atTop (𝓝 0)) :
    Tendsto (fun n => a n + b n) atTop (𝓝 0) := by
  simpa using ha.add hb

end PldrTrainingDynamics.LimitCriteria
