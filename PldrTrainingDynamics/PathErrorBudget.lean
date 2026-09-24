import PldrTrainingDynamics.LimitCriteria

/-!
Manuscript correspondence: `model:cor:row-path-error`.

Checked clause and supplied hypotheses: Finite scalar telescoping with assumed one-step errors, nonnegative full-budget comparison, additive remainder, and normalized convergence under positive scales; scalar ratio invariance under nonzero scaling.

Written obligations: No matrix differentiability, orthogonal-projection identity, generator remainder estimate, or native scaling law is inferred.
These kernels double-check selected clauses; the written proof is independent.
-/

/-! Prefix error from assumed one-step analytic bounds. No differentiability,
projection formula, or generator remainder estimate is inferred here. -/
namespace PldrTrainingDynamics.PathErrorBudget
open scoped BigOperators
open Filter Topology

theorem prefix_budget (u d b : ℕ → ℝ)
    (h : ∀ k, |u (k+1) - u k - d k| ≤ b k) (n : ℕ) :
    |u n - u 0 - ∑ k ∈ Finset.range n, d k| ≤ ∑ k ∈ Finset.range n, b k := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    calc
      |u (n+1) - u 0 - ((∑ k ∈ Finset.range n, d k) + d n)| =
          |(u n - u 0 - ∑ k ∈ Finset.range n, d k) + (u (n+1) - u n - d n)| := by congr 1 <;> ring
      _ ≤ |u n - u 0 - ∑ k ∈ Finset.range n, d k| + |u (n+1) - u n - d n| := abs_add_le _ _
      _ ≤ (∑ k ∈ Finset.range n, b k) + b n := add_le_add ih (h n)

theorem prefix_le_full (b : ℕ → ℝ) (hb : ∀ k, 0 ≤ b k) {m n : ℕ} (hmn : m ≤ n) :
    (∑ k ∈ Finset.range m, b k) ≤ ∑ k ∈ Finset.range n, b k := by
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hmn) (fun k _ _ => hb k)

theorem add_remainder (e r b c : ℝ) (he : |e| ≤ b) (hr : |r| ≤ c) :
    |e+r| ≤ b+c := le_trans (abs_add_le e r) (add_le_add he hr)

theorem scaled_budget_suffices (e b s : ℕ → ℝ) (hs : ∀ n, 0 < s n)
    (h : ∀ n, |e n| ≤ b n) (hb : Tendsto (fun n => b n / s n) atTop (𝓝 0)) :
    Tendsto (fun n => |e n| / s n) atTop (𝓝 0) :=
  LimitCriteria.envelope_suffices _ _
    (fun n => div_nonneg (abs_nonneg _) (le_of_lt (hs n)))
    (fun n => div_le_div_of_nonneg_right (h n) (le_of_lt (hs n))) hb

/-- Scalar energy-ratio kernel; identifying both squared norms' scaling is written analysis. -/
theorem ratio_rescale (c e a : ℝ) (ha : a ≠ 0) :
    (a^2*c)/(a^2*e) = c/e := by
  exact mul_div_mul_left c e (pow_ne_zero _ ha)

end PldrTrainingDynamics.PathErrorBudget
