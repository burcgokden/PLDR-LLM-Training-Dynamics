import Mathlib

/-!
Manuscript correspondence: `row:cor:linear-stochastic-forcing`.

Checked clause and supplied hypotheses: Finite weighted Hilbert-space second moments. Normalized nonnegative weights, zero vector means, and vanishing expected pairwise inner products are explicit hypotheses. Deterministic transports are absorbed in the increments.

Written obligations: General conditional expectation, martingale-to-orthogonality, and random state-dependent propagators are not formalized.
These kernels double-check selected clauses; the written proof is independent.
-/

/-! Finite probability-space kernels. Deterministic transports are absorbed in
`X`. Zero means and pairwise expected orthogonality are explicit hypotheses;
conditional expectations and native state-dependent transport are not inferred. -/
namespace PldrTrainingDynamics.StochasticTransport
open scoped BigOperators
variable {Ω H : Type*} [Fintype Ω] [NormedAddCommGroup H] [InnerProductSpace ℝ H]

noncomputable def avg (w : Ω → ℝ) (f : Ω → ℝ) : ℝ := ∑ ω, w ω * f ω

theorem weighted_add_sq (w : Ω → ℝ) (x y : Ω → H) :
    avg w (fun ω => ‖x ω + y ω‖ ^ 2) =
      avg w (fun ω => ‖x ω‖ ^ 2) +
      2 * avg w (fun ω => inner (𝕜 := ℝ) (x ω) (y ω)) +
      avg w (fun ω => ‖y ω‖ ^ 2) := by
  unfold avg
  simp_rw [norm_add_sq_real, mul_add]
  simp only [Finset.sum_add_distrib]
  congr 1
  congr 1
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _
  ring

theorem centered_second_moment (w : Ω → ℝ) (X : ℕ → Ω → H)
    (orth : ∀ i j, i < j → avg w (fun ω => inner (𝕜 := ℝ) (X i ω) (X j ω)) = 0)
    (n : ℕ) :
    avg w (fun ω => ‖∑ i ∈ Finset.range n, X i ω‖ ^ 2) =
      ∑ i ∈ Finset.range n, avg w (fun ω => ‖X i ω‖ ^ 2) := by
  induction n with
  | zero => simp [avg]
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    rw [weighted_add_sq, ih]
    have hz : avg w (fun ω => inner (𝕜 := ℝ)
        (∑ i ∈ Finset.range n, X i ω) (X n ω)) = 0 := by
      simp only [avg, sum_inner, Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_eq_zero
      intro i hi
      exact orth i n (Finset.mem_range.mp hi)
    rw [hz]
    ring

theorem mean_plus_variance (w : Ω → ℝ) (hw : ∀ ω, 0 ≤ w ω)
    (hn : ∑ ω, w ω = 1) (μ : H) (X : ℕ → Ω → H)
    (centered : ∀ i, ∑ ω, w ω • X i ω = 0)
    (orth : ∀ i j, i < j → avg w (fun ω => inner (𝕜 := ℝ) (X i ω) (X j ω)) = 0)
    (n : ℕ) :
    avg w (fun ω => ‖μ + ∑ i ∈ Finset.range n, X i ω‖ ^ 2) =
      ‖μ‖ ^ 2 + ∑ i ∈ Finset.range n, avg w (fun ω => ‖X i ω‖ ^ 2) := by
  have hmean : ∑ ω, w ω • (∑ i ∈ Finset.range n, X i ω) = 0 := by
    simp only [Finset.smul_sum]
    rw [Finset.sum_comm]
    simp [centered]
  have hcross : avg w (fun ω => inner (𝕜 := ℝ) μ (∑ i ∈ Finset.range n, X i ω)) = 0 := by
    simp only [avg, ← inner_smul_right, ← inner_sum, hmean, inner_zero_right]
  have hconstant : avg w (fun _ => ‖μ‖ ^ 2) = ‖μ‖ ^ 2 := by
    simp [avg, ← Finset.sum_mul, hn]
  rw [weighted_add_sq, hconstant, hcross, centered_second_moment w X orth n]
  ring

end PldrTrainingDynamics.StochasticTransport
