import Mathlib

namespace ModelRG.CacheRisk
open scoped BigOperators

/-- Finite weighted cache error splits into scatter and calibration displacement.
This is algebra; nonnegative weights are needed for the risk interpretation. -/
theorem empirical_budget {ι : Type*} [Fintype ι]
    (w x : ι → ℝ) (mu c : ℝ)
    (hw : ∑ i, w i = 1) (hm : ∑ i, w i * x i = mu) :
    (∑ i, w i * (x i - c)^2) =
      (∑ i, w i * (x i - mu)^2) + (mu - c)^2 := by
  calc
    (∑ i, w i * (x i - c)^2) =
        (∑ i, w i * (x i - mu)^2) +
        2 * (mu - c) * (∑ i, w i * x i) +
        (c^2 - mu^2) * (∑ i, w i) := by
      simp_rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by rw [hw, hm]; ring

/-- Fixed linear averaging commutes with the scalar sign action. -/
theorem cache_sign {ι : Type*} [Fintype ι] (a x : ι → ℝ) (sign : ℝ) :
    (∑ i, a i * (sign * x i)) = sign * ∑ i, a i * x i := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Finite Euclidean empirical cache budget, with a coordinatewise weighted mean.
Nonnegative weights are needed for the statistical interpretation, not this identity. -/
theorem empirical_budget_euclidean {ι κ : Type*} [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : ι → κ → ℝ) (mu c : κ → ℝ)
    (hw : ∑ i, w i = 1) (hm : ∀ k, ∑ i, w i * x i k = mu k) :
    (∑ i, w i * ∑ k, (x i k - c k)^2) =
      (∑ i, w i * ∑ k, (x i k - mu k)^2) + ∑ k, (mu k - c k)^2 := by
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm, Finset.sum_comm (s := Finset.univ) (t := Finset.univ) (f := fun i k => w i * (x i k - mu k)^2)]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  exact ModelRG.CacheRisk.empirical_budget w (fun i => x i k) (mu k) (c k) hw (hm k)

/-- Initial-to-current cache displacement, coordinate by coordinate. -/
theorem state_displacement (current initial cache : ℝ) :
    (current-cache)^2 = (current-initial)^2 + (initial-cache)^2 +
      2*(current-initial)*(initial-cache) := by ring

/-- Finite Euclidean displacement keeps the signed cross term. -/
theorem state_displacement_euclidean {κ : Type*} [Fintype κ]
    (current initial cache : κ → ℝ) :
    (∑ k, (current k - cache k)^2) =
      (∑ k, (current k - initial k)^2) +
      (∑ k, (initial k - cache k)^2) +
      2 * ∑ k, (current k - initial k) * (initial k - cache k) := by
  calc
    (∑ k, (current k - cache k)^2) =
        ∑ k, ((current k - initial k)^2 + (initial k - cache k)^2 +
          2 * ((current k - initial k) * (initial k - cache k))) := by
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by simp only [Finset.sum_add_distrib, Finset.mul_sum]

/-- Full finite cache risk transport. Nonnegative weights supply the
statistical interpretation but are unnecessary for the algebraic equality. -/
theorem empirical_state_transport_euclidean {ι κ : Type*}
    [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : ι → κ → ℝ) (current initial cache : κ → ℝ)
    (hw : ∑ i, w i = 1)
    (hm : ∀ k, ∑ i, w i * x i k = current k) :
    (∑ i, w i * ∑ k, (x i k - cache k)^2) =
      (∑ i, w i * ∑ k, (x i k - current k)^2) +
      (∑ k, (current k - initial k)^2) +
      (∑ k, (initial k - cache k)^2) +
      2 * ∑ k, (current k - initial k) * (initial k - cache k) := by
  rw [ModelRG.CacheRisk.empirical_budget_euclidean w x current cache hw hm]
  rw [state_displacement_euclidean current initial cache]
  ring


end ModelRG.CacheRisk
