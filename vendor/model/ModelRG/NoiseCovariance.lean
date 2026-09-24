import ModelRG.Heads

/-! Selected finite second-moment checks for covariance transport.
For centered variables under normalized nonnegative weights these are variance
identities. Probability, centering, adaptation and native closure are not
asserted by this purely algebraic interface. -/

open scoped BigOperators
noncomputable section
namespace ModelRG

theorem weighted_second_moment_transport {ι : Type*} [Fintype ι]
    (w x e : ι → ℝ) (f : ℝ) :
    (∑ i, w i * (f * x i + e i)^2) =
      f^2 * (∑ i, w i * (x i)^2) + (∑ i, w i * (e i)^2) +
      2 * f * (∑ i, w i * x i * e i) := by
  calc
    (∑ i, w i * (f * x i + e i)^2) =
        ∑ i, (f^2 * (w i * (x i)^2) + w i * (e i)^2 +
          2 * f * (w i * x i * e i)) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum]

theorem weighted_second_moment_orthogonal {ι : Type*} [Fintype ι]
    (w x e : ι → ℝ) (f : ℝ) (hcross : ∑ i, w i * x i * e i = 0) :
    (∑ i, w i * (f * x i + e i)^2) =
      f^2 * (∑ i, w i * (x i)^2) + (∑ i, w i * (e i)^2) := by
  rw [weighted_second_moment_transport, hcross]
  ring

end ModelRG

#print axioms ModelRG.weighted_second_moment_transport
#print axioms ModelRG.weighted_second_moment_orthogonal
