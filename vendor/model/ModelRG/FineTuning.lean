import Mathlib

/-! Finite identities for source mixtures, paired discrepancies and coupled
sectors. These statements do not assert native thermodynamic limits. -/

open scoped BigOperators
namespace ModelRG

theorem bernoulli_two_step_polynomial (r a b c d : ℝ) :
    (1-r)^2*a + r*(1-r)*(b+c) + r^2*d =
      a + r*(b+c-2*a) + r^2*(a-b-c+d) := by ring

theorem bernoulli_score_numerator (r a b : ℝ) :
    (1-r)*a*(-r) + r*b*(1-r) = r*(1-r)*(b-a) := by ring

theorem paired_discrepancy_centering {ι : Type*} [Fintype ι]
    (w x y : ι → ℝ) (mx my : ℝ)
    (hw : ∑ i, w i = 1)
    (hx : ∑ i, w i * (x i-mx) = 0)
    (hy : ∑ i, w i * (y i-my) = 0) :
    (∑ i, w i * (x i-y i)^2) =
      (∑ i, w i * (x i-mx)^2) + (∑ i, w i * (y i-my)^2) -
      2*(∑ i, w i*(x i-mx)*(y i-my)) + (mx-my)^2 := by
  have h : ∀ i, w i * (x i-y i)^2 =
      w i*(x i-mx)^2 + w i*(y i-my)^2 - 2*(w i*(x i-mx)*(y i-my)) +
      2*(mx-my)*(w i*(x i-mx)) - 2*(mx-my)*(w i*(y i-my)) +
      (mx-my)^2*w i := by intro i; ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hx, hy, hw]
  ring

theorem source_quadratic_knots (a b c : ℝ) :
    (1-(0:ℝ))*a+0*c+4*0*(1-0)*(b-(a+c)/2) = a ∧
    (1-(1/2:ℝ))*a+(1/2)*c+4*(1/2)*(1-1/2)*(b-(a+c)/2) = b ∧
    (1-(1:ℝ))*a+1*c+4*1*(1-1)*(b-(a+c)/2) = c := by
  constructor
  · ring
  constructor <;> ring

theorem two_source_gram_determinant (x₁ x₂ y₁ y₂ : ℝ) :
    (x₁^2+x₂^2)*(y₁^2+y₂^2) - (x₁*y₁+x₂*y₂)^2 =
      (x₁*y₂-x₂*y₁)^2 := by ring

theorem coupled_gap_characteristic (a b g z : ℝ) :
    (a-z)*(b-z)-g^2 = z^2-(a+b)*z+a*b-g^2 := by ring

theorem two_zero_positive_gaps (a b g : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hdet : g^2 ≤ a*b)
    (htrace : a+b = 0) : a = 0 ∧ b = 0 ∧ g = 0 := by
  have a0 : a = 0 := by linarith
  have b0 : b = 0 := by linarith
  have g0 : g = 0 := by rw [a0, b0] at hdet; nlinarith [sq_nonneg g]
  exact ⟨a0,b0,g0⟩

end ModelRG
