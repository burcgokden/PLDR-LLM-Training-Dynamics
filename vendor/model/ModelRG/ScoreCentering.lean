import Mathlib

/-! Finite algebra for the fixed-mean moment-score decomposition.
The manuscript supplies the stand-alone centered-vector proof.
There is no formal claim about population moments, positive definiteness,
log determinants, inverse matrices, native execution, or statistical validity. -/
noncomputable section
open scoped BigOperators
namespace ModelRG

private theorem weighted_cross_about {ι : Type*} [Fintype ι]
    (w x y : ι → ℝ) (mx my a b : ℝ)
    (hw : ∑ i, w i = 1) (hx : ∑ i, w i * x i = mx)
    (hy : ∑ i, w i * y i = my) :
    (∑ i, w i * (x i - a) * (y i - b)) =
      (∑ i, w i * x i * y i) - b * mx - a * my + a * b := by
  have hi (i : ι) : w i * (x i - a) * (y i - b) =
      w i * x i * y i - b * (w i * x i) - a * (w i * y i) +
        (a * b) * w i := by ring
  simp_rw [hi, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  simp only [← Finset.mul_sum, hw, hx, hy, mul_one]

/-- Finite centered bilinear identity with the fitted-mean error retained. -/
theorem weighted_cross_centering {ι : Type*} [Fintype ι]
    (w x y : ι → ℝ) (mx my a b : ℝ)
    (hw : ∑ i, w i = 1) (hx : ∑ i, w i * x i = mx)
    (hy : ∑ i, w i * y i = my) :
    (∑ i, w i * (x i - a) * (y i - b)) =
      (∑ i, w i * (x i - mx) * (y i - my)) + (mx - a) * (my - b) := by
  rw [weighted_cross_about w x y mx my a b hw hx hy,
      weighted_cross_about w x y mx my mx my hw hx hy]
  ring

/-- Coordinate form of a finite quadratic score's covariance/mean split.
The coefficient array is arbitrary; it may later be interpreted as a
precision difference after the manuscript supplies analytic hypotheses. -/
theorem finite_quadratic_centering {ι κ : Type*} [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : ι → κ → ℝ) (mean fitted : κ → ℝ)
    (a : κ → κ → ℝ) (hw : ∑ i, w i = 1)
    (hm : ∀ j, ∑ i, w i * x i j = mean j) :
    (∑ j, ∑ k, a j k *
      (∑ i, w i * (x i j - fitted j) * (x i k - fitted k))) =
    (∑ j, ∑ k, a j k *
      (∑ i, w i * (x i j - mean j) * (x i k - mean k))) +
    (∑ j, ∑ k, a j k * ((mean j - fitted j) * (mean k - fitted k))) := by
  have h (j k : κ) := weighted_cross_centering w (fun i => x i j)
    (fun i => x i k) (mean j) (mean k) (fitted j) (fitted k) hw (hm j) (hm k)
  simp_rw [h, mul_add, Finset.sum_add_distrib]

end ModelRG
