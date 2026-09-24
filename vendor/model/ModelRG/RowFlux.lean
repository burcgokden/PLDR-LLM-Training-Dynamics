import Mathlib

/-! Finite row-energy algebra. The finite cross term uses the outgoing
energy denominator. The matrix directional derivative uses the incoming
energy denominator. A parameter directional derivative additionally requires
the generator Jacobian and the chain rule. Analytic derivatives, remainder
estimates, probability limits and native execution are outside this module. -/
namespace ModelRG.RowFlux

noncomputable section

open scoped BigOperators

/-- Exact finite increment: both right-hand terms use outgoing energy.
The first is the finite cross term, not a matrix directional derivative. -/
theorem row_ratio_identity (E C a b r s : ℝ)
    (hE : E ≠ 0) (hNext : E + 2*r + s ≠ 0) :
    (C + 2*a + b) / (E + 2*r + s) - C/E =
      (2*a - 2*(C/E)*r) / (E + 2*r + s) +
      (b - (C/E)*s) / (E + 2*r + s) := by
  field_simp
  ring

/-- Rational difference between the finite increment `(L+Q)/E'` and `L/E`.
Identifying `L/E` with a matrix directional derivative is written analysis;
this theorem neither differentiates a generator nor bounds its remainder. -/
theorem derivative_defect_identity (E E' L Q : ℝ)
    (hE : E ≠ 0) (hNext : E' ≠ 0) :
    (L+Q)/E' - L/E = Q/E' - ((E'-E)/E')*(L/E) := by
  field_simp
  ring

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

def energy (A : ι → κ → ℝ) : ℝ := ∑ i, ∑ j, (A i j)^2
def pairing (A D : ι → κ → ℝ) : ℝ := ∑ i, ∑ j, A i j * D i j
def center (A : ι → κ → ℝ) (i : ι) (j : κ) : ℝ :=
  A i j - (∑ r, A r j) / (Fintype.card ι : ℝ)

omit [Fintype κ] in
theorem center_add (A D : ι → κ → ℝ) :
    center (fun i j => A i j + D i j) = fun i j => center A i j + center D i j := by
  funext i j
  simp only [center, Finset.sum_add_distrib, add_div]
  ring

theorem two_increment_cross (A D : ι → κ → ℝ) :
    energy (fun i j => A i j + D i j) = energy A + energy D + 2 * pairing A D := by
  simp only [energy, pairing, add_sq, Finset.sum_add_distrib, Finset.mul_sum]
  ring

theorem matrix_row_identity (A D : ι → κ → ℝ)
    (hE : energy A ≠ 0) (hNext : energy (fun i j => A i j + D i j) ≠ 0) :
    energy (center (fun i j => A i j + D i j)) / energy (fun i j => A i j + D i j) -
      energy (center A) / energy A =
    (2 * pairing (center A) (center D) - 2 * (energy (center A) / energy A) * pairing A D) /
      energy (fun i j => A i j + D i j) +
    (energy (center D) - (energy (center A) / energy A) * energy D) /
      energy (fun i j => A i j + D i j) := by
  rw [center_add, two_increment_cross, two_increment_cross]
  have hn : energy A + 2 * pairing A D + energy D ≠ 0 := by
    rw [two_increment_cross] at hNext
    convert hNext using 1 <;> ring
  convert row_ratio_identity (energy A) (energy (center A))
    (pairing (center A) (center D)) (energy (center D)) (pairing A D) (energy D) hE hn using 1 <;> ring

/-- Fixed weights preserve finite identities. Nonnegative normalized weights
are additional hypotheses when transferring analytic absolute-error bounds. -/
theorem weighted_increment {σ : Type*} [Fintype σ]
    (w before after cross quadratic : σ → ℝ)
    (h : ∀ i, after i - before i = cross i + quadratic i) :
    (∑ i, w i * after i) - (∑ i, w i * before i) =
      (∑ i, w i * cross i) + (∑ i, w i * quadratic i) := by
  rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [← mul_sub, h i, mul_add]

/-- Chronological endpoint identity; differential-error and scaling estimates
need separate analytic hypotheses and are not part of this export. -/
theorem finite_telescope (u : ℕ → ℝ) (n : ℕ) :
    (∑ i ∈ Finset.range n, (u (i+1)-u i)) = u n-u 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    ring

end

end ModelRG.RowFlux
