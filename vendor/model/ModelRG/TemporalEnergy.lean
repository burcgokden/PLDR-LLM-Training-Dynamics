import Mathlib

/-! Finite temporal-energy algebra in a real inner-product space.
Frobenius matrices are the finite Euclidean specialization. The normalized
physical statistic requires positive incoming energy. No analytic upper-bound,
predictive closure, critical limit, or native measurement is formalized here. -/

noncomputable section
open scoped InnerProductSpace
namespace ModelRG.TemporalEnergy

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

def diagonal (ds : List H) : ℝ := (ds.map (fun d => ‖d‖ ^ 2)).sum

/-- Twice the ordered pair sum. Expanding the tail sum gives i < j. -/
def cross : List H → ℝ
  | [] => 0
  | d :: ds => 2 * ⟪d, ds.sum⟫_ℝ + cross ds

theorem energy_identity (ds : List H) :
    ‖ds.sum‖ ^ 2 = diagonal ds + cross ds := by
  induction ds with
  | nil => simp [diagonal, cross]
  | cons d ds ih =>
    simp only [List.sum_cons, norm_add_sq_real, cross, diagonal,
      List.map_cons, List.sum_cons] at *
    linarith

omit [InnerProductSpace ℝ H] in
theorem diagonal_append (xs ys : List H) :
    diagonal (xs ++ ys) = diagonal xs + diagonal ys := by
  simp [diagonal]

theorem cross_append (xs ys : List H) :
    cross (xs ++ ys) = cross xs + cross ys + 2 * ⟪xs.sum, ys.sum⟫_ℝ := by
  have h := energy_identity (xs ++ ys)
  rw [List.sum_append, norm_add_sq_real, diagonal_append] at h
  have hx := energy_identity xs
  have hy := energy_identity ys
  linarith

theorem cross_lower_bound (ds : List H) : -diagonal ds ≤ cross ds := by
  have h := sq_nonneg ‖ds.sum‖
  rw [energy_identity] at h
  linarith

theorem normalized_energy_identity (ds : List H) (E : ℝ) :
    ‖ds.sum‖ ^ 2 / E = diagonal ds / E + cross ds / E := by
  rw [energy_identity, add_div]

theorem normalized_lower_bound (ds : List H) (E : ℝ) (hE : 0 < E) :
    -(diagonal ds / E) ≤ cross ds / E := by
  have h := div_le_div_of_nonneg_right (cross_lower_bound ds) (le_of_lt hE)
  simpa only [neg_div] using h

end ModelRG.TemporalEnergy

