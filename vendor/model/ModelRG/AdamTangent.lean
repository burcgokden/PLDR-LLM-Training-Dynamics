import Mathlib

/-! Algebraic cores of the clipped augmented Adam tangent.
The derivative, admissible moment-boundary and full-graph arguments have
stand-alone proofs in the manuscript. -/

noncomputable section
namespace ModelRG

theorem adam_first_direction (beta m g dm dg radius : ℝ) :
    beta * (m + radius * dm) + (1-beta) * (g + radius * dg) -
      (beta * m + (1-beta) * g) =
      radius * (beta * dm + (1-beta) * dg) := by
  ring

theorem adam_second_centered_difference (beta v g dv dg radius : ℝ)
    (hr : radius ≠ 0) :
    ((beta * (v + radius * dv) + (1-beta) * (g + radius * dg)^2) -
      (beta * (v - radius * dv) + (1-beta) * (g - radius * dg)^2)) /
      (2 * radius) = beta * dv + 2 * (1-beta) * g * dg := by
  field_simp <;> ring

theorem adam_ratio_direction (a d m dm dd : ℝ) (ha : a ≠ 0) (hd : d ≠ 0) :
    (dm * d - m * dd) / (a * d^2) = dm / (a*d) - m*dd/(a*d^2) := by
  field_simp <;> ring

theorem clip_radial_direction (bound norm offset component : ℝ)
    (h : norm + offset ≠ 0) :
    (bound/(norm+offset) - bound*norm/(norm+offset)^2)*component =
      bound*offset/(norm+offset)^2*component := by
  field_simp <;> ring

end ModelRG
