/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Segment-valid row domains

This module formalizes the convex interval hull and the direct segment hull
used by every global mean-value bridge in the row-map theory.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SegmentTube

/-- Coordinatewise closed interval hull in a finite row space. -/
def intervalHull {n : ℕ} (lower upper : Fin n → ℝ) : Set (Fin n → ℝ) :=
  {x | ∀ i, lower i ≤ x i ∧ x i ≤ upper i}

/-- Every convex interpolation of two points in the interval hull remains in
the interval hull.  This is the segment condition required by the mean-value
argument. -/
theorem segment_mem_intervalHull {n : ℕ} {lower upper x y : Fin n → ℝ}
    (hx : x ∈ intervalHull lower upper)
    (hy : y ∈ intervalHull lower upper)
    {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    (fun i => (1 - s) * x i + s * y i) ∈ intervalHull lower upper := by
  intro i
  constructor
  · have hx' := (hx i).1
    have hy' := (hy i).1
    nlinarith [mul_nonneg (sub_nonneg.mpr hs1) (sub_nonneg.mpr hx'),
      mul_nonneg hs0 (sub_nonneg.mpr hy')]
  · have hx' := (hx i).2
    have hy' := (hy i).2
    nlinarith [mul_nonneg (sub_nonneg.mpr hs1) (sub_nonneg.mpr hx'),
      mul_nonneg hs0 (sub_nonneg.mpr hy')]

/-- Union of all closed segments whose endpoints lie in a registered row
set. -/
def segmentHull {E : Type*} [AddCommGroup E] [Module ℝ E]
    (rows : Set E) : Set E :=
  {z | ∃ x ∈ rows, ∃ y ∈ rows, ∃ s : ℝ,
    0 ≤ s ∧ s ≤ 1 ∧ z = (1 - s) • x + s • y}

/-- The segment joining any two registered rows belongs to their segment
hull by construction. -/
theorem segment_mem_segmentHull {E : Type*} [AddCommGroup E] [Module ℝ E]
    {rows : Set E} {x y : E} (hx : x ∈ rows) (hy : y ∈ rows)
    {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    (1 - s) • x + s • y ∈ segmentHull rows := by
  exact ⟨x, hx, y, hy, s, hs0, hs1, rfl⟩

end SegmentTube
end PldrLlmCurvatureSandpile
