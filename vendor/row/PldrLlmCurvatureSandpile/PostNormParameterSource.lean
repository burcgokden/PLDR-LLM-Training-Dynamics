/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Ordered affine parameter sources in a post-normalized chain

The analytic row-map derivative has the recurrence v_{k+1} = M_k v_k + b_k.
This module checks the ordered homogeneous product and the suffix transport
of every parameter source without making any claim about how the runtime
Jacobians are constructed.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PostNormParameterSource

variable {E : Type*} [AddCommMonoid E]

/-- Apply the linear parts of chronologically ordered affine units. -/
def applyLinear (state : E) : List (E →+ E) → E
  | [] => state
  | map :: tail => applyLinear (map state) tail

/-- Execute chronologically ordered affine units. -/
def affineFold (state : E) : List ((E →+ E) × E) → E
  | [] => state
  | unit :: tail => affineFold (unit.1 state + unit.2) tail

/-- Transport each local source through every later linear factor. -/
def transportedSources : List ((E →+ E) × E) → E
  | [] => 0
  | unit :: tail =>
      applyLinear unit.2 (tail.map Prod.fst) + transportedSources tail

/-- A chronological linear chain preserves addition. -/
theorem applyLinear_add (left right : E) (maps : List (E →+ E)) :
    applyLinear (left + right) maps =
      applyLinear left maps + applyLinear right maps := by
  induction maps generalizing left right with
  | nil => rfl
  | cons map tail inductionHypothesis =>
      simp only [applyLinear]
      rw [map.map_add, inductionHypothesis]

/-- The affine chain equals its ordered homogeneous transport plus the sum
of all parameter sources transported by their ordered suffix products. -/
theorem affine_source_unroll (state : E) (units : List ((E →+ E) × E)) :
    affineFold state units =
      applyLinear state (units.map Prod.fst) + transportedSources units := by
  induction units generalizing state with
  | nil => simp [affineFold, applyLinear, transportedSources]
  | cons unit tail inductionHypothesis =>
      rw [affineFold, inductionHypothesis]
      simp only [List.map_cons, applyLinear, transportedSources]
      rw [applyLinear_add]
      ac_rfl

/-- The first three units display the source ordering explicitly: the first
source crosses both later maps, the second crosses only the last map, and
the last source is untransported. -/
theorem three_unit_source_order
    (state b1 b2 b3 : E) (M1 M2 M3 : E →+ E) :
    affineFold state [(M1, b1), (M2, b2), (M3, b3)] =
      M3 (M2 (M1 state)) +
        (M3 (M2 b1) + M3 b2 + b3) := by
  simp [affineFold, map_add, add_left_comm, add_comm]

end PostNormParameterSource
end PldrLlmCurvatureSandpile
