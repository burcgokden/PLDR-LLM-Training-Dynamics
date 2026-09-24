/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact finite source transport

This module checks the algebraic unrolling of endpoint increments. Runtime
construction of the nonlinear chords is deliberately outside this abstract
additive recurrence.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace FiniteSourceTransport

variable {E : Type*} [AddCommMonoid E]

/-- Execute the linear part of a chronological finite-increment chain. -/
def applyLinear (state : E) : List (E →+ E) → E
  | [] => state
  | map :: tail => applyLinear (map state) tail

/-- Execute a chronological affine endpoint-increment recurrence. -/
def finiteFold (state : E) : List ((E →+ E) × E) → E
  | [] => state
  | unit :: tail => finiteFold (unit.1 state + unit.2) tail

/-- Transport every local endpoint source through all later linear chords. -/
def transportedSources : List ((E →+ E) × E) → E
  | [] => 0
  | unit :: tail =>
      applyLinear unit.2 (tail.map Prod.fst) + transportedSources tail

theorem applyLinear_add (left right : E) (maps : List (E →+ E)) :
    applyLinear (left + right) maps =
      applyLinear left maps + applyLinear right maps := by
  induction maps generalizing left right with
  | nil => rfl
  | cons map tail inductionHypothesis =>
      simp only [applyLinear]
      rw [map.map_add, inductionHypothesis]

/-- One finite affine unit has exactly one transported input and one local
source, with no remainder term. -/
theorem finite_affine_step (state source : E) (map : E →+ E) :
    finiteFold state [(map, source)] = map state + source := by
  rfl

/-- The exact endpoint recurrence equals the homogeneous transport plus all
local sources transported by their chronological suffixes. -/
theorem finite_source_unroll (state : E) (units : List ((E →+ E) × E)) :
    finiteFold state units =
      applyLinear state (units.map Prod.fst) + transportedSources units := by
  induction units generalizing state with
  | nil => simp [finiteFold, applyLinear, transportedSources]
  | cons unit tail inductionHypothesis =>
      rw [finiteFold, inductionHypothesis]
      simp only [List.map_cons, applyLinear, transportedSources]
      rw [applyLinear_add]
      ac_rfl

/-- Eight units are covered by the same exact recurrence in the implemented
chronological order. -/
theorem eight_unit_finite_order
    (state b1 b2 b3 b4 b5 b6 b7 b8 : E)
    (M1 M2 M3 M4 M5 M6 M7 M8 : E →+ E) :
    finiteFold state
      [(M1, b1), (M2, b2), (M3, b3), (M4, b4),
       (M5, b5), (M6, b6), (M7, b7), (M8, b8)] =
      M8 (M7 (M6 (M5 (M4 (M3 (M2 (M1 state)))))))
      + (M8 (M7 (M6 (M5 (M4 (M3 (M2 b1))))))
      + M8 (M7 (M6 (M5 (M4 (M3 b2)))))
      + M8 (M7 (M6 (M5 (M4 b3))))
      + M8 (M7 (M6 (M5 b4)))
      + M8 (M7 (M6 b5))
      + M8 (M7 b6)
      + M8 b7
      + b8) := by
  simp [finiteFold, map_add]
  ac_rfl

end FiniteSourceTransport
end PldrLlmCurvatureSandpile
