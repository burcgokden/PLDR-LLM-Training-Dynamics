import Mathlib.Logic.Function.Iterate

/-!
# Gauge fibers and deterministic closure

This module checks the abstract symmetry argument used by the manuscript. It
does not assert that the concrete represented PLDR program has an exact
symmetry or that any quotient observable is closed.
-/

namespace RowRGMap
namespace GaugeFiber

/-- An observable closes deterministically at scale `b` when equal source
observables force equal observable successors after `b` updates. -/
def DeterministicClosesAt {X A : Type*}
    (update : X → X) (observable : X → A) (b : ℕ) : Prop :=
  ∀ x y, observable x = observable y →
    observable ((update^[b]) x) = observable ((update^[b]) y)

/-- Equivariance of one update implies equivariance of every finite iterate. -/
theorem iterate_equivariant {G X : Type*}
    (act : G → X → X) (update : X → X)
    (hupdate : ∀ g x, update (act g x) = act g (update x)) :
    ∀ b g x, (update^[b]) (act g x) = act g ((update^[b]) x) := by
  intro b
  induction b with
  | zero =>
      intro g x
      simp
  | succ b inductionHypothesis =>
      intro g x
      simp only [Function.iterate_succ_apply]
      rw [hupdate, inductionHypothesis]

/-- A composition of two commuting-with-update actions also commutes with the
update. No commutativity assumption between the actions is needed. -/
theorem compose_equivariant {X : Type*}
    (first second update : X → X)
    (hfirst : ∀ x, update (first x) = first (update x))
    (hsecond : ∀ x, update (second x) = second (update x))
    (x : X) :
    update ((first ∘ second) x) = (first ∘ second) (update x) := by
  simp only [Function.comp_apply]
  rw [hfirst, hsecond]

/-- A composition of two observable-invariant actions is observable invariant. -/
theorem compose_invariant {X E : Type*}
    (first second : X → X) (energy : X → E)
    (hfirst : ∀ x, energy (first x) = energy x)
    (hsecond : ∀ x, energy (second x) = energy x)
    (x : X) :
    energy ((first ∘ second) x) = energy x := by
  simp only [Function.comp_apply]
  rw [hfirst, hsecond]

/-- An invariant observable has identical successors along every orbit of an
equivariant update, at every finite scale. -/
theorem gauge_orbit_successor_equal {G X E : Type*}
    (act : G → X → X) (update : X → X) (energy : X → E)
    (hupdate : ∀ g x, update (act g x) = act g (update x))
    (henergy : ∀ g x, energy (act g x) = energy x)
    (b : ℕ) (g : G) (x : X) :
    energy ((update^[b]) (act g x)) = energy ((update^[b]) x) := by
  rw [iterate_equivariant act update hupdate b g x]
  exact henergy g ((update^[b]) x)

/-- If a parameter-only action factors as a consistent symmetry after a
moment-only action, then their observable successors agree at every finite
scale. -/
theorem paired_gauge_successor_equal {X E : Type*}
    (consistent parameterOnly momentOnly update : X → X) (energy : X → E)
    (hfactor : parameterOnly = consistent ∘ momentOnly)
    (hupdate : ∀ x, update (consistent x) = consistent (update x))
    (henergy : ∀ x, energy (consistent x) = energy x)
    (b : ℕ) (x : X) :
    energy ((update^[b]) (parameterOnly x)) =
      energy ((update^[b]) (momentOnly x)) := by
  rw [hfactor]
  calc
    energy ((update^[b]) ((consistent ∘ momentOnly) x)) =
        energy ((update^[b]) (consistent (momentOnly x))) := rfl
    _ = energy (consistent ((update^[b]) (momentOnly x))) := by
      congr 1
      exact iterate_equivariant
        (fun _ : Unit => consistent) update (fun _ => hupdate)
        b () (momentOnly x)
    _ = energy ((update^[b]) (momentOnly x)) :=
      henergy ((update^[b]) (momentOnly x))

/-- One equal augmented-state pair with unequal energy successors refutes
deterministic closure of that augmented observable. -/
theorem augmented_fiber_witness_not_closed {X A E : Type*}
    (update : X → X) (augmented : X → A) (energyOf : A → E)
    (b : ℕ) (x y : X)
    (hsource : augmented x = augmented y)
    (hsuccessor :
      energyOf (augmented ((update^[b]) x)) ≠
        energyOf (augmented ((update^[b]) y))) :
    ¬ DeterministicClosesAt update augmented b := by
  intro hcloses
  exact hsuccessor (congrArg energyOf (hcloses x y hsource))

end GaugeFiber
end RowRGMap
