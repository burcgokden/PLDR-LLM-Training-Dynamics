/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Trajectory-compatible slices

Written correspondence labels:
- row:eq:realized-chart-step
- row:eq:realized-chart-observation
- row:eq:fixed-slice-defect
- row:eq:fixed-slice-observation-defect
- row:sec:trajectory-chart
- row:thm:observable-complete-state
- row:prop:stratified-itinerary

These are algebraic connections on admitted domain types. They do not prove
chart existence, domain admission, differentiability, or predictive closure.
The one-step identities allow time-indexed types; finite transport below uses
a fixed additive coordinate space, as does the imported transport kernel.
-/
import PldrLlmCurvatureSandpile.ObservableCompleteStateCocycle

namespace PldrTrainingDynamics.TrajectoryChart

/-- A selected slice successor, on the supplied admitted domain types. -/
def sliceMap {S N : ℕ → Type*}
    (F : (t : ℕ) → S t → S (t + 1))
    (extract : (t : ℕ) → S t → N t)
    (insert : (t : ℕ) → N t → S t) (t : ℕ) (z : N t) : N (t + 1) :=
  extract (t + 1) (F t (insert t z))

/-- A right inverse alone is insufficient: this connection uses the realized lift. -/
theorem realized_step_of_lift {S N : ℕ → Type*}
    (F : (t : ℕ) → S t → S (t + 1))
    (extract : (t : ℕ) → S t → N t)
    (insert : (t : ℕ) → N t → S t)
    (s : (t : ℕ) → S t) (z : (t : ℕ) → N t)
    (hevolve : ∀ t, s (t + 1) = F t (s t))
    (hcoordinate : ∀ t, z t = extract t (s t))
    (hlift : ∀ t, s t = insert t (z t)) (t : ℕ) :
    z (t + 1) = sliceMap F extract insert t (z t) := by
  rw [hcoordinate (t + 1), hevolve t, hlift t]
  rfl

/-- The same lift connects the physical and sliced observations. -/
theorem realized_observation_of_lift {S N Y : Type*}
    (H : S → Y) (insert : N → S) (s : S) (z : N)
    (hlift : s = insert z) : H s = (H ∘ insert) z := by
  rw [hlift]
  rfl

/-- A prescribed slice retains an additive successor defect, without a size bound. -/
theorem fixed_slice_defect {S S' E E' : Type*} [AddCommGroup E']
    (F : S → S') (extractNext : S' → E') (insert : E → S)
    (A : E → E') (s : S) (z : E) (reference : E) :
    extractNext (F s) =
      A z + extractNext (F (insert reference)) +
      (extractNext (F (insert z)) - extractNext (F (insert reference)) - A z) +
      (extractNext (F s) - extractNext (F (insert z))) := by
  abel

/-- Observation mismatch is separate from the successor mismatch. -/
theorem observation_defect {S E Y : Type*} [AddCommGroup Y]
    (H : S → Y) (insert : E → S) (s : S) (z : E) :
    H s = H (insert z) + (H s - H (insert z)) := by
  abel

/-- Exact real counterexample to deriving the realized step from a right inverse. -/
theorem right_inverse_not_enough :
    let F : ℝ × ℝ → ℝ × ℝ := fun p => (p.1 + p.2, p.2)
    let extract : ℝ × ℝ → ℝ := Prod.fst
    let insert : ℝ → ℝ × ℝ := fun z => (z, 0)
    (∀ z, extract (insert z) = z) ∧
      extract (F (1, 1)) ≠ extract (F (insert (extract (1, 1)))) := by
  norm_num

/-- The adapted slice recovers the successor and its unit forcing. -/
theorem adapted_slice_recovers_step :
    let F : ℝ × ℝ → ℝ × ℝ := fun p => (p.1 + p.2, p.2)
    let insert : ℝ → ℝ × ℝ := fun z => (z, 1)
    (1, 1) = insert 1 ∧ (∀ z, (F (insert z)).1 = z + 1) ∧
      (F (insert 0)).1 = 1 := by
  norm_num

/-- Exact affine/remainder decomposition derived from the trajectory lift. -/
theorem normal_step_of_lift {S N : ℕ → Type*} [∀ t, AddCommGroup (N t)]
    (F : (t : ℕ) → S t → S (t + 1))
    (extract : (t : ℕ) → S t → N t)
    (insert : (t : ℕ) → N t → S t)
    (A : (t : ℕ) → N t →+ N (t + 1))
    (s : (t : ℕ) → S t) (z : (t : ℕ) → N t)
    (hevolve : ∀ t, s (t + 1) = F t (s t))
    (hcoordinate : ∀ t, z t = extract t (s t))
    (hlift : ∀ t, s t = insert t (z t)) (t : ℕ) :
    z (t + 1) = A t (z t) + sliceMap F extract insert t 0 +
      (sliceMap F extract insert t (z t) - sliceMap F extract insert t 0 - A t (z t)) := by
  rw [realized_step_of_lift F extract insert s z hevolve hcoordinate hlift t]
  abel

open PldrLlmCurvatureSandpile.ObservableCompleteStateCocycle

/-- Supply the derived recurrence to the existing fixed-space transport theorem. -/
theorem transport_of_lift {S : ℕ → Type*} {E : Type*} [AddCommGroup E]
    (F : (t : ℕ) → S t → S (t + 1))
    (extract : (t : ℕ) → S t → E) (insert : (t : ℕ) → E → S t)
    (A : ℕ → E →+ E) (s : (t : ℕ) → S t) (z : ℕ → E)
    (hevolve : ∀ t, s (t + 1) = F t (s t))
    (hcoordinate : ∀ t, z t = extract t (s t))
    (hlift : ∀ t, s t = insert t (z t)) :
    ∀ n, z n = homogeneous A (z 0) n +
      transported A (fun t => sliceMap F extract insert t 0) n +
      transported A (fun t =>
        sliceMap F extract insert t (z t) - sliceMap F extract insert t 0 - A t (z t)) n := by
  exact complete_state_variation_of_constants A z _ _ (z 0) rfl
    (normal_step_of_lift F extract insert A s z hevolve hcoordinate hlift)

/-- State and observation lifts feed the same fixed-space observable transport. -/
theorem observable_transport_of_lift {S : ℕ → Type*} {E Y : Type*}
    [AddCommGroup E] [AddCommGroup Y]
    (F : (t : ℕ) → S t → S (t + 1))
    (extract : (t : ℕ) → S t → E) (insert : (t : ℕ) → E → S t)
    (A : ℕ → E →+ E) (H : (t : ℕ) → S t → Y) (O : ℕ → E →+ Y)
    (s : (t : ℕ) → S t) (z : ℕ → E)
    (hevolve : ∀ t, s (t + 1) = F t (s t))
    (hcoordinate : ∀ t, z t = extract t (s t))
    (hlift : ∀ t, s t = insert t (z t)) :
    ∀ n, H n (s n) = H n (insert n 0) + O n (homogeneous A (z 0) n) +
      O n (transported A (fun t => sliceMap F extract insert t 0) n) +
      O n (transported A (fun t =>
        sliceMap F extract insert t (z t) - sliceMap F extract insert t 0 - A t (z t)) n) +
      (H n (insert n (z n)) - H n (insert n 0) - O n (z n)) := by
  apply observable_variation_of_constants A O z _ _ (fun t => H t (s t))
    (fun t => H t (insert t 0)) _ (z 0) rfl
    (normal_step_of_lift F extract insert A s z hevolve hcoordinate hlift)
  intro t
  rw [hlift t]
  abel

end PldrTrainingDynamics.TrajectoryChart
