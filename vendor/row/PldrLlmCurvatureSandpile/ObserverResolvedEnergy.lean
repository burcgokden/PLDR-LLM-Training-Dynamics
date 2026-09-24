/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Observer-resolved energy near the exact row-constant face

Finite real-arithmetic kernels separating exact zero from positive values
below an empirical effect floor. The floor never changes the mathematical
state or the branch of the exact work-charge identity.
-/
import Mathlib
import PldrLlmCurvatureSandpile.DirectObservableEnergy

namespace PldrLlmCurvatureSandpile
namespace ObserverResolvedEnergy

open DirectObservableEnergy

/-- The exact face is a real-arithmetic zero, not an empirical threshold. -/
def ExactFace (energyValue : ℝ) : Prop := energyValue = 0

/-- Positive energy that is censored by a strictly positive observer floor. -/
def FloorCensoredPositive (floor energyValue : ℝ) : Prop :=
  0 < energyValue ∧ energyValue ≤ floor

/-- Positive energy resolved above the observer floor. -/
def ResolvedPositive (floor energyValue : ℝ) : Prop :=
  floor < energyValue

/-- Every nonnegative energy lies in exactly one observer stratum. -/
theorem observer_stratum_partition
    (floor energyValue : ℝ) (henergy : 0 ≤ energyValue) :
    ExactFace energyValue ∨
      FloorCensoredPositive floor energyValue ∨
      ResolvedPositive floor energyValue := by
  by_cases hzero : energyValue = 0
  · exact Or.inl hzero
  · have hpositive : 0 < energyValue := lt_of_le_of_ne henergy (Ne.symm hzero)
    by_cases hcensored : energyValue ≤ floor
    · exact Or.inr (Or.inl ⟨hpositive, hcensored⟩)
    · exact Or.inr (Or.inr (lt_of_not_ge hcensored))

/-- The three observer strata are pairwise disjoint. -/
theorem observer_strata_disjoint
    (floor energyValue : ℝ) (hfloor : 0 < floor) :
    ¬ (ExactFace energyValue ∧ FloorCensoredPositive floor energyValue) ∧
    ¬ (ExactFace energyValue ∧ ResolvedPositive floor energyValue) ∧
    ¬ (FloorCensoredPositive floor energyValue ∧
      ResolvedPositive floor energyValue) := by
  constructor
  · rintro ⟨hzero, hpositive, _⟩
    exact (ne_of_gt hpositive) hzero
  constructor
  · rintro ⟨hzero, hresolved⟩
    dsimp [ExactFace, ResolvedPositive] at hzero hresolved
    rw [hzero] at hresolved
    exact (not_lt_of_ge (le_of_lt hfloor)) hresolved
  · rintro ⟨⟨_, hcensored⟩, hresolved⟩
    dsimp [ResolvedPositive] at hresolved
    linarith

/-- A nonnegative energy whose certified upper endpoint is zero is exactly on
the row-constant face. -/
theorem upper_zero_certifies_exact_face
    (energyValue upper : ℝ)
    (henergy : 0 ≤ energyValue)
    (henclosedAbove : energyValue ≤ upper)
    (hupper : upper = 0) :
    ExactFace energyValue := by
  dsimp [ExactFace]
  linarith

/-- A strictly positive certified lower endpoint proves strict positivity of
the true energy. It does not by itself certify resolution above a separate
observer floor. -/
theorem positive_lower_certifies_strict_positivity
    (lower energyValue : ℝ)
    (henclosedBelow : lower ≤ energyValue)
    (hlower : 0 < lower) :
    0 < energyValue := by
  linarith

/-- An enclosure is floor-censored when its lower endpoint is positive and
its upper endpoint does not exceed the observer floor. -/
theorem enclosure_certifies_floor_censored
    (floor lower energyValue upper : ℝ)
    (hlower : 0 < lower)
    (henclosedBelow : lower ≤ energyValue)
    (henclosedAbove : energyValue ≤ upper)
    (hupper : upper ≤ floor) :
    FloorCensoredPositive floor energyValue := by
  exact ⟨positive_lower_certifies_strict_positivity
    lower energyValue henclosedBelow hlower, le_trans henclosedAbove hupper⟩

/-- Resolution above the observer floor requires the certified lower endpoint
itself to exceed that floor. -/
theorem lower_above_floor_certifies_resolved
    (floor lower energyValue : ℝ)
    (henclosedBelow : lower ≤ energyValue)
    (hlower : floor < lower) :
    ResolvedPositive floor energyValue := by
  dsimp [ResolvedPositive]
  linarith

/-- Finite squared energy is zero exactly when every coordinate is zero. -/
theorem energy_eq_zero_iff_state_zero
    {Coordinate : Type*} [Fintype Coordinate]
    (state : Coordinate → ℝ) :
    energy state = 0 ↔ ∀ coordinate, state coordinate = 0 :=
  face_preserved_iff state

/-- Unnormalized dissipative work is defined at every source state. -/
def work {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) : ℝ :=
  -2 * pairing state increment

/-- Unnormalized finite-step charge is defined at every source state. -/
def charge {Coordinate : Type*} [Fintype Coordinate]
    (increment : Coordinate → ℝ) : ℝ :=
  energy increment

/-- The branch-free work-charge balance is valid at positive, censored, and
exact-face states. -/
theorem observer_safe_work_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    energy (fun coordinate => state coordinate + increment coordinate)
      = energy state - work state increment + charge increment := by
  have hledger := exact_work_charge state increment
  unfold work charge
  linarith

/-- Strict contraction is exactly strict excess of work over charge. -/
theorem strict_contraction_iff_work_gt_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    energy (fun coordinate => state coordinate + increment coordinate)
        < energy state ↔
      charge increment < work state increment := by
  have hbalance := observer_safe_work_charge state increment
  constructor <;> intro h <;> linarith

/-- Exact preservation is exactly equality of work and charge. -/
theorem preservation_iff_work_eq_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    energy (fun coordinate => state coordinate + increment coordinate)
        = energy state ↔
      work state increment = charge increment := by
  have hbalance := observer_safe_work_charge state increment
  constructor <;> intro h <;> linarith

/-- Strict reopening is exactly strict excess of charge over work. -/
theorem strict_reopening_iff_work_lt_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) :
    energy state <
        energy (fun coordinate => state coordinate + increment coordinate) ↔
      work state increment < charge increment := by
  have hbalance := observer_safe_work_charge state increment
  constructor <;> intro h <;> linarith

/-- At the exact face, work vanishes and the endpoint is pure charge. -/
theorem exact_face_work_charge
    {Coordinate : Type*} [Fintype Coordinate]
    (state increment : Coordinate → ℝ) (hface : energy state = 0) :
    work state increment = 0 ∧
      energy (fun coordinate => state coordinate + increment coordinate)
        = charge increment := by
  classical
  have hzero : ∀ coordinate, state coordinate = 0 :=
    (energy_eq_zero_iff_state_zero state).mp hface
  constructor
  · simp [work, pairing, hzero]
  · simp [energy, charge, hzero]

/-- Separated work and charge intervals certify a true contraction. -/
theorem interval_certifies_contraction
    (workValue chargeValue workEstimate chargeEstimate
      workError chargeError : ℝ)
    (hwork : |workEstimate - workValue| ≤ workError)
    (hcharge : |chargeEstimate - chargeValue| ≤ chargeError)
    (hseparate : workEstimate - workError >
      chargeEstimate + chargeError) :
    chargeValue < workValue := by
  have hworkBounds := (abs_le.mp hwork).2
  have hchargeBounds := (abs_le.mp hcharge).1
  linarith

/-- The reverse interval separation certifies a true reopening. -/
theorem interval_certifies_reopening
    (workValue chargeValue workEstimate chargeEstimate
      workError chargeError : ℝ)
    (hwork : |workEstimate - workValue| ≤ workError)
    (hcharge : |chargeEstimate - chargeValue| ≤ chargeError)
    (hseparate : workEstimate + workError <
      chargeEstimate - chargeError) :
    workValue < chargeValue := by
  have hworkBounds := (abs_le.mp hwork).1
  have hchargeBounds := (abs_le.mp hcharge).2
  linarith

end ObserverResolvedEnergy
end PldrLlmCurvatureSandpile
