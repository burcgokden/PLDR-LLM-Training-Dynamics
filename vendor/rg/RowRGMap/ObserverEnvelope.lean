import Mathlib

/-!
# Observer energy enclosures

The observer reports an estimate and an absolute error radius. These kernels
check the nonnegative enclosure, the strictly positive and exact-face decision
rules, convergence under a vanishing upper envelope, and the impossibility of
bounding a ratio when the denominator interval reaches zero.
-/

namespace RowRGMap
namespace ObserverEnvelope

open Filter

def lowerEnergy (estimate radius : ℝ) : ℝ :=
  max 0 (estimate - radius)

def upperEnergy (estimate radius : ℝ) : ℝ :=
  estimate + radius

theorem energy_enclosed {energy estimate radius : ℝ}
    (henergy : 0 ≤ energy)
    (herror : |energy - estimate| ≤ radius) :
    lowerEnergy estimate radius ≤ energy ∧
      energy ≤ upperEnergy estimate radius := by
  constructor
  · apply max_le henergy
    have := (abs_le.mp herror).1
    linarith
  · have hdifference : energy - estimate ≤ radius :=
      (le_abs_self (energy - estimate)).trans herror
    simpa [upperEnergy, add_comm] using
      (sub_le_iff_le_add.mp hdifference)

theorem positive_of_positive_lower {energy lower : ℝ}
    (hlower : 0 < lower) (henclosed : lower ≤ energy) :
    0 < energy :=
  hlower.trans_le henclosed

theorem exact_face_of_zero_upper {energy upper : ℝ}
    (henergy : 0 ≤ energy) (henclosed : energy ≤ upper)
    (hupper : upper = 0) :
    energy = 0 := by
  linarith

theorem convergence_from_upper_envelope
    {energy upper : ℕ → ℝ}
    (henergy : ∀ n, 0 ≤ energy n)
    (henclosed : ∀ n, energy n ≤ upper n)
    (hupper : Tendsto upper atTop (nhds 0)) :
    Tendsto energy atTop (nhds 0) := by
  exact squeeze_zero henergy henclosed hupper

theorem fixed_tolerance_only_bounds
    {energy upper : ℕ → ℝ} {tolerance : ℝ}
    (henclosed : ∀ n, energy n ≤ upper n)
    (hupper : ∀ᶠ n in atTop, upper n ≤ tolerance) :
    ∀ᶠ n in atTop, energy n ≤ tolerance := by
  filter_upwards [hupper] with n hn
  exact (henclosed n).trans hn

theorem unresolved_interval_has_zero_and_positive
    {lower upper : ℝ} (hlower : lower = 0) (hupper : 0 < upper) :
    (lower ≤ 0 ∧ 0 ≤ upper) ∧
      ∃ value : ℝ, 0 < value ∧ lower ≤ value ∧ value ≤ upper := by
  constructor
  · constructor <;> linarith
  · refine ⟨upper / 2, ?_, ?_, ?_⟩ <;> linarith

/-- A fixed positive tolerance can contain a positive constant sequence and
therefore cannot establish an asymptotic zero limit. -/
theorem fixed_tolerance_does_not_imply_convergence
    {tolerance : ℝ} (htolerance : 0 < tolerance) :
    ∃ energy : ℕ → ℝ,
      (∀ n, 0 < energy n ∧ energy n ≤ tolerance) ∧
      ¬ Tendsto energy atTop (nhds 0) := by
  refine ⟨fun _ => tolerance / 2, ?_, ?_⟩
  · intro n
    constructor <;> linarith
  · intro hzero
    have hconstant :
        Tendsto (fun _ : ℕ => tolerance / 2) atTop
          (nhds (tolerance / 2)) :=
      tendsto_const_nhds
    have hequal : (0 : ℝ) = tolerance / 2 :=
      tendsto_nhds_unique hzero hconstant
    linarith


end ObserverEnvelope
end RowRGMap
