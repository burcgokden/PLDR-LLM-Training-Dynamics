/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Observable complete-state cocycle identities
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ObservableCompleteStateCocycle

open Filter

variable {E Y : Type*}

/-- Chronological propagation of an initial complete-state displacement. -/
def homogeneous [AddCommGroup E]
    (operator : ℕ → E →+ E) (initial : E) : ℕ → E
  | 0 => initial
  | step + 1 => operator step (homogeneous operator initial step)

/-- Chronological transport of an additive input. -/
def transported [AddCommGroup E]
    (operator : ℕ → E →+ E) (input : ℕ → E) : ℕ → E
  | 0 => 0
  | step + 1 =>
      operator step (transported operator input step) + input step

/-- Exact complete-state variation of constants, with forcing and nonlinear
remainder kept as separate transported vectors. -/
theorem complete_state_variation_of_constants
    [AddCommGroup E]
    (operator : ℕ → E →+ E)
    (state force remainder : ℕ → E)
    (initial : E)
    (hinitial : state 0 = initial)
    (hstep : ∀ step,
      state (step + 1) =
        operator step (state step) + force step + remainder step) :
    ∀ step,
      state step =
        homogeneous operator initial step
        + transported operator force step
        + transported operator remainder step := by
  intro step
  induction step with
  | zero =>
      simp [homogeneous, transported, hinitial]
  | succ step ih =>
      rw [hstep step, ih]
      simp only [homogeneous, transported, map_add]
      abel

/-- Applying a physical observation to the exact complete-state identity
preserves the vector sum and therefore all possible cancellations. -/
theorem observable_variation_of_constants
    [AddCommGroup E] [AddCommGroup Y]
    (operator : ℕ → E →+ E)
    (observe : ℕ → E →+ Y)
    (state force remainder : ℕ → E)
    (physical baseline nonlinear : ℕ → Y)
    (initial : E)
    (hinitial : state 0 = initial)
    (hstep : ∀ step,
      state (step + 1) =
        operator step (state step) + force step + remainder step)
    (hobserve : ∀ step,
      physical step =
        baseline step + observe step (state step) + nonlinear step) :
    ∀ step,
      physical step =
        baseline step
        + observe step (homogeneous operator initial step)
        + observe step (transported operator force step)
        + observe step (transported operator remainder step)
        + nonlinear step := by
  intro step
  rw [
    hobserve step,
    complete_state_variation_of_constants operator state force remainder
      initial hinitial hstep step,
  ]
  simp only [map_add]
  abel

/-- Vanishing of every displayed observable component is sufficient for
physical collapse.  It is deliberately not stated as a necessary condition,
because vector components can cancel. -/
theorem componentwise_observable_vanishing
    [AddCommGroup Y] [TopologicalSpace Y] [IsTopologicalAddGroup Y]
    (physical baseline homogeneousPart forcePart remainderPart nonlinear :
      ℕ → Y)
    (hidentity : ∀ step,
      physical step =
        baseline step + homogeneousPart step + forcePart step
        + remainderPart step + nonlinear step)
    (hbaseline : Tendsto baseline atTop (nhds 0))
    (hhomogeneous : Tendsto homogeneousPart atTop (nhds 0))
    (hforce : Tendsto forcePart atTop (nhds 0))
    (hremainder : Tendsto remainderPart atTop (nhds 0))
    (hnonlinear : Tendsto nonlinear atTop (nhds 0)) :
    Tendsto physical atTop (nhds 0) := by
  have hsum :
      Tendsto
        (fun step =>
          baseline step + homogeneousPart step + forcePart step
          + remainderPart step + nonlinear step)
        atTop (nhds 0) := by
    simpa using
      (((hbaseline.add hhomogeneous).add hforce).add hremainder).add hnonlinear
  exact hsum.congr' (
    Eventually.of_forall fun step => (hidentity step).symm
  )

/-- A uniform lower comparison with the scheduled metric is enough to
transfer scheduled decay to a fixed norm.  No upper metric comparison is
used in this direction. -/
theorem lower_metric_transfer
    (fixed scheduled : ℕ → ℝ) (lower : ℝ)
    (hlower : 0 < lower)
    (hfixed : ∀ step, 0 ≤ fixed step)
    (hcompare : ∀ step, lower * fixed step ≤ scheduled step)
    (hscheduled : Tendsto scheduled atTop (nhds 0)) :
    Tendsto fixed atTop (nhds 0) := by
  have hinv : 0 ≤ lower⁻¹ := inv_nonneg.mpr (le_of_lt hlower)
  have hupper : ∀ step, fixed step ≤ lower⁻¹ * scheduled step := by
    intro step
    calc
      fixed step = lower⁻¹ * (lower * fixed step) := by
        field_simp
      _ ≤ lower⁻¹ * scheduled step :=
        mul_le_mul_of_nonneg_left (hcompare step) hinv
  have hupperLimit :
      Tendsto (fun step => lower⁻¹ * scheduled step) atTop (nhds 0) := by
    simpa using tendsto_const_nhds.mul hscheduled
  exact squeeze_zero hfixed hupper hupperLimit

/-- Lower-metric transfer followed by a fixed-norm physical cover proves
physical collapse.  No upper comparison with the scheduled metric is used. -/
theorem fixed_norm_cover_collapse
    (fixed scheduled physical baseline : ℕ → ℝ)
    (lower coverGain : ℝ)
    (hlower : 0 < lower)
    (hfixedNonnegative : ∀ step, 0 ≤ fixed step)
    (hcompare : ∀ step, lower * fixed step ≤ scheduled step)
    (hscheduled : Tendsto scheduled atTop (nhds 0))
    (hphysicalNonnegative : ∀ step, 0 ≤ physical step)
    (hcover : ∀ step,
      physical step ≤ baseline step + coverGain * fixed step)
    (hbaseline : Tendsto baseline atTop (nhds 0)) :
    Tendsto physical atTop (nhds 0) := by
  have hfixed : Tendsto fixed atTop (nhds 0) :=
    lower_metric_transfer fixed scheduled lower hlower
      hfixedNonnegative hcompare hscheduled
  have hscaled :
      Tendsto (fun step => coverGain * fixed step) atTop (nhds 0) := by
    simpa using tendsto_const_nhds.mul hfixed
  have hupper :
      Tendsto (fun step => baseline step + coverGain * fixed step)
        atTop (nhds 0) := by
    simpa using hbaseline.add hscaled
  exact squeeze_zero hphysicalNonnegative hcover hupper

end ObservableCompleteStateCocycle
end PldrLlmCurvatureSandpile
