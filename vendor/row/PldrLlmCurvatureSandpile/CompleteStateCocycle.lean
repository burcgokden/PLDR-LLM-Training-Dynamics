/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Complete-state forced cocycle and physical collapse
-/
import Mathlib
import PldrLlmCurvatureSandpile.NonautonomousAttraction

namespace PldrLlmCurvatureSandpile
namespace CompleteStateCocycle

open Filter

/-- A complete-state one-step affine estimate iterates through the same
chronological product-convolution used by the implemented successor. -/
theorem complete_state_affine_envelope
    (state gain force : ℕ → ℝ) {initial : ℝ}
    (hstart : state 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep : ∀ step,
      state (step + 1) ≤ gain step * state step + force step) :
    ∀ step, state step ≤
      NonautonomousAttraction.orderedEnvelope gain force initial step := by
  exact NonautonomousAttraction.ordered_product_convolution
    state gain force hstart hgain hstep

/-- If the chronological affine envelope vanishes, any nonnegative scheduled
state below that envelope also vanishes.  No fixed metric enters this claim. -/
theorem scheduled_affine_collapse
    (state gain force : ℕ → ℝ) {initial : ℝ}
    (hstateNonnegative : ∀ step, 0 ≤ state step)
    (hstart : state 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep : ∀ step,
      state (step + 1) ≤ gain step * state step + force step)
    (henvelope : Tendsto
      (NonautonomousAttraction.orderedEnvelope gain force initial)
      atTop (nhds 0)) :
    Tendsto state atTop (nhds 0) := by
  have hupper : ∀ step, state step ≤
      NonautonomousAttraction.orderedEnvelope gain force initial step :=
    complete_state_affine_envelope state gain force hstart hgain hstep
  exact squeeze_zero hstateNonnegative hupper henvelope

/-- Uniform control of a physical row quotient by a vanishing baseline and
a vanishing complete-state normal proves physical collapse. -/
theorem physical_cover_collapse
    (state physical baseline : ℕ → ℝ) (coverGain : ℝ)
    (hphysicalNonnegative : ∀ step, 0 ≤ physical step)
    (hcover : ∀ step,
      physical step ≤ baseline step + coverGain * state step)
    (hstate : Tendsto state atTop (nhds 0))
    (hbaseline : Tendsto baseline atTop (nhds 0)) :
    Tendsto physical atTop (nhds 0) := by
  have hscaled :
      Tendsto (fun step => coverGain * state step) atTop (nhds 0) := by
    simpa using tendsto_const_nhds.mul hstate
  have hupper :
      Tendsto (fun step => baseline step + coverGain * state step)
        atTop (nhds 0) := by
    simpa using hbaseline.add hscaled
  exact squeeze_zero hphysicalNonnegative hcover hupper

/-- If the next state, homogeneous response, and nonlinear remainder vanish,
then the invariance force in the exact successor identity must vanish. -/
theorem vanishing_invariance_force_necessary
    {E : Type*} [NormedAddCommGroup E]
    (nextState linear remainder force : ℕ → E)
    (hidentity : ∀ step,
      force step = nextState step - linear step - remainder step)
    (hnext : Tendsto nextState atTop (nhds 0))
    (hlinear : Tendsto linear atTop (nhds 0))
    (hremainder : Tendsto remainder atTop (nhds 0)) :
    Tendsto force atTop (nhds 0) := by
  have hright :
      Tendsto (fun step => nextState step - linear step - remainder step)
        atTop (nhds 0) := by
    simpa using (hnext.sub hlinear).sub hremainder
  exact hright.congr' (Eventually.of_forall fun step => (hidentity step).symm)

/-- A nonvanishing invariance force obstructs convergence whenever the
homogeneous and remainder terms vanish. -/
theorem nonvanishing_force_obstructs_collapse
    {E : Type*} [NormedAddCommGroup E]
    (nextState linear remainder force : ℕ → E)
    (hidentity : ∀ step,
      force step = nextState step - linear step - remainder step)
    (hlinear : Tendsto linear atTop (nhds 0))
    (hremainder : Tendsto remainder atTop (nhds 0))
    (hforce : ¬ Tendsto force atTop (nhds 0)) :
    ¬ Tendsto nextState atTop (nhds 0) := by
  intro hnext
  exact hforce (vanishing_invariance_force_necessary
    nextState linear remainder force hidentity hnext hlinear hremainder)

/-- A sequence can agree with exact collapse on every point of an arbitrary
finite observation horizon and then reopen permanently. -/
def delayedReopening (horizon step : ℕ) : ℝ :=
  if step ≤ horizon then 0 else 1

theorem delayed_reopening_matches_finite_prefix
    (horizon step : ℕ) (hstep : step ≤ horizon) :
    delayedReopening horizon step = 0 := by
  simp [delayedReopening, hstep]

theorem delayed_reopening_not_tendsto_zero (horizon : ℕ) :
    ¬ Tendsto (delayedReopening horizon) atTop (nhds 0) := by
  intro hzero
  have hevent : ∀ᶠ step in atTop,
      delayedReopening horizon step = 1 := by
    apply eventually_atTop.2
    refine ⟨horizon + 1, ?_⟩
    intro step hstep
    simp [delayedReopening, Nat.not_le.mpr (Nat.lt_of_lt_of_le
      (Nat.lt_succ_self horizon) hstep)]
  have hconstant : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) :=
    tendsto_const_nhds
  have hone : Tendsto (delayedReopening horizon) atTop (nhds 1) :=
    hconstant.congr' (hevent.mono fun step hstep => hstep.symm)
  have : (0 : ℝ) = 1 := tendsto_nhds_unique hzero hone
  norm_num at this

end CompleteStateCocycle
end PldrLlmCurvatureSandpile
