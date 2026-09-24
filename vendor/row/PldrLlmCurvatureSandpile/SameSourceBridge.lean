/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Same-source observable bridge

Stagewise Lipschitz and additive-defect transport for two executions that
retain one source-pair identity through tensors, attention, and logits.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SameSourceBridge

/-- Exact recursively transported upper bound for a sequence of downstream
stages. -/
def defectEnvelope (gain defect : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => gain n * defectEnvelope gain defect initial n + defect n

/-- Stagewise inequalities telescope along the same paired executions. -/
theorem stagewise_defect_telescoping
    (distance gain defect : ℕ → ℝ) {initial : ℝ}
    (hstart : distance 0 ≤ initial)
    (hgain : ∀ n, 0 ≤ gain n)
    (hstep : ∀ n, distance (n + 1) ≤ gain n * distance n + defect n) :
    ∀ n, distance n ≤ defectEnvelope gain defect initial n := by
  intro n
  induction n with
  | zero => simpa [defectEnvelope] using hstart
  | succ n ih =>
      calc
        distance (n + 1) ≤ gain n * distance n + defect n := hstep n
        _ ≤ gain n * defectEnvelope gain defect initial n + defect n :=
          add_le_add (mul_le_mul_of_nonneg_left ih (hgain n)) le_rfl
        _ = defectEnvelope gain defect initial (n + 1) := by
          rw [defectEnvelope]

/-- A declared absolute cache budget follows from the transported row-map
and stage-defect terms. -/
theorem absolute_cache_error
    {rowGain direct diameter transportedDefect error budget : ℝ}
    (herror : error ≤ rowGain * direct * diameter + transportedDefect)
    (hbudget : rowGain * direct * diameter + transportedDefect ≤ budget) :
    error ≤ budget :=
  herror.trans hbudget

/-- Optional normalization is valid only with an independently positive mean
floor. -/
theorem normalized_bridge_of_mean_floor
    {absolute mean meanFloor normalizedBound : ℝ}
    (hfloor : 0 < meanFloor)
    (hmean : meanFloor ≤ mean)
    (habsolute : absolute ≤ normalizedBound * meanFloor)
    (hnormalized : 0 ≤ normalizedBound) :
    absolute / mean ≤ normalizedBound := by
  have hmeanpos : 0 < mean := hfloor.trans_le hmean
  apply (div_le_iff₀ hmeanpos).2
  calc
    absolute ≤ normalizedBound * meanFloor := habsolute
    _ ≤ normalizedBound * mean := mul_le_mul_of_nonneg_left hmean hnormalized

/-- Normalization with a signed tensor mean uses an independently positive
absolute-mean floor. -/
theorem normalized_bridge_of_abs_mean_floor
    {absolute mean meanFloor normalizedBound : ℝ}
    (hfloor : 0 < meanFloor)
    (hmean : meanFloor ≤ |mean|)
    (habsolute : absolute ≤ normalizedBound * meanFloor)
    (hnormalized : 0 ≤ normalizedBound) :
    absolute / |mean| ≤ normalizedBound := by
  have hmeanpos : 0 < |mean| := hfloor.trans_le hmean
  apply (div_le_iff₀ hmeanpos).2
  calc
    absolute ≤ normalizedBound * meanFloor := habsolute
    _ ≤ normalizedBound * |mean| :=
      mul_le_mul_of_nonneg_left hmean hnormalized

/-- The row-map gain occurs once before all downstream gains. -/
theorem row_map_applied_once
    {delta0 delta1 delta2 direct rowDefect gain stageDefect : ℝ}
    (hgain : 0 ≤ gain)
    (hrow : delta1 ≤ direct * delta0 + rowDefect)
    (hstage : delta2 ≤ gain * delta1 + stageDefect) :
    delta2 ≤ gain * (direct * delta0 + rowDefect) + stageDefect := by
  exact hstage.trans
    (add_le_add
      (mul_le_mul_of_nonneg_left hrow hgain)
      (le_refl stageDefect))

end SameSourceBridge
end PldrLlmCurvatureSandpile
