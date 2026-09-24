import RowRGMap.CriticalFlow

namespace RowRGMap
namespace CriticalSource

open Filter
open AffineCocycle
open CriticalFlow

def blockRG (blockFactor : ℕ) (edge : Edge) : Edge :=
  block (List.replicate blockFactor edge)

@[simp] theorem blockRG_zero (edge : Edge) :
    blockRG 0 edge = identity := by
  simp [blockRG]

@[simp] theorem blockRG_succ (blockFactor : ℕ) (edge : Edge) :
    blockRG (blockFactor + 1) edge =
      compose (blockRG blockFactor edge) edge := by
  simp [blockRG, List.replicate_succ]

/-- Blocking an arbitrary number of critical edges preserves unit gain and
adds their sources. -/
theorem blockRG_critical (blockFactor : ℕ) (source : ℝ) :
    blockRG blockFactor { gain := 1, source := source } =
      { gain := 1, source := (blockFactor : ℝ) * source } := by
  induction blockFactor with
  | zero =>
      ext <;> simp [blockRG, identity]
  | succ blockFactor inductionHypothesis =>
      rw [blockRG_succ, inductionHypothesis]
      ext <;> simp [compose]
      ring

def blockIterate (blockFactor : ℕ) : ℕ → Edge → Edge
  | 0, edge => edge
  | n + 1, edge => blockRG blockFactor (blockIterate blockFactor n edge)

/-- After n arbitrary-factor critical blocking steps the source scaling field
is multiplied by blockFactor^n exactly. -/
theorem blockIterate_critical (blockFactor : ℕ) (source : ℝ) :
    ∀ n,
      blockIterate blockFactor n { gain := 1, source := source } =
        { gain := 1,
          source := (blockFactor : ℝ) ^ n * source } := by
  intro n
  induction n with
  | zero => simp [blockIterate]
  | succ n inductionHypothesis =>
      rw [blockIterate, inductionHypothesis, blockRG_critical]
      apply Edge.ext
      · simp
      · simp [pow_succ]
        ring

/-- A positive critical source is relevant for every integer blocking factor
at least two. -/
theorem critical_source_tendsto_atTop_arbitrary
    {blockFactor : ℕ} (hblock : 2 ≤ blockFactor)
    {source : ℝ} (hsource : 0 < source) :
    Tendsto
      (fun n =>
        (blockIterate blockFactor n
          { gain := 1, source := source }).source)
      atTop atTop := by
  have hbase : (1 : ℝ) < (blockFactor : ℝ) := by
    exact_mod_cast (lt_of_lt_of_le (by norm_num : 1 < 2) hblock)
  have hp : Tendsto (fun n : ℕ => (blockFactor : ℝ) ^ n) atTop atTop :=
    tendsto_pow_atTop_atTop_of_one_lt hbase
  have hscaled := hp.atTop_mul_const hsource
  simpa only [blockIterate_critical] using hscaled

def binaryIterate : ℕ → Edge → Edge
  | 0, edge => edge
  | n + 1, edge => binaryRG (binaryIterate n edge)

theorem binaryIterate_critical (source : ℝ) :
    ∀ n,
      binaryIterate n { gain := 1, source := source } =
        { gain := 1, source := (2 : ℝ) ^ n * source } := by
  intro n
  induction n with
  | zero => simp [binaryIterate]
  | succ n inductionHypothesis =>
      rw [binaryIterate, inductionHypothesis]
      apply Edge.ext
      · simp [binaryRG, compose]
      · simp [binaryRG, compose, pow_succ]
        ring

theorem critical_source_tendsto_atTop {source : ℝ} (hsource : 0 < source) :
    Tendsto
      (fun n => (binaryIterate n { gain := 1, source := source }).source)
      atTop atTop := by
  have hp : Tendsto (fun n : ℕ => (2 : ℝ) ^ n) atTop atTop :=
    tendsto_pow_atTop_atTop_of_one_lt (by norm_num)
  have hscaled := hp.atTop_mul_const hsource
  simpa only [binaryIterate_critical] using hscaled

end CriticalSource
end RowRGMap


