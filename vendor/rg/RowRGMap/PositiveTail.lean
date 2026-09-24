import Mathlib

namespace RowRGMap
namespace PositiveTail

open Filter

def cumulativeLog (increment : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ t ∈ Finset.range n, increment t

noncomputable def energy (initial : ℝ) (increment : ℕ → ℝ) (n : ℕ) : ℝ :=
  initial * Real.exp (cumulativeLog increment n)

@[simp] theorem cumulativeLog_zero (increment : ℕ → ℝ) :
    cumulativeLog increment 0 = 0 := by
  simp [cumulativeLog]

theorem cumulativeLog_succ (increment : ℕ → ℝ) (n : ℕ) :
    cumulativeLog increment (n + 1) =
      cumulativeLog increment n + increment n := by
  simp [cumulativeLog, Finset.sum_range_succ]

@[simp] theorem energy_zero (initial : ℝ) (increment : ℕ → ℝ) :
    energy initial increment 0 = initial := by
  simp [energy]

theorem energy_succ (initial : ℝ) (increment : ℕ → ℝ) (n : ℕ) :
    energy initial increment (n + 1) =
      Real.exp (increment n) * energy initial increment n := by
  rw [energy, energy, cumulativeLog_succ, Real.exp_add]
  ring

theorem log_energy_ratio {initial : ℝ} (hinitial : 0 < initial)
    (increment : ℕ → ℝ) (n : ℕ) :
    Real.log (energy initial increment n / initial) =
      cumulativeLog increment n := by
  simp [energy, ne_of_gt hinitial]

theorem energy_tendsto_zero_iff
    {initial : ℝ} (hinitial : 0 < initial) (increment : ℕ → ℝ) :
    Tendsto (energy initial increment) atTop (nhds 0) ↔
      Tendsto (cumulativeLog increment) atTop atBot := by
  constructor
  · intro henergy
    apply Real.tendsto_exp_comp_nhds_zero.mp
    have hscaled := henergy.div_const initial
    simpa [energy, ne_of_gt hinitial] using hscaled
  · intro hsum
    have hexp : Tendsto
        (fun n => Real.exp (cumulativeLog increment n)) atTop (nhds 0) :=
      Real.tendsto_exp_comp_nhds_zero.mpr hsum
    change Tendsto
      (fun n => initial * Real.exp (cumulativeLog increment n))
      atTop (nhds 0)
    simpa using tendsto_const_nhds.mul hexp

end PositiveTail
end RowRGMap
