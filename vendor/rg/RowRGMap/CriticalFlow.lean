import RowRGMap.AffineCocycle

/-!
# Clean positive-affine RG flow

Binary blocking is enough to check the fixed points and the two relevant
linearized coordinates at the marginal edge `(gain, source) = (1, 0)`.
-/

namespace RowRGMap
namespace CriticalFlow

open AffineCocycle

def binaryRG (edge : Edge) : Edge :=
  compose edge edge

@[simp] theorem binary_gain (edge : Edge) :
    (binaryRG edge).gain = edge.gain ^ 2 := by
  simp [binaryRG, compose]
  ring

@[simp] theorem binary_source (edge : Edge) :
    (binaryRG edge).source = (edge.gain + 1) * edge.source := by
  simp [binaryRG, compose]
  ring

theorem binary_gain_fixed_iff (gain : ℝ) :
    gain ^ 2 = gain ↔ gain = 0 ∨ gain = 1 := by
  constructor
  · intro h
    have hfactor : gain * (gain - 1) = 0 := by
      nlinarith
    rcases mul_eq_zero.mp hfactor with hzero | hone
    · exact Or.inl hzero
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl) <;> norm_num

theorem nonnegative_binary_fixed_points {edge : Edge}
    (hfixed : binaryRG edge = edge) :
    edge.gain = 0 ∨ edge.gain = 1 := by
  apply (binary_gain_fixed_iff edge.gain).1
  have hcomponent := congrArg Edge.gain hfixed
  simpa using hcomponent

/-- The full fixed set of binary blocking: the memoryless line and the
source-free marginal point. -/
theorem binary_fixed_iff (edge : Edge) :
    binaryRG edge = edge ↔
      edge.gain = 0 ∨ (edge.gain = 1 ∧ edge.source = 0) := by
  constructor
  · intro hfixed
    rcases nonnegative_binary_fixed_points hfixed with hzero | hone
    · exact Or.inl hzero
    · right
      refine ⟨hone, ?_⟩
      have hsource := congrArg Edge.source hfixed
      rw [binary_source, hone] at hsource
      linarith
  · rintro (hzero | ⟨hone, hsource⟩)
    · apply Edge.ext <;> simp [binaryRG, compose, hzero]
    · apply Edge.ext <;> simp [binaryRG, compose, hone, hsource]

/-- The source-free marginal edge is an RG fixed point. -/
theorem critical_edge_fixed :
    binaryRG { gain := 1, source := 0 } =
      ({ gain := 1, source := 0 } : Edge) := by
  ext <;> norm_num [binaryRG, compose]

/-- The absorbing zero-memory edge is an RG fixed point. -/
theorem absorbing_edge_fixed :
    binaryRG { gain := 0, source := 0 } =
      ({ gain := 0, source := 0 } : Edge) := by
  ext <;> norm_num [binaryRG, compose]

/-- Exact nonlinear correction to the gain scaling field near one. -/
theorem gain_perturbation_scaling (delta : ℝ) :
    (1 + delta) ^ 2 - 1 = 2 * delta + delta ^ 2 := by
  ring

/-- At critical gain, binary blocking doubles the reopening field. -/
theorem critical_source_scaling (source : ℝ) :
    (binaryRG { gain := 1, source := source }).source = 2 * source := by
  simp [binaryRG, compose]
  ring

/-- The clean distance from criticality has the expected first-order factor
two and the exact quadratic correction. -/
theorem subcritical_distance_scaling (tau : ℝ) :
    1 - (1 - tau) ^ 2 = 2 * tau - tau ^ 2 := by
  ring

noncomputable def stationaryEnergy (gain source : ℝ) : ℝ :=
  source / (1 - gain)

theorem stationary_energy_fixed (gain source : ℝ)
    (hdenominator : 1 - gain ≠ 0) :
    gain * stationaryEnergy gain source + source =
      stationaryEnergy gain source := by
  simp [stationaryEnergy]
  field_simp [hdenominator]
  ring

theorem stationary_energy_nonnegative {gain source : ℝ}
    (hgain : gain < 1) (hsource : 0 ≤ source) :
    0 ≤ stationaryEnergy gain source := by
  exact div_nonneg hsource (le_of_lt (sub_pos.mpr hgain))

end CriticalFlow
end RowRGMap
