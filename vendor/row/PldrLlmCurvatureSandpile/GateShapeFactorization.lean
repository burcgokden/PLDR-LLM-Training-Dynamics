/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Mixed gate-shape factorization
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace GateShapeFactorization

open Filter
open scoped BigOperators

/-- One nonnegative coordinate of the final affine LayerNorm row energy. -/
def coordinateEnergy {d : ℕ}
    (gate shapeNorm : Fin d → ℝ) (coordinate : Fin d) : ℝ :=
  (gate coordinate * shapeNorm coordinate) ^ 2

/-- The exact diagonal gate-shape energy. -/
def gateShapeEnergy {d : ℕ} (gate shapeNorm : Fin d → ℝ) : ℝ :=
  ∑ coordinate, coordinateEnergy gate shapeNorm coordinate

theorem coordinate_energy_nonnegative {d : ℕ}
    (gate shapeNorm : Fin d → ℝ) (coordinate : Fin d) :
    0 ≤ coordinateEnergy gate shapeNorm coordinate := by
  exact sq_nonneg _

theorem gate_shape_energy_nonnegative {d : ℕ}
    (gate shapeNorm : Fin d → ℝ) :
    0 ≤ gateShapeEnergy gate shapeNorm := by
  unfold gateShapeEnergy
  exact Finset.sum_nonneg fun coordinate _ =>
    coordinate_energy_nonnegative gate shapeNorm coordinate

theorem coordinate_energy_le_gate_shape_energy {d : ℕ}
    (gate shapeNorm : Fin d → ℝ) (coordinate : Fin d) :
    coordinateEnergy gate shapeNorm coordinate ≤
      gateShapeEnergy gate shapeNorm := by
  unfold gateShapeEnergy
  exact Finset.single_le_sum
    (fun index _ => coordinate_energy_nonnegative gate shapeNorm index)
    (Finset.mem_univ coordinate)

/-- A finite gate-shape energy vanishes exactly when every mixed coordinate
vanishes. Each coordinate may close through either factor or their product. -/
theorem gate_shape_energy_zero_iff {d : ℕ}
    (gate shapeNorm : Fin d → ℝ) :
    gateShapeEnergy gate shapeNorm = 0 ↔
      ∀ coordinate, gate coordinate * shapeNorm coordinate = 0 := by
  constructor
  · intro hzero coordinate
    have hle : coordinateEnergy gate shapeNorm coordinate ≤
        gateShapeEnergy gate shapeNorm :=
      coordinate_energy_le_gate_shape_energy gate shapeNorm coordinate
    have hnonnegative := coordinate_energy_nonnegative gate shapeNorm coordinate
    unfold coordinateEnergy at hle hnonnegative
    nlinarith
  · intro hcoordinate
    unfold gateShapeEnergy coordinateEnergy
    apply Finset.sum_eq_zero
    intro coordinate _
    rw [hcoordinate coordinate]
    norm_num

/-- On a finite feature set, total mixed energy collapses exactly when every
gate-shape coordinate energy collapses.  No coordinate is required to close
through the gate factor alone or the normalized-shape factor alone. -/
theorem coordinate_energy_collapse_iff {d : ℕ}
    (gate shapeNorm : ℕ → Fin d → ℝ) :
    Tendsto
        (fun step => gateShapeEnergy (gate step) (shapeNorm step))
        atTop (nhds 0) ↔
      ∀ coordinate, Tendsto
        (fun step => coordinateEnergy (gate step) (shapeNorm step) coordinate)
        atTop (nhds 0) := by
  constructor
  · intro htotal coordinate
    apply squeeze_zero
    · intro step
      exact coordinate_energy_nonnegative
        (gate step) (shapeNorm step) coordinate
    · intro step
      exact coordinate_energy_le_gate_shape_energy
        (gate step) (shapeNorm step) coordinate
    · exact htotal
  · intro hcoordinate
    have hsum := tendsto_finsetSum (Finset.univ : Finset (Fin d))
      (fun coordinate _ => hcoordinate coordinate)
    simpa [gateShapeEnergy] using hsum

/-- Coordinatewise gate convergence collapses the complete finite energy when
the normalized-shape coordinates have one uniform bound. -/
theorem gate_shape_energy_collapse_of_gate_collapse {d : ℕ}
    (gate shapeNorm : ℕ → Fin d → ℝ) (shapeBound : ℝ)
    (hshapeBound : 0 ≤ shapeBound)
    (hshape : ∀ step coordinate,
      |shapeNorm step coordinate| ≤ shapeBound)
    (hgate : ∀ coordinate,
      Tendsto (fun step => gate step coordinate) atTop (nhds 0)) :
    Tendsto
      (fun step => gateShapeEnergy (gate step) (shapeNorm step))
      atTop (nhds 0) := by
  apply (coordinate_energy_collapse_iff gate shapeNorm).2
  intro coordinate
  apply squeeze_zero
  · intro step
    exact coordinate_energy_nonnegative
      (gate step) (shapeNorm step) coordinate
  · intro step
    unfold coordinateEnergy
    have habsolute :
        |gate step coordinate * shapeNorm step coordinate| ≤
          shapeBound * |gate step coordinate| := by
      rw [abs_mul]
      have := mul_le_mul_of_nonneg_left
        (hshape step coordinate) (abs_nonneg (gate step coordinate))
      simpa [mul_comm] using this
    have hupper :
        0 ≤ shapeBound * |gate step coordinate| :=
      mul_nonneg hshapeBound (abs_nonneg _)
    calc
      (gate step coordinate * shapeNorm step coordinate) ^ 2 =
          |gate step coordinate * shapeNorm step coordinate| ^ 2 := by
        rw [sq_abs]
      _ ≤ (shapeBound * |gate step coordinate|) ^ 2 := by
        nlinarith [abs_nonneg
          (gate step coordinate * shapeNorm step coordinate)]
  · have habsolute :
        Tendsto (fun step => |gate step coordinate|)
          atTop (nhds 0) := by
      simpa using (hgate coordinate).abs
    have hscaled :
        Tendsto (fun step => shapeBound * |gate step coordinate|)
          atTop (nhds 0) := by
      simpa using tendsto_const_nhds.mul habsolute
    simpa using hscaled.pow 2

end GateShapeFactorization
end PldrLlmCurvatureSandpile
