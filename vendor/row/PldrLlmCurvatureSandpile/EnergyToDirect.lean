/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Energy-to-direct projection

Projection from complete nonnegative block energies to grid coordinates,
physical row covers, and strict entry above the persistent floor.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace EnergyToDirect

/-- A positive lower weight converts a complete energy upper bound into a
bound for each registered scalar coordinate. -/
theorem block_energy_controls_grid
    {coordinate energy lowerWeight : ℝ}
    (hlower : 0 < lowerWeight)
    (hcoordinate : lowerWeight * coordinate ^ 2 ≤ energy) :
    coordinate ^ 2 ≤ energy / lowerWeight := by
  exact (le_div_iff₀ hlower).2 (by simpa [mul_comm] using hcoordinate)

/-- Adding the independent cover remainder transfers a grid estimate to the
direct row-map seminorm. -/
theorem energy_cover_controls_direct
    {direct grid cover energyBound : ℝ}
    (hgrid : grid ≤ energyBound)
    (hcover : direct ≤ grid + cover) :
    direct ≤ energyBound + cover := by
  exact hcover.trans (add_le_add hgrid le_rfl)

/-- A vanishing transient and a criterion strictly above the full floor give
eventual strict direct entry. -/
theorem entry_above_energy_floor
    (transient : ℕ → ℝ) {floor criterion : ℝ}
    (hgap : floor < criterion)
    (hvanish : Filter.Tendsto transient Filter.atTop (nhds 0)) :
    ∃ N, ∀ n ≥ N, floor + transient n < criterion := by
  have hopen : Set.Iio (criterion - floor) ∈ nhds (0 : ℝ) := by
    exact Iio_mem_nhds (sub_pos.mpr hgap)
  obtain ⟨N, hN⟩ := Filter.eventually_atTop.1 (hvanish hopen)
  refine ⟨N, ?_⟩
  intro n hn
  have ht : transient n < criterion - floor := hN n hn
  linarith

end EnergyToDirect
end PldrLlmCurvatureSandpile
