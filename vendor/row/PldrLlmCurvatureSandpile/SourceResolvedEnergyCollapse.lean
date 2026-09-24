/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Global source-resolved energy comparison
-/
import Mathlib
import PldrLlmCurvatureSandpile.NonautonomousAttraction

namespace PldrLlmCurvatureSandpile
namespace SourceResolvedEnergyCollapse

open scoped BigOperators

/-- The physical-energy envelope is the shared chronological affine envelope. -/
abbrev energyEnvelope := NonautonomousAttraction.orderedEnvelope

/-- Nonnegative source gains and native charges transport in their actual
order. Individual gains are not required to be below one. -/
theorem global_source_resolved_energy_envelope
    (energy gain nativeCharge : ℕ → ℝ) {initial : ℝ}
    (hstart : energy 0 ≤ initial)
    (hgain : ∀ step, 0 ≤ gain step)
    (hstep : ∀ step,
      energy (step + 1) ≤ gain step * energy step + nativeCharge step) :
    ∀ step, energy step ≤ energyEnvelope gain nativeCharge initial step := by
  exact NonautonomousAttraction.ordered_product_convolution
    energy gain nativeCharge hstart hgain hstep

/-- Any registered nonnegative map energy is bounded by the total energy of
a finite registry. This is the finite-registry transfer kernel. -/
theorem map_energy_le_registry_total
    {Map : Type*} [Fintype Map]
    (energy : Map → ℝ) (hnonnegative : ∀ map, 0 ≤ energy map)
    (map : Map) :
    energy map ≤ ∑ index, energy index := by
  classical
  exact Finset.single_le_sum
    (fun index _ => hnonnegative index) (Finset.mem_univ map)

end SourceResolvedEnergyCollapse
end PldrLlmCurvatureSandpile
