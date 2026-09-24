/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Intermittent radial-face block closure

Selected scalar kernels for composing the canonical face cocycle on arbitrary
finite blocks, comparing block anchors, and recovering complete finite-block
maxima from anchors and exact excursions.
-/
import Mathlib
import PldrLlmCurvatureSandpile.BlockExcursion
import PldrLlmCurvatureSandpile.ObservableEnergyDissipation
import PldrLlmCurvatureSandpile.RadialTangentialEnergy

namespace PldrLlmCurvatureSandpile
namespace RadialFaceBlockClosure

open Filter
open DirectObservableEnergy
open RadialTangentialEnergy

/-- The canonical affine energy envelope restarted at an arbitrary block
anchor. -/
noncomputable def shiftedFaceEnvelope
    (energySequence : ℕ → ℝ) (start : ℕ) : ℕ → ℝ :=
  NonautonomousAttraction.orderedEnvelope
    (fun offset => faceGain
      (energySequence (start + offset))
      (energySequence (start + offset + 1)))
    (fun offset => faceReopening
      (energySequence (start + offset))
      (energySequence (start + offset + 1)))
    (energySequence start)

/-- The canonical radial-face cocycle composes exactly from every finite block
anchor, including blocks containing true face restarts. -/
theorem exact_shifted_face_cocycle
    (energySequence : ℕ → ℝ) (start : ℕ) :
    ∀ horizon,
      energySequence (start + horizon)
        = shiftedFaceEnvelope energySequence start horizon := by
  intro horizon
  induction horizon with
  | zero => simp [shiftedFaceEnvelope, NonautonomousAttraction.orderedEnvelope]
  | succ horizon inductionHypothesis =>
      rw [shiftedFaceEnvelope,
        NonautonomousAttraction.orderedEnvelope]
      have hinduction :
          energySequence (start + horizon)
            = NonautonomousAttraction.orderedEnvelope
              (fun offset => faceGain
                (energySequence (start + offset))
                (energySequence (start + offset + 1)))
              (fun offset => faceReopening
                (energySequence (start + offset))
                (energySequence (start + offset + 1)))
              (energySequence start) horizon := by
        simpa [shiftedFaceEnvelope] using inductionHypothesis
      rw [← hinduction]
      have hstep := exact_face_affine_step
        (energySequence (start + horizon))
        (energySequence (start + horizon + 1))
      simpa [Nat.add_assoc] using hstep

/-- A uniform nonnegative block gain and bounded block injection give the
finite geometric convolution bound for block anchors. -/
theorem uniform_anchor_convolution
    (anchor injection : ℕ → ℝ) (rho injectionBound : ℝ)
    (hrho : 0 ≤ rho)
    (hinjection : ∀ block, injection block ≤ injectionBound)
    (hstep : ∀ block,
      anchor (block + 1) ≤ rho * anchor block + injection block) :
    ∀ block,
      anchor block ≤ rho ^ block * anchor 0
        + ∑ offset ∈ Finset.range block,
            rho ^ offset * injectionBound := by
  exact ObservableEnergyDissipation.uniform_block_forcing_convolution
    anchor injection rho injectionBound hrho hinjection hstep

/-- A nonnegative block-anchor recurrence with a uniform gain below one and a
vanishing forcing term has vanishing anchors. The proof closes the geometric
tail explicitly rather than treating the finite convolution bound as a limit
statement. -/
theorem anchor_tendsto_zero_of_uniform_gain
    (anchor injection : ℕ → ℝ) (rho : ℝ)
    (hrho : 0 ≤ rho) (hcontract : rho < 1)
    (hanchor : ∀ block, 0 ≤ anchor block)
    (hinjectionLimit : Tendsto injection atTop (nhds 0))
    (hstep : ∀ block,
      anchor (block + 1) ≤ rho * anchor block + injection block) :
    Tendsto anchor atTop (nhds 0) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hgap : 0 < 1 - rho := sub_pos.mpr hcontract
  let δ : ℝ := (1 - rho) * (ε / 2)
  have hδ : 0 < δ := mul_pos hgap (half_pos hε)
  obtain ⟨N, hN⟩ := (Metric.tendsto_atTop.1 hinjectionLimit) δ hδ
  have hgeometric :
      Tendsto (fun block : ℕ => rho ^ block * (anchor N + 1))
        atTop (nhds 0) := by
    simpa using
      (tendsto_pow_atTop_nhds_zero_of_lt_one hrho hcontract).mul_const
        (anchor N + 1)
  obtain ⟨K, hK⟩ :=
    (Metric.tendsto_atTop.1 hgeometric) (ε / 2) (half_pos hε)
  refine ⟨N + K, ?_⟩
  intro n hn
  obtain ⟨k, rfl⟩ :=
    Nat.exists_eq_add_of_le (Nat.le_trans (Nat.le_add_right N K) hn)
  have hk : K ≤ k := by omega
  have hinjectionBound : ∀ offset, injection (N + offset) ≤ δ := by
    intro offset
    have habs := hN (N + offset) (Nat.le_add_right N offset)
    exact (abs_lt.mp (by simpa [Real.dist_eq] using habs)).2.le
  have hshift : ∀ offset,
      anchor (N + (offset + 1)) ≤
        rho * anchor (N + offset) + injection (N + offset) := by
    intro offset
    simpa [Nat.add_assoc] using hstep (N + offset)
  have hbound := ObservableEnergyDissipation.uniform_block_forcing_bound
    (fun offset => anchor (N + offset))
    (fun offset => injection (N + offset))
    rho δ hrho hcontract hinjectionBound hshift k
  have hpowNonnegative : 0 ≤ rho ^ k := pow_nonneg hrho k
  have hpowLeOne : rho ^ k ≤ 1 := pow_le_one₀ hrho (le_of_lt hcontract)
  have hfirst : rho ^ k * anchor N < ε / 2 := by
    calc
      rho ^ k * anchor N ≤ rho ^ k * (anchor N + 1) := by
        apply mul_le_mul_of_nonneg_left _ hpowNonnegative
        linarith
      _ < ε / 2 := by
        have hmetric := hK k hk
        have hconstant : 0 ≤ anchor N + 1 := by linarith [hanchor N]
        simpa only [Real.dist_eq, sub_zero,
          abs_of_nonneg (mul_nonneg hpowNonnegative hconstant)] using hmetric
  have hforceIdentity :
      δ * (1 - rho ^ k) / (1 - rho)
        = (ε / 2) * (1 - rho ^ k) := by
    dsimp [δ]
    field_simp [ne_of_gt hgap]
  have hforcing : δ * (1 - rho ^ k) / (1 - rho) ≤ ε / 2 := by
    rw [hforceIdentity]
    nlinarith
  simp only [Nat.add_zero] at hbound
  have htotal : anchor (N + k) < ε := lt_of_le_of_lt hbound (by linarith)
  simpa [Real.dist_eq, abs_of_nonneg (hanchor (N + k))] using htotal

/-- The scalar geometric convolution generated by a restart sequence. -/
def geometricConvolution (rho : ℝ) (injection : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | block + 1 => rho * geometricConvolution rho injection block
      + injection block

/-- A geometric convolution of nonnegative gains and injections remains
nonnegative. -/
theorem geometric_convolution_nonnegative
    (rho : ℝ) (injection : ℕ → ℝ)
    (hrho : 0 ≤ rho) (hinjection : ∀ block, 0 ≤ injection block) :
    ∀ block, 0 ≤ geometricConvolution rho injection block := by
  intro block
  induction block with
  | zero => simp [geometricConvolution]
  | succ block inductionHypothesis =>
      simp only [geometricConvolution]
      exact add_nonneg (mul_nonneg hrho inductionHypothesis) (hinjection block)

/-- If the restart sequence vanishes and the geometric gain is uniformly
below one, then its exact convolution tail vanishes. -/
theorem geometric_convolution_tendsto_zero
    (rho : ℝ) (injection : ℕ → ℝ)
    (hrho : 0 ≤ rho) (hcontract : rho < 1)
    (hinjection : ∀ block, 0 ≤ injection block)
    (hinjectionLimit : Tendsto injection atTop (nhds 0)) :
    Tendsto (geometricConvolution rho injection) atTop (nhds 0) := by
  apply anchor_tendsto_zero_of_uniform_gain
    (geometricConvolution rho injection) injection rho hrho hcontract
    (geometric_convolution_nonnegative rho injection hrho hinjection)
    hinjectionLimit
  intro block
  simp [geometricConvolution]

/-- Vanishing block anchors and exact excursions are equivalent to vanishing
attained block maxima. This is the finite-block recovery kernel after an
anchor recurrence has been closed. -/
theorem anchor_excursion_maximum_collapse
    (energySequence : ℕ → ℝ) (start horizon : ℕ → ℕ)
    (henergy : ∀ block, 0 ≤ energySequence (start block)) :
    Tendsto
      (fun block => BlockExcursion.blockMaximum energySequence
        (start block) (horizon block)) atTop (nhds 0) ↔
      Tendsto (fun block => energySequence (start block))
        atTop (nhds 0) ∧
      Tendsto
        (fun block => BlockExcursion.blockExcursion energySequence
          (start block) (horizon block)) atTop (nhds 0) := by
  exact BlockExcursion.block_excursion_collapse_iff
    energySequence start horizon henergy

/-- Uniform block contraction with vanishing restart injection closes the
anchor limit; adding vanishing exact within-block excursions closes the
attained block maxima. -/
theorem intermittent_block_maximum_tendsto_zero
    (energySequence : ℕ → ℝ) (start horizon : ℕ → ℕ)
    (injection : ℕ → ℝ) (rho : ℝ)
    (hrho : 0 ≤ rho) (hcontract : rho < 1)
    (henergy : ∀ block, 0 ≤ energySequence (start block))
    (hinjectionLimit : Tendsto injection atTop (nhds 0))
    (hstep : ∀ block,
      energySequence (start (block + 1)) ≤
        rho * energySequence (start block) + injection block)
    (hexcursion : Tendsto
      (fun block => BlockExcursion.blockExcursion energySequence
        (start block) (horizon block)) atTop (nhds 0)) :
    Tendsto
      (fun block => BlockExcursion.blockMaximum energySequence
        (start block) (horizon block)) atTop (nhds 0) := by
  apply (anchor_excursion_maximum_collapse
    energySequence start horizon henergy).2
  exact ⟨anchor_tendsto_zero_of_uniform_gain
    (fun block => energySequence (start block)) injection rho
    hrho hcontract henergy hinjectionLimit hstep, hexcursion⟩

end RadialFaceBlockClosure
end PldrLlmCurvatureSandpile
