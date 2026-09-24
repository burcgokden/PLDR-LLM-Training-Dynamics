/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Observable moving-metric energy dissipation

Finite identities and comparison kernels used by the observable row-map
collapse argument. Every source and metric-change term remains explicit.
-/
import Mathlib

open scoped BigOperators

namespace PldrLlmCurvatureSandpile
namespace ObservableEnergyDissipation

/-- Energy in a finite diagonal metric. -/
def diagonalEnergy {Coordinate : Type*} [Fintype Coordinate]
    (metric state : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, metric coordinate * (state coordinate) ^ 2

/-- Bilinear pairing induced by a finite diagonal metric. -/
def diagonalPairing {Coordinate : Type*} [Fintype Coordinate]
    (metric left right : Coordinate → ℝ) : ℝ :=
  ∑ coordinate, metric coordinate * left coordinate * right coordinate

/-- Exact work caused by changing the diagonal metric at the successor state. -/
def metricShapeWork {Coordinate : Type*} [Fintype Coordinate]
    (metric metricNext stateNext : Coordinate → ℝ) : ℝ :=
  ∑ coordinate,
    (metricNext coordinate - metric coordinate) * (stateNext coordinate) ^ 2

/-- Exact finite energy identity for an AdamW-form step in a moving diagonal
metric. The direction may contain decay, loss, clipping, preconditioning,
moment-lag, and residual source terms; the identity hides none of them.
-/
theorem moving_metric_adamw_identity
    {Coordinate : Type*} [Fintype Coordinate]
    (metric metricNext state direction : Coordinate → ℝ) (eta : ℝ) :
    diagonalEnergy metricNext
          (fun coordinate => state coordinate - eta * direction coordinate)
        - diagonalEnergy metric state
      = -2 * eta * diagonalPairing metric state direction
        + eta ^ 2 * diagonalEnergy metric direction
        + metricShapeWork metric metricNext
          (fun coordinate => state coordinate - eta * direction coordinate) := by
  classical
  simp only [diagonalEnergy, diagonalPairing, metricShapeWork,
    ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib,
    Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- An explicit observable dissipation margin gives one-step contraction.
The hypothesis is exactly the signed balance obtained from the moving-metric
identity: direct alignment must dominate the quadratic step charge and the
moving-metric work by margin times the current energy.
-/
theorem explicit_one_step_margin_contraction
    {Coordinate : Type*} [Fintype Coordinate]
    (metric metricNext state direction : Coordinate → ℝ)
    (eta margin : ℝ)
    (hmargin :
      2 * eta * diagonalPairing metric state direction
          - eta ^ 2 * diagonalEnergy metric direction
          - metricShapeWork metric metricNext
            (fun coordinate => state coordinate - eta * direction coordinate)
        ≥ margin * diagonalEnergy metric state) :
    diagonalEnergy metricNext
        (fun coordinate => state coordinate - eta * direction coordinate)
      ≤ (1 - margin) * diagonalEnergy metric state := by
  have hidentity :=
    moving_metric_adamw_identity metric metricNext state direction eta
  linarith

/-- Pairing with an explicitly enumerated source ledger is the sum of the
individual source pairings. -/
theorem source_ledger_pairing
    {Coordinate Source : Type*} [Fintype Coordinate] [Fintype Source]
    (metric state : Coordinate → ℝ) (source : Source → Coordinate → ℝ) :
    diagonalPairing metric state (fun coordinate => ∑ item, source item coordinate)
      = ∑ item, diagonalPairing metric state (source item) := by
  classical
  simp only [diagonalPairing, Finset.mul_sum]
  rw [Finset.sum_comm]

/-- Weighted finite edge-shape energy. The supplied contrasts can already
include the graph incidence map and the row-map shape operator. -/
def shapeEnergy {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ) (contrast : Edge → Coordinate → ℝ) : ℝ :=
  ∑ edge, ∑ coordinate,
    weight edge * (contrast edge coordinate) ^ 2

/-- Secant cross term in finite edge-shape coordinates. -/
def shapeCross {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ)
    (contrast contrastIncrement : Edge → Coordinate → ℝ) : ℝ :=
  ∑ edge, ∑ coordinate,
    weight edge * contrast edge coordinate * contrastIncrement edge coordinate

/-- Signed shape gain, with the convention used by the collapse ledger. -/
def shapeGain {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ)
    (contrast contrastIncrement : Edge → Coordinate → ℝ) : ℝ :=
  -2 * shapeCross weight contrast contrastIncrement

/-- Nonnegative quadratic charge of a finite shape increment. -/
def shapeCharge {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ) (contrastIncrement : Edge → Coordinate → ℝ) : ℝ :=
  ∑ edge, ∑ coordinate,
    weight edge * (contrastIncrement edge coordinate) ^ 2

/-- Exact finite shape-work algebra. There is no discarded Taylor remainder:
successor energy minus current energy is signed secant gain plus quadratic
charge. -/
theorem exact_shape_work
    {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ)
    (contrast contrastIncrement : Edge → Coordinate → ℝ) :
    shapeEnergy weight
          (fun edge coordinate =>
            contrast edge coordinate + contrastIncrement edge coordinate)
        - shapeEnergy weight contrast
      = -shapeGain weight contrast contrastIncrement
        + shapeCharge weight contrastIncrement := by
  classical
  rw [show -shapeGain weight contrast contrastIncrement =
      2 * shapeCross weight contrast contrastIncrement by
        simp only [shapeGain]
        ring]
  simp only [shapeEnergy, shapeCross, shapeCharge,
    ← Finset.sum_sub_distrib, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro edge _
  apply Finset.sum_congr rfl
  intro coordinate _
  ring

/-- The exact shape charge is nonnegative when all edge weights are
nonnegative. -/
theorem shape_charge_nonnegative
    {Edge Coordinate : Type*} [Fintype Edge] [Fintype Coordinate]
    (weight : Edge → ℝ) (contrastIncrement : Edge → Coordinate → ℝ)
    (hweight : ∀ edge, 0 ≤ weight edge) :
    0 ≤ shapeCharge weight contrastIncrement := by
  exact Finset.sum_nonneg fun edge _ =>
    Finset.sum_nonneg fun coordinate _ =>
      mul_nonneg (hweight edge) (sq_nonneg _)

/-- Ordered scalar envelope for a nonautonomous affine recurrence. -/
def orderedEnvelope (gain forcing : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | step + 1 => gain step * orderedEnvelope gain forcing initial step + forcing step

/-- Comparison with the chronological affine envelope. Nonnegative gains are
the only monotonicity assumption needed to retain the order of the recursion.
-/
theorem ordered_affine_recursion
    (energy gain forcing : ℕ → ℝ)
    (hgain : ∀ step, 0 ≤ gain step)
    (hinitial : energy 0 ≤ orderedEnvelope gain forcing (energy 0) 0)
    (hstep : ∀ step,
      energy (step + 1) ≤ gain step * energy step + forcing step) :
    ∀ step, energy step ≤ orderedEnvelope gain forcing (energy 0) step := by
  intro step
  induction step with
  | zero => exact hinitial
  | succ step inductionHypothesis =>
      rw [orderedEnvelope]
      exact (hstep step).trans <|
        add_le_add
          (mul_le_mul_of_nonneg_left inductionHypothesis (hgain step))
          le_rfl

/-- Zero block forcing yields geometric decay under a fixed block gain. -/
theorem zero_block_forcing_geometric
    (energy : ℕ → ℝ) (rho : ℝ) (hrho : 0 ≤ rho)
    (hstep : ∀ block, energy (block + 1) ≤ rho * energy block) :
    ∀ block, energy block ≤ rho ^ block * energy 0 := by
  intro block
  induction block with
  | zero => simp
  | succ block inductionHypothesis =>
      calc
        energy (block + 1) ≤ rho * energy block := hstep block
        _ ≤ rho * (rho ^ block * energy 0) :=
          mul_le_mul_of_nonneg_left inductionHypothesis hrho
        _ = rho ^ (block + 1) * energy 0 := by rw [pow_succ]; ring

/-- Uniform block forcing gives the finite ordered geometric convolution.
This conclusion is valid without taking an asymptotic limit. -/
theorem uniform_block_forcing_convolution
    (energy forcing : ℕ → ℝ) (rho forcingBound : ℝ)
    (hrho : 0 ≤ rho)
    (hforcing : ∀ block, forcing block ≤ forcingBound)
    (hstep : ∀ block,
      energy (block + 1) ≤ rho * energy block + forcing block) :
    ∀ block,
      energy block ≤
        rho ^ block * energy 0
          + ∑ offset ∈ Finset.range block, rho ^ offset * forcingBound := by
  intro block
  induction block with
  | zero => simp
  | succ block inductionHypothesis =>
      calc
        energy (block + 1)
            ≤ rho * energy block + forcing block := hstep block
        _ ≤ rho *
              (rho ^ block * energy 0
                + ∑ offset ∈ Finset.range block,
                    rho ^ offset * forcingBound)
              + forcingBound := by
          exact add_le_add
            (mul_le_mul_of_nonneg_left inductionHypothesis hrho)
            (hforcing block)
        _ = rho ^ (block + 1) * energy 0
              + ∑ offset ∈ Finset.range (block + 1),
                  rho ^ offset * forcingBound := by
          simp_rw [← Finset.sum_mul]
          rw [geom_sum_succ, pow_succ]
          ring

/-- Under a strict block contraction, the uniform-forcing convolution has the
closed finite form used to read off its forcing floor. -/
theorem uniform_block_forcing_bound
    (energy forcing : ℕ → ℝ) (rho forcingBound : ℝ)
    (hrho : 0 ≤ rho) (hcontract : rho < 1)
    (hforcing : ∀ block, forcing block ≤ forcingBound)
    (hstep : ∀ block,
      energy (block + 1) ≤ rho * energy block + forcing block) :
    ∀ block,
      energy block ≤
        rho ^ block * energy 0
          + forcingBound * (1 - rho ^ block) / (1 - rho) := by
  intro block
  have hconvolution :=
    uniform_block_forcing_convolution energy forcing rho forcingBound
      hrho hforcing hstep block
  have hrho_ne : rho ≠ 1 := ne_of_lt hcontract
  rw [← Finset.sum_mul, geom_sum_eq hrho_ne] at hconvolution
  calc
    energy block ≤
        rho ^ block * energy 0
          + ((rho ^ block - 1) / (rho - 1)) * forcingBound :=
      hconvolution
    _ = rho ^ block * energy 0
          + forcingBound * (1 - rho ^ block) / (1 - rho) := by
      rw [show 1 - rho ^ block = -(rho ^ block - 1) by ring,
        show 1 - rho = -(rho - 1) by ring,
        mul_div_assoc, neg_div_neg_eq]
      ring

/-- A supplied pair attaining a finite diameter transfers any uniform
pairwise energy bound to that diameter. This isolates the exact kernel needed
after graph or effective-resistance estimates have supplied pairwiseBound.
-/
theorem supplied_pairwise_energy_to_diameter
    {Vertex : Type*} [Fintype Vertex]
    (pairwiseSq : Vertex → Vertex → ℝ)
    (diameterSq pairwiseBound : ℝ) (left right : Vertex)
    (hpairwise : ∀ first second, pairwiseSq first second ≤ pairwiseBound)
    (hdiameter : diameterSq = pairwiseSq left right) :
    diameterSq ≤ pairwiseBound := by
  rw [hdiameter]
  exact hpairwise left right

/-- Squared Frobenius deviation of a finite stack from a supplied anchor row. -/
def stackedDeviationSq
    {Vertex Coordinate : Type*} [Fintype Vertex] [Fintype Coordinate]
    (row : Vertex → Coordinate → ℝ) (anchor : Vertex) : ℝ :=
  ∑ vertex, ∑ coordinate,
    (row vertex coordinate - row anchor coordinate) ^ 2

/-- A uniform row-to-anchor squared bound controls the squared Frobenius norm
of the complete finite stack. -/
theorem stacked_generator_sq_bound
    {Vertex Coordinate : Type*} [Fintype Vertex] [Fintype Coordinate]
    (row : Vertex → Coordinate → ℝ) (anchor : Vertex) (bound : ℝ)
    (hrow : ∀ vertex,
      (∑ coordinate,
        (row vertex coordinate - row anchor coordinate) ^ 2) ≤ bound) :
    stackedDeviationSq row anchor ≤ (Fintype.card Vertex : ℝ) * bound := by
  unfold stackedDeviationSq
  calc
    (∑ vertex, ∑ coordinate,
        (row vertex coordinate - row anchor coordinate) ^ 2)
        ≤ ∑ _vertex : Vertex, bound :=
      Finset.sum_le_sum fun vertex _ => hrow vertex
    _ = (Fintype.card Vertex : ℝ) * bound := by simp

end ObservableEnergyDissipation
end PldrLlmCurvatureSandpile
