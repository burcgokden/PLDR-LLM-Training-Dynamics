import RowRGMap.AffineCocycle
import RowRGMap.RecurrentFaces

/-!
# Homogeneous and transported-source sectors

The exact affine orbit splits into a propagated initial condition and a
transported source sector.  Nonnegativity makes the two-sector extinction
criterion branch free.
-/

namespace RowRGMap
namespace CollapseCriterion

open Filter
open AffineCocycle

def orbit (gain source : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => gain n * orbit gain source initial n + source n

def homogeneous (gain : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => gain n * homogeneous gain initial n

def transported (gain source : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => gain n * transported gain source n + source n

theorem orbit_decomposition (gain source : ℕ → ℝ) (initial : ℝ) :
    ∀ n, orbit gain source initial n =
      homogeneous gain initial n + transported gain source n := by
  intro n
  induction n with
  | zero => simp [orbit, homogeneous, transported]
  | succ n inductionHypothesis =>
      simp only [orbit, homogeneous, transported, inductionHypothesis]
      ring

theorem homogeneous_nonnegative
    {gain : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hinitial : 0 ≤ initial) :
    ∀ n, 0 ≤ homogeneous gain initial n := by
  intro n
  induction n with
  | zero => simpa [homogeneous] using hinitial
  | succ n inductionHypothesis =>
      simp only [homogeneous]
      exact mul_nonneg (hgain n) inductionHypothesis

theorem transported_nonnegative
    {gain source : ℕ → ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hsource : ∀ n, 0 ≤ source n) :
    ∀ n, 0 ≤ transported gain source n := by
  intro n
  induction n with
  | zero => simp [transported]
  | succ n inductionHypothesis =>
      simp only [transported]
      exact add_nonneg (mul_nonneg (hgain n) inductionHypothesis) (hsource n)

theorem orbit_nonnegative
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hsource : ∀ n, 0 ≤ source n)
    (hinitial : 0 ≤ initial) :
    ∀ n, 0 ≤ orbit gain source initial n := by
  intro n
  rw [orbit_decomposition]
  exact add_nonneg
    (homogeneous_nonnegative hgain hinitial n)
    (transported_nonnegative hgain hsource n)

theorem sectors_suffice
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hhomogeneous : Tendsto (homogeneous gain initial) atTop (nhds 0))
    (htransported : Tendsto (transported gain source) atTop (nhds 0)) :
    Tendsto (orbit gain source initial) atTop (nhds 0) := by
  have hsum := hhomogeneous.add htransported
  have hfunction : orbit gain source initial =
      fun n => homogeneous gain initial n + transported gain source n := by
    funext n
    exact orbit_decomposition gain source initial n
  rw [hfunction]
  simpa using hsum

theorem homogeneous_le_orbit
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hsource : ∀ n, 0 ≤ source n)
    (n : ℕ) :
    homogeneous gain initial n ≤ orbit gain source initial n := by
  rw [orbit_decomposition]
  exact le_add_of_nonneg_right (transported_nonnegative hgain hsource n)

theorem transported_le_orbit
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hinitial : 0 ≤ initial)
    (n : ℕ) :
    transported gain source n ≤ orbit gain source initial n := by
  rw [orbit_decomposition]
  exact le_add_of_nonneg_left (homogeneous_nonnegative hgain hinitial n)

theorem sectors_necessary
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hsource : ∀ n, 0 ≤ source n)
    (hinitial : 0 ≤ initial)
    (horbit : Tendsto (orbit gain source initial) atTop (nhds 0)) :
    Tendsto (homogeneous gain initial) atTop (nhds 0) ∧
      Tendsto (transported gain source) atTop (nhds 0) := by
  constructor
  · exact squeeze_zero
      (homogeneous_nonnegative hgain hinitial)
      (homogeneous_le_orbit hgain hsource)
      horbit
  · exact squeeze_zero
      (transported_nonnegative hgain hsource)
      (transported_le_orbit hgain hinitial)
      horbit

theorem orbit_tendsto_zero_iff_sectors
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n) (hsource : ∀ n, 0 ≤ source n)
    (hinitial : 0 ≤ initial) :
    Tendsto (orbit gain source initial) atTop (nhds 0) ↔
      Tendsto (homogeneous gain initial) atTop (nhds 0) ∧
      Tendsto (transported gain source) atTop (nhds 0) := by
  constructor
  · exact sectors_necessary hgain hsource hinitial
  · rintro ⟨hhomogeneous, htransported⟩
    exact sectors_suffice hhomogeneous htransported

/-- At every step the nonnegative source is bounded by the successor energy.
This is the finite inequality behind the necessity of vanishing restarts. -/
theorem source_le_successor
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n)
    (hsource : ∀ n, 0 ≤ source n)
    (hinitial : 0 ≤ initial) (n : ℕ) :
    source n ≤ orbit gain source initial (n + 1) := by
  simp only [orbit]
  have horbit := orbit_nonnegative hgain hsource hinitial n
  nlinarith [mul_nonneg (hgain n) horbit]

/-- Pointwise extinction of a nonnegative two-sector decomposition is
equivalent to separate extinction of both sectors. -/
theorem zero_orbit_iff_zero_sectors
    {gain source : ℕ → ℝ} {initial : ℝ}
    (hgain : ∀ n, 0 ≤ gain n)
    (hsource : ∀ n, 0 ≤ source n)
    (hinitial : 0 ≤ initial) (n : ℕ) :
    orbit gain source initial n = 0 ↔
      homogeneous gain initial n = 0 ∧
      transported gain source n = 0 := by
  rw [orbit_decomposition]
  constructor
  · intro hsum
    have hhom := homogeneous_nonnegative hgain hinitial n
    have htrans := transported_nonnegative hgain hsource n
    constructor <;> linarith
  · rintro ⟨hhom, htrans⟩
    simp [hhom, htrans]


/-- Vanishing restart used in the canonical recurrent-face obstruction. -/
noncomputable def recurrentRestart (m : ℕ) : ℝ :=
  1 / ((m : ℝ) + 1)

theorem recurrent_restart_positive (m : ℕ) :
    0 < recurrentRestart m := by
  simp [recurrentRestart]
  positivity

theorem recurrent_restart_tendsto_zero :
    Tendsto recurrentRestart atTop (nhds 0) := by
  change Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) atTop (nhds 0)
  exact tendsto_one_div_add_atTop_nhds_zero_nat

/-- An exact orbit with recurrent face hits, vanishing restarts, and unit
excursions. -/
noncomputable def recurrentFaceEnergy (n : ℕ) : ℝ :=
  if n % 3 = 0 then 0
  else if n % 3 = 1 then recurrentRestart (n / 3)
  else 1

@[simp] theorem recurrent_face_energy_zero (m : ℕ) :
    recurrentFaceEnergy (3 * m) = 0 := by
  simp [recurrentFaceEnergy]

@[simp] theorem recurrent_face_energy_restart (m : ℕ) :
    recurrentFaceEnergy (3 * m + 1) = recurrentRestart m := by
  simp [recurrentFaceEnergy]
  congr 1
  omega

@[simp] theorem recurrent_face_energy_excursion (m : ℕ) :
    recurrentFaceEnergy (3 * m + 2) = 1 := by
  simp [recurrentFaceEnergy]

theorem recurrent_face_canonical_edges (m : ℕ) :
    canonical 0 (recurrentRestart m) =
        { gain := 0, source := recurrentRestart m } ∧
      canonical (recurrentRestart m) 1 =
        { gain := (m : ℝ) + 1, source := 0 } ∧
      canonical 1 0 = { gain := 0, source := 0 } := by
  have hpositive := recurrent_restart_positive m
  have hm : 0 < (m : ℝ) + 1 := by positivity
  constructor
  · simp [canonical]
  constructor
  · ext <;> simp [canonical, recurrentRestart, hm]
  · simp [canonical]

theorem recurrent_face_edges_replay (m : ℕ) :
    act (canonical 0 (recurrentRestart m)) 0 = recurrentRestart m ∧
      act (canonical (recurrentRestart m) 1) (recurrentRestart m) = 1 ∧
      act (canonical 1 0) 1 = 0 := by
  constructor
  · exact canonical_act (le_refl 0)
  constructor
  · exact canonical_act (le_of_lt (recurrent_restart_positive m))
  · exact canonical_act (by norm_num)

theorem recurrent_face_energy_not_tendsto_zero :
    ¬ Tendsto recurrentFaceEnergy atTop (nhds 0) := by
  intro hcollapse
  rw [Metric.tendsto_atTop] at hcollapse
  obtain ⟨N, hN⟩ := hcollapse (1 / 2) (by norm_num)
  have hfar := hN (3 * N + 2) (by omega)
  norm_num at hfar

open RecurrentFaces

theorem recurrent_face_energy_nonnegative (n : ℕ) :
    0 ≤ recurrentFaceEnergy n := by
  by_cases hzero : n % 3 = 0
  · simp [recurrentFaceEnergy, hzero]
  by_cases hone : n % 3 = 1
  · simp [recurrentFaceEnergy, hone]
    exact le_of_lt (recurrent_restart_positive (n / 3))
  · simp [recurrentFaceEnergy, hzero, hone]

/-- The canonical obstruction orbit, now instantiated as a genuine recurrent
face schedule rather than merely listed as a scalar sequence. -/
def obstructionSchedule : RecurrentFaceSchedule recurrentFaceEnergy where
  time := fun j => 3 * j
  strictMono_time := by
    intro a b hab
    exact (Nat.mul_lt_mul_left (by decide : 0 < 3)).2 hab
  face := recurrent_face_energy_zero
  successive := by
    intro j n hleft hright
    have hn : n = 3 * j + 1 ∨ n = 3 * j + 2 := by
      omega
    rcases hn with rfl | rfl
    · rw [recurrent_face_energy_restart]
      exact ne_of_gt (recurrent_restart_positive j)
    · rw [recurrent_face_energy_excursion]
      norm_num

@[simp] theorem obstruction_restart (j : ℕ) :
    excursionRestart obstructionSchedule j = recurrentRestart j := by
  simp [excursionRestart, obstructionSchedule,
    recurrent_face_energy_restart]

theorem obstruction_restart_tendsto_zero :
    Tendsto (excursionRestart obstructionSchedule) atTop (nhds 0) := by
  convert recurrent_restart_tendsto_zero using 1
  funext j
  exact obstruction_restart j

/-- Vanishing restarts are not sufficient for collapse, even for a valid
recurrent-face schedule with nonnegative energy. -/
theorem restart_tendsto_zero_not_sufficient :
    ∃ energy : ℕ → ℝ, ∃ schedule : RecurrentFaceSchedule energy,
      (∀ n, 0 ≤ energy n) ∧
      Tendsto (excursionRestart schedule) atTop (nhds 0) ∧
      ¬ Tendsto energy atTop (nhds 0) := by
  exact ⟨recurrentFaceEnergy, obstructionSchedule,
    recurrent_face_energy_nonnegative,
    obstruction_restart_tendsto_zero,
    recurrent_face_energy_not_tendsto_zero⟩

end CollapseCriterion
end RowRGMap
