import Mathlib

namespace RowRGMap
namespace RecurrentFaces

open Filter

structure RecurrentFaceSchedule (energy : ℕ → ℝ) where
  time : ℕ → ℕ
  strictMono_time : StrictMono time
  face : ∀ j, energy (time j) = 0
  successive : ∀ j n, time j < n → n < time (j + 1) → energy n ≠ 0

def excursionInterval {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) : Finset ℕ :=
  Finset.Ioc (schedule.time j) (schedule.time (j + 1))

theorem excursionInterval_nonempty {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) :
    (excursionInterval schedule j).Nonempty := by
  refine ⟨schedule.time (j + 1), ?_⟩
  simp only [excursionInterval, Finset.mem_Ioc, le_rfl, and_true]
  exact schedule.strictMono_time (Nat.lt_succ_self j)

noncomputable def excursionPeak {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) : ℝ :=
  (excursionInterval schedule j).sup'
    (excursionInterval_nonempty schedule j) energy

noncomputable def excursionRestart {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) : ℝ :=
  energy (schedule.time j + 1)

noncomputable def excursionAmplification {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) : ℝ :=
  if excursionRestart schedule j = 0 then 0
  else excursionPeak schedule j / excursionRestart schedule j

theorem face_times_unbounded {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) :
    Tendsto schedule.time atTop atTop :=
  schedule.strictMono_time.tendsto_atTop

theorem interval_cover {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) {n : ℕ}
    (hstart : schedule.time 0 < n) :
    ∃ j, n ∈ excursionInterval schedule j := by
  have hexists : ∃ m, n ≤ schedule.time m := by
    exact ⟨n, schedule.strictMono_time.id_le n⟩
  let m := Nat.find hexists
  have hm : n ≤ schedule.time m := Nat.find_spec hexists
  have hmpos : 0 < m := by
    by_contra hnot
    have hmzero : m = 0 := Nat.eq_zero_of_not_pos hnot
    rw [hmzero] at hm
    exact (not_le_of_gt hstart) hm
  obtain ⟨j, hj⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hmpos)
  have hright : n ≤ schedule.time (j + 1) := by
    calc
      n ≤ schedule.time m := hm
      _ = schedule.time (j + 1) := by rw [hj, Nat.succ_eq_add_one]
  have hleft : schedule.time j < n := by
    by_contra hnot
    have hnle : n ≤ schedule.time j := le_of_not_gt hnot
    have hminimal : m ≤ j := Nat.find_min' hexists hnle
    omega
  exact ⟨j, by simpa [excursionInterval] using And.intro hleft hright⟩

theorem energy_le_excursionPeak {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) {j n : ℕ}
    (hn : n ∈ excursionInterval schedule j) :
    energy n ≤ excursionPeak schedule j := by
  exact Finset.le_sup' energy hn

theorem excursionPeak_nonnegative {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy)
    (henergy : ∀ n, 0 ≤ energy n) (j : ℕ) :
    0 ≤ excursionPeak schedule j := by
  let n := schedule.time (j + 1)
  have hn : n ∈ excursionInterval schedule j := by
    simp only [n, excursionInterval, Finset.mem_Ioc, le_rfl, and_true]
    exact schedule.strictMono_time (Nat.lt_succ_self j)
  exact (henergy n).trans (energy_le_excursionPeak schedule hn)

theorem peak_zero_of_restart_zero {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ)
    (hzero : excursionRestart schedule j = 0) :
    excursionPeak schedule j = 0 := by
  have hnext : schedule.time j + 1 = schedule.time (j + 1) := by
    have hle : schedule.time j + 1 ≤ schedule.time (j + 1) :=
      Nat.succ_le_of_lt (schedule.strictMono_time (Nat.lt_succ_self j))
    apply le_antisymm hle
    by_contra hnot
    have hlt : schedule.time j + 1 < schedule.time (j + 1) :=
      Nat.lt_of_not_ge hnot
    have hne := schedule.successive j (schedule.time j + 1)
      (Nat.lt_succ_self _) hlt
    exact hne hzero
  unfold excursionPeak
  apply Finset.sup'_eq_of_forall
  intro n hn
  have hbounds := (Finset.mem_Ioc.mp hn)
  have hn_eq : n = schedule.time (j + 1) := by omega
  simpa [hn_eq] using schedule.face (j + 1)

theorem peak_eq_restart_mul_amplification {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) :
    excursionPeak schedule j =
      excursionRestart schedule j * excursionAmplification schedule j := by
  by_cases hzero : excursionRestart schedule j = 0
  · simp [excursionAmplification, hzero, peak_zero_of_restart_zero schedule j hzero]
  · simp [excursionAmplification, hzero]
    field_simp

theorem restart_mem_excursionInterval {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) :
    schedule.time j + 1 ∈ excursionInterval schedule j := by
  simp only [excursionInterval, Finset.mem_Ioc]
  exact ⟨Nat.lt_succ_self _,
    Nat.succ_le_of_lt (schedule.strictMono_time (Nat.lt_succ_self j))⟩

theorem excursionRestart_nonnegative {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy)
    (henergy : ∀ n, 0 ≤ energy n) (j : ℕ) :
    0 ≤ excursionRestart schedule j := by
  exact henergy (schedule.time j + 1)

theorem excursionRestart_le_peak {energy : ℕ → ℝ}
    (schedule : RecurrentFaceSchedule energy) (j : ℕ) :
    excursionRestart schedule j ≤ excursionPeak schedule j := by
  exact energy_le_excursionPeak schedule
    (restart_mem_excursionInterval schedule j)

theorem energy_tendsto_zero_iff_excursionPeak
    {energy : ℕ → ℝ} (schedule : RecurrentFaceSchedule energy)
    (henergy : ∀ n, 0 ≤ energy n) :
    Tendsto energy atTop (nhds 0) ↔
      Tendsto (excursionPeak schedule) atTop (nhds 0) := by
  constructor
  · intro hcollapse
    rw [Metric.tendsto_atTop] at hcollapse ⊢
    intro ε hε
    obtain ⟨N, hN⟩ := hcollapse ε hε
    refine ⟨N, ?_⟩
    intro j hj
    have hnonneg := excursionPeak_nonnegative schedule henergy j
    have hlt : excursionPeak schedule j < ε := by
      unfold excursionPeak
      rw [Finset.sup'_lt_iff]
      intro n hn
      have hjn : N ≤ n := by
        have hinterval := Finset.mem_Ioc.mp hn
        have hjtime : j ≤ schedule.time j :=
          schedule.strictMono_time.id_le j
        omega
      have hdistance := hN n hjn
      simpa [Real.dist_eq, abs_of_nonneg (henergy n)] using hdistance
    simpa [Real.dist_eq, abs_of_nonneg hnonneg] using hlt
  · intro hpeaks
    rw [Metric.tendsto_atTop] at hpeaks ⊢
    intro ε hε
    obtain ⟨J, hJ⟩ := hpeaks ε hε
    refine ⟨schedule.time J + 1, ?_⟩
    intro n hn
    have hn_after : schedule.time J < n := by omega
    have htime0 : schedule.time 0 ≤ schedule.time J :=
      schedule.strictMono_time.monotone (Nat.zero_le J)
    have hstart : schedule.time 0 < n := lt_of_le_of_lt htime0 hn_after
    obtain ⟨j, hjmem⟩ := interval_cover schedule hstart
    have hJj : J ≤ j := by
      by_contra hnot
      have hsucc : j + 1 ≤ J := by omega
      have htimes : schedule.time (j + 1) ≤ schedule.time J :=
        schedule.strictMono_time.monotone hsucc
      have hright : n ≤ schedule.time (j + 1) :=
        (Finset.mem_Ioc.mp hjmem).2
      omega
    have hp_distance := hJ j hJj
    have hp_nonnegative := excursionPeak_nonnegative schedule henergy j
    have hp_lt : excursionPeak schedule j < ε := by
      simpa [Real.dist_eq, abs_of_nonneg hp_nonnegative] using hp_distance
    have he_le := energy_le_excursionPeak schedule hjmem
    have he_lt : energy n < ε := he_le.trans_lt hp_lt
    simpa [Real.dist_eq, abs_of_nonneg (henergy n)] using he_lt

theorem excursionRestart_tendsto_zero_of_collapse
    {energy : ℕ → ℝ} (schedule : RecurrentFaceSchedule energy)
    (henergy : ∀ n, 0 ≤ energy n)
    (hcollapse : Tendsto energy atTop (nhds 0)) :
    Tendsto (excursionRestart schedule) atTop (nhds 0) := by
  have hpeaks :=
    (energy_tendsto_zero_iff_excursionPeak schedule henergy).mp hcollapse
  exact squeeze_zero
    (excursionRestart_nonnegative schedule henergy)
    (excursionRestart_le_peak schedule)
    hpeaks

theorem restart_times_amplification_tendsto_zero_of_collapse
    {energy : ℕ → ℝ} (schedule : RecurrentFaceSchedule energy)
    (henergy : ∀ n, 0 ≤ energy n)
    (hcollapse : Tendsto energy atTop (nhds 0)) :
    Tendsto
      (fun j => excursionRestart schedule j * excursionAmplification schedule j)
      atTop (nhds 0) := by
  have hpeaks :=
    (energy_tendsto_zero_iff_excursionPeak schedule henergy).mp hcollapse
  have hequality :
      (fun j => excursionRestart schedule j * excursionAmplification schedule j) =
        excursionPeak schedule := by
    funext j
    exact (peak_eq_restart_mul_amplification schedule j).symm
  rw [hequality]
  exact hpeaks

end RecurrentFaces
end RowRGMap

