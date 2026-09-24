import Mathlib

/- Finite algebra behind the stand-alone potential-activity arguments.
   The symbols b and b' represent the logarithms of strictly positive bases. -/
namespace ModelRG
open scoped BigOperators

theorem potential_log_increment (p p' b b' : ℝ) :
    p' * b' - p * b = (p' - p) * b + p' * (b' - b) := by
  ring

theorem potential_increment_energy (n : ℕ) (w u v : Fin n → ℝ) :
    (∑ i, w i * (u i + v i)^2) =
      (∑ i, w i * (u i)^2) + (∑ i, w i * (v i)^2) +
        2 * (∑ i, w i * u i * v i) := by
  rw [Finset.mul_sum]
  simp only [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem potential_temporal_block (n : ℕ) (q : ℕ → ℝ) :
    (∑ i ∈ Finset.range n, (q (i+1) - q i)) = q n - q 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    ring

theorem potential_schedule_threshold (a r threshold : ℝ) (ha : 0 < a) :
    threshold < a * r ↔ threshold / a < r := by
  rw [div_lt_iff₀ ha]
  rw [mul_comm r a]

def dampedMoment (beta initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n+1 => beta * dampedMoment beta initial n

theorem zero_gradient_moment (beta initial : ℝ) (n : ℕ) :
    dampedMoment beta initial n = beta^n * initial := by
  induction n with
  | zero => simp [dampedMoment]
  | succ n ih =>
    simp only [dampedMoment, ih, pow_succ]
    ring

theorem potential_symmetric_increment (p p' b b' : ℝ) :
    p' * b' - p * b =
      (p' - p) * ((b + b') / 2) + ((p + p') / 2) * (b' - b) := by
  ring

theorem potential_corner_interaction (p p' b b' : ℝ) :
    p' * b' - p' * b - p * b' + p * b = (p' - p) * (b' - b) := by
  ring

theorem potential_endpoint_half_difference (p p' b b' : ℝ) :
    (p' - p) * ((b + b') / 2) - (p' - p) * b =
      (p' - p) * (b' - b) / 2 := by
  ring

end ModelRG
