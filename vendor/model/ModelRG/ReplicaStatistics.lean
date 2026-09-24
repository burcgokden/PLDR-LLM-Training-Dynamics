import ModelRG.FiniteEmission

/- Finite observation identities. The manuscript proves the conditional-law,
   single-pass continuity and limiting interpretations separately. -/
noncomputable section
namespace ModelRG.ReplicaStatistics
open scoped BigOperators
variable {ι κ : Type*} [Fintype ι] [Fintype κ]

theorem weighted_pair_variance (w x : ι → ℝ) (hw : ∑ i, w i = 1) :
    (1 / 2 : ℝ) * (∑ i, ∑ j, w i * w j * (x i - x j)^2) =
      ModelRG.fisherEnergy w x := by
  have inner (i : ι) : (∑ j, w j * (x i - x j)^2) =
      ModelRG.fisherEnergy w x + (ModelRG.weightedMean w x - x i)^2 := by
    calc
      _ = ∑ j, w j * (x j - x i)^2 := by
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = _ := ModelRG.FiniteEmission.centered_square w x (x i) hw
  have variance : (∑ i, w i * (ModelRG.weightedMean w x - x i)^2) =
      ModelRG.fisherEnergy w x := by
    unfold ModelRG.fisherEnergy
    apply Finset.sum_congr rfl
    intro i _
    ring
  simp_rw [mul_assoc, ← Finset.mul_sum, inner, mul_add]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, hw, one_mul, variance]
  ring

theorem binary_mixture_variance (p v0 v1 m0 m1 : ℝ) :
    (1-p) * (v0 + (m0-((1-p)*m0+p*m1))^2) +
      p * (v1 + (m1-((1-p)*m0+p*m1))^2) =
      (1-p)*v0 + p*v1 + p*(1-p)*(m1-m0)^2 := by
  ring

theorem binary_peak_bound (n delta p : ℝ) (hn : 0 ≤ n)
    (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ n*delta^2*p*(1-p) ∧ n*delta^2*p*(1-p) ≤ n*delta^2/4 := by
  constructor
  · positivity
  · nlinarith [mul_nonneg (mul_nonneg hn (sq_nonneg delta)) (sq_nonneg (p-1/2))]

theorem gram_sum_square (w : ι → ℝ) (b : ι → κ → ℝ) :
    (∑ i, ∑ j, w i * w j * (∑ k, b i k * b j k)) =
      ∑ k, (∑ i, w i * b i k)^2 := by
  classical
  have expansion (k : κ) : (∑ i, w i * b i k)^2 =
      ∑ i, ∑ j, (w i * b i k) * (w j * b j k) := by
    rw [pow_two, Finset.sum_mul]
    simp_rw [Finset.mul_sum]
  simp_rw [expansion]
  calc
    _ = ∑ i, ∑ j, ∑ k, (w i * b i k) * (w j * b j k) := by
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = ∑ i, ∑ k, ∑ j, (w i * b i k) * (w j * b j k) := by
      apply Finset.sum_congr rfl
      intro i _
      rw [Finset.sum_comm]
    _ = _ := by rw [Finset.sum_comm]

theorem gram_quadratic_nonnegative (w : ι → ℝ) (b : ι → κ → ℝ) :
    0 ≤ ∑ i, ∑ j, w i * w j * (∑ k, b i k * b j k) := by
  rw [gram_sum_square]
  exact Finset.sum_nonneg (fun k _ => sq_nonneg _)

theorem empirical_mean_correction (s mean2 variance : ℝ) (hs : s ≠ 0) :
    mean2 + (s-1)/s * variance = (mean2-variance/s)+variance := by
  field_simp
  ring

def hellingerSquared (p q : ι → ℝ) : ℝ :=
  (1/2 : ℝ) * ∑ i, (Real.sqrt (p i)-Real.sqrt (q i))^2

theorem hellinger_affinity (p q : ι → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    hellingerSquared p q = 1-∑ i, Real.sqrt (p i)*Real.sqrt (q i) := by
  have point (i : ι) : (Real.sqrt (p i)-Real.sqrt (q i))^2 =
      p i + q i - 2*(Real.sqrt (p i)*Real.sqrt (q i)) := by
    nlinarith [Real.sq_sqrt (hp i), Real.sq_sqrt (hq i)]
  unfold hellingerSquared
  simp_rw [point]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hp1, hq1]
  ring

theorem hellinger_bounds (p q : ι → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    0 ≤ hellingerSquared p q ∧ hellingerSquared p q ≤ 1 := by
  constructor
  · unfold hellingerSquared
    positivity
  · rw [hellinger_affinity p q hp hq hp1 hq1]
    have h : 0 ≤ ∑ i, Real.sqrt (p i)*Real.sqrt (q i) :=
      Finset.sum_nonneg (fun i _ => mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
    linarith

theorem stochastic_affinity (K : κ → ι → ℝ) (p q : ι → ℝ)
    (hK : ∀ j i, 0 ≤ K j i) (hK1 : ∀ i, ∑ j, K j i = 1)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i) :
    (∑ i, Real.sqrt (p i)*Real.sqrt (q i)) ≤
      ∑ j, Real.sqrt (∑ i, K j i*p i)*Real.sqrt (∑ i, K j i*q i) := by
  have term (j : κ) : (∑ i, K j i*Real.sqrt (p i)*Real.sqrt (q i)) ≤
      Real.sqrt (∑ i, K j i*p i)*Real.sqrt (∑ i, K j i*q i) := by
    have bound := Real.sum_sqrt_mul_sqrt_le Finset.univ
      (fun i => mul_nonneg (hK j i) (hp i))
      (fun i => mul_nonneg (hK j i) (hq i))
    have point (i : ι) : Real.sqrt (K j i*p i)*Real.sqrt (K j i*q i) =
        K j i*Real.sqrt (p i)*Real.sqrt (q i) := by
      rw [Real.sqrt_mul (hK j i), Real.sqrt_mul (hK j i)]
      calc
        _ = (Real.sqrt (K j i))^2*Real.sqrt (p i)*Real.sqrt (q i) := by ring
        _ = _ := by rw [Real.sq_sqrt (hK j i)]
    simpa only [point] using bound
  have sum_bound := Finset.sum_le_sum (s := Finset.univ) (fun j _ => term j)
  rw [Finset.sum_comm] at sum_bound
  simp_rw [← Finset.sum_mul, hK1, one_mul] at sum_bound
  exact sum_bound

theorem hellinger_channel_contraction (K : κ → ι → ℝ) (p q : ι → ℝ)
    (hK : ∀ j i, 0 ≤ K j i) (hK1 : ∀ i, ∑ j, K j i = 1)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    hellingerSquared (fun j => ∑ i, K j i*p i) (fun j => ∑ i, K j i*q i) ≤
      hellingerSquared p q := by
  have positive (v : ι → ℝ) (hv : ∀ i, 0 ≤ v i) (j : κ) :
      0 ≤ ∑ i, K j i*v i :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hK j i) (hv i))
  have normalized (v : ι → ℝ) (hv : ∑ i, v i = 1) :
      (∑ j, ∑ i, K j i*v i) = 1 := by
    rw [Finset.sum_comm]
    simp_rw [← Finset.sum_mul, hK1, one_mul]
    exact hv
  rw [hellinger_affinity _ _ (positive p hp) (positive q hq)
      (normalized p hp1) (normalized q hq1), hellinger_affinity p q hp hq hp1 hq1]
  linarith [stochastic_affinity K p q hK hK1 hp hq]

end ModelRG.ReplicaStatistics
