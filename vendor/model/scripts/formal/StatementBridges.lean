import ModelRG
open scoped BigOperators

example (n diagonal cross : ℝ) (hn : 1 ≤ n) (hc : cross ≤ diagonal) :
    ModelRG.headSusceptibility n diagonal cross ≤ n * diagonal :=
  ModelRG.head_susceptibility_upper n diagonal cross hn hc

example {E : Type*} [NormedAddCommGroup E] (x e : E) (N s : ℝ)
    (hN : 0 ≤ N) (hs : 1 < s) :
    |N/(s-1)*‖x+e‖^2 - N/(s-1)*‖x‖^2| ≤
      N/(s-1)*(2*‖x‖*‖e‖+‖e‖^2) :=
  ModelRG.sample_norm_transfer x e N s hN hs

example {ι : Type*} [Fintype ι] (p v : ι → ℝ)
    (hp : ∀ i, 0 < p i) (hp1 : ∑ i, p i = 1) (hv : ∑ i, v i = 0) :
    2 * (∑ i, (v i)^2) ≤ ∑ i, (v i)^2 / p i :=
  ModelRG.simplex_tangent_hessian p v hp hp1 hv


-- A concrete two-step consequence retains the inequality and its forcing.
example (e : ℕ → ℝ) (h : ∀ n, e (n + 1) ≤ (1 / 2 : ℝ) * e n + 1 / 10) :
    e 2 ≤ (1 / 4 : ℝ) * e 0 + 3 / 20 := by
  have bound := ModelRG.accumulated_error_bound e (1 / 2) (1 / 10)
    (by norm_num) (by norm_num) h 2
  norm_num at bound
  linarith

example (w : ℕ → ℝ) (h : ∀ n, w (n + 1) = (9 / 10 : ℝ) * w n) :
    w 3 = (729 / 1000 : ℝ) * w 0 := by
  have bound := ModelRG.autonomous_decay w (9 / 10) h 3
  norm_num at bound
  linarith


example (c a cc aa : ℝ)
    (hc : |cc - c| ≤ (1 / 10 : ℝ))
    (ha : |aa - a| ≤ (1 / 10 : ℝ)) (margin : 1 / 5 < c - a) :
    aa < cc := by
  exact ModelRG.score_margin_positive c a cc aa (1 / 10) hc ha (by linarith)

example (w a b e : Fin 2 → ℝ) (hw : ∀ i, 0 ≤ w i)
    (h : ∀ i, |a i - b i| ≤ 2 * e i) :
    |w 0 * (a 0 - b 0) + w 1 * (a 1 - b 1)| ≤
      2 * (w 0 * e 0 + w 1 * e 1) := by
  simpa [Fin.sum_univ_two] using ModelRG.weighted_prefix_loss_bound w a b e hw h

-- Empty, one-step, and chronological two-step consequences of the evolving-law bound.
example (e d L : ℕ → ℝ) (hL : ∀ n, 0 ≤ L n)
    (hstep : ∀ n, e (n + 1) ≤ L n * e n + d n) : e 0 ≤ e 0 := by
  simpa using ModelRG.nonstationary_error_bound e d L hL hstep 0

example (e d L : ℕ → ℝ) (hL : ∀ n, 0 ≤ L n)
    (hstep : ∀ n, e (n + 1) ≤ L n * e n + d n) :
    e 1 ≤ e 0 * L 0 + d 0 := by
  simpa [Nat.Ico_zero_eq_range, Finset.sum_range_succ, Finset.prod_range_succ]
    using ModelRG.nonstationary_error_bound e d L hL hstep 1

example (e : ℕ → ℝ)
    (hstep : ∀ n, e (n + 1) ≤ (if n = 0 then 2 else 3) * e n +
      (if n = 0 then 5 else 7)) : e 2 ≤ 6 * e 0 + 22 := by
  have bound := ModelRG.nonstationary_error_bound e
    (fun n => if n = 0 then 5 else 7) (fun n => if n = 0 then 2 else 3)
    (by intro n; split_ifs <;> norm_num) hstep 2
  norm_num [Nat.Ico_zero_eq_range, Finset.sum_range_succ,
    Finset.prod_range_succ, Finset.prod_Ico_succ_top] at bound
  linarith

-- An incoming state equal to its forcing retains the cross moment.
example (x : Fin 2 → ℝ) :
    (∑ i, (1 / 2 : ℝ) * ((1 / 2 : ℝ) * x i + x i)^2) =
      (9 / 4 : ℝ) * (∑ i, (1 / 2 : ℝ) * (x i)^2) := by
  have h := ModelRG.weighted_second_moment_transport
    (fun _ : Fin 2 => (1 / 2 : ℝ)) x x (1 / 2 : ℝ)
  simp only [Fin.sum_univ_two] at h ⊢
  nlinarith

example {ι : Type*} [Fintype ι] (w x e : ι → ℝ)
    (orthogonal : ∑ i, w i * x i * e i = 0) :
    (∑ i, w i * (2 * x i + e i)^2) =
      4 * (∑ i, w i * (x i)^2) + (∑ i, w i * (e i)^2) := by
  have h := ModelRG.weighted_second_moment_orthogonal w x e 2 orthogonal
  norm_num at h
  exact h

-- A common origin and all interval pairs are explicit in these sample identities.
example {n m : ℕ} (d : Fin n → Fin m → ℝ) (r : Fin n → ℝ)
    (r0 : ℝ) (hn : 1 < n) (telescopes : ∀ i, ∑ j, d i j = r i-r0) :
    (∑ j, ∑ k, ModelRG.sampleCovR (fun i => d i j) (fun i => d i k)) =
      ModelRG.sampleCovR r r :=
  ModelRG.endpoint_sample_covariance d r r0 hn telescopes

example {n m : ℕ} (d : Fin n → Fin m → ℝ) :
    ModelRG.sampleCovR (fun i => ∑ j, d i j) (fun i => ∑ j, d i j) =
      ∑ j, ∑ k, ModelRG.sampleCovR (fun i => d i j) (fun i => d i k) :=
  (ModelRG.sample_covariance_sum d).symm

example {ι : Type*} [Fintype ι] (w x : ι → ℝ) :
    (∑ i, w i * (x i+x i)^2) - (∑ i, w i * (x i)^2) =
      3 * (∑ i, w i * (x i)^2) := by
  have h := ModelRG.signed_second_moment_difference w x x
  simp only [← pow_two, mul_assoc] at h
  linarith

-- Uniform empirical weights and the divisor-63 to divisor-64 conversion.
example (x y : Fin 64 → ℝ) (a b : ℝ) :
    (∑ i, (1 / 64 : ℝ) * (x i - a) * (y i - b)) =
      (63 / 64 : ℝ) * ModelRG.sampleCovR x y +
      (ModelRG.sampleMeanR x - a) * (ModelRG.sampleMeanR y - b) := by
  have hx : (∑ i, (1 / 64 : ℝ) * x i) = ModelRG.sampleMeanR x := by
    rw [← Finset.mul_sum]
    simp only [ModelRG.sampleMeanR, Nat.cast_ofNat]
    ring
  have hy : (∑ i, (1 / 64 : ℝ) * y i) = ModelRG.sampleMeanR y := by
    rw [← Finset.mul_sum]
    simp only [ModelRG.sampleMeanR, Nat.cast_ofNat]
    ring
  have h := ModelRG.weighted_cross_centering (fun _ : Fin 64 => (1 / 64 : ℝ))
    x y (ModelRG.sampleMeanR x) (ModelRG.sampleMeanR y) a b (by norm_num) hx hy
  rw [h]
  congr 1
  simp only [ModelRG.sampleCovR, Nat.cast_ofNat]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  ring

-- A nonzero fitted-mean error survives even with only two observations.
example (x y a : ℝ) :
    ((x-a)^2+(y-a)^2)/2 =
      ((x-(x+y)/2)^2+(y-(x+y)/2)^2)/2 + ((x+y)/2-a)^2 := by
  ring


-- Rectangular maps and arbitrary cross covariance remain visible in the type.
example (A : Matrix (Fin 2) (Fin 3) ℝ) (B : Matrix (Fin 3) (Fin 4) ℝ)
    (C : Matrix (Fin 4) (Fin 4) ℝ) :
    A * (B * C * B.transpose) * A.transpose =
      (A * B) * C * (A * B).transpose :=
  ModelRG.covariance_congruence_comp A B C

-- Exact images, with no coordinate-box approximation between the maps.
example (A : Matrix (Fin 2) (Fin 3) ℝ) (B : Matrix (Fin 3) (Fin 4) ℝ)
    (T : Set (Fin 4 → ℝ)) :
    A.mulVec '' (B.mulVec '' T) = (A * B).mulVec '' T :=
  ModelRG.linear_set_image_comp A B T


-- Both ordered cross terms survive a non-square observation map.
example (A : Matrix (Fin 2) (Fin 3) ℝ)
    (X : Matrix (Fin 3) (Fin 5) ℝ) (E : Matrix (Fin 2) (Fin 5) ℝ)
    (W : Matrix (Fin 5) (Fin 5) ℝ) :
    (A * X + E) * W * (A * X + E).transpose =
      A * (X * W * X.transpose) * A.transpose +
      A * (X * W * E.transpose) +
      (E * W * X.transpose) * A.transpose + E * W * E.transpose :=
  ModelRG.covariance_add_perturbation A X E W

-- The sample convention remains s/(s-1), and centering has a real premise.
example (x : Fin 4 → ℝ) (mu : ℝ) (hmean : ∑ i, x i = 4 * mu) :
    (17 / 3 : ℝ) * (∑ i, ‖x i - mu‖ ^ 2) ≤
      (68 / 3 : ℝ) * ((∑ i, ‖x i‖ ^ 2) / 4) := by
  have bound := ModelRG.sample_uncentered_bound x mu 17 (by norm_num) (by norm_num) hmean
  norm_num at bound ⊢
  exact bound


-- Independently stated pulse-error consequences with fixed, nonzero budgets.
example {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (x y : E) (hx : ‖x‖ ≤ 2) (hy : ‖y‖ ≤ 2) :
    ‖(1 / 2 : ℝ) • (x-y)‖ ≤ 2 :=
  ModelRG.odd_error_bound x y 2 hx hy

example {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (x y z : E) (hx : ‖x‖ ≤ 2) (hy : ‖y‖ ≤ 2) (hz : ‖z‖ ≤ 2) :
    ‖(1 / 2 : ℝ) • (x+y)-z‖ ≤ 4 := by
  convert ModelRG.even_error_bound x y z 2 hx hy hz using 1 <;> norm_num

example {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (r x y : E) (hr : ‖r‖ ≤ 7) (hx : ‖x‖ ≤ 2) (hy : ‖y‖ ≤ 2) :
    ‖r+x-(2 : ℝ) • y‖ ≤ 13 := by
  convert ModelRG.halving_arithmetic_bound r x y 7 2 hr hx hy using 1 <;> norm_num

-- Finite moment interventions retain both bias factors and all mixed sectors.
example (m g u : ℝ) :
    ((9/10 : ℝ)*(m+u)+(1/10 : ℝ)*g) - ((9/10 : ℝ)*m+(1/10 : ℝ)*g) = (9/10 : ℝ)*u := by ring

example (x e o y f p : ℝ) :
    (x+e-o)*(y+f-p) = x*y+e*f+o*p+x*f+e*y-(x*p+o*y+e*p+o*f) := by ring

example : (3 : ℝ) + ((11+7)/2-3)+(11-7)/2 = 11 :=
  ModelRG.pulse_reconstruct_plus 3 11 7

example (theta m g u : ℝ) :
    ((1-(1/100 : ℝ)*(1/10 : ℝ))*theta - (1/100 : ℝ)*((9/10 : ℝ)*(m+u)+(1-(9/10 : ℝ))*g)/((3/4 : ℝ)*2)) -
      ((1-(1/100 : ℝ)*(1/10 : ℝ))*theta - (1/100 : ℝ)*((9/10 : ℝ)*m+(1-(9/10 : ℝ))*g)/((3/4 : ℝ)*2)) =
      -(1/100 : ℝ)*(9/10 : ℝ)*u/((3/4 : ℝ)*2) :=
  ModelRG.adam_finite_first_moment theta (1/100) (1/10) (9/10) m g u (3/4) 2

-- Fine-tuning identities in independently fixed, nondegenerate cases.
example (a b : ℝ) :
    (3/4 : ℝ)*a*(-1/4)+(1/4)*b*(3/4) = (3/16 : ℝ)*(b-a) := by
  convert ModelRG.bernoulli_score_numerator (1/4) a b using 1 <;> ring

example : ((1:ℝ)^2+0^2)*(0^2+3^2)-(1*0+0*3)^2 = 9 := by
  have h := ModelRG.two_source_gram_determinant 1 0 0 3
  norm_num at h ⊢

example (a b g : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hd : g^2 ≤ a*b) (ht : a+b = 0) : g = 0 :=
  (ModelRG.two_zero_positive_gaps a b g ha hb hd ht).2.2

example (x y : Fin 2 → ℝ) :
    (∑ i, (1/2 : ℝ)*(x i-y i)^2) =
      (∑ i, (1/2 : ℝ)*(x i-(x 0+x 1)/2)^2) +
      (∑ i, (1/2 : ℝ)*(y i-(y 0+y 1)/2)^2) -
      2*(∑ i, (1/2 : ℝ)*(x i-(x 0+x 1)/2)*(y i-(y 0+y 1)/2)) +
      ((x 0+x 1)/2-(y 0+y 1)/2)^2 := by
  apply ModelRG.paired_discrepancy_centering
  · norm_num
  · simp only [Fin.sum_univ_two]; ring
  · simp only [Fin.sum_univ_two]; ring

-- A dependent pair of constant dictionary columns: z_i = a + 2b.
-- Nonuniform weights, and an exactly orthogonal nonzero residual.
example (a b : ℝ) :
    (∑ i : Fin 3, (![1, 2, 1] : Fin 3 → ℝ) i * ((![1, 2, 3] : Fin 3 → ℝ) i - 2)^2) ≤
    (∑ i : Fin 3, (![1, 2, 1] : Fin 3 → ℝ) i * ((![1, 2, 3] : Fin 3 → ℝ) i - (a + 2*b))^2) := by
  apply ModelRG.weighted_projection_lower_bound
    (![1, 2, 1] : Fin 3 → ℝ) (![1, 2, 3] : Fin 3 → ℝ) (fun _ => 2) (fun _ => a + 2*b)
  · intro i; fin_cases i <;> norm_num
  · simp [Fin.sum_univ_succ]; ring

example (a b : ℝ) :
    (∑ i : Fin 3, (![1, 2, 1] : Fin 3 → ℝ) i * ((![1, 2, 3] : Fin 3 → ℝ) i - (a + 2*b))^2) =
    (∑ i : Fin 3, (![1, 2, 1] : Fin 3 → ℝ) i * ((![1, 2, 3] : Fin 3 → ℝ) i - 2)^2) +
    (∑ i : Fin 3, (![1, 2, 1] : Fin 3 → ℝ) i * (2 - (a + 2*b))^2) := by
  apply ModelRG.weighted_residual_split
    (![1, 2, 1] : Fin 3 → ℝ) (![1, 2, 3] : Fin 3 → ℝ) (fun _ => 2) (fun _ => a + 2*b)
  simp [Fin.sum_univ_succ]; ring

-- A nonvacuous positive interval for a normalized observable.
example (x : ℝ) (h : |x - 2| ≤ (1/4 : ℝ)*2) : 0 < x := by
  exact (ModelRG.physical_relative_bracket 2 x (1/4) (by norm_num) (by norm_num) h).2.2

-- Identity observation retains a visible scaling direction.
example {V : Type*} [AddCommGroup V] [Module ℝ V]
    (A : V →ₗ[ℝ] V) (v : V) (lam : ℝ)
    (hv : v ≠ 0) (he : A v = lam • v) :
    (LinearMap.id : V →ₗ[ℝ] V) v ≠ 0 ∧ A v = lam • v := by
  simpa using ModelRG.physical_visible_eigenvector A A LinearMap.id v lam (by intro x; rfl) he hv

-- Complete desired readout types, independent of the exporter serialization.
example {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (v e : E) :
    |‖v + e‖ ^ 2 - ‖v‖ ^ 2| ≤ 2 * ‖v‖ * ‖e‖ + ‖e‖ ^ 2 :=
  ModelRG.readout_squared_norm_error_bound v e

example (s d r : ℝ) (hs : 0 ≤ s) (hd : 0 ≤ d) (hr : 0 ≤ r) (he : d ≤ r*s) :
    2*s*d+d^2 ≤ (2*r+r^2)*s^2 :=
  ModelRG.readout_relative_scale_budget s d r hs hd hr he

example (a b c D s d : ℝ) (hl : |a-b| ≤ D) (hr : |b-c| ≤ 2*s*d+d^2) :
    |a-c| ≤ D+2*s*d+d^2 :=
  ModelRG.readout_combined_law_budget a b c D s d hl hr

-- Equal scalar emissions can have distinct next-step factor dynamics.
example (eta : ℝ) :
    ((2 : ℝ) - eta / 2) * ((1 / 2 : ℝ) - 2 * eta) -
      ((1 - eta) * (1 - eta)) = -9 * eta / 4 := by
  have h := ModelRG.factor_gradient_displacement (2 : ℝ) (1 / 2) eta 1
  have k := ModelRG.factor_gradient_displacement (1 : ℝ) 1 eta 1
  ring_nf at h k ⊢

-- Binary color normalization is an independent specialization of the written identity.
example (j l r : Fin 2 → ℝ) (hl : ∑ a, l a = 1) (hr : ∑ a, r a = 1) :
    2 * (∑ a, j a) - 1 =
      2 * (∑ a, (j a - l a * r a)) +
      2 * (∑ a, (l a - 1 / 2) * (r a - 1 / 2)) := by
  have h := ModelRG.normalized_spatial_decomposition 2 2 (by norm_num) (by norm_num) j l r hl hr
  norm_num at h ⊢
  exact h

example (a b c d : ℝ) (ha : 0 < a) (hc : 0 < c) (hb : 0 ≤ b) :
    |d / c^2 - b / a^2| ≤ |d-b| / c^2 + b * |c-a| * (c+a) / (a^2 * c^2) :=
  ModelRG.moment_ratio_error_bound a b c d ha hc hb


-- Potential activity must retain both moving factors and the cross term.
example (p b dp db : ℝ) :
    (p+dp)*(b+db)-p*b = dp*b + p*db + dp*db := by
  have h := ModelRG.potential_log_increment p (p+dp) b (b+db)
  nlinarith [h]

example (q : ℕ → ℝ) : (q 1-q 0)+(q 2-q 1) = q 2-q 0 := by
  simpa [Finset.sum_range_succ] using ModelRG.potential_temporal_block 2 q

example (r : ℝ) : (3 : ℝ) < 2*r ↔ (3/2 : ℝ) < r :=
  ModelRG.potential_schedule_threshold 2 r 3 (by norm_num)

-- Independent complete types for finite ordered-area transport.
open ModelRG.PotentialBlock
example (x y z : ModelRG.PotentialBlock) :
    comp (comp x y) z = comp x (comp y z) := comp_assoc x y z
example (x : ModelRG.PotentialBlock) :
    comp zero x = x ∧ comp x zero = x := comp_identity x
example (x : ModelRG.PotentialBlock) :
    comp x (inverse x) = zero ∧ comp (inverse x) x = zero := comp_inverse x
example (r s : ℝ) (x y : ModelRG.PotentialBlock) :
    scale r s (comp x y) = comp (scale r s x) (scale r s y) := scale_comp r s x y
example (p b : ℝ) (x y : ModelRG.PotentialBlock) :
    exponentAllocation b (comp x y) =
      exponentAllocation b x + exponentAllocation (b + x.e) y ∧
    baseAllocation p (comp x y) =
      baseAllocation p x + baseAllocation (p + x.d) y := allocation_comp p b x y
example (p b : ℝ) (x : ModelRG.PotentialBlock) :
    exponentAllocation b x + baseAllocation p x =
      (p + x.d) * (b + x.e) - p * b := allocation_product p b x
example (x y : ModelRG.PotentialBlock) :
    (comp x y).area - (comp y x).area = y.d*x.e-x.d*y.e := area_order_difference x y

-- Finite predictive geometry, with complete independently stated types.
namespace FiniteEmissionBridges
open ModelRG
example {ι : Type*} [Fintype ι] (p v : ι → ℝ) (c : ℝ)
    (hp : ∑ i, p i = 1) :
    (∑ i, p i * (v i - c)^2) = fisherEnergy p v + (weightedMean p v - c)^2 :=
  FiniteEmission.centered_square p v c hp

example {ι : Type*} [Fintype ι] (p q v : ι → ℝ) (a b : ℝ)
    (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1) (ha : 0 ≤ a)
    (hl : ∀ i, a*p i ≤ q i) (hu : ∀ i, q i ≤ b*p i) :
    a*fisherEnergy p v ≤ fisherEnergy q v ∧ fisherEnergy q v ≤ b*fisherEnergy p v :=
  FiniteEmission.fisher_comparison p q v a b hp hq ha hl hu

example {E : Type*} [AddCommGroup E] [Module ℝ E]
    (B0 B1 : E →ₗ[ℝ] E →ₗ[ℝ] ℝ) (d e : E) (hsym : B0 d e = B0 e d) :
    (B1 (d+e) (d+e) - B0 d d)/2 =
      B0 e d + B0 e e/2 + (B1 (d+e) (d+e)-B0 (d+e) (d+e))/2 :=
  FiniteEmission.signed_bilinear_contrast B0 B1 d e hsym

-- A nonconstant binary observation makes the variance comparison nonvacuous.
example : ModelRG.fisherEnergy (fun _ : Fin 2 => (1/2 : ℝ))
    (fun i => if i = 0 then (0 : ℝ) else 2) = 1 := by
  norm_num [ModelRG.fisherEnergy, ModelRG.weightedMean, Fin.sum_univ_two]
end FiniteEmissionBridges


-- A uniform four-realization pair identity retains both normalization factors.
example (x : Fin 4 → ℝ) :
    (1/32 : ℝ) * (∑ i, ∑ j, (x i-x j)^2) =
      ModelRG.fisherEnergy (fun _ : Fin 4 => (1/4 : ℝ)) x := by
  have h := ModelRG.ReplicaStatistics.weighted_pair_variance
    (fun _ : Fin 4 => (1/4 : ℝ)) x (by norm_num)
  simp only [← Finset.mul_sum, mul_assoc] at h
  convert h using 1 <;> ring

example (n delta : ℝ) (hn : 0 ≤ n) :
    n*delta^2*(1/2 : ℝ)*(1-1/2) ≤ n*delta^2/4 :=
  (ModelRG.ReplicaStatistics.binary_peak_bound n delta (1/2) hn
    (by norm_num) (by norm_num)).2

example (p q : Fin 3 → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) :
    ModelRG.ReplicaStatistics.hellingerSquared p q ≤ 1 :=
  (ModelRG.ReplicaStatistics.hellinger_bounds p q hp hq hp1 hq1).2

-- A rectangular observation channel has a normalized column for every token.
example (K : Fin 2 → Fin 3 → ℝ) (p q : Fin 3 → ℝ)
    (hK : ∀ j i, 0 ≤ K j i) (hK1 : ∀ i, K 0 i+K 1 i=1)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hp1 : ∑ i, p i=1) (hq1 : ∑ i, q i=1) :
    ModelRG.ReplicaStatistics.hellingerSquared
      (fun j => ∑ i, K j i*p i) (fun j => ∑ i, K j i*q i) ≤
      ModelRG.ReplicaStatistics.hellingerSquared p q := by
  exact ModelRG.ReplicaStatistics.hellinger_channel_contraction K p q hK
    (by intro i; simpa [Fin.sum_univ_two] using hK1 i) hp hq hp1 hq1

-- Signed temporal weights require the complete Gram covariance, including crosses.
example (b : Fin 2 → Fin 3 → ℝ) :
    0 ≤ ∑ i, ∑ j, (if i=0 then (1 : ℝ) else -2) *
      (if j=0 then (1 : ℝ) else -2) * (∑ k, b i k*b j k) :=
  ModelRG.ReplicaStatistics.gram_quadratic_nonnegative
    (fun i : Fin 2 => if i=0 then (1 : ℝ) else -2) b

example (mean2 variance : ℝ) :
    mean2+(5/6 : ℝ)*variance=(mean2-variance/6)+variance := by
  have h := ModelRG.ReplicaStatistics.empirical_mean_correction
    (6 : ℝ) mean2 variance (by norm_num)
  norm_num at h ⊢
  exact h

-- Count-weighted variance, normalized and unnormalized, in finite coordinates.
example {ι : Type*} [Fintype ι] (c x : ι → ℝ) (s : ℝ)
    (hs : s ≠ 0) (hs1 : s-1 ≠ 0) (hc : ∑ i, c i = s) :
    (∑ i, ∑ j, c i*c j*(x i-x j)^2)/(2*s*(s-1)) =
      s/(s-1)*ModelRG.fisherEnergy (fun i => c i/s) x :=
  ModelRG.ReplicaBootstrap.count_pair_variance c x s hs hs1 hc

example {ι : Type*} [Fintype ι] {d : ℕ} (c : ι → ℝ) (x : ι → Fin d → ℝ)
    (s : ℝ) (hs : s ≠ 0) (hs1 : s-1 ≠ 0) (hc : ∑ i, c i = s) :
    (∑ k, (∑ i, ∑ j, c i*c j*(x i k-x j k)^2)/(2*s*(s-1)))/(d : ℝ) =
      (s/(s-1)*∑ k, ModelRG.fisherEnergy (fun i => c i/s) (fun i => x i k))/(d : ℝ) :=
  ModelRG.ReplicaBootstrap.count_vector_variance c x s hs hs1 hc

example {ι : Type*} [Fintype ι] (c x : ι → ℝ) (k : ι)
    (hc : ∀ i, i ≠ k → c i = 0) :
    (∑ i, ∑ j, c i*c j*(x i-x j)^2) = 0 :=
  ModelRG.ReplicaBootstrap.single_support_pair_zero c x k hc

example : (∑ i : Fin 2, ∑ j : Fin 2,
    ((if i=0 then 0 else 2 : ℝ)-(if j=0 then 0 else 2))^2)/(2*2*(2-1)) = 2 := by
  norm_num [Fin.sum_univ_two]

example : (∑ k : Fin 2, (∑ i : Fin 2, ∑ j : Fin 2,
    ((if i=0 then 0 else if k=0 then 2 else 4 : ℝ)-
     (if j=0 then 0 else if k=0 then 2 else 4))^2)/(2*2*(2-1)))/2 = 5 := by
  norm_num [Fin.sum_univ_two]

-- Actual finite tail definitions, with nested sets and one common reference.
example {ι : Type*} [DecidableEq ι] (J L : Finset ι) (r p : ι → ℝ)
    (hL : L ⊆ J) (hr : ModelRG.CategoricalScale.mass L r ≠ 0) :
    ModelRG.CategoricalScale.project L r (ModelRG.CategoricalScale.project J r p) =
      ModelRG.CategoricalScale.project J r p :=
  ModelRG.CategoricalScale.project_fine_after_coarse J L r p hL hr

example {ι : Type*} [DecidableEq ι] (J : Finset ι) (r : ι → ℝ)
    (q : ModelRG.CategoricalScale.Coarse J) (hr : ModelRG.CategoricalScale.mass J r ≠ 0) :
    ModelRG.CategoricalScale.aggregate J (ModelRG.CategoricalScale.restore J r q) = q :=
  ModelRG.CategoricalScale.aggregate_restore J r q hr

example (x : Fin 6 → ℝ) :
    (∑ s, (x s - (∑ t, x t)/6)^2)/5 =
      (6/5 : ℝ)*((∑ s, (x s)^2)/6-((∑ s, x s)/6)^2) := by
  have h := ModelRG.CategoricalScale.empirical_bias_variance x 6 (by simp) (by norm_num)
  norm_num at h ⊢
  exact h


-- The finite vocabulary reference is positive on the declared nonempty tail.
example {ι : Type*} [DecidableEq ι] (J : Finset ι) (r p : ι → ℝ)
    (hJ : J.Nonempty) (hr : ∀ i, 0 < r i) :
    0 < ModelRG.CategoricalScale.mass J r ∧
    ModelRG.CategoricalScale.tailDensityAverage J r p =
      ModelRG.CategoricalScale.mass J p / ModelRG.CategoricalScale.mass J r := by
  exact ⟨ModelRG.CategoricalScale.positive_mass J r hJ hr,
    ModelRG.CategoricalScale.tailDensityAverage_eq J r p
      (fun i _ => ne_of_gt (hr i))⟩

example {ι : Type*} [DecidableEq ι] (J : Finset ι) (r p : ι → ℝ)
    (i : ι) (hi : i ∈ J) (hr : ∀ j ∈ J, r j ≠ 0) :
    ModelRG.CategoricalScale.project J r p i / r i =
      (∑ j ∈ J, r j * (p j / r j)) / (∑ j ∈ J, r j) :=
  ModelRG.CategoricalScale.project_density_expectation J r p i hi hr

example {ι : Type*} [DecidableEq ι] (J : Finset ι) (r p : ι → ℝ)
    (i : ι) (hi : i ∉ J) :
    ModelRG.CategoricalScale.project J r p i / r i = p i / r i :=
  ModelRG.CategoricalScale.project_density_outside J r p i hi


-- Explicit positive block size and unequal residuals, coordinatewise in a covariance matrix.
example (B : ℝ) (r : Fin 2 → ℝ) :
    (∑ i : Fin 2, ∑ j : Fin 2, (B + if i=j then r i else 0)) / 2 =
      2 * B + (r 0 + r 1) / 2 := by
  have h := ModelRG.LatentCovariance.heterogeneous_susceptibility 2 (by decide) B r
    (fun i j => B + if i=j then r i else 0) (by intros; rfl)
  simpa [Fin.sum_univ_two] using h

example (B W : ℝ) :
    3 * B + (∑ _ : Fin 3, W) / 3 = 3 * B + W := by
  simpa using ModelRG.LatentCovariance.common_covariance_specialization 3 (by decide) B W

-- Context probabilities are native-variance weights, with a fixed nonempty panel.
example (p e : Fin 2 → ℝ) (hp : ∀ i, 0 < p i) :
    (e 0 + e 1)/(p 0 + p 1) =
      (p 0/(p 0+p 1))*(e 0/p 0)+(p 1/(p 0+p 1))*(e 1/p 1) := by
  simpa [Fin.sum_univ_two] using ModelRG.LatentCovariance.context_weighted_ratio p e hp

/- Exact finite context-risk contracts and nonuniform numerical bridges. -/
open scoped BigOperators

example : ∀ {ι : Type} [Fintype ι]
    (mass native residual : ι → ℝ), (∀ i, 0 < native i) →
    (∑ i, mass i * residual i) / (∑ i, mass i * native i) =
      ∑ i, ((mass i * native i) / (∑ j, mass j * native j)) *
        (residual i / native i) := @ModelRG.ContextRisk.weighted_ratio

example : ∀ {ι : Type} [Fintype ι]
    (weight risk : ι → ℝ) (threshold : ℝ),
    (∀ i, 0 ≤ weight i) → (∀ i, 0 ≤ risk i) →
    (∑ i, weight i = 1) → (0 < threshold) →
    (∑ i ∈ Finset.univ.filter (fun i => threshold < risk i), weight i) ≤
      min 1 ((∑ i, weight i * risk i) / threshold) := @ModelRG.ContextRisk.weighted_tail

example : ∀ (native residual count : ℝ), count ≠ 0 →
    (residual / count) / (native / count) = residual / native :=
    ModelRG.ContextRisk.means_ratio

-- Nonuniform weights, unequal native variances, and nonzero residuals.
example : ((1/4 : ℝ)*1 + (3/4)*9) / ((1/4)*1 + (3/4)*4) = (28/13 : ℝ) := by
  norm_num

-- The strict tail convention excludes equality at the threshold.
example : (1/5 : ℝ) ≤ min 1 (((4/5)*0 + (1/5)*1) / (1/2)) := by
  norm_num


/- Independent exact contracts for finite cache risk and stability. -/
example {ι : Type*} [Fintype ι] (w x : ι → ℝ) (mean cache : ℝ)
    (normalized : ∑ i, w i = 1) (actualMean : ∑ i, w i * x i = mean) :
    (∑ i, w i * (x i - cache)^2) =
      (∑ i, w i * (x i - mean)^2) + (mean - cache)^2 :=
  ModelRG.CacheRisk.empirical_budget w x mean cache normalized actualMean

example {ι : Type*} [Fintype ι] (a x : ι → ℝ) (s : ℝ) :
    (∑ i, a i * (s * x i)) = s * ∑ i, a i * x i :=
  ModelRG.CacheRisk.cache_sign a x s

example (e : ℕ → ℝ) (L d : ℝ) (nonnegative : 0 ≤ L) (initial : e 0 = 0)
    (step : ∀ t, e (t+1) ≤ L * e t + d) (t : ℕ) :
    e t ≤ d * ∑ k ∈ Finset.range t, L^k :=
  ModelRG.constant_stability_bound e L d nonnegative initial step t

/- Summing scalar identities supplies the actual finite coordinate bridge. -/
example {ι κ : Type*} [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : κ → ι → ℝ) (mean cache : κ → ℝ)
    (hw : ∑ i, w i = 1) (hm : ∀ k, ∑ i, w i * x k i = mean k) :
    (∑ k, ∑ i, w i * (x k i - cache k)^2) =
      (∑ k, ∑ i, w i * (x k i - mean k)^2) + ∑ k, (mean k - cache k)^2 := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  exact ModelRG.CacheRisk.empirical_budget w (x k) (mean k) (cache k) hw (hm k)

example : (1/4 : ℝ)*(1-2)^2 + (3/4 : ℝ)*(3-2)^2 =
    (1/4 : ℝ)*(1-5/2)^2 + (3/4 : ℝ)*(3-5/2)^2 + (5/2-2)^2 := by norm_num

example : (3 : ℝ) * ∑ j ∈ Finset.range 3, (2 : ℝ)^j = 21 := by norm_num [Finset.sum_range_succ]

example {ι κ : Type*} [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : ι → κ → ℝ) (mu c : κ → ℝ)
    (hw : ∑ i, w i = 1) (hm : ∀ k, ∑ i, w i * x i k = mu k) :
    (∑ i, w i * ∑ k, (x i k - c k)^2) =
      (∑ i, w i * ∑ k, (x i k - mu k)^2) + ∑ k, (mu k - c k)^2 :=
  ModelRG.CacheRisk.empirical_budget_euclidean w x mu c hw hm
example (a b c : ℝ) : (a-c)^2 = (a-b)^2+(b-c)^2+2*(a-b)*(b-c) :=
  ModelRG.CacheRisk.state_displacement a b c
example (u v : ℝ) : (1-u^2)*(1-v^2) ≤ (1-u*v)^2 :=
  ModelRG.OptimizerMemory.bias_product_bound u v

/- Independent exact types for finite vector cache transport. -/
example {κ : Type*} [Fintype κ] (a b c : κ → ℝ) :
    (∑ j, (a j-c j)^2) = (∑ j, (a j-b j)^2) +
      (∑ j, (b j-c j)^2) + 2 * ∑ j, (a j-b j)*(b j-c j) :=
  ModelRG.CacheRisk.state_displacement_euclidean a b c

example {ι κ : Type*} [Fintype ι] [Fintype κ]
    (w : ι → ℝ) (x : ι → κ → ℝ) (a b c : κ → ℝ)
    (normalized : ∑ i, w i = 1) (actualMean : ∀ j, ∑ i, w i*x i j = a j) :
    (∑ i, w i * ∑ j, (x i j-c j)^2) =
      (∑ i, w i * ∑ j, (x i j-a j)^2) +
      (∑ j, (a j-b j)^2) + (∑ j, (b j-c j)^2) +
      2 * ∑ j, (a j-b j)*(b j-c j) :=
  ModelRG.CacheRisk.empirical_state_transport_euclidean w x a b c normalized actualMean

/- Unequal coordinates, nonuniform weights, and a negative cross term. -/
example :
    ((1/4 : ℝ)*((1-2)^2+(2-(-1))^2) +
      (3/4 : ℝ)*((3-2)^2+(6-(-1))^2)) =
    ((1/4 : ℝ)*((1-5/2)^2+(2-5)^2) +
      (3/4 : ℝ)*((3-5/2)^2+(6-5)^2)) +
    ((5/2-1)^2+(5-3)^2) + ((1-2)^2+(3-(-1))^2) +
    2*((5/2-1)*(1-2)+(5-3)*(3-(-1))) := by norm_num

-- Finite-population bridges deliberately retain the actual ordered off-diagonal sum.
example {ι : Type*} [Fintype ι] [DecidableEq ι] (u : ι → ℝ)
    (h : ∑ i, u i = 0) :
    (∑ i, ∑ j ∈ Finset.univ.erase i, u i * u j) = -(∑ i, (u i)^2) :=
  ModelRG.FinitePopulation.centered_offdiagonal u h

example (M B d o : ℝ) (hB : B ≠ 0) (hM : M ≠ 1) :
    (M-B) / (B*(M-1)) * d - o / (M-1) =
      ((M/B) * d - (d+o)) / (M-1) :=
  ModelRG.FinitePopulation.batch_covariance_collect M B d o hB hM

example : (4-2 : ℝ) / (2*(4-1)) * 7 - 3/(4-1) = 4/3 := by
  have h := ModelRG.FinitePopulation.batch_covariance_collect
    4 2 7 3 (by norm_num) (by norm_num)
  norm_num at h ⊢


-- Finite row transport: the bridge states actual row means and coordinate sums.
-- Finite cross terms use outgoing energy. The rational L/E bridge is
-- algebra only; it asserts no matrix or parameter differentiability.
section RowFluxBridges
open scoped BigOperators
example (A D : Fin 2 → Fin 3 → ℝ)
    (hE : (∑ i, ∑ j, (A i j)^2) ≠ 0)
    (hNext : (∑ i, ∑ j, (A i j + D i j)^2) ≠ 0) :
    (∑ i, ∑ j, (A i j + D i j - (∑ r, (A r j + D r j)) / 2)^2) /
      (∑ i, ∑ j, (A i j + D i j)^2) -
      (∑ i, ∑ j, (A i j - (∑ r, A r j) / 2)^2) / (∑ i, ∑ j, (A i j)^2) =
    (2 * (∑ i, ∑ j, (A i j - (∑ r, A r j) / 2) * (D i j - (∑ r, D r j) / 2)) -
      2 * ((∑ i, ∑ j, (A i j - (∑ r, A r j) / 2)^2) / (∑ i, ∑ j, (A i j)^2)) *
      (∑ i, ∑ j, A i j * D i j)) / (∑ i, ∑ j, (A i j + D i j)^2) +
    ((∑ i, ∑ j, (D i j - (∑ r, D r j) / 2)^2) -
      ((∑ i, ∑ j, (A i j - (∑ r, A r j) / 2)^2) / (∑ i, ∑ j, (A i j)^2)) *
      (∑ i, ∑ j, (D i j)^2)) / (∑ i, ∑ j, (A i j + D i j)^2) := by
  simpa only [ModelRG.RowFlux.energy, ModelRG.RowFlux.pairing, ModelRG.RowFlux.center,
    Fintype.card_fin, Nat.cast_ofNat] using ModelRG.RowFlux.matrix_row_identity A D hE hNext

example (E E' L Q : ℝ) (hE : E ≠ 0) (hNext : E' ≠ 0) :
    (L+Q)/E' - L/E = Q/E' - ((E'-E)/E')*(L/E) :=
  ModelRG.RowFlux.derivative_defect_identity E E' L Q hE hNext

example (u : ℕ → ℝ) (n : ℕ) :
    (∑ i ∈ Finset.range n, (u (i+1)-u i)) = u n-u 0 :=
  ModelRG.RowFlux.finite_telescope u n
end RowFluxBridges

-- Independent explicit formulas, without newly defined coordinate aliases.
example (x y z X Y Z : ℝ) (hz : 1+z ≠ 0) (hZ : 1+Z ≠ 0) :
    (x+y)/(1+z) - (X+Y)/(1+Z) =
      ((x-X)+(y-Y)-(z-Z)*((X+Y)/(1+Z)))/(1+z) :=
  ModelRG.FluxForecast.error_identity x y z X Y Z hz hZ
example (E F G C D H : ℝ) (hE : E ≠ 0) (hF : F ≠ 0) :
    ((D-C)/E+(F/E)*((H-D)/F), (F/E)*(G/F)) = ((H-C)/E,G/E) :=
  ModelRG.FluxForecast.energy_composition E F G C D H hE hF
example : ((1:ℝ)/2+(3:ℝ))/4 = (1+2*3)/(2*4) := by
  have h := ModelRG.FluxForecast.row_action 0 1 3 2 4 (by norm_num) (by norm_num)
  norm_num at h ⊢

-- Independent temporal formulas specialize to Euclidean/Frobenius coordinates.
section TemporalEnergyBridges
open scoped InnerProductSpace
example {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (xs ys : List H) :
    ModelRG.TemporalEnergy.cross (xs ++ ys) =
      ModelRG.TemporalEnergy.cross xs + ModelRG.TemporalEnergy.cross ys +
        2 * ⟪xs.sum, ys.sum⟫_ℝ :=
  ModelRG.TemporalEnergy.cross_append xs ys
example {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    (xs : List H) (e : ℝ) (he : 0 < e) :
    -((xs.map (fun x => ‖x‖^2)).sum / e) ≤ ModelRG.TemporalEnergy.cross xs / e :=
  ModelRG.TemporalEnergy.normalized_lower_bound xs e he
example (x y : ℝ) :
    ModelRG.TemporalEnergy.cross [x,y] = 2*x*y := by
  simp [ModelRG.TemporalEnergy.cross, real_inner_comm]; ring
end TemporalEnergyBridges
