import ModelRG.ReplicaBootstrap

/- Finite reference-conditioned categorical lifts. The KL and limiting
   probability arguments are supplied as stand-alone manuscript proofs. -/
noncomputable section
namespace ModelRG.CategoricalScale
open scoped BigOperators
variable {ι : Type*} [DecidableEq ι]

def mass (J : Finset ι) (p : ι → ℝ) : ℝ := ∑ i ∈ J, p i

structure Coarse (J : Finset ι) where
  kept : {i // i ∉ J} → ℝ
  tail : ℝ

def aggregate (J : Finset ι) (p : ι → ℝ) : Coarse J :=
  ⟨fun i => p i, mass J p⟩

def restore (J : Finset ι) (r : ι → ℝ) (q : Coarse J) (i : ι) : ℝ :=
  if h : i ∈ J then q.tail * r i / mass J r else q.kept ⟨i, h⟩

def project (J : Finset ι) (r p : ι → ℝ) : ι → ℝ :=
  restore J r (aggregate J p)

theorem project_apply (J : Finset ι) (r p : ι → ℝ) (i : ι) :
    project J r p i = if i ∈ J then mass J p * r i / mass J r else p i := by
  simp [project, restore, aggregate]

theorem mass_restore (J : Finset ι) (r : ι → ℝ) (q : Coarse J)
    (hr : mass J r ≠ 0) : mass J (restore J r q) = q.tail := by
  calc
    _ = ∑ i ∈ J, q.tail * r i / mass J r := by
      apply Finset.sum_congr rfl
      intro i hi
      simp [restore, hi]
    _ = q.tail * mass J r / mass J r := by
      rw [← Finset.sum_div, ← Finset.mul_sum]
      rfl
    _ = q.tail := by field_simp

theorem aggregate_restore (J : Finset ι) (r : ι → ℝ) (q : Coarse J)
    (hr : mass J r ≠ 0) : aggregate J (restore J r q) = q := by
  cases q with
  | mk kept tail =>
    have hk : (aggregate J (restore J r ⟨kept, tail⟩)).kept = kept := by
      funext i
      simp [aggregate, restore, i.property]
    have ht := mass_restore J r (Coarse.mk kept tail) hr
    exact congrArg₂ Coarse.mk hk ht

theorem mass_project (J : Finset ι) (r p : ι → ℝ)
    (hr : mass J r ≠ 0) : mass J (project J r p) = mass J p :=
  mass_restore J r (aggregate J p) hr

theorem mass_project_superset (J L : Finset ι) (r p : ι → ℝ)
    (hL : L ⊆ J) (hr : mass L r ≠ 0) :
    mass J (project L r p) = mass J p := by
  have outside : mass (J \ L) (project L r p) = mass (J \ L) p := by
    apply Finset.sum_congr rfl
    intro i hi
    rw [project_apply, if_neg (Finset.mem_sdiff.mp hi).2]
  have a := Finset.sum_sdiff hL (f := project L r p)
  have b := Finset.sum_sdiff hL (f := p)
  change mass (J \ L) (project L r p) + mass L (project L r p) = mass J (project L r p) at a
  change mass (J \ L) p + mass L p = mass J p at b
  rw [outside, mass_project L r p hr] at a
  linarith

theorem project_coarse_after_fine (J L : Finset ι) (r p : ι → ℝ)
    (hL : L ⊆ J) (hr : mass L r ≠ 0) :
    project J r (project L r p) = project J r p := by
  funext i
  rw [project_apply, project_apply, mass_project_superset J L r p hL hr]
  by_cases hi : i ∈ J
  · simp [hi, project_apply]
  · have hn : i ∉ L := fun h => hi (hL h)
    simp [hi, project_apply, hn]

theorem mass_project_subset (J L : Finset ι) (r p : ι → ℝ)
    (hL : L ⊆ J) :
    mass L (project J r p) = mass J p * mass L r / mass J r := by
  calc
    _ = ∑ i ∈ L, mass J p * r i / mass J r := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [project_apply, if_pos (hL hi)]
    _ = _ := by rw [← Finset.sum_div, ← Finset.mul_sum]; rfl

theorem project_fine_after_coarse (J L : Finset ι) (r p : ι → ℝ)
    (hL : L ⊆ J) (hr : mass L r ≠ 0) :
    project L r (project J r p) = project J r p := by
  funext i
  rw [project_apply]
  by_cases hi : i ∈ L
  · rw [if_pos hi, mass_project_subset J L r p hL, project_apply, if_pos (hL hi)]
    field_simp
  · rw [if_neg hi]

theorem positive_mass (J : Finset ι) (r : ι → ℝ)
    (hJ : J.Nonempty) (hr : ∀ i, 0 < r i) : 0 < mass J r := by
  exact Finset.sum_pos (fun i _ => hr i) hJ

theorem empirical_bias_variance {σ : Type*} [Fintype σ]
    (x : σ → ℝ) (S : ℝ) (hS : (Fintype.card σ : ℝ) = S)
    (hs : 1 < S) :
    (∑ s, (x s - (∑ t, x t)/S)^2)/(S-1) =
      S/(S-1) * ((∑ s, (x s)^2)/S - ((∑ s, x s)/S)^2) := by
  have hn : S ≠ 0 := by linarith
  have hn1 : S-1 ≠ 0 := by linarith
  have point (s : σ) : (x s - (∑ t, x t)/S)^2 =
      (x s)^2 - 2*((∑ t, x t)/S)*x s + ((∑ t, x t)/S)^2 := by ring
  simp_rw [point]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, hS]
  field_simp
  ring

theorem empirical_vector_bias_variance {σ κ ν : Type*}
    [Fintype σ] [Fintype κ] [Fintype ν]
    (x : σ → κ → ν → ℝ) (w : κ → ℝ) (S : ℝ)
    (hS : (Fintype.card σ : ℝ) = S) (hs : 1 < S) :
    (∑ c, w c * ∑ i, (∑ s, (x s c i - (∑ t, x t c i)/S)^2)/(S-1)) =
      S/(S-1) * ((∑ c, w c * ∑ i, (∑ s, (x s c i)^2)/S) -
        (∑ c, w c * ∑ i, ((∑ s, x s c i)/S)^2)) := by
  simp_rw [empirical_bias_variance (fun s => x s _ _) S hS hs,
    Finset.mul_sum, mul_sub, Finset.sum_sub_distrib]
  simp_rw [Finset.mul_sum]
  congr 1
  · apply Finset.sum_congr rfl
    intro c hc
    apply Finset.sum_congr rfl
    intro i hi
    ring
  · apply Finset.sum_congr rfl
    intro c hc
    apply Finset.sum_congr rfl
    intro i hi
    ring

def tailDensityAverage (J : Finset ι) (r p : ι → ℝ) : ℝ :=
  (∑ i ∈ J, r i * (p i / r i)) / mass J r

omit [DecidableEq ι] in
theorem tailDensityAverage_eq (J : Finset ι) (r p : ι → ℝ)
    (hr : ∀ i ∈ J, r i ≠ 0) :
    tailDensityAverage J r p = mass J p / mass J r := by
  unfold tailDensityAverage mass
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  field_simp [hr i hi]

theorem project_density_on_tail (J : Finset ι) (r p : ι → ℝ)
    (i : ι) (hi : i ∈ J) (hri : r i ≠ 0) :
    project J r p i / r i = mass J p / mass J r := by
  rw [project_apply, if_pos hi]
  field_simp

theorem project_density_expectation (J : Finset ι) (r p : ι → ℝ)
    (i : ι) (hi : i ∈ J) (hr : ∀ j ∈ J, r j ≠ 0) :
    project J r p i / r i = tailDensityAverage J r p := by
  rw [project_density_on_tail J r p i hi (hr i hi), tailDensityAverage_eq J r p hr]

theorem project_density_outside (J : Finset ι) (r p : ι → ℝ)
    (i : ι) (hi : i ∉ J) :
    project J r p i / r i = p i / r i := by
  rw [project_apply, if_neg hi]


end ModelRG.CategoricalScale
