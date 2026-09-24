import Mathlib

/-!
# Positive-affine energy cocycles

An edge acts on a nonnegative scalar energy by `E |-> gain * E + source`.
Chronological blocking is the semidirect-product law used by the manuscript.
-/

namespace RowRGMap
namespace AffineCocycle

@[ext]
structure Edge where
  gain : ℝ
  source : ℝ

/-- The later edge is the first argument. -/
def compose (later earlier : Edge) : Edge where
  gain := later.gain * earlier.gain
  source := later.gain * earlier.source + later.source

def identity : Edge where
  gain := 1
  source := 0

def act (edge : Edge) (energy : ℝ) : ℝ :=
  edge.gain * energy + edge.source

def Nonnegative (edge : Edge) : Prop :=
  0 ≤ edge.gain ∧ 0 ≤ edge.source

/-- The canonical realized cone: at least one affine coordinate vanishes. -/
def Canonical (edge : Edge) : Prop :=
  edge.gain * edge.source = 0

@[simp] theorem compose_gain (later earlier : Edge) :
    (compose later earlier).gain = later.gain * earlier.gain := rfl

@[simp] theorem compose_source (later earlier : Edge) :
    (compose later earlier).source =
      later.gain * earlier.source + later.source := rfl

@[simp] theorem act_identity (energy : ℝ) :
    act identity energy = energy := by
  simp [act, identity]

@[simp] theorem compose_identity_left (edge : Edge) :
    compose identity edge = edge := by
  cases edge
  simp [compose, identity]

@[simp] theorem compose_identity_right (edge : Edge) :
    compose edge identity = edge := by
  cases edge
  simp [compose, identity]

theorem compose_assoc (third second first : Edge) :
    compose (compose third second) first =
      compose third (compose second first) := by
  apply Edge.ext <;> simp [compose] <;> ring

theorem act_compose (later earlier : Edge) (energy : ℝ) :
    act (compose later earlier) energy =
      act later (act earlier energy) := by
  simp [act, compose]
  ring

theorem compose_nonnegative {later earlier : Edge}
    (hlater : Nonnegative later) (hearlier : Nonnegative earlier) :
    Nonnegative (compose later earlier) := by
  constructor
  · exact mul_nonneg hlater.1 hearlier.1
  · exact add_nonneg (mul_nonneg hlater.1 hearlier.2) hlater.2

theorem canonical_iff_gain_zero_or_source_zero (edge : Edge) :
    Canonical edge ↔ edge.gain = 0 ∨ edge.source = 0 := by
  exact mul_eq_zero

/-- The canonical cone is closed by chronological affine composition. -/
theorem compose_canonical {later earlier : Edge}
    (hlater : Canonical later) (hearlier : Canonical earlier) :
    Canonical (compose later earlier) := by
  simp only [Canonical, compose_gain, compose_source]
  calc
    later.gain * earlier.gain *
          (later.gain * earlier.source + later.source) =
        later.gain ^ 2 * (earlier.gain * earlier.source) +
          earlier.gain * (later.gain * later.source) := by ring
    _ = 0 := by rw [hearlier, hlater]; ring

@[simp] theorem identity_canonical : Canonical identity := by
  simp [Canonical, identity]

theorem act_nonnegative {edge : Edge} {energy : ℝ}
    (hedge : Nonnegative edge) (henergy : 0 ≤ energy) :
    0 ≤ act edge energy := by
  exact add_nonneg (mul_nonneg hedge.1 henergy) hedge.2

/-- Ordered blocking of a chronological list. -/
def block : List Edge → Edge
  | [] => identity
  | edge :: rest => compose (block rest) edge

@[simp] theorem block_nil : block [] = identity := rfl

@[simp] theorem block_cons (edge : Edge) (rest : List Edge) :
    block (edge :: rest) = compose (block rest) edge := rfl

theorem block_append (earlier later : List Edge) :
    block (earlier ++ later) =
      compose (block later) (block earlier) := by
  induction earlier with
  | nil => simp
  | cons edge rest inductionHypothesis =>
      simp only [List.cons_append, block_cons, inductionHypothesis]
      exact compose_assoc (block later) (block rest) edge

theorem block_act (edges : List Edge) (energy : ℝ) :
    act (block edges) energy =
      edges.foldl (fun current edge => act edge current) energy := by
  induction edges generalizing energy with
  | nil => simp
  | cons edge rest inductionHypothesis =>
      simp only [block_cons, act_compose, List.foldl_cons]
      exact inductionHypothesis (act edge energy)

theorem block_nonnegative : ∀ {edges : List Edge},
    (∀ edge ∈ edges, Nonnegative edge) → Nonnegative (block edges) := by
  intro edges
  induction edges with
  | nil =>
      intro _
      exact ⟨by norm_num [identity], by norm_num [identity]⟩
  | cons edge rest inductionHypothesis =>
      intro hall
      apply compose_nonnegative
      · exact inductionHypothesis (by
          intro candidate hcandidate
          exact hall candidate (by simp [hcandidate]))
      · exact hall edge (by simp)

theorem block_canonical : ∀ {edges : List Edge},
    (∀ edge ∈ edges, Canonical edge) → Canonical (block edges) := by
  intro edges
  induction edges with
  | nil => intro _; exact identity_canonical
  | cons edge rest inductionHypothesis =>
      intro hall
      exact compose_canonical
        (inductionHypothesis fun candidate h => hall candidate (by simp [h]))
        (hall edge (by simp))

/-- Positive endpoint gauges change energy units at the two endpoints. -/
noncomputable def normalize (edge : Edge)
    (sourceGauge targetGauge : ℝ) : Edge where
  gain := edge.gain * sourceGauge / targetGauge
  source := edge.source / targetGauge

theorem normalize_compose (later earlier : Edge)
    (sourceGauge sharedGauge targetGauge : ℝ)
    (hshared : sharedGauge ≠ 0) (htarget : targetGauge ≠ 0) :
    compose
        (normalize later sharedGauge targetGauge)
        (normalize earlier sourceGauge sharedGauge) =
      normalize (compose later earlier) sourceGauge targetGauge := by
  ext <;> simp [normalize, compose]
  <;> field_simp [hshared, htarget]

/-- Normalize every edge by consecutive gauges along a finite block. -/
noncomputable def normalizeChain :
    List Edge → (ℕ → ℝ) → List Edge
  | [], _ => []
  | edge :: rest, gauge =>
      normalize edge (gauge 0) (gauge 1) ::
        normalizeChain rest (fun index => gauge (index + 1))

/-- Fine-edge gauge normalization commutes with arbitrary finite blocking. -/
theorem block_normalizeChain
    (edges : List Edge) (gauge : ℕ → ℝ)
    (hnonzero : ∀ index, index ≤ edges.length → gauge index ≠ 0) :
    block (normalizeChain edges gauge) =
      normalize (block edges) (gauge 0) (gauge edges.length) := by
  induction edges generalizing gauge with
  | nil =>
      have hgauge : gauge 0 ≠ 0 := hnonzero 0 (by simp)
      ext <;> simp [normalizeChain, block, normalize, identity, hgauge]
  | cons edge rest inductionHypothesis =>
      let shifted : ℕ → ℝ := fun index => gauge (index + 1)
      have hshifted :
          ∀ index, index ≤ rest.length → shifted index ≠ 0 := by
        intro index hindex
        exact hnonzero (index + 1) (by simp; omega)
      have hshared : gauge 1 ≠ 0 :=
        hnonzero 1 (by simp)
      have htarget : gauge (rest.length + 1) ≠ 0 :=
        hnonzero (rest.length + 1) (by simp)
      simp only [normalizeChain, block_cons]
      rw [inductionHypothesis shifted hshifted]
      simpa [shifted, Nat.add_comm] using
        normalize_compose (block rest) edge
          (gauge 0) (gauge 1) (gauge (rest.length + 1))
          hshared htarget

/-- The canonical edge of two nonnegative endpoint energies. -/
noncomputable def canonical (sourceEnergy targetEnergy : ℝ) : Edge :=
  if 0 < sourceEnergy then
    { gain := targetEnergy / sourceEnergy, source := 0 }
  else
    { gain := 0, source := targetEnergy }

theorem canonical_act {sourceEnergy targetEnergy : ℝ}
    (hsource : 0 ≤ sourceEnergy) :
    act (canonical sourceEnergy targetEnergy) sourceEnergy = targetEnergy := by
  by_cases hpositive : 0 < sourceEnergy
  · simp [canonical, hpositive, act]
    field_simp [ne_of_gt hpositive]
  · have hzero : sourceEnergy = 0 := le_antisymm (le_of_not_gt hpositive) hsource
    simp [canonical, hzero, act]

theorem canonical_nonnegative {sourceEnergy targetEnergy : ℝ}
    (hsource : 0 ≤ sourceEnergy) (htarget : 0 ≤ targetEnergy) :
    Nonnegative (canonical sourceEnergy targetEnergy) := by
  by_cases hpositive : 0 < sourceEnergy
  · simp [canonical, hpositive, Nonnegative]
    exact div_nonneg htarget hsource
  · simp [canonical, hpositive, Nonnegative, htarget]

@[simp] theorem canonical_is_canonical (sourceEnergy targetEnergy : ℝ) :
    Canonical (canonical sourceEnergy targetEnergy) := by
  by_cases hpositive : 0 < sourceEnergy
  · simp [canonical, hpositive, Canonical]
  · simp [canonical, hpositive, Canonical]

end AffineCocycle
end RowRGMap
