import Mathlib
import RowRGMap.AffineCocycle

/-!
# Nonnegative interval affine RG

Rectangular intervals enclose the gain and source coordinates of a positive
affine edge. Because every coordinate is nonnegative, endpoint arithmetic is
monotone and chronological composition uses only endpoint products and sums.
-/

namespace RowRGMap
namespace IntervalAffine

@[ext]
structure Interval where
  lower : ℝ
  upper : ℝ

def Valid (interval : Interval) : Prop :=
  0 ≤ interval.lower ∧ interval.lower ≤ interval.upper

def Contains (interval : Interval) (value : ℝ) : Prop :=
  interval.lower ≤ value ∧ value ≤ interval.upper

def singleton (value : ℝ) : Interval where
  lower := value
  upper := value

def multiply (left right : Interval) : Interval where
  lower := left.lower * right.lower
  upper := left.upper * right.upper

def add (left right : Interval) : Interval where
  lower := left.lower + right.lower
  upper := left.upper + right.upper

@[ext]
structure Edge where
  gain : Interval
  source : Interval

/-- The later interval edge is the first argument. -/
def compose (later earlier : Edge) : Edge where
  gain := multiply later.gain earlier.gain
  source := add (multiply later.gain earlier.source) later.source

def act (edge : Edge) (energy : Interval) : Interval :=
  add (multiply edge.gain energy) edge.source

def ExactEdge (gain source : ℝ) : Edge where
  gain := singleton gain
  source := singleton source

@[simp] theorem multiply_lower (left right : Interval) :
    (multiply left right).lower = left.lower * right.lower := rfl

@[simp] theorem multiply_upper (left right : Interval) :
    (multiply left right).upper = left.upper * right.upper := rfl

@[simp] theorem add_lower (left right : Interval) :
    (add left right).lower = left.lower + right.lower := rfl

@[simp] theorem add_upper (left right : Interval) :
    (add left right).upper = left.upper + right.upper := rfl

theorem multiply_valid {left right : Interval}
    (hleft : Valid left) (hright : Valid right) :
    Valid (multiply left right) := by
  constructor
  · exact mul_nonneg hleft.1 hright.1
  · exact mul_le_mul hleft.2 hright.2 hright.1
      (le_trans hleft.1 hleft.2)

theorem add_valid {left right : Interval}
    (hleft : Valid left) (hright : Valid right) :
    Valid (add left right) := by
  exact ⟨add_nonneg hleft.1 hright.1, add_le_add hleft.2 hright.2⟩

theorem contains_nonnegative {interval : Interval} {value : ℝ}
    (hvalid : Valid interval) (hcontains : Contains interval value) :
    0 ≤ value :=
  hvalid.1.trans hcontains.1

theorem multiply_contains {left right : Interval} {x y : ℝ}
    (hleft : Valid left) (hright : Valid right)
    (hx : Contains left x) (hy : Contains right y) :
    Contains (multiply left right) (x * y) := by
  constructor
  · exact mul_le_mul hx.1 hy.1 hright.1
      (contains_nonnegative hleft hx)
  · change x * y ≤ left.upper * right.upper
    have hynonnegative := contains_nonnegative hright hy
    have hleftupper : 0 ≤ left.upper := hleft.1.trans hleft.2
    exact (mul_le_mul_of_nonneg_right hx.2 hynonnegative).trans
      (mul_le_mul_of_nonneg_left hy.2 hleftupper)

theorem add_contains {left right : Interval} {x y : ℝ}
    (hx : Contains left x) (hy : Contains right y) :
    Contains (add left right) (x + y) := by
  exact ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

theorem compose_assoc (third second first : Edge) :
    compose (compose third second) first =
      compose third (compose second first) := by
  ext <;> simp [compose, multiply, add] <;> ring

theorem compose_valid {later earlier : Edge}
    (hlaterGain : Valid later.gain) (hlaterSource : Valid later.source)
    (hearlierGain : Valid earlier.gain) (hearlierSource : Valid earlier.source) :
    Valid (compose later earlier).gain ∧
      Valid (compose later earlier).source := by
  constructor
  · exact multiply_valid hlaterGain hearlierGain
  · exact add_valid (multiply_valid hlaterGain hearlierSource) hlaterSource

theorem compose_contains_exact
    {later earlier : Edge} {laterGain laterSource earlierGain earlierSource : ℝ}
    (hlaterGain : Valid later.gain)
    (hearlierGain : Valid earlier.gain) (hearlierSource : Valid earlier.source)
    (hlg : Contains later.gain laterGain)
    (hls : Contains later.source laterSource)
    (heg : Contains earlier.gain earlierGain)
    (hes : Contains earlier.source earlierSource) :
    Contains (compose later earlier).gain (laterGain * earlierGain) ∧
      Contains (compose later earlier).source
        (laterGain * earlierSource + laterSource) := by
  constructor
  · exact multiply_contains hlaterGain hearlierGain hlg heg
  · exact add_contains
      (multiply_contains hlaterGain hearlierSource hlg hes) hls

theorem act_contains_exact
    {edge : Edge} {energy : Interval} {gain source value : ℝ}
    (hgainValid : Valid edge.gain)
    (henergyValid : Valid energy)
    (hgain : Contains edge.gain gain)
    (hsource : Contains edge.source source)
    (henergy : Contains energy value) :
    Contains (act edge energy) (gain * value + source) := by
  exact add_contains
    (multiply_contains hgainValid henergyValid hgain henergy) hsource

@[simp] theorem compose_exact (laterGain laterSource earlierGain earlierSource : ℝ) :
    compose (ExactEdge laterGain laterSource)
        (ExactEdge earlierGain earlierSource) =
      ExactEdge (laterGain * earlierGain)
        (laterGain * earlierSource + laterSource) := by
  ext <;> simp [compose, ExactEdge, singleton, multiply, add]

@[simp] theorem act_exact (gain source value : ℝ) :
    act (ExactEdge gain source) (singleton value) =
      singleton (gain * value + source) := by
  ext <;> simp [act, ExactEdge, singleton, multiply, add]


/-- A valid interval edge has valid gain and source coordinates. -/
def EdgeValid (edge : Edge) : Prop :=
  Valid edge.gain ∧ Valid edge.source

/-- Componentwise interval inclusion, with the outer interval first. -/
def Includes (outer inner : Interval) : Prop :=
  outer.lower ≤ inner.lower ∧ inner.upper ≤ outer.upper

/-- The exact interval identity. -/
def identity : Edge :=
  ExactEdge 1 0

@[simp] theorem compose_identity_left (edge : Edge) :
    compose identity edge = edge := by
  ext <;> simp [compose, identity, ExactEdge, singleton, multiply, add]

@[simp] theorem compose_identity_right (edge : Edge) :
    compose edge identity = edge := by
  ext <;> simp [compose, identity, ExactEdge, singleton, multiply, add]

theorem singleton_valid {value : ℝ} (hvalue : 0 ≤ value) :
    Valid (singleton value) :=
  ⟨hvalue, le_rfl⟩

theorem identity_valid : EdgeValid identity := by
  exact ⟨singleton_valid (by norm_num), singleton_valid (by norm_num)⟩

theorem includes_refl (interval : Interval) :
    Includes interval interval :=
  ⟨le_rfl, le_rfl⟩

theorem multiply_includes
    {outerLeft innerLeft outerRight innerRight : Interval}
    (houterLeft : Valid outerLeft) (hinnerLeft : Valid innerLeft)
    (houterRight : Valid outerRight) (hinnerRight : Valid innerRight)
    (hleft : Includes outerLeft innerLeft)
    (hright : Includes outerRight innerRight) :
    Includes (multiply outerLeft outerRight)
      (multiply innerLeft innerRight) := by
  constructor
  · exact mul_le_mul hleft.1 hright.1 houterRight.1 hinnerLeft.1
  · exact mul_le_mul hleft.2 hright.2
      (hinnerRight.1.trans hinnerRight.2)
      (houterLeft.1.trans houterLeft.2)

theorem add_includes
    {outerLeft innerLeft outerRight innerRight : Interval}
    (hleft : Includes outerLeft innerLeft)
    (hright : Includes outerRight innerRight) :
    Includes (add outerLeft outerRight) (add innerLeft innerRight) :=
  ⟨add_le_add hleft.1 hright.1, add_le_add hleft.2 hright.2⟩

theorem compose_includes
    {outerLater innerLater outerEarlier innerEarlier : Edge}
    (houterLater : EdgeValid outerLater)
    (hinnerLater : EdgeValid innerLater)
    (houterEarlier : EdgeValid outerEarlier)
    (hinnerEarlier : EdgeValid innerEarlier)
    (hlaterGain : Includes outerLater.gain innerLater.gain)
    (hlaterSource : Includes outerLater.source innerLater.source)
    (hearlierGain : Includes outerEarlier.gain innerEarlier.gain)
    (hearlierSource : Includes outerEarlier.source innerEarlier.source) :
    Includes (compose outerLater outerEarlier).gain
        (compose innerLater innerEarlier).gain ∧
      Includes (compose outerLater outerEarlier).source
        (compose innerLater innerEarlier).source := by
  constructor
  · exact multiply_includes
      houterLater.1 hinnerLater.1 houterEarlier.1 hinnerEarlier.1
      hlaterGain hearlierGain
  · exact add_includes
      (multiply_includes
        houterLater.1 hinnerLater.1 houterEarlier.2 hinnerEarlier.2
        hlaterGain hearlierSource)
      hlaterSource

/-- A pair stores an interval edge and one exact affine edge it encloses. -/
def Encloses (intervalEdge : Edge)
    (exactEdge : RowRGMap.AffineCocycle.Edge) : Prop :=
  Contains intervalEdge.gain exactEdge.gain ∧
    Contains intervalEdge.source exactEdge.source

/-- Chronological blocking for paired interval and exact edges. -/
def intervalBlock : List (Edge × RowRGMap.AffineCocycle.Edge) → Edge
  | [] => identity
  | pair :: rest => compose (intervalBlock rest) pair.1

def exactBlock :
    List (Edge × RowRGMap.AffineCocycle.Edge) →
      RowRGMap.AffineCocycle.Edge
  | [] => RowRGMap.AffineCocycle.identity
  | pair :: rest =>
      RowRGMap.AffineCocycle.compose (exactBlock rest) pair.2

theorem intervalBlock_valid
    (pairs : List (Edge × RowRGMap.AffineCocycle.Edge))
    (hvalid : ∀ pair ∈ pairs, EdgeValid pair.1) :
    EdgeValid (intervalBlock pairs) := by
  induction pairs with
  | nil => exact identity_valid
  | cons pair rest inductionHypothesis =>
      have hpair := hvalid pair (by simp)
      have hrest : ∀ candidate ∈ rest, EdgeValid candidate.1 := by
        intro candidate hcandidate
        exact hvalid candidate (by simp [hcandidate])
      exact compose_valid
        (inductionHypothesis hrest).1 (inductionHypothesis hrest).2
        hpair.1 hpair.2

theorem intervalBlock_encloses_exact
    (pairs : List (Edge × RowRGMap.AffineCocycle.Edge))
    (hvalid : ∀ pair ∈ pairs, EdgeValid pair.1)
    (hencloses : ∀ pair ∈ pairs, Encloses pair.1 pair.2) :
    Encloses (intervalBlock pairs) (exactBlock pairs) := by
  induction pairs with
  | nil =>
      constructor <;>
        simp [intervalBlock, exactBlock, identity, ExactEdge,
          singleton, RowRGMap.AffineCocycle.identity, Contains]
  | cons pair rest inductionHypothesis =>
      have hpairValid := hvalid pair (by simp)
      have hrestValid : ∀ candidate ∈ rest, EdgeValid candidate.1 := by
        intro candidate hcandidate
        exact hvalid candidate (by simp [hcandidate])
      have hrestEncloses : ∀ candidate ∈ rest,
          Encloses candidate.1 candidate.2 := by
        intro candidate hcandidate
        exact hencloses candidate (by simp [hcandidate])
      have ih := inductionHypothesis hrestValid hrestEncloses
      have hblockValid := intervalBlock_valid rest hrestValid
      have hpairEncloses := hencloses pair (by simp)
      exact compose_contains_exact
        hblockValid.1 hpairValid.1 hpairValid.2
        ih.1 ih.2 hpairEncloses.1 hpairEncloses.2

/-- Positive endpoint intervals give the sharp elementary gain bracket. -/
theorem gain_bracket
    {source target lowerSource upperSource lowerTarget upperTarget : ℝ}
    (hlowerSource : 0 < lowerSource)
    (hlowerTarget : 0 ≤ lowerTarget)
    (hsource : lowerSource ≤ source ∧ source ≤ upperSource)
    (htarget : lowerTarget ≤ target ∧ target ≤ upperTarget) :
    lowerTarget / upperSource ≤ target / source ∧
      target / source ≤ upperTarget / lowerSource := by
  have hsourcePositive : 0 < source := hlowerSource.trans_le hsource.1
  have htargetNonnegative : 0 ≤ target :=
    hlowerTarget.trans htarget.1
  have hupperTargetNonnegative : 0 ≤ upperTarget :=
    htargetNonnegative.trans htarget.2
  constructor
  · exact div_le_div₀ htargetNonnegative htarget.1
      hsourcePositive hsource.2
  · exact div_le_div₀ hupperTargetNonnegative htarget.2
      hlowerSource hsource.1

/-- A denominator interval reaching zero and a positive admissible successor
cannot impose a finite upper gain bound. -/
theorem no_finite_gain_bound_with_positive_successor
    (sourceUpper targetUpper bound : ℝ)
    (hsourceUpper : 0 < sourceUpper)
    (htargetUpper : 0 < targetUpper)
    (hbound : 0 ≤ bound) :
    ∃ source target : ℝ,
      0 < source ∧ source ≤ sourceUpper ∧
      0 < target ∧ target ≤ targetUpper ∧
      bound < target / source := by
  let scale := min sourceUpper (targetUpper / (bound + 1))
  have hden : 0 < bound + 1 := by linarith
  have hratio : 0 < targetUpper / (bound + 1) :=
    div_pos htargetUpper hden
  have hscale : 0 < scale := by
    exact lt_min hsourceUpper hratio
  refine ⟨scale / 2, targetUpper, by positivity, ?_,
    htargetUpper, le_rfl, ?_⟩
  · calc
      scale / 2 ≤ scale := by linarith
      _ ≤ sourceUpper := min_le_left _ _
  · have hsmall : scale / 2 < targetUpper / (bound + 1) := by
      calc
        scale / 2 < scale := by linarith
        _ ≤ targetUpper / (bound + 1) := min_le_right _ _
    have hproduct : (scale / 2) * (bound + 1) < targetUpper :=
      (lt_div_iff₀ hden).mp hsmall
    have hmore : bound + 1 < targetUpper / (scale / 2) :=
      (lt_div_iff₀ (by positivity)).2 (by nlinarith)
    linarith

theorem zero_successor_iff_zero_gain {source target : ℝ}
    (hsource : 0 < source) :
    target / source = 0 ↔ target = 0 := by
  constructor
  · intro hquotient
    exact (div_eq_zero_iff.mp hquotient).resolve_right (ne_of_gt hsource)
  · intro htarget
    simp [htarget]

end IntervalAffine
end RowRGMap
