import Mathlib

/- Finite formalization of the finite scalar area-composition algebra.
   No stochastic closure, native implementation, or criticality is claimed. -/
namespace ModelRG
noncomputable section

@[ext] structure PotentialBlock where
  d : ℝ
  e : ℝ
  area : ℝ

namespace PotentialBlock

def comp (x y : PotentialBlock) : PotentialBlock :=
  ⟨x.d + y.d, x.e + y.e, x.area + y.area + (y.d * x.e - x.d * y.e) / 2⟩

def zero : PotentialBlock := ⟨0, 0, 0⟩
def inverse (x : PotentialBlock) : PotentialBlock := ⟨-x.d, -x.e, -x.area⟩
def scale (r s : ℝ) (x : PotentialBlock) : PotentialBlock :=
  ⟨r * x.d, s * x.e, (r * s) * x.area⟩
def exponentAllocation (b : ℝ) (x : PotentialBlock) : ℝ :=
  x.d * (b + x.e / 2) + x.area
def baseAllocation (p : ℝ) (x : PotentialBlock) : ℝ :=
  (p + x.d / 2) * x.e - x.area

theorem comp_assoc (x y z : PotentialBlock) :
    comp (comp x y) z = comp x (comp y z) := by
  ext <;> simp [comp] <;> ring

theorem comp_identity (x : PotentialBlock) :
    comp zero x = x ∧ comp x zero = x := by
  constructor <;> ext <;> simp [comp, zero]

theorem comp_inverse (x : PotentialBlock) :
    comp x (inverse x) = zero ∧ comp (inverse x) x = zero := by
  constructor <;> ext <;> simp [comp, inverse, zero]

theorem scale_comp (r s : ℝ) (x y : PotentialBlock) :
    scale r s (comp x y) = comp (scale r s x) (scale r s y) := by
  ext <;> simp [comp, scale] <;> ring

theorem allocation_comp (p b : ℝ) (x y : PotentialBlock) :
    exponentAllocation b (comp x y) =
      exponentAllocation b x + exponentAllocation (b + x.e) y ∧
    baseAllocation p (comp x y) =
      baseAllocation p x + baseAllocation (p + x.d) y := by
  constructor <;> simp [exponentAllocation, baseAllocation, comp] <;> ring

theorem allocation_product (p b : ℝ) (x : PotentialBlock) :
    exponentAllocation b x + baseAllocation p x =
      (p + x.d) * (b + x.e) - p * b := by
  simp [exponentAllocation, baseAllocation]
  ring

theorem area_order_difference (x y : PotentialBlock) :
    (comp x y).area - (comp y x).area = y.d * x.e - x.d * y.e := by
  simp [comp]
  ring

end PotentialBlock
end
end ModelRG
