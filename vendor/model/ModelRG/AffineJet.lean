import Mathlib

namespace ModelRG

variable {V U : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup U] [Module ℝ U]

/-- The source space is shared across layers. Matrices may mix every head. -/
@[ext] structure Jet (V U : Type*) [AddCommGroup V] [Module ℝ V]
    [AddCommGroup U] [Module ℝ U] where
  A : V →ₗ[ℝ] V
  B : U →ₗ[ℝ] V

namespace Jet

def comp (later earlier : Jet V U) : Jet V U :=
  ⟨later.A.comp earlier.A, later.A.comp earlier.B + later.B⟩

def act (j : Jet V U) (x : V) (eta : U) : V := j.A x + j.B eta

theorem comp_act (j k : Jet V U) (x : V) (eta : U) :
    act (comp j k) x eta = act j (act k x eta) eta := by
  simp [act, comp, map_add, add_assoc]

theorem comp_assoc (j k l : Jet V U) :
    comp (comp j k) l = comp j (comp k l) := by
  ext x <;> simp [comp, map_add, add_assoc]

def identity : Jet V U := ⟨LinearMap.id, 0⟩

theorem identity_act (x : V) (eta : U) : act (identity : Jet V U) x eta = x := by
  simp [act, identity]

end Jet

/-- Eliminating an unresolved scalar already creates a memory term at step two. -/
theorem two_step_memory (a b c d x y : ℝ) :
    a*(a*x+b*y)+b*(c*x+d*y) = a*(a*x+b*y) + (b*c)*x + (b*d)*y := by
  ring

end ModelRG
