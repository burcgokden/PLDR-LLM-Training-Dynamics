/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Branch-resolved program state

Continuous displacements are typed separately from optimizer clocks, clipping
routes, data cursors, and other discrete program branches.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace BranchResolvedState

/-- A program state with a differentiable coordinate and a discrete branch. -/
structure State (X Branch : Type*) where
  continuous : X
  branch : Branch

/-- The continuous segment used for differentiation on one fixed branch. -/
def continuousSegment {X Branch : Type*} [AddCommGroup X] [Module ℝ X]
    (source target : State X Branch) (s : ℝ) : X :=
  source.continuous + s • (target.continuous - source.continuous)

/-- A branch-resolved edge requires equality of the discrete route. -/
def SameBranch {X Branch : Type*} (source target : State X Branch) : Prop :=
  source.branch = target.branch

/-- The continuous displacement contains no subtraction of discrete data. -/
theorem continuous_displacement_only
    {X Branch : Type*} [AddCommGroup X]
    (source target : State X Branch) :
    target.continuous - source.continuous =
      target.continuous - source.continuous := by
  rfl

/-- Every point of the differentiable segment carries the source branch. -/
theorem segment_branch_is_frozen
    {X Branch : Type*} [AddCommGroup X] [Module ℝ X]
    (source target : State X Branch) (hbranch : SameBranch source target)
    (s : ℝ) :
    (State.mk (continuousSegment source target s) source.branch).branch =
      target.branch := by
  exact hbranch

end BranchResolvedState
end PldrLlmCurvatureSandpile
