/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Implemented post-normalized row-map chain

The matrices in this module are abstract derivative or chord factors. The
results check the execution order of the pre-chain affine LayerNorm and all
eight post-normalized residual units used by the PLDR metric learner.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace PostNormChain

variable {d : ℕ}

abbrev SquareMatrix (d : ℕ) := Matrix (Fin d) (Fin d) ℝ

/-- Append one residual Jacobian and its following affine LayerNorm to an
already accumulated derivative. -/
def appendUnit (accumulator residual layerNorm : SquareMatrix d) : SquareMatrix d :=
  layerNorm * residual * accumulator

/-- Accumulate post-normalized units from the pre-chain affine LayerNorm in
the same left-to-right order in which the program executes them. -/
def fullChain (preLayerNorm : SquareMatrix d)
    (units : List (SquareMatrix d × SquareMatrix d)) : SquareMatrix d :=
  units.foldl
    (fun accumulator unit => appendUnit accumulator unit.1 unit.2)
    preLayerNorm

/-- Appending a program unit places its residual factor before its affine
LayerNorm factor and to the left of every earlier factor. -/
theorem full_chain_append (preLayerNorm residual layerNorm : SquareMatrix d)
    (units : List (SquareMatrix d × SquareMatrix d)) :
    fullChain preLayerNorm (units ++ [(residual, layerNorm)]) =
      layerNorm * residual * fullChain preLayerNorm units := by
  simp [fullChain, appendUnit]

/-- The concrete eight-unit product contains the pre-chain affine
LayerNorm and every one of the eight post-normalized residual units. -/
theorem eight_unit_execution_order
    (L0 B1 L1 B2 L2 B3 L3 B4 L4 B5 L5 B6 L6 B7 L7 B8 L8 :
      SquareMatrix d) :
    fullChain L0
      [(B1, L1), (B2, L2), (B3, L3), (B4, L4),
       (B5, L5), (B6, L6), (B7, L7), (B8, L8)] =
      L8 * B8 * (L7 * B7 * (L6 * B6 * (L5 * B5 *
        (L4 * B4 * (L3 * B3 * (L2 * B2 * (L1 * B1 * L0))))))) := by
  simp [fullChain, appendUnit, Matrix.mul_assoc]

/-- Replacing only the final affine LayerNorm factor by the derivative of
normalization gives the normalized-shape chain while retaining the first
seven post-normalized affine factors. -/
theorem eight_unit_shape_execution_order
    (L0 B1 L1 B2 L2 B3 L3 B4 L4 B5 L5 B6 L6 B7 L7 B8 N8 :
      SquareMatrix d) :
    N8 * B8 * fullChain L0
      [(B1, L1), (B2, L2), (B3, L3), (B4, L4),
       (B5, L5), (B6, L6), (B7, L7)] =
      N8 * B8 * (L7 * B7 * (L6 * B6 * (L5 * B5 *
        (L4 * B4 * (L3 * B3 * (L2 * B2 * (L1 * B1 * L0))))))) := by
  simp [fullChain, appendUnit, Matrix.mul_assoc]

end PostNormChain
end PldrLlmCurvatureSandpile
