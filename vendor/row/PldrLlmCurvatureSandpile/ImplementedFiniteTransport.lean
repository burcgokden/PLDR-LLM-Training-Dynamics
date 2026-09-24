/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Exact finite endpoint transport kernels

These identities model the affine and multiplicative primitives used in the
implemented row program. They contain no Taylor remainder.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace ImplementedFiniteTransport

/-- Exact endpoint increment of a scalar affine primitive, including source
changes in the weight and bias. -/
theorem affine_endpoint_increment
    (weight weightIncrement input inputIncrement bias biasIncrement : ℝ) :
    (weight + weightIncrement) * (input + inputIncrement)
        + (bias + biasIncrement) - (weight * input + bias) =
      weight * inputIncrement + weightIncrement * input
        + weightIncrement * inputIncrement + biasIncrement := by
  ring

/-- Exact endpoint increment of a product primitive. The mixed endpoint term
is retained explicitly. -/
theorem product_endpoint_increment
    (left leftIncrement right rightIncrement : ℝ) :
    (left + leftIncrement) * (right + rightIncrement) - left * right =
      left * rightIncrement + right * leftIncrement
        + leftIncrement * rightIncrement := by
  ring

/-- Exact centered quadratic-energy charge of an endpoint increment. -/
theorem quadratic_energy_increment (state increment : ℝ) :
    (state + increment) ^ 2 - state ^ 2 =
      2 * state * increment + increment ^ 2 := by
  ring

/-- An affine primitive followed by a product primitive has the exact
chronological endpoint expansion used by finite source transport. -/
theorem affine_product_endpoint_increment
    (weight weightIncrement input inputIncrement bias biasIncrement
      gate gateIncrement : ℝ) :
    ((weight + weightIncrement) * (input + inputIncrement)
          + bias + biasIncrement) * (gate + gateIncrement)
        - (weight * input + bias) * gate =
      (weight * inputIncrement + weightIncrement * input
          + weightIncrement * inputIncrement + biasIncrement) * gate
        + (weight * input + bias) * gateIncrement
        + (weight * inputIncrement + weightIncrement * input
          + weightIncrement * inputIncrement + biasIncrement)
            * gateIncrement := by
  ring

end ImplementedFiniteTransport
end PldrLlmCurvatureSandpile
