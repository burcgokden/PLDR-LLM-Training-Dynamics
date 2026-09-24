/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Affine transition-kernel blocking

Selected scalar algebra for the mean and innovation covariance of an affine
Gaussian transition. The probability-law, projection-closure, and
continuous-flow arguments remain analytic obligations of the manuscript.
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace TransitionKernelRG

/-- Scalar parameters of an affine state update with independent noise. -/
@[ext]
structure GaussianKernel where
  gain : ℝ
  bias : ℝ
  variance : ℝ

/-- Composition after integrating the independent intermediate innovation. -/
def compose (later earlier : GaussianKernel) : GaussianKernel where
  gain := later.gain * earlier.gain
  bias := later.gain * earlier.bias + later.bias
  variance := later.gain ^ 2 * earlier.variance + later.variance

/-- The deterministic identity transition. -/
def identity : GaussianKernel where
  gain := 1
  bias := 0
  variance := 0

@[simp] theorem compose_gain (later earlier : GaussianKernel) :
    (compose later earlier).gain = later.gain * earlier.gain := rfl

@[simp] theorem compose_bias (later earlier : GaussianKernel) :
    (compose later earlier).bias =
      later.gain * earlier.bias + later.bias := rfl

@[simp] theorem compose_variance (later earlier : GaussianKernel) :
    (compose later earlier).variance =
      later.gain ^ 2 * earlier.variance + later.variance := rfl

/-- Exact transition-kernel blocking is associative. -/
theorem compose_assoc (third second first : GaussianKernel) :
    compose (compose third second) first =
      compose third (compose second first) := by
  apply GaussianKernel.ext <;> simp [compose] <;> ring

@[simp] theorem identity_left (kernel : GaussianKernel) :
    compose identity kernel = kernel := by
  cases kernel
  simp [compose, identity]

@[simp] theorem identity_right (kernel : GaussianKernel) :
    compose kernel identity = kernel := by
  cases kernel
  simp [compose, identity]

/-- Nonnegative innovation variance is preserved by blocking. -/
theorem compose_variance_nonnegative {later earlier : GaussianKernel}
    (hlater : 0 ≤ later.variance) (hearlier : 0 ≤ earlier.variance) :
    0 ≤ (compose later earlier).variance := by
  exact add_nonneg (mul_nonneg (sq_nonneg later.gain) hearlier) hlater

/-- In stationary unit-variance coordinates the innovation variance is the
complement of squared memory. -/
def stationaryInnovation (gain : ℝ) : ℝ :=
  1 - gain ^ 2

/-- Two exact blocks retain the stationary covariance identity. -/
theorem stationary_two_block (gain : ℝ) :
    gain ^ 2 * stationaryInnovation gain
        + stationaryInnovation gain =
      stationaryInnovation (gain ^ 2) := by
  simp [stationaryInnovation]
  ring

/-- Binary blocking squares the scalar memory coefficient. -/
theorem binary_memory (kernel : GaussianKernel) :
    (compose kernel kernel).gain = kernel.gain ^ 2 := by
  simp [compose]
  ring

/-- Centering by a stationary mean removes the affine bias. -/
theorem stationary_mean_centering (gain bias mean : ℝ)
    (hmean : gain * mean + bias = mean) :
    gain * mean + bias - mean = 0 := by
  linarith

end TransitionKernelRG
end PldrLlmCurvatureSandpile
