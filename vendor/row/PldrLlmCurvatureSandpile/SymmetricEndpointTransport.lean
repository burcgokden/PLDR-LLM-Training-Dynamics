/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Symmetric finite gate-shape transport
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SymmetricEndpointTransport

/-- The midpoint product identity does not privilege either endpoint. -/
theorem midpoint_product_identity (a₀ a₁ b₀ b₁ : ℝ) :
    a₁ * b₁ - a₀ * b₀ =
      ((a₁ + a₀) / 2) * (b₁ - b₀)
      + (a₁ - a₀) * ((b₁ + b₀) / 2) := by
  ring

/-- Symmetric shape contribution for one scalar gate-shape coordinate. -/
noncomputable def shapeIncrement (shape₀ shape₁ gate₀ gate₁ : ℝ) : ℝ :=
  (shape₁ - shape₀) * ((gate₁ + gate₀) / 2)

/-- Symmetric gate contribution for one scalar gate-shape coordinate. -/
noncomputable def gateIncrement (shape₀ shape₁ gate₀ gate₁ : ℝ) : ℝ :=
  ((shape₁ + shape₀) / 2) * (gate₁ - gate₀)

theorem symmetric_gate_shape_increment
    (shape₀ shape₁ gate₀ gate₁ : ℝ) :
    shape₁ * gate₁ - shape₀ * gate₀ =
      shapeIncrement shape₀ shape₁ gate₀ gate₁
      + gateIncrement shape₀ shape₁ gate₀ gate₁ := by
  unfold shapeIncrement gateIncrement
  ring

/-- Exact work, self-charge, and interaction ledger for the symmetric split. -/
theorem symmetric_energy_ledger
    (source shapePart gatePart : ℝ) :
    (source + shapePart + gatePart) ^ 2 - source ^ 2 =
      2 * source * shapePart + 2 * source * gatePart
      + shapePart ^ 2 + gatePart ^ 2
      + 2 * shapePart * gatePart := by
  ring

end SymmetricEndpointTransport
end PldrLlmCurvatureSandpile
