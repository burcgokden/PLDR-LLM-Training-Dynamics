/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Source-to-native energy envelope
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace SourceNativeEnergy

/-- A source-predicted state plus an outward native radius gives the squared
norm envelope used by the physical energy recurrence. -/
theorem source_native_energy_envelope
    {E : Type*} [SeminormedAddCommGroup E]
    (source predictedIncrement nativeResidual : E) (radius : ℝ)
    (hradius : 0 ≤ radius) (hresidual : ‖nativeResidual‖ ≤ radius) :
    ‖source + predictedIncrement + nativeResidual‖ ^ 2 ≤
      ‖source + predictedIncrement‖ ^ 2
      + 2 * ‖source + predictedIncrement‖ * radius + radius ^ 2 := by
  have hnorm :
      ‖source + predictedIncrement + nativeResidual‖ ≤
        ‖source + predictedIncrement‖ + radius := by
    calc
      ‖source + predictedIncrement + nativeResidual‖ ≤
          ‖source + predictedIncrement‖ + ‖nativeResidual‖ := norm_add_le _ _
      _ ≤ ‖source + predictedIncrement‖ + radius :=
        add_le_add_right hresidual _
  have hright : 0 ≤ ‖source + predictedIncrement‖ + radius :=
    add_nonneg (norm_nonneg _) hradius
  calc
    ‖source + predictedIncrement + nativeResidual‖ ^ 2 ≤
        (‖source + predictedIncrement‖ + radius) ^ 2 :=
      (sq_le_sq₀ (norm_nonneg _) hright).2 hnorm
    _ = ‖source + predictedIncrement‖ ^ 2
        + 2 * ‖source + predictedIncrement‖ * radius + radius ^ 2 := by
      ring

/-- A zero native residual specializes the envelope to exact equality. -/
theorem zero_native_radius_exact
    {E : Type*} [SeminormedAddCommGroup E]
    (source predictedIncrement : E) :
    ‖source + predictedIncrement + (0 : E)‖ ^ 2 =
      ‖source + predictedIncrement‖ ^ 2 := by
  simp

end SourceNativeEnergy
end PldrLlmCurvatureSandpile
