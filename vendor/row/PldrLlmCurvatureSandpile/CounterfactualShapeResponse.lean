/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license. -/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace CounterfactualShapeResponse

/-- The full two-arm response is the fixed-source-shape gate response plus
the difference of the two arms' exact shape works. -/
theorem two_arm_energy_identity
    (baselineFixed removalFixed baselineShapeWork removalShapeWork : ℝ) :
    (removalFixed + removalShapeWork)
        - (baselineFixed + baselineShapeWork)
      = (removalFixed - baselineFixed)
        + (removalShapeWork - baselineShapeWork) := by
  ring

/-- If the shape-work correction is smaller than a positive fixed-shape
response, the live two-arm response keeps its sign and has the charged
minimum magnitude. -/
theorem charged_source_removal
    {fixedResponse shapeCorrection liveResponse charge : ℝ}
    (hlive : liveResponse = fixedResponse + shapeCorrection)
    (hcharge : |shapeCorrection| ≤ charge)
    (hdominates : charge < fixedResponse) :
    0 < liveResponse ∧ fixedResponse - charge ≤ liveResponse := by
  rw [hlive]
  rw [abs_le] at hcharge
  constructor <;> linarith

end CounterfactualShapeResponse
end PldrLlmCurvatureSandpile
