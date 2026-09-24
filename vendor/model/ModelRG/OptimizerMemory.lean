import Mathlib

namespace ModelRG.OptimizerMemory
/-- Bias-correction algebra only; the recurrence and limit are written analysis. -/
theorem bias_product_bound (u v : ℝ) :
    (1-u^2)*(1-v^2) ≤ (1-u*v)^2 := by
  nlinarith [sq_nonneg (u-v)]
end ModelRG.OptimizerMemory
