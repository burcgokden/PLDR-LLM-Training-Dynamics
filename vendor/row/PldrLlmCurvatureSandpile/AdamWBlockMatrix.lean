/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# A genuine Fin (3*d) block matrix
-/
import Mathlib

namespace PldrLlmCurvatureSandpile
namespace AdamWBlockMatrix

/-- Three-by-three blocks whose entries are `d`-dimensional matrices. -/
def threeBlockMatrix {d : ℕ}
    (block : Fin 3 → Fin 3 → Matrix (Fin d) (Fin d) ℝ) :
    Matrix (Fin 3 × Fin d) (Fin 3 × Fin d) ℝ :=
  fun row column => block row.1 column.1 row.2 column.2

/-- Reindex the three-block matrix as a square `Fin (3*d)` matrix. -/
def fullBlockFin {d : ℕ}
    (block : Fin 3 → Fin 3 → Matrix (Fin d) (Fin d) ℝ) :
    Matrix (Fin (3 * d)) (Fin (3 * d)) ℝ :=
  Matrix.reindex finProdFinEquiv finProdFinEquiv (threeBlockMatrix block)

/-- Reindexing preserves every block entry exactly. -/
theorem full_block_fin_entry {d : ℕ}
    (block : Fin 3 → Fin 3 → Matrix (Fin d) (Fin d) ℝ)
    (row column : Fin 3 × Fin d) :
    fullBlockFin block (finProdFinEquiv row) (finProdFinEquiv column)
      = block row.1 column.1 row.2 column.2 := by
  simp [fullBlockFin, threeBlockMatrix, Matrix.reindex]

/-- The `Fin (3*d)` representation has the advertised row dimension. -/
theorem full_block_fin_rows {d : ℕ}
    (_block : Fin 3 → Fin 3 → Matrix (Fin d) (Fin d) ℝ) :
    Fintype.card (Fin (3 * d)) = 3 * d := by
  simp

end AdamWBlockMatrix
end PldrLlmCurvatureSandpile
