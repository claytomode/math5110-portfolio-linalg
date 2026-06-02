/-
  Two-by-two PSD criterion (Sylvester leading minors). Hand-checkable case aligned with
  Python `np.linalg.eigh` on 2×2 sample blocks.
-/

import PortfolioLinAlg.Basic
import Mathlib.Data.Matrix.Notation

namespace PortfolioLinAlg

open Matrix

/-- Symmetric 2×2 matrix with entries a, b, b, c. -/
def sym2 (a b c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, b; b, c]

theorem sym2_isSymm (a b c : ℝ) : Matrix.IsSymm (sym2 a b c) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [sym2, Matrix.IsSymm, conjTranspose_apply, star_trivial]

/-- PSD ↔ nonnegative leading principal minors (n = 2 Sylvester criterion). -/
theorem sym2_posSemidef_iff (a b c : ℝ) :
    (sym2 a b c).PosSemidef ↔ 0 ≤ a ∧ 0 ≤ a * c - b ^ 2 := by
  sorry

end PortfolioLinAlg
