/-
  Thin wrappers around Mathlib positive semidefinite matrix facts.
  Portfolio variance xᵀΣx ≥ 0 when Σ is PSD — no full KKT or optimality formalization.
-/

import PortfolioLinAlg.Basic

open scoped ComplexOrder

namespace PortfolioLinAlg

variable {n : Type*} [Fintype n] [DecidableEq n] {𝕜 : Type*} [RCLike 𝕜]

/-- PSD matrices yield nonnegative real quadratic forms (portfolio variance ≥ 0). -/
theorem posSemidef_quadraticForm_nonneg {A : Matrix n n 𝕜} (hA : A.PosSemidef) (x : n → 𝕜) :
    0 ≤ RCLike.re (dotProduct (star x) (A.mulVec x)) :=
  hA.re_dotProduct_nonneg x

/-- Eigenvalues of a PSD matrix are nonnegative. -/
theorem posSemidef_eigenvalue_nonneg {A : Matrix n n 𝕜} (hA : A.PosSemidef) (i : n) :
    0 ≤ hA.1.eigenvalues i :=
  hA.eigenvalues_nonneg i

end PortfolioLinAlg
