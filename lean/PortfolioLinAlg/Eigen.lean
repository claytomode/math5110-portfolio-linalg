/-
  Eigenvalue facts matching `np.linalg.eigh` PSD check in `covariance.analyze_sigma`.
-/

import PortfolioLinAlg.Covariance

namespace PortfolioLinAlg

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- PSD ⇒ all eigenvalues ≥ 0 (Python: `min(eigenvalues) >= -tol`). -/
theorem eigenvalues_nonneg_of_posSemidef {cov : Matrix n n ℝ} (hcov : cov.PosSemidef) (i : n) :
    0 ≤ hcov.1.eigenvalues i :=
  posSemidef_eigenvalue_nonneg hcov i

/-- If `cov` is symmetric and all eigenvalues ≥ 0, then `cov` is PSD (converse of the `eigh` test). -/
theorem posSemidef_of_eigenvalues_nonneg {cov : Matrix n n ℝ} (hcov : cov.IsHermitian)
    (heig : ∀ i : n, 0 ≤ hcov.eigenvalues i) : cov.PosSemidef :=
  hcov.posSemidef_of_eigenvalues_nonneg heig

/-- Sample scatter has nonnegative eigenvalues (special case of Gram PSD). -/
theorem scatter_eigenvalues_nonneg {T n : Type*} [Fintype T] [Fintype n] [DecidableEq n]
    (X : ReturnMatrix T n) (i : n) :
    0 ≤ (scatter_matrix_isHermitian (X := X)).eigenvalues i :=
  eigenvalues_nonneg_of_posSemidef (scatter_matrix_posSemidef X) i

end PortfolioLinAlg
