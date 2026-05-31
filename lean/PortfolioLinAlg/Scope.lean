/-
  What this Lean project proves vs what Python computes numerically.

  | Python (`portfolio_linalg`)     | Formalized in Lean? |
  |---------------------------------|---------------------|
  | `np.cov` → Σ                    | `sampleCovariance` PSD if `k > 0` |
  | `np.linalg.eigh` PSD check      | `eigenvalues_nonneg_of_posSemidef` / converse |
  | `quad_form(x, sigma)` ≥ 0       | `portfolio_variance_nonneg` |
  | `mu @ x`                        | `portfolio_mean_eq_dot` (linear algebra only) |
  | CVXPY Markowitz QP / frontier   | **Not formalized** (convex solver, KKT out of scope) |
  | Long-only constraints, `r_min`  | **Not formalized** |
  | Floating-point tolerance        | **Not formalized** (exact ℝ) |
-/

import PortfolioLinAlg.Eigen

namespace PortfolioLinAlg

-- This file exists for documentation imports; no extra theorems.

end PortfolioLinAlg
