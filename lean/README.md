# Lean 4 — PortfolioLinAlg

Formalizes the **real linear algebra** behind the Python pipeline. Numerical fetching, `np.cov` floating-point details, and CVXPY are **not** verified here.

## Build

```bash
lake update   # first time only
lake build
```

## Python ↔ Lean map

| Python | Lean |
|--------|------|
| `np.cov` → Σ (Gram / scatter × `1/(T-1)`) | `scatter_matrix_posSemidef`, `sample_covariance_posSemidef` |
| `np.linalg.eigh`, `min(λ) ≥ -tol` | `eigenvalues_nonneg_of_posSemidef`, `posSemidef_of_eigenvalues_nonneg` |
| `0.5 * quad_form(x, sigma)` in CVXPY | `portfolio_variance_nonneg` |
| `mu @ x` | `portfolio_mean_eq_dot` |
| CVXPY frontier loop | *not formalized* |

## Modules

| File | Content |
|------|---------|
| `PSD.lean` | Generic PSD ⇒ quadratic form / eigenvalues |
| `Covariance.lean` | Scatter, sample Σ, portfolio variance |
| `Eigen.lean` | Eigenvalue PSD criterion (matches `eigh` logic) |
| `PosDef2.lean` | 2×2 Sylvester minors (`sym2_posSemidef_iff`, proof pending) |

## References

- [Mathlib.LinearAlgebra.Matrix.PosDef](https://leanprover-community.github.io/mathlib4_docs/Mathlib/LinearAlgebra/Matrix/PosDef.html)
