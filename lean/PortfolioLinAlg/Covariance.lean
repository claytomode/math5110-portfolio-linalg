/-
  Matrix facts used by the Python pipeline (`covariance.py`, `frontier.py`).

  Python uses floating-point `np.cov` / `np.linalg.eigh` / CVXPY; here we formalize the
  exact real-linear-algebra statements those steps rely on.
-/

import PortfolioLinAlg.PSD
import Mathlib.Data.Real.StarOrdered

namespace PortfolioLinAlg

open Matrix

/-! ## Sample covariance = (positive scalar) × scatter matrix -/

/-- Centered return matrix `X` (rows = dates, columns = assets), shape `T × n`. -/
abbrev ReturnMatrix (T n : Type*) [Fintype T] [Fintype n] :=
  Matrix T n ℝ

/-- Scatter matrix `Xᵀ X` (assets × assets), the numerator of `np.cov(..., ddof=1)`. -/
def scatterMatrix {T n : Type*} [Fintype T] [Fintype n] (X : ReturnMatrix T n) : Matrix n n ℝ :=
  X.transpose * X

/-- Sample covariance `invk • (Xᵀ X)` with `invk = 1/(T-1)` in Python (`ddof=1`). -/
noncomputable def sampleCovariance {T n : Type*} [Fintype T] [Fintype n]
    (X : ReturnMatrix T n) (invk : ℝ) : Matrix n n ℝ :=
  invk • scatterMatrix X

lemma isHermitian_smul_real {n : Type*} [Fintype n] {c : ℝ} {A : Matrix n n ℝ}
    (hA : A.IsHermitian) : (c • A).IsHermitian := by
  ext i j
  simp only [IsHermitian, conjTranspose_smul, hA.eq, smul_apply, star_trivial]

/-- `Xᵀ X` is PSD — Gram matrix of asset return vectors over time. -/
theorem scatter_matrix_posSemidef {T n : Type*} [Fintype T] [Fintype n] (X : ReturnMatrix T n) :
    (scatterMatrix X).PosSemidef :=
  posSemidef_conjTranspose_mul_self X

/-- Scatter matrices are symmetric (used implicitly by `np.linalg.eigh`). -/
theorem scatter_matrix_isHermitian {T n : Type*} [Fintype T] [Fintype n] (X : ReturnMatrix T n) :
    (scatterMatrix X).IsHermitian :=
  isHermitian_transpose_mul_self X

/-- Scaling a PSD matrix by `c ≥ 0` preserves PSD (Python: multiply by `1/(T-1) > 0`). -/
theorem posSemidef_smul_nonneg {n : Type*} [Fintype n] {c : ℝ} (hc : 0 ≤ c) {A : Matrix n n ℝ}
    (hA : A.PosSemidef) : (c • A).PosSemidef := by
  rcases hA with ⟨hH, hquad⟩
  refine ⟨isHermitian_smul_real hH, fun x => ?_⟩
  have h_eq : star x ⬝ᵥ (c • A) *ᵥ x = c * (star x ⬝ᵥ A *ᵥ x) := by
    simp only [mulVec, smul_apply, dotProduct, Finset.smul_sum, Pi.smul_apply, smul_eq_mul,
      Finset.mul_sum, star_smul, star_trivial]
    congr 1
    ext i
    ring
  rw [h_eq]
  exact mul_nonneg hc (hquad x)

/-- Sample covariance is PSD when `invk ≥ 0` (Python: `invk = 1/(T-1)` with `T ≥ 2`). -/
theorem sample_covariance_posSemidef {T n : Type*} [Fintype T] [Fintype n] (X : ReturnMatrix T n)
    {invk : ℝ} (hk : 0 ≤ invk) : (sampleCovariance X invk).PosSemidef := by
  dsimp [sampleCovariance, scatterMatrix]
  exact posSemidef_smul_nonneg hk (scatter_matrix_posSemidef X)

/-! ## Portfolio variance `wᵀ Σ w` (CVXPY `quad_form`) -/

/-- Portfolio variance is nonnegative when the covariance matrix is PSD. -/
theorem portfolio_variance_nonneg {n : Type*} [Fintype n] (cov : Matrix n n ℝ)
    (hcov : cov.PosSemidef) (w : n → ℝ) : 0 ≤ dotProduct w (cov.mulVec w) := by
  simpa [dotProduct_mulVec, star_trivial] using hcov.re_dotProduct_nonneg w

/-- Expected return `μᵀ w` (Python: `mu @ x`). -/
theorem portfolio_mean_eq_dot {n : Type*} [Fintype n] (μ w : n → ℝ) :
    dotProduct μ w = ∑ i, μ i * w i := rfl

end PortfolioLinAlg
