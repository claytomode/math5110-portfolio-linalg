# Project proposal — Formal matrix analysis for portfolio optimization

**Course:** Applied linear algebra / matrix analysis  
**Student:** [Name]  
**Date:** May 2026

## One-sentence pitch

Survey quadratic forms and PSD matrices; formalize key matrix facts in Lean 4; compute the mean–variance efficient frontier on real return data in Python; relate to OR (constrained QP) and KKT structure without claiming novelty.

## Structure (syllabus Parts 1–3 + Lean)

| Part | Weight | Content |
|------|--------|---------|
| 1 — Theory | ~40% | PSD, eigenvalues, Markowitz QP, KKT block form, OR context |
| 2 — Computation | ~30% | NumPy `eigh`, Polars/httpx data, CVXPY frontier |
| 3 — Application | ~30% | ~11 ETFs, cited Yahoo data (httpx), frontier + interpretation |
| Lean (parallel) | — | 2+ sorry-free PSD lemmas in Mathlib |

## Research questions

1. What does PSD of \(\Sigma\) mean geometrically and algebraically?
2. Why is mean–variance optimization a convex QP?
3. What does the KKT system look like for \(\min \frac12 x^\top \Sigma x\) s.t. \(\mathbf 1^\top x = 1\), \(\mu^\top x \ge r_{\min}\), \(x \ge 0\)?
4. On real ETF returns, what does the efficient frontier look like numerically?
5. Which matrix facts can be proved in Lean 4 + Mathlib vs left to numerics?

## Data plan

- **Source:** Yahoo Finance chart API via `httpx` (documented in `python/data/metadata.json`)
- **Universe:** SPY, QQQ, IWM, XLF, XLE, XLK, EFA, EEM, AGG, TLT, GLD (~11 ETFs)
- **Window:** ~2 years daily returns from 2024-01-02
- **Outputs:** \(\mu\), \(\Sigma\), efficient frontier plot, eigenvalue spectrum, correlation heatmap

## Success criteria

- [ ] Written report with Parts 1–2–3
- [ ] Real cited data; frontier from code
- [ ] Lean: ≥2 sorry-free PSD / quadratic-form lemmas
- [ ] Presentation survey-heavy; application last
- [ ] Explicit: industrial solvers at scale; this is matrix analysis + small QP

## Related work

- Markowitz (1952) — portfolio selection
- Boyd & Vandenberghe — convex optimization, QP, KKT
- Golub & Van Loan — matrix computations
- Mathlib / *Mathematics in Lean*