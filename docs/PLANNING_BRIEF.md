# Project planning brief — Applied linear algebra course project

> Canonical spec for this repository. See [proposal.md](proposal.md) for the 1–2 page course proposal.

## Student context

- **Course:** Applied linear algebra / matrix analysis (not ML-focused).
- **Syllabus expects:**
  1. **Part 1:** Theory survey (textbook linear algebra techniques used).
  2. **Part 2:** Numerical / matrix computation on computer.
  3. **Part 3:** Application on **particular example/data** (real or benchmark, cited).
  4. **Presentation:** Mostly survey; **application at the end**.
  5. Written work can be **individual** even if labs were grouped.
- **Student interests:** Operations research, optimization, formal methods (Lean). Prior **Macaulay2** work (Gröbner bases, LP encoded as ideals). Does **not** want CS2 / territory-control / TCI path.
- **Topic approval:** **Done** — Lean + Python mean–variance matrix analysis; proceed with report and presentation.

## Recommended project title

**Formal matrix analysis for portfolio optimization: PSD structure in Lean, efficient frontier in Python**

## One-sentence pitch

Survey quadratic forms and PSD matrices; **formalize key matrix facts in Lean 4**; **compute mean–variance efficient frontier on real return data in Python**; relate to OR (constrained QP) and KKT structure without claiming novelty or beating industrial solvers.

## Three-part report structure

### Part 1 — Theory survey (~40%)

PSD matrices, eigenvalues, quadratic forms, Markowitz QP, KKT block systems, OR context, Lean/formal methods paragraph.

### Part 2 — Computation (~30%)

Sample \(\Sigma\) from returns; `eigh`; CVXPY frontier; plots. **Stack:** uv, Polars, httpx, NumPy, matplotlib, cvxpy.

### Part 3 — Application (~30%)

~11 ETFs via Yahoo Finance chart API (`httpx`), frontier interpretation, figures. Single-period; not investment advice.

### Lean parallel track

2–4 target theorems; **do not** formalize full KKT or portfolio optimality. See `lean/PortfolioLinAlg/PSD.lean`.

## Repository structure

Implemented as in root [README.md](../README.md).

## Explicit non-goals

No Gröbner main deliverable; no CS2; no ML classification; no full KKT in Lean; no beating Gurobi/CVXPY; no Lean-only without Python numerics.

## Timeline (8-week template)

| Week | Milestone |
|------|-----------|
| 1 | ~~Proposal; topic approval~~ **done**; tickers locked |
| 2 | Theory outline; fetch data; \(\Sigma\) eigenvalues |
| 3 | Python frontier + plots |
| 4 | Lean Mathlib; first PSD lemma |
| 5 | More Lean; polish notebook |
| 6 | Draft report |
| 7 | Slides |
| 8 | Final report + cleanup |

## Success criteria

- [ ] Report Parts 1–2–3
- [x] Topic approved (Lean + Python)
- [ ] Real cited data; frontier from code
- [x] Lean: pipeline-aligned PSD / covariance / eigenvalue lemmas (see `lean/README.md`)
- [ ] Survey-heavy presentation
