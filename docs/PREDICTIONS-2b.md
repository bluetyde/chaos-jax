# Phase 2 follow-up: does the slope range explain 0.96 against 1.003?

Written and committed before the run. This is a diagnostic. It adds no gate to `PLAN.md` and does
not change any Phase 2 gate or result.

## Question

Lea et al. 2000 report about 0.96 for d<z>/dr. They get it from a straight-line fit over r of
roughly 25 to 50, using one long run (averaging time 131.36, RK4 dt = 0.005, start point
(-2.40, -3.70, 14.98), r grid step 0.005). The Phase 2 oracle gives 1.0029 +/- 0.0021 as a local
slope at rho = 28. The two numbers differ by about 4%.

## Method (fixed now)

Lorenz sigma = 10, beta = 8/3. Phase 1 RK4 code.

- **A. Ensemble over a wide range.** Same pipeline as Phase 2 (N = 1000 start points, key
  `PRNGKey(0)`, dt = 0.01, spin-up 50, average 100). rho = 25, 26, ..., 50 (26 values). Least-squares
  line through mean<z>(rho). Bootstrap SE over start points, 2000 resamples. Also report the local
  line slope on 5 windows [25,30], [30,35], [35,40], [40,45], [45,50], and a quadratic fit.
- **B. Lea-like single run.** dt = 0.005, rho from 25 to 50 in steps of 0.005 (5001 values), averaging
  time 131.36 (26272 steps), no spin-up, start point (-2.40, -3.70, 14.98). Least-squares line slope.
  Then repeat with 19 more random start points (key `PRNGKey(2)`, same ranges as Phase 2) to see how much
  the slope changes with the start point. Report the 20 slopes.

Expected compute: about 3 minutes on CPU. Stop and say so if it takes more than 10 minutes.

## Decision rule for A (fixed now)

- Range explains the gap: A line slope between 0.94 and 0.98.
- Range explains part of the gap: A line slope between 0.98 and 1.00.
- Range does not explain the gap: A line slope above 1.00 or below 0.94.

In the third case I report the gap as unexplained. I do not tune anything to reach 0.96.

## Predictions

- P2b.a A line slope is about 0.985, so "part of the gap". Reason: Figure 5b of arXiv 1204.0159 has
  <z> near 92 at rho = 100 against 23.5 at rho = 28. That gives a global slope near 0.95 over [28, 100].
  So the slope probably falls as rho rises, but the fall is smaller over [25, 50].
- P2b.b The local window slopes fall from about 1.0 at [25,30] to below 0.98 at [45,50].
- P2b.c B single-run slope is within 0.03 of the A line slope. The 20 slopes from B spread by more than 0.01
  (min to max), because one run of 131 time units is noisy.
