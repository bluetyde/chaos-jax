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

## Results (added after the run, 2026-10-26)

`python experiments/phase2b_wide_slope.py`, 39 s on CPU. Output in `results/phase2b_wide_slope.json`.

| Item | Result |
|---|---|
| A: line slope over rho 25 to 50 | 0.9888 (bootstrap SE 0.0001) |
| A: quadratic fit, local slope at 28 | 1.0101 (SE 0.0005) |
| A: window slopes | [25,30] 0.9906, [30,35] 1.0074, [35,40] 0.9917, [40,45] 0.9759, [45,50] 0.9627 (SE 0.0006 to 0.0017) |
| B: Lea start point, tau = 131.36, dt = 0.005 | line slope 0.9925 |
| B: 20 start points | min 0.9698, max 0.9925, mean 0.9862, sd 0.0067 |

Decision rule for A: 0.9888 is in (0.98, 1.00], so **the range explains part of the gap**.

| Prediction | Result |
|---|---|
| P2b.a A line slope about 0.985, "part of the gap" | 0.9888. Hit. |
| P2b.b Window slopes fall from about 1.0 to below 0.98 | [25,30] is 0.991 and [45,50] is 0.963. Hit at the ends. The fall is not steady, because [30,35] is the highest at 1.007. |
| P2b.c B within 0.03 of A, spread above 0.01 | 0.9925 against 0.9888, spread 0.0227. Hit. |

What this shows:
- Over [25, 50] the slope is 0.989, so about 1.5 of the 4 points of gap come from the range. Over [45, 50] it is 0.963, close to 0.96.
- Running the Lea et al. setup as I read it (their start point, tau, dt, and my guess of 25 to 50 for the range) gives 0.9925, not 0.96. A gap of about 0.03 remains. It is not explained here. Their fit range is only "roughly 25 to 50" in my reading of the text. A wider range would lower the slope. I have not tested that.
- The local slope is not the same everywhere. It differs between windows by more than the standard errors. The Phase 2 slope at 28 (1.0029 +/- 0.0021) and the quadratic estimate at 28 (1.0101 +/- 0.0005) differ by about 3 SE of the Phase 2 value. The quadratic is a fit over a wide range, so it is not a local estimate, and I did not look further.
- The Phase 2 gates and results stand. G2.2 only claimed linearity over [27, 29].
