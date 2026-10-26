# Phase 2 predictions and gates

Written and committed before any Phase 2 experiment ran. Gates G2.1 to G2.3 come from `PLAN.md`.
This file only makes them exact. Do not edit a gate after results. Add a dated Amendment instead.

## Disclosure

Before writing this file I opened Figures 5 and 6 of `reference/papers/arxiv-1204.0159.pdf`
(pages 15 and 16) to find the sanity band. The text has no number for d<z>/drho at rho = 28.
The figures show a value near 1.0 to 1.02. This is a value read from a plot, not a stated number.
So the prediction below is not independent of that paper. The oracle is not tuned to it.

## Method (fixed now)

Lorenz: sigma = 10, beta = 8/3, dt = 0.01, RK4 (Phase 1 code).

1. Draw N = 1000 random start points. Key `PRNGKey(0)`. x, y uniform in [-15, 15], z uniform in [5, 45].
2. For each rho in {27, 27.5, 28, 28.5, 29}: run each start point 50 time units (5000 steps) to reach the attractor. Then average z over 100 time units (10000 steps). The same start points are used at every rho.
3. Estimate d<z>/drho at 28 as (mean<z>(29) - mean<z>(27)) / 2, where the mean is over the N start points.
4. Standard error: bootstrap over start points, 2000 resamples, key `PRNGKey(1)`. A resample uses the same start points for all rho.

Expected compute: about 3 minutes on CPU, 7.5e7 RK4 steps. If it takes over 10 minutes, I stop and say so.

## Gates

- **G2.1** Bootstrap standard error of the estimate is below 10% of the estimate. If not, raise N or the averaging time, and record the cost. The gate stays the same.
- **G2.2** Linearity over [27, 29]. Fit a quadratic to mean<z>(rho) at the five rho values. The gate passes if |c2| < 2 x bootstrap SE(c2), where c2 is the coefficient of (rho - 28)^2. If it fails, report it and stop. Phases 3 and later need rethinking.
- **G2.3** Control. Driven damped oscillator x'' + c x' + k x = F cos(w t), with c = 0.5, k = 4, F = 1, w = 1.5. Quantity: time average of x^2. Closed form: F^2 / (2 D), D = (k - w^2)^2 + c^2 w^2. The derivative in w comes from the closed form. The same pipeline (random start points, spin-up, average, central difference, bootstrap) must match it with relative error below 1e-6. Details for the control:
  - The averaging window is exactly 10 periods, 2 pi 10 / w, with dt = window / 20000. A whole number of periods makes the finite-window average exact. The window length therefore depends on w, and this is the only difference from the Lorenz pipeline.
  - Spin-up is 3 windows. Step h = 1e-4. 4 random start points.
  - The control runs first. If it fails, fix the pipeline before reading any Lorenz result.

## Predictions (not gates)

- P2.a d<z>/drho at 28 is between 0.95 and 1.08.
- P2.b Bootstrap SE is about 0.02, so about 2% of the estimate. Reason: the standard deviation of one 100-unit average of z is about 0.8, so one start point gives a slope noise of about 0.57, and 0.57 / sqrt(1000) = 0.018.
- P2.c G2.2 passes. The curve of <z> against rho looks close to linear in Figure 5b of the paper.
- P2.d G2.3 passes with relative error near 1e-8 or smaller.
