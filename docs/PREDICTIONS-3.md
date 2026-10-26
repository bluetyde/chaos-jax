# Phase 3 predictions and gates

Written and committed before any Phase 3 experiment ran. P3.1 to P3.4 come from `PLAN.md`.
This file makes them exact. Do not edit a prediction after results. Add a dated Amendment instead.

## What I knew before predicting

- Phase 1: lambda_1 = 0.9064 and 0.9018 (tangent method). Reference value for P3.1: the mean, **0.9041**.
- Phase 2: oracle d<z>/drho at rho = 28 is **1.0029** (SE 0.0021). Slopes over wider ranges are lower, down to 0.963 on [45, 50]. This does not move the oracle.
- Lea et al. 2000 (read for the log): the gradient of a time average grows like exp(lambda tau) with lambda about 0.93 for r from 25 to 30. They report that single-run gradients are of order 1e4 at tau = 2.26 and order 1e100 at tau = 131. I did not use those numbers to set the thresholds below.

## Method (fixed now)

Lorenz sigma = 10, rho = 28, beta = 8/3, dt = 0.01, RK4 (Phase 1 code). Start points: N = 200 random points, key `PRNGKey(30)`, x and y uniform in [-15, 15], z uniform in [5, 45], then 50 time units at rho = 28 (not differentiated). The same start points are used in every part. A start point is fixed. Only rho is the differentiated input in P3.3 and P3.4.

- **P3.1 and P3.2 quantity:** the largest singular value of the state Jacobian dx(T)/dx0, computed by pushing the identity through the tangent step (Phase 1 `step_and_tangent`), at T = 2, 2.5, ..., 20. For each system, average log of that value over the start points. Fit a line over T in [2, 20]. The slope is the result.
- **P3.3 quantity:** g(T) = d/drho of (1/T) integral of z over [0, T], from the fixed start point, by reverse-mode autodiff (`jax.grad`) through the unrolled RK4 scan. T = 1, 2, ..., 30. The median over start points of |g(T)| is compared with the oracle 1.0029.
- **P3.4 quantity:** the same g(T) computed in float32 in a separate process (x64 off, same start points cast to float32). Compare with float64.
- **Controls first.** P3.2 runs before P3.1. Controls: damped oscillator c = 0.4, k = 4, start points (x, v) uniform in [-2, 2]. Low-energy pendulum: theta1, theta2 uniform in [0.05, 0.15], both at rest, dt = 0.00125 (Phase 1). 50 start points each.

## Definitions of the departure times (fixed now)

- e64(T) = median over start points of |g64(T) - 1.0029| / 1.0029.
- e32(T) = median over start points of |g32(T) - g64(T)| / |g64(T)|.
- T_leave64 = smallest T >= 2 on the grid with e64(T') > 0.25 for every grid T' >= T.
- T_leave32 = smallest T on the grid with e32(T') > 0.25 for every grid T' >= T.
- T* (for P3.3) = smallest T on the grid with median |g64(T)| > 10 x 1.0029 = 10.029.

If a quantity never crosses its threshold on the grid, the result is "not reached by T = 30".

## Gates (from the plan)

- **P3.1** The Jacobian slope for Lorenz is within 15% of 0.9041, so in [0.7685, 1.0397].
- **P3.2** Both controls have slope at most 0.02.
- **P3.3** Median |g| exceeds 10 x the oracle at some T on the grid. The one predicted number for T* is below.
- **P3.4** T_leave32 < T_leave64.

## Numerical predictions

- P3.1 slope: **0.92**, range 0.88 to 0.96. The slope of a mean log norm should be close to lambda_1.
- P3.2 damped oscillator slope: **-0.2** (exact decay rate c/2). Pendulum slope: **about +0.01**, with real risk of exceeding 0.02. A near-linear system with a small amplitude dependence of frequency makes the norm grow roughly linearly in t, so log norm grows like log t. Over [2, 20] that can add up to 0.02 or more at larger amplitude. If it fails I report it and do not change the threshold.
- P3.3 **T\* = 6** (plausible range 4 to 9). Reasoning: g is an average of a sensitivity that grows like A exp(0.9 t) with A of order 3. Then g(T) is about A exp(0.9 T) / (0.9 T), which passes 10 near T = 6.
- P3.4 **T_leave64 = 4** (range 2 to 7). **T_leave32 = 16** (range 13 to 19). Reasoning: float32 round-off is about 6e-8. It grows like exp(0.9 T) and reaches 25% near T = ln(4e6) / 0.9, about 17. Float64 leaves the oracle earlier, because of sampling noise from the chaotic growth, not round-off. So I expect **P3.4 to fail**: float32 leaves float64 *later* than float64 leaves the oracle. I am writing this down because it contradicts the plan's wording.

Expected compute: under 5 minutes on CPU for all parts. Stop and say so if it takes more than 10.

## Results (added after the run, 2026-10-26)

Commands: `python experiments/phase3_blowup.py f64` (34 s), then `f32` (32 s, separate process, dtype checked as float32), then `report`.
Output in `results/phase3_f64.json`, `results/phase3.json` and the `.npz` files. N = 200 start points for Lorenz, 50 for each control.

| Gate | Result | Pass |
|---|---|---|
| P3.2 (controls first) damped oscillator slope <= 0.02 | -0.2004 | yes |
| P3.2 low-energy pendulum slope <= 0.02 | -0.0146 (SE 0.0009) | yes |
| P3.1 Lorenz slope within 15% of 0.9041 | 0.9057 (SE 0.0037), +0.18% | yes |
| P3.3 median \|g\| > 10 x 1.0029 at some T | T* = 8 (median \|g\| = 13.5) | yes |
| P3.4 T_leave32 < T_leave64 | T_leave32 = 14, T_leave64 = 2 | **no** |

| Prediction | Result |
|---|---|
| P3.1 slope 0.92, range 0.88 to 0.96 | 0.9057. In range, point 0.014 low. |
| P3.2 oscillator slope -0.2 | -0.2004. Hit. |
| P3.2 pendulum slope about +0.01, risk above 0.02 | -0.0146. **Miss on the sign.** The gate passes. |
| P3.3 T* = 6, range 4 to 9 | 8. **Point miss by 2.** In range. |
| P3.4 T_leave64 = 4, range 2 to 7 | 2. Point miss. At the edge of the range. |
| P3.4 T_leave32 = 16, range 13 to 19 | 14. Point miss by 2. In range. |
| P3.4 expected to fail | Failed. Hit. |
| Compute under 5 minutes | About 70 s. |

Notes on these results:
- **T_leave64 = 2 is not a sharp finding.** The float64 gradient is already 40% off the oracle in median at T = 1 (median |g| = 0.81, e64 = 0.40) and at T = 2 (0.87, e64 = 0.41). That is the short-average bias that Lea et al. describe, plus scatter between start points. So the float64 side of P3.4 never had a window where the gradient agreed with the oracle within 25%. The comparison in P3.4 is therefore between a time that is the lower edge of my grid and T_leave32. P3.4 fails as the plan states it.
- **Float32 holds up longer than I wrote the gate to expect.** e32 stays below 1e-3 up to T = 7, reaches 0.016 at T = 10 and 0.2 at T = 13. Float32 round-off is not what ends the usefulness of the naive gradient. Sampling noise does.
- **The sign of the gradient is random at long T.** The fraction of start points with a negative gradient is 0.375 at T = 5, 0.53 at T = 10, 0.485 at T = 20 and 0.53 at T = 30. The oracle is +1.0029. This matches a gradient whose sign carries no information.
- **Exploratory, not predicted:** over T = 10 to 30, the slope of ln(median |g|) against T is 0.850. The slope of ln(T x median |g|) is 0.903. This is close to the Phase 1 lambda_1 of 0.9041. It is consistent with g being about exp(lambda_1 T) / T.
- **What is not shown here:** no truncation or ensemble result (Phase 4). Median |g| is a measure of size, not of error, because the signs are random. The comparison with the oracle at T = 8 means "13 times too big in magnitude", not "13 times off".
