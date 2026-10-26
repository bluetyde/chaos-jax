# Phase 4 predictions and gates

Written and committed before any Phase 4 code ran. Gates come from `PLAN.md`. This file makes them exact.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Equation check (done before writing this file)

LSS formulation: `reference/papers/arxiv-1204.0159.pdf`, Sections 4.3 and 5.1 to 5.4 (PDF pages 8 to 12). I read the page images for pages 10 and 11 and the text for pages 8 to 12.
The printed equations are not all consistent with each other. I derived my own discretization from the continuous problem (Eq. 13 and Eq. 15), which I checked on the page and by a short argument (time dilation d tau/dt = 1 + eta gives both). What I found:

1. Eq. 17 gives E_i and G_i without a factor 1/2 on the Jacobian term, while b_i is an average of two terms. A trapezoid rule needs 1/2 on the Jacobian terms. I use E_i = -I/dt - A(u_{i-1})/2 and G_i = I/dt - A(u_i)/2.
2. Eq. 16 has f_i eta_i on the left with the same sign as the other terms. That gives the opposite sign of eta compared with Eq. 13. I use the sign of Eq. 13: E_i v_{i-1} + G_i v_i - f_i eta_i = b_i.
3. Eq. 19 gives B B^T w = b. Eq. 18 and Eq. 21 only agree with the constraint if the system is B B^T w = -b. In my sign convention I derived it again: B B^T w = -b, v_i = -(G_i^T w_i + E_{i+1}^T w_{i+1}), eta_i = f_i^T w_i / alpha^2.
4. Eq. 20 has m rows (E_m, G_m) but Eq. 16 has m-1 constraints. I use m-1.

Because of 1 to 3, the LSS code is only trusted after it matches (a) a dense solve of the full KKT system on a small case, (b) the closed-form control, and (c) the Phase 2 oracle band.

## Quantity and methods

Q_b: d/drho of the time average of z over [0, T], at rho = 28, from a start point on the attractor. Oracle: **1.0029** (Phase 2). Lorenz sigma = 10, beta = 8/3, dt = 0.01, RK4. Horizons T in {5, 10, 20, 40, 80}.

Start points: 5000 random points, key `PRNGKey(40)`, x and y uniform in [-15, 15], z uniform in [5, 45], then 50 time units at rho = 28.
Single-start statistics use the first 200 points. Ensembles use all 5000 as R = 50 disjoint ensembles of M = 100.

- **(a) naive:** reverse-mode autodiff through the unrolled RK4 scan (the Phase 3 quantity). The average uses the T/dt post-step states.
- **(b) truncated:** the same, with `stop_gradient` on the state at every segment boundary. Segment length L = **2.0** time units, which is about 1.8 Lyapunov times (1/lambda_1 = 1.1). The last segment may be shorter. The gate uses L = 2.0 only. Other L values may be shown as an extra, and they are never used for a gate.
- **(c) ensemble:** the mean of (a) over the M = 100 start points of an ensemble, and the mean of (b) the same way. One estimate per ensemble. 50 estimates.
- **(d) LSS:** the discretization above, with trapezoid rule on the samples of the RK4 trajectory (the T/dt post-step states). Block-tridiagonal solve by a block Thomas scan. No dense matrix. Derivative from Eq. 22 and 23 of the paper. alpha = **10** for Lorenz. I chose it because eta f must compete with v in units of the state size, which is 10 to 30. The paper text I found gives no value for Lorenz. The effect of alpha is checked in an exploratory sweep (alpha in 1, 3, 10, 30, 100) and not used for any gate.
- **(e) NILSS:** stretch goal. Not attempted in this phase.

## Gate for "works at horizon T"

The median of the method's estimates satisfies |median - 1.0029| <= 0.25 x 1.0029, and the interquartile range (Q3 - Q1, `numpy.percentile`, linear) is at most 1.0 x 1.0029. Estimates are per start point for (a), (b), (d) and per ensemble for (c).
Cost: wall time per estimate (warm, after compile) and peak memory above a bare process, each measured in its own process.

## Control first

Driven damped oscillator x'' + c x' + k x = F cos(w t) with c = 0.5, k = 4, F = 1, w = 1.5. State (x, v, theta), theta' = w. Quantity: time average of x^2. Closed form derivative in w: 0.37098692 (from Phase 2). The window is exactly 10 whole periods, 20000 samples, three windows of spin-up, 4 random start points.
Gate: every method matches the closed form with relative error below **1e-4**.

How the control applies to each method (fixed now):
- (a) naive and (c) ensemble: the window length depends on w (whole periods for every w, as in Phase 2), and the derivative runs through dt.
- (d) LSS: fixed dt, window set at the reference w.
- (b) truncated: **L = the whole window** (no truncation), which makes it equal to (a). This only tests the code. A truncated gradient with a short L cannot be right for a periodic system. The phase sensitivity grows without decay there, and truncation discards it. I also report (b) with L = 2.0 on the control. It is not a gate, and I expect it to fail.

Do not read any Lorenz result before the control passes for all methods.

## Predictions (not gates)

- **P4.1** (a) naive works at no horizon.
- **P4.2** (b) truncated, L = 2.0: median biased low by about 15% (range -35% to +5%). It works at T = 80 only (possibly T = 40 too). It fails at small T on the IQR criterion.
- **P4.3** (c) ensemble of naive works at T = 5 only (possibly T = 10), and fails for T >= 20 because the heavy tail makes the mean of 100 samples random.
- **P4.4** (c) ensemble of truncated works at all five horizons.
- **P4.5** (d) LSS works at T = 10, 20, 40, 80 and fails at T = 5 because of the free boundaries. At T = 80 the median error is below 5% and IQR / 1.0029 is below 0.15. At T = 20 IQR / 1.0029 is about 0.3.
- **P4.6** Control: all pass 1e-4. LSS control error about 1e-5 (range 1e-8 to 1e-4). Truncated with L = 2.0 fails the control (error above 1e-4).
- **P4.7** Cost per single estimate at T = 80, warm, one process: naive about 0.05 s, truncated about 0.05 s, LSS about 0.3 s (range 0.05 to 2 s). An ensemble of 100 costs about 100 times a single one. Peak memory above a bare process stays below 500 MB for every single estimate.
- **P4.8** (exploratory) LSS median at T = 80 changes by less than 5% for alpha in {3, 10, 30}.
- **Break-it** (each is planted after committing, restored with git):
  1. Drop the time-dilation term from LSS (no eta f in the constraint and no eta term in the derivative).
  2. Shifted average window: pair v_k with the observable at sample k + m/10.
  3. Swap the sign of the constraint: use +f_i eta_i in the constraint (the sign printed in Eq. 16) while keeping Eq. 22 as is.
  Prediction: all three are caught by the control test and by the Lorenz LSS test. A bug no test catches is reported and the tests are fixed.

Expected compute: about 15 minutes for the 5000-start-point gradients (chunked to limit memory), a few minutes for LSS. One heavy job at a time. Stop and report if a step takes over twice its estimate.

## Results (added after the runs, 2026-10-26)

Commands (all on CPU): `python experiments/phase4_fixes.py gen`, `grads` (59 s), `lss` (13 s), `alpha` (10 s), `report`, `cost` (33 s).
Output in `results/phase4*.json` and `results/phase4_g_T*.npz`. NILSS (e) was not attempted.

### Control (ran first)

| Method | Relative error against the closed form | Gate 1e-4 |
|---|---|---|
| (a) naive, 4 start points | 1.9e-11 | pass |
| (b) truncated with L = whole window | identical to (a) (tested for equality) | pass (code check only) |
| (c) ensemble mean of (a) | same as (a) | pass |
| (d) LSS, 10 periods, 4 start points | **4.49e-2**, same for all four | **FAIL** |
| (b) truncated with L = 2.0, not a gate | 0.897 | fails as expected |

LSS control failure, not fixed and not adjusted. The error is a finite-window effect. It halves each time the window doubles (200 samples per period, alpha = 1): 4.5e-2 at 10 periods, 1.1e-2 at 40, 2.5e-3 at 160 (about 1/T). It does not depend on the sample density (2000 against 200 samples per period gives the same error) and is not a boundary layer of the damping, since stronger damping (c = 2) made it larger (0.29). The solver and the formulas match a dense solve of the full constrained problem (`tests/test_fixes_1_lss_solver.py`), so I think the code is right and the 1e-4 gate cannot be met by LSS on a window of this length. I did not find a way to meet it without changing the gate. The test is kept as a strict `xfail`, and a passing test checks the 1/T convergence.
Detail for the control: the naive gradient differentiates through the spin-up as well, so no transient is left. This was needed for 1e-4 and does not change the gate.

### Lorenz, oracle 1.0029 (median over estimates; IQR / oracle)

| T | (a) naive | (b) truncated L = 2 | (c) ensemble naive | (c) ensemble truncated | (d) LSS |
|---|---|---|---|---|---|
| 5 | 0.868; 3.45 no | 0.861; 0.84 **yes** | 0.573; 5.9 no | 0.914; 0.63 **yes** | 0.989; 0.050 **yes** |
| 10 | 2.73; 168 no | 0.884; 1.08 no | -3.7; 241 no | 0.859; 0.62 **yes** | 1.004; 0.026 **yes** |
| 20 | 1.0e4; 1.1e6 no | 0.859; 0.92 **yes** | 3.1e4; 1.6e6 no | 0.914; 0.57 **yes** | 1.012; 0.012 **yes** |
| 40 | -1.2e12; 2.4e13 no | 0.850; 0.85 **yes** | -2.7e12; 7.8e13 no | 0.846; 0.56 **yes** | 1.014; 0.008 **yes** |
| 80 | 1.1e25; 7.4e28 no | 0.853; 0.83 **yes** | 3.8e28; 2.7e29 no | 0.851; 0.60 **yes** | 1.016; 0.003 **yes** |

Single-estimate methods use 200 start points. Ensembles are 50 ensembles of 100. Full numbers in `results/phase4.json`.

### Cost per single estimate (warm, one start point, one process; bare process 183 MB)

| T | naive (ms, MB above bare) | truncated | LSS |
|---|---|---|---|
| 5 | 0.5, 52 | 0.5, 51 | 4.1, 89 |
| 20 | 1.1, 51 | 0.8, 49 | 13.9, 98 |
| 80 | 2.8, 50 | 2.9, 51 | 78.8, 105 |

An ensemble of 100 costs 100 times a single estimate: about 0.29 s at T = 80 for (c), against 0.079 s for one LSS estimate.

### Predictions against results

| Prediction | Result |
|---|---|
| P4.1 naive works at no horizon | Hit. |
| P4.2 truncated: bias about -15% (range -35% to +5%) | Hit: median is 12% to 15% low. |
| P4.2 truncated works at T = 80 only (maybe 40) | **Miss.** It passes at T = 5, 20, 40 and 80 and fails at T = 10. Its IQR / oracle is 0.83 to 1.08 at every T, right at the gate of 1.0, so the verdict flips on noise. |
| P4.3 ensemble naive works at T = 5 only | **Miss.** It fails at T = 5 too (median 43% low, IQR / oracle 5.9). |
| P4.4 ensemble truncated works at all five | Hit (bias 9% to 16%, IQR / oracle 0.56 to 0.63). |
| P4.5 LSS works at T = 10 to 80 | Hit. |
| P4.5 LSS fails at T = 5 | **Miss.** It works: error 1.4%, IQR / oracle 0.05. |
| P4.5 LSS median error below 5% and IQR / oracle below 0.15 at T = 80 | Hit: 1.3% and 0.003. |
| P4.5 LSS IQR / oracle about 0.3 at T = 20 | **Miss by 25 times**: 0.012. |
| P4.6 control: all pass, LSS error about 1e-5 | **Miss: LSS fails the control** (4.5e-2). Others hit. Truncated L = 2 fails: hit. |
| P4.7 naive 0.05 s, truncated 0.05 s, LSS 0.3 s at T = 80 | **Miss on all three**: 2.8 ms, 2.9 ms and 79 ms (LSS is inside the stated range 0.05 to 2 s). |
| P4.7 peak memory below 500 MB above bare | Hit: at most 105 MB. |
| P4.8 LSS median changes under 5% for alpha 3 to 30 | Hit: 1.0169, 1.0157, 1.0146 (0.2%). |
| Compute (gradients 15 min, LSS a few minutes) | **Miss**: 59 s and 13 s. |
| Break-it: each caught by the control and the Lorenz LSS test | Partly. See below. |

### Break-it (each planted after committing, then restored with `git checkout`)

| Bug | Result |
|---|---|
| 1. Drop the time-dilation term | `3 failed, 37 passed, 1 xfailed`. Caught by the dense KKT test, the 1/T control test and the Lorenz LSS test. |
| 2. Shifted window (observable taken at sample k + m/10) | First run: `1 failed, 39 passed, 1 xfailed`. Caught **only** by the Lorenz LSS test. The control missed it because m/10 is exactly one period there. I added an independent assembly of Eq. 22 to the dense test. With bug 2 planted again that test fails. |
| 3. Swap the sign of the constraint (eta sign) | `3 failed, 37 passed, 1 xfailed`. Caught by the dense KKT test, the 1/T control test and the Lorenz LSS test. |

### Other findings

- **LSS and the oracle disagree by 1.3%.** LSS settles at 1.014 to 1.016, with IQR / oracle 0.003 at T = 80. The Phase 2 oracle is 1.0029 +/- 0.0021. The gap is far larger than the LSS spread. The gate cannot see it, because the gate allows 25%. Figure 6a of arXiv 1204.0159 shows an LSS plateau near 1.015 (read from a plot). The Phase 2b quadratic fit gave a local slope of 1.0101 +/- 0.0005 at rho = 28. Possible cause: the oracle is a central difference with h = 1, so it is a coarser slope than the local derivative LSS computes. The window slopes of Phase 2b differ by more than their errors, which fits a rough response. I have not tested this. A test would be the oracle at smaller h.
- **Truncation does not average out.** The per-start truncated gradient has a spread (IQR / oracle about 0.85) that does not shrink as T grows from 5 to 80, and the ensemble of 100 only brings it to about 0.6. The per-start values are heavy-tailed (largest 13102 at T = 5). This agrees in kind with the heavy tails Eyink et al. 2004 describe. I did not fit a tail exponent.
- **Truncation is biased low by 9% to 16% here and is wrong for a periodic system** (control error 0.90).
- LSS needs one trajectory and gives an IQR / oracle of 0.003 to 0.05. Its cost per estimate is 8 to 28 times that of one naive gradient.
