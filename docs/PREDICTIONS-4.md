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
