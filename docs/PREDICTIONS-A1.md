# Task A1 predictions and gates: oracle v2

Written and committed before any A1 code ran. Gates follow `TODO.md` (task A1) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Disclosure: what I had seen before predicting

- My own results: Phase 2 oracle d<z>/d(rho) = 1.0029 +- 0.0021 (h = 1). Phase 2b: line slope over rho 25 to 50 is 0.9888, quadratic-fit local slope at 28 is 1.0101 +- 0.0005. Phase 4: LSS gives 1.014 to 1.016 at T = 40 to 80. Its T = 10, 20, 40, 80 medians rise with T (1.004, 1.012, 1.014, 1.016).
- From `TODO.md` (written by the planning session from the nilss-jax README, not checked by me): nilss-jax reports d<z>/d(rho) "near 1.02" and d<z>/d(sigma) = +0.133. It quotes finite differences of +0.150 +- 0.004 for sigma, and says rho and beta are within 1.5% of finite differences. I have seen no beta value.
- The sigma prediction below is therefore **not independent** of that README. I did not read any nilss-jax source or output. I do not tune anything to these numbers.

## Method (fixed now)

Lorenz sigma = 10, rho = 28, beta = 8/3. dt = 0.01, RK4 (Phase 1 code). Quantity: time average of z.
The pipeline is the Phase 2 pipeline: random start points (x, y uniform in [-15, 15], z uniform in [5, 45]), 50 time units of spin-up at each parameter value, then average z over 100 time units. The same start points are used at theta - h, theta and theta + h.

Parameters and step ladders (rung k = 0, 1, 2, 3):
- rho: h = 1, 0.5, 0.25, 0.125.
- sigma: h = 1, 0.5, 0.25, 0.125.
- beta: h = 0.25, 0.125, 0.0625, 0.03125.

Start points per rung: N_k = 2000 x 4^k, so 2000, 8000, 32000, 128000. The noise of the slope grows like 1/h, so this keeps it near constant for rho and sigma. For beta the step is 4 times smaller than the rho step, so its noise is about 4 times larger at the same N. I accept that.
Each (parameter, rung) uses an independent set of start points: key `fold_in(PRNGKey(60), 10 x parameter_index + k)`, with parameter_index 0, 1, 2 for rho, sigma, beta.

Per rung I report:
- slope = (mean z(theta + h) - mean z(theta - h)) / (2 h),
- curvature = (mean z(theta + h) + mean z(theta - h) - 2 mean z(theta)) / h^2,
- bootstrap standard errors for both (2000 resamples, start points resampled jointly across the three runs, key `PRNGKey(61)`).

Expected compute: about 10 minutes on CPU for all three parameters, from 7.7e9 RK4 steps per parameter at the Phase 2 speed of about 4e7 steps per second. I run one parameter at a time. If a parameter takes more than 15 minutes I stop and report.

## Control first

The same three-point pipeline on the driven damped oscillator (c = 0.5, k = 4, F = 1, w = 1.5), average of x^2, whole-period windows as in Phase 2. Closed-form first and second derivative in w (by `jax.grad` and `jax.hessian` of the closed form).
Gates: slope with h = 1e-4 has relative error below 1e-6. Curvature with h = 0.02 has relative error below 1e-3. 4 random start points.
Do not read any Lorenz result before the control passes.

## Gates

- **A1.1 Convergence, per parameter.** For each pair of successive rungs (k, k + 1): |slope_k - slope_{k+1}| <= 2 sqrt(SE_k^2 + SE_{k+1}^2). The parameter is "converged" if all three pairs pass. If not, I say "not converged" and report which pairs failed.
- **A1.2 Precision.** At the smallest step, SE < 5% of |slope|, for each parameter. If not, I raise N and report the cost. The gate stays.
- **Final value for a parameter:** the slope at the smallest step with its SE. If converged I also give the inverse-variance weighted mean of all four rungs as a second number.
- **How I will describe the rho result (fixed now):** slope at h = 0.125 at or above 1.012: "above the Phase 2 oracle, consistent with LSS". At or below 1.006: "stays near the Phase 2 oracle". Between: "in between, not settled".

## Predictions (not gates)

- **P-A1.1 rho:** slope at h = 0.125 is **1.008** (range 0.996 to 1.022), so the description is "in between". SE at h = 0.125 about 0.0015.
- **P-A1.2 rho convergence:** the pair (h = 1, 0.5) is the one most likely to fail. I predict the ladder is converged for h <= 0.5 and that the h = 1 rung is low by less than 0.01.
- **P-A1.3 sigma:** slope at h = 0.125 is **+0.15** (range +0.10 to +0.20). Not independent, see the disclosure. sigma is not converged at the first pair, because h = 1 is 10% of sigma.
- **P-A1.4 beta:** slope at h = 0.03125 is **negative, about -3** (range -8 to -0.5). This is a guess from the sign of the -beta z term. I have no number to base it on. beta converged.
- **P-A1.5 curvature:** the curvature in rho is within 2 SE of zero at h = 1 and has an SE too large to say more at h = 0.125. Reason: Phase 2 found c2 = -0.0014 +- 0.0032 (so curvature about -0.003).
- **P-A1.6 control:** both gates pass.
- **P-A1.7 compute:** about 10 minutes in total.
