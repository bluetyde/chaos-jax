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

## Results (added after the runs, 2026-10-26)

Commands: `python -m pytest -q tests/test_oracle_v2_control.py` (control, first), then `python experiments/phaseA1_oracle_v2.py run rho|sigma|beta` (88 s per parameter) and `report`.
Summary in `results/phaseA1_oracle_v2.json`. The raw per-start-point arrays (12 MB) are not committed. They are regenerated by the `run` command, which is deterministic for the fixed keys.
Compute: rho 88 s, then sigma and beta together 178 s. Total about 4.4 minutes of CPU.

### Control (ran first)

Slope with h = 1e-4: relative error 5.9e-8 (gate 1e-6). Curvature with h = 0.02: relative error 8.0e-4 (gate 1e-3). Both pass. The curvature margin is small.

### Slopes (central difference; bootstrap SE; same pipeline as Phase 2)

| Parameter | h | N | slope +- SE | curvature +- SE |
|---|---|---|---|---|
| rho | 1 | 2000 | +1.00331 +- 0.00151 | +0.0016 +- 0.0052 |
| rho | 0.5 | 8000 | +1.00305 +- 0.00150 | +0.0205 +- 0.0099 |
| rho | 0.25 | 32000 | +1.00486 +- 0.00147 | -0.0081 +- 0.0199 |
| rho | 0.125 | 128000 | **+1.00150 +- 0.00145** | +0.0314 +- 0.0408 |
| sigma | 1 | 2000 | +0.15145 +- 0.00138 | -0.0789 +- 0.0052 |
| sigma | 0.5 | 8000 | +0.14887 +- 0.00142 | -0.0797 +- 0.0100 |
| sigma | 0.25 | 32000 | +0.14530 +- 0.00146 | -0.0901 +- 0.0198 |
| sigma | 0.125 | 128000 | **+0.14936 +- 0.00147** | -0.0727 +- 0.0404 |
| beta | 0.25 | 2000 | -1.63064 +- 0.00583 | -0.470 +- 0.077 |
| beta | 0.125 | 8000 | -1.65030 +- 0.00581 | -0.495 +- 0.158 |
| beta | 0.0625 | 32000 | -1.66141 +- 0.00579 | -0.441 +- 0.323 |
| beta | 0.03125 | 128000 | **-1.65814 +- 0.00595** | +0.112 +- 0.637 |

### Gates

| Gate | rho | sigma | beta |
|---|---|---|---|
| A1.1 pair 1 (diff / limit) | 0.00026 / 0.00425 ok | 0.00257 / 0.00396 ok | 0.01966 / 0.01647 **FAIL** |
| A1.1 pair 2 | 0.00181 / 0.00420 ok | 0.00357 / 0.00407 ok | 0.01111 / 0.01642 ok |
| A1.1 pair 3 | 0.00336 / 0.00412 ok | 0.00406 / 0.00414 ok | 0.00327 / 0.01661 ok |
| A1.1 converged | **yes** | **yes** | **no** (one pair, the one with the largest steps) |
| A1.2 SE below 5% at smallest h | yes (0.14%) | yes (1.0%) | yes (0.36%) |
| Final value (smallest h) | 1.0015 +- 0.0015 | 0.1494 +- 0.0015 | -1.658 +- 0.006 |
| Weighted mean of all rungs | 1.0032 +- 0.0007 | 0.1488 +- 0.0007 | -1.650 +- 0.003 (not converged, so not a gate result) |

rho description (rule fixed in advance): the slope at h = 0.125 is 1.0015, below 1.006, so **the oracle stays near the Phase 2 value**. The converged answer did not move to 1.012 or above.

### Predictions against results

| Prediction | Result |
|---|---|
| P-A1.1 rho slope 1.008 (range 0.996 to 1.022), "in between" | 1.0015. In range, point 0.007 high, and the description is "stays near" instead of "in between". **Miss on the description.** SE 0.00145 against 0.0015: hit. |
| P-A1.2 rho: converged for h <= 0.5, h = 1 low by under 0.01 | The whole ladder is converged and the h = 1 rung is not low (1.0033 against 1.0015). **Partial miss.** |
| P-A1.3 sigma +0.15 (range +0.10 to +0.20) | 0.1494. Hit (not independent of the README). |
| P-A1.3 sigma not converged at the first pair | **Miss.** Converged, with pair 3 close to its limit (0.00406 against 0.00414). |
| P-A1.4 beta about -3 (range -8 to -0.5) | -1.658. In range, point miss. |
| P-A1.4 beta converged | **Miss.** One pair failed. |
| P-A1.5 rho curvature within 2 SE of zero at h = 1 | 0.0016 +- 0.0052. Hit. At h = 0.5 it is 2.1 SE from zero. |
| P-A1.6 control gates pass | Hit. |
| P-A1.7 compute about 10 minutes | **Miss.** About 4.4 minutes. |

### What this shows, and what it does not

- **The Phase 2 oracle for rho was not biased by h = 1.** Four rungs from h = 1 to 0.125 agree within noise. The best number is 1.003 (weighted mean 1.0032 +- 0.0007) or 1.0015 +- 0.0015 at the smallest step.
- **The disagreement with LSS grows.** Phase 4 LSS gives 1.014 to 1.016 and the nilss-jax README is about 1.02. The weighted-mean oracle is 1.0032 +- 0.0007. The gap to our LSS is 0.013, about 18 times the oracle SE. I did not find the cause. Candidate causes, not tested: a bias of the shadowing estimators on this system (my LSS estimate rises with T from 1.004 at T = 10 to 1.016 at T = 80, so it may not have converged either), or a limit to what the finite-difference oracle measures at this window length (T = 100 average, 50 spin-up). The 25 to 50 fit and the wide-window slopes from Phase 2b already show that the local slope varies by about 1% between windows.
- **sigma:** 0.1494 +- 0.0015 agrees with the finite-difference value +0.150 +- 0.004 that the README quotes (as relayed in `TODO.md`). It sits 12% above the nilss-jax value +0.133 relayed in `TODO.md`. This is the number A2 will check.
- **beta:** -1.658 +- 0.006 at the smallest step. The ladder is not converged at the largest step. The slope moves from -1.631 (h = 0.25) to -1.650 and -1.661, and then -1.658. That is a curvature effect in the slope. Two of three pairs pass, and one failure in nine such tests (3 parameters x 3 pairs) is not unusual by chance. I still report it as "not converged" under the gate.
- **Limits:** one averaging window (100 time units, 50 spin-up) and one dt (0.01). I did not vary them. The random start points come from a box and a short spin-up, not from a long run on the attractor.
