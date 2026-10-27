# chaos-jax: results

Written 2026-10-27. This file reports what was measured, with the numbers, the error bars and the failures.
Every number here comes from a file in `results/` or from a predictions file in `docs/`. The appendix maps each headline number to its file. `tests/test_results_doc.py` checks the headline numbers against the result files.

## What this is

This repository is a small lab and an independent audit. It is not a new method.
It measures when autodiff gradients through chaotic simulations fail on Lorenz-63. It then compares ways to fix the failure. It then audits a young JAX package for shadowing sensitivities (`nilss-jax`) against a finite-difference oracle.
Almost all of this was known in outline before this work. The prior art is named in section 8. The parts that are measurements made here are listed in section 9.

## 1. Summary

1. **Plain gradients blow up as theory says.** For Lorenz-63 at rho = 28, the gradient of a time average of z with respect to rho has a median size more than 10 times the oracle at an averaging time T* = 8. At long T the sign of the gradient is random (53% negative at T = 30). The growth of the state Jacobian matches the Lyapunov exponent: slope 0.9057 against 0.9041.
2. **Of the fixes tried, only least squares shadowing (LSS) worked at every horizon from 5 to 80 time units.** Truncation was biased 12% to 15% low and noisy. Ensembles of plain gradients failed. Ensembles of truncated gradients passed the gate, with a median 9% to 16% too low.
3. **A finite-difference oracle gives d<z>/d(rho) = 1.003, d<z>/d(sigma) = 0.149 and d<z>/d(beta) = -1.658 at the standard parameters.** It is stable against its step, averaging window, dt and spin-up for rho and sigma. One check on beta is not stable for a short spin-up.
4. **Three shadowing codes agree with each other and differ from the oracle.** The codes are `nilss-jax` 0.1.0, my own LSS and my own NILSS (written from the papers). They agree within 0.5% for all three parameters. They sit about 1.4% above the oracle for rho and about 10% below it for sigma. For beta all agree. A noise-regularised finite-difference check sided with the oracle, under an assumption I did not test.
5. **These two gaps are not new findings.** The `nilss-jax` README already states the sigma gap (+0.133 against +0.150) and a rho difference of about 1.5%. A published paper (arXiv 1806.02077) reports a reproducible 2% rho bias between finite differences and all shadowing methods it tried, with the same observable and parameters. This repository replicates both on its own oracle. I did not find the cause. A hyperbolicity test (section 4.10) found that at rho = 28 the unstable and stable directions stay at least 0.12 rad apart, so a near-tangency between them does not explain the gap there. Another candidate, a missing unstable contribution, was not tested.
6. **`nilss-jax` is stable against its settings** (dt, segment length, number of tangents, trajectory length) and an independent NILSS written from the papers reproduces it to 0.11% or better.
7. **Its reliability report ordered its verdicts by error on the cases I could score, and I could score few.** On the 4 scored cases, `ok` came with errors of 0.8% to 1.5%, `caution` with 12.3% and `unreliable` with 38% or more. Only 4 of 13 finite-difference oracles converged in my calibration. All 8 `ok` verdicts that could be scored come from one system and one parameter point.

### Gates that failed or were weak, in one list

- Phase 4: the LSS control gate fails (error 4.5e-2 against 1e-4 on a 10-period window). It is kept as a strict `xfail`.
- Phase 3: P3.4 fails as the plan states it (float32 leaves float64 at T = 14, float64 leaves the oracle at T = 2).
- A1: the beta finite-difference ladder is not converged at its largest step.
- A2: gate A2.1 fails for rho and sigma (and passes for beta).
- A3: A3.3 fails on the strict criterion for all three parameters. A3.4 fails for rho and sigma.
- A4: 9 of 13 oracles did not converge, so the calibration gate is weakly tested.
- A5: one control (C-A5.4) failed at first because of a bug in my own test. It was repaired. The gate was not changed.

## 2. What was built

| Part | Files | Notes |
|---|---|---|
| Core: RK4 with `jax.lax.scan`, Lorenz-63, damped and driven oscillators, double pendulum, Lorenz-96, Roessler | `src/chaos_jax/integrate.py`, `systems.py` | float64 only |
| Tangent propagation, Benettin Lyapunov exponents (tangent method and two-trajectory method) | `tangent.py`, `lyapunov.py` | |
| Ensemble finite-difference pipeline, bootstrap | `response.py` | |
| Gradient methods: naive, truncated, block-tridiagonal LSS | `fixes.py`, `controls.py` | LSS discretised from the continuous problem. See section 4.5. |
| Noisy integrator | `noisy.py` | for the noise-regularised check |
| NILSS from the papers | `nilss.py` | |
| Tests | `tests/` | 67 pass and 1 is an expected failure (`xfail`) |
| Experiments | `experiments/` | results in `results/` |

The last full test run is quoted in section 11.

## 3. How the work was done

- **Predictions and gates were committed before each experiment ran.** Each task has a `docs/PREDICTIONS-*.md` file. Results and failed predictions are appended after the run. Gates were never edited after results. Where something had to change, a dated Amendment records it.
- **Controls ran first.** Each task has controls with known answers. If a control failed, I stopped and found the cause.
- **Planted-bug checks.** For Phase 1, Phase 4 and A5 I planted bugs after committing, confirmed that tests failed and restored the tree with git. Details are in sections 4.1, 4.5 and 4.9.
- **Disclosure.** Each predictions file for the audit states which `nilss-jax` numbers I had already seen. Those predictions are weak evidence when they hit.
- **Independence of the NILSS code.** I did not read `nilss-jax` source at any time. I read its README, changelog, `docs/usage.md` and `docs/reliability.md`. The pinned commit is `ad5b635f5798d89f5d31afa4118d31b778ab86f7`.
- **Equation checks.** Equations from papers were checked against PDF pages. The LSS paper (arXiv 1204.0159) has printed equations that disagree with each other (section 4.5). The NILSS papers did not.

## 4. Results

### 4.1 Phase 1: core and controls

All five gates pass. `docs/PHASE1-CHOICES.md` has the choices, written before any exponent was measured.

| Gate | Result |
|---|---|
| G1.1 RK4 order in [3.8, 4.2] | passes on the damped oscillator and on a logistic ODE for every dt halving from 0.1 to 0.0125 |
| G1.2 autodiff against central finite difference below 1e-6 | passes for the oscillator, Lorenz (state Jacobian and d z / d rho) and the pendulum |
| G1.3 Lorenz lambda_1 from two start points | 0.9064 and 0.9018 (tangent method), 0.9039 and 0.9043 (two-trajectory method). Difference 0.0046, limit 0.02. |
| G1.4 controls | damped oscillator -0.2022 and -0.1978 (exact -0.2). Low-energy pendulum +0.0015. |
| G1.5 high-energy pendulum | lambda_1 = 1.3626. Energy drift 2.06e-8 over T = 10 at dt = 0.00125. |

The sum of the Lorenz spectrum is -13.66656. The exact value is -13.66667. The plan's 0.906 was used only as a sanity band.
Planted bugs: a flipped sign of the beta term (6 tests failed), a wrong dt in one RK4 stage (7 failed) and float32 (15 failed). All were caught.

### 4.2 Phase 2: finite-difference oracle for d<z>/d(rho)

- Central difference over rho = 27 and 29, 1000 start points, average over 100 time units after 50 of spin-up.
- **d<z>/d(rho) = 1.0029 +- 0.0021** (bootstrap). Gates G2.1, G2.2 and G2.3 pass. The driven-oscillator control matches its closed form with relative error 5.9e-8.
- The quadratic coefficient is -0.0014 +- 0.0032, so the response is linear over [27, 29] within the error bars.
- Follow-up over rho = 25 to 50: the line slope is 0.9888 and the local slope in windows of width 5 falls from 1.007 on [30, 35] to 0.963 on [45, 50]. A single run in the setup of Lea et al. gives 0.9925.

### 4.3 Oracle v2 and its checks (A1, A1b, A3b)

**A1: smaller steps.** Four rungs of h and up to 128000 start points per rung.

| Parameter | Smallest step | Weighted mean of rungs | Ladder converged |
|---|---|---|---|
| rho | 1.0015 +- 0.0015 | 1.0032 +- 0.0007 | yes |
| sigma | 0.1494 +- 0.0015 | 0.1488 +- 0.0007 | yes |
| beta | -1.658 +- 0.006 | not used | **no** (one pair fails: 0.0197 against a limit of 0.0165) |

The audit uses 1.0032 for rho, 0.1488 for sigma and -1.658 for beta.

**A1b: window, dt, spin-up.** One factor changed at a time (average over 25 or 400, dt 0.02 or 0.005, spin-up 10 or 200). Across all variants rho ranges from 0.9976 to 1.0033 and sigma from 0.1456 to 0.1514. The largest difference to the baseline for rho or sigma is 2.03 standard errors. **The gap to the shadowing values is not explained by these settings.** Beta with a spin-up of 10 gives -1.615 (5.1 standard errors off). Spin-ups of 50 and 200 agree.

**A3b: noise-regularised finite differences.** I added white noise of size eps to Lorenz and looked at the slope as eps falls. The slope falls with more noise and plateaus as eps goes to 0.

| eps | rho slope (h = 0.5) | sigma slope (h = 0.5) |
|---|---|---|
| 4 | 0.9245 | 0.1057 |
| 1 | 0.9678 | 0.1456 |
| 0.125 | 1.0012 +- 0.0015 | 0.1466 +- 0.0014 |

The step sizes h = 1 and h = 0.5 agree at all 14 noise levels and parameters. By the decision rule written in advance, both parameters land on the finite-difference side. This reading assumes that the small-noise derivative continues to eps = 0. I did not test that assumption.

### 4.4 Phase 3: gradient blow-up

| Gate | Result |
|---|---|
| P3.2 controls: slope of log norm of the state Jacobian at most 0.02 | oscillator -0.2004, low-energy pendulum -0.0146. Pass. |
| P3.1 Lorenz slope within 15% of 0.9041 | 0.9057 +- 0.0037. Pass. |
| P3.3 median size of the gradient above 10 times the oracle | T* = 8 (median 13.5 against 1.0029). Pass. |
| P3.4 float32 leaves float64 before float64 leaves the oracle | T_leave32 = 14, T_leave64 = 2. **Fail.** |

At T = 80 the median naive gradient is 1.1e25. The sign is random at long T (37.5% negative at T = 5, 53% at T = 30). Float64 is already 40% off the oracle at T = 1 because short averages are biased low. So the float64 side of P3.4 has no clean window. Float32 stays within 1e-3 of float64 up to T = 7. After the fact I found that over T = 10 to 30 the median grows like exp(lambda_1 T) / T (slope of ln(T x median) = 0.903). This was not predicted.

### 4.5 Phase 4: the fixes

Oracle 1.0029. A method "works" at horizon T if the median is within 25% of the oracle and the interquartile range is within 100% of it.

| Method | Works at T = | At T = 80: median (IQR / oracle) |
|---|---|---|
| naive autodiff | none | 1.1e25 (7e28) |
| truncated, L = 2 | 5, 20, 40, 80 (not 10: IQR 1.08) | 0.853 (0.83) |
| ensemble of naive | none | 3.8e28 |
| ensemble of truncated | 5, 10, 20, 40, 80 | 0.851 (0.60) |
| LSS | 5, 10, 20, 40, 80 | 1.016 (0.003) |

- Cost at T = 80, one estimate: naive 2.8 ms, truncated 2.9 ms, LSS 79 ms. Memory above a bare process: 50, 51 and 105 MB.
- The spread of the truncated gradient does not shrink with T. The per-start values are heavy-tailed (largest 13102 at T = 5). I did not fit a tail exponent.
- **Failed control: LSS on the driven oscillator.** Error 4.5e-2 on a 10-period window against the gate 1e-4. It falls like 1/T (4.5e-2, 2.2e-2, 1.1e-2, 5.4e-3, 2.5e-3 at 10, 20, 40, 80, 160 periods) and does not depend on the sample density. The LSS code matches a dense solve of the full problem. The cause of the 1/T term is not known. The test is a strict `xfail`.
- **Printed equations that disagree.** In arXiv 1204.0159 (as in the library copy), Eq. 16 to 21 disagree with each other in four places: a missing factor 1/2 in E and G, the sign of eta, the sign of the right-hand side of the Schur system, and the row count in Eq. 20. I derived my own discretisation from Eq. 13 and 15. I did not check errata or other versions.
- Planted bugs in LSS: dropping time dilation (3 tests failed), a shifted window (1 failed, then a test was added), the wrong sign of the constraint (3 failed). All were caught.

### 4.6 A2: `nilss-jax` on Lorenz-63

32 orbits, T = 1000, documented settings (dt 0.005, T_seg 0.5, one tangent). The controls pass (exponent 0.9053, mean z 23.552).

| Parameter | `nilss-jax` | Oracle | Difference | A2.1 (2 combined SE) |
|---|---|---|---|---|
| rho | 1.01708 +- 0.00007 | 1.00317 +- 0.00074 | +1.39% | fail |
| sigma | 0.13375 +- 0.00008 | 0.14882 +- 0.00072 | -10.13% | fail |
| beta | -1.65927 +- 0.00116 | -1.65814 +- 0.00595 | -0.07% | pass |

The README claim on sigma reproduces (0.13375 against 0.133). The tool's own error bars are tiny (orbit-to-orbit spread 0.0004 for rho). The gaps are not noise.

### 4.7 A3: settings and a second shadowing code

- **Settings do not matter** over dt 0.0025 to 0.01, T_seg 0.25 to 1, one or two tangents and T 250 to 4000. Rho ranges from 1.0170 to 1.0178, sigma from 0.1336 to 0.1338 and beta from -1.658 to -1.663. The sigma gap stays at about 21 standard errors.
- **My LSS** (T = 320, 200 start points) gives rho 1.01637, sigma 0.13429 and beta -1.65591. This differs from `nilss-jax` by -0.07%, +0.41% and +0.20%. A3.3 (agreement within 2 combined SE) fails for all three because both error bars are tiny. A3.4 (converged in T) fails for rho and sigma: the LSS value still rises with T.

### 4.8 A4: calibration of the reliability report

13 cases (Lorenz-63 at rho 28, 45, 50, 60 and sigma and beta; Lorenz-96 N = 8 at F = 6 and 10 and N = 16 at F = 8; Roessler c = 5, 7, 9), four ensembles of 16 orbits each, own oracle.

- On the 4 oracle-converged cases: `ok` came with errors of 0.8% to 1.5% (8 ensembles), `caution` with 12.3% (4) and `unreliable` with 38% to 2946% (4).
- **A4.1 passes**: no `ok` verdict with an error above 25%. A4.2 is met: 8 `ok` ensembles, the minimum was 6. There are no misses and no false alarms among the scored cases.
- The four ensembles of a case gave the same verdict in 13 of 13 cases.
- **Weak test.** Only 4 of 13 oracles converged (I predicted at least 10). All 8 `ok` ensembles are Lorenz-63 at rho = 28 (parameters rho and beta). The `ok` for rho carries a 1.4% bias in all four ensembles. This is the tool author's own documented blind spot.
- Not scored, after the fact: at rho = 45 the means are within a few percent of the finite-difference range while the verdict is `unreliable`. The report may be conservative there.

### 4.9 A5: my own NILSS

Written from arXiv 1611.00880 and 1711.06633, with no `nilss-jax` source read. The controls pass (Schur solve against a dense KKT solve, tangents against finite differences, continuity, formula 41 against formula 15 at 1.4e-4).

| Parameter | My NILSS | `nilss-jax` | Difference | Oracle |
|---|---|---|---|---|
| rho | 1.01701 +- 0.00006 | 1.01708 | -0.007% | +1.38% |
| sigma | 0.13363 +- 0.00009 | 0.13375 | -0.089% | -10.21% |
| beta | -1.65739 +- 0.00112 | -1.65927 | +0.113% | +0.05% |

A5.1a, A5.1b, A5.2 and A5.3 pass. Planted bugs (drop the xi term, wrong continuity sign, skip the reset) were all caught. The continuity control missed the third one at first. It was fixed.

### 4.10 A5b: hyperbolicity test

I computed covariant Lyapunov vectors with the algorithm of Ginelli et al. (arXiv 1212.3961), written by me. The controls pass after repairs: a linear flow with known eigenvectors (1e-8), the neutral vector against the flow direction (1e-6 at dt = 0.005), exponents from the vectors (0.9085, -0.0002, -14.575), independence of the backward start, and covariance under the true tangent flow. Two controls failed at first, one from a test that was too short and one because the RK4 map's neutral direction differs from the flow at fourth order in dt. A third problem was a precision floor in my angle function. Details are in `docs/PREDICTIONS-A5b.md`.

| At rho = 28, two orbits of T = 10000 | Smallest angle | Verdict |
|---|---|---|
| unstable against stable (theta1) | 0.123, 0.134 rad | bounded away from zero |
| stable against the centre-unstable plane (theta2) | 0.120, 0.131 rad | bounded away from zero |
| unstable against the flow (theta3) | 0.007, 0.008 rad (1.1% of samples below 0.05) | near-tangency recurrences |

- **Segments near a tangency or near the equilibrium do not carry the gap.** Removing the 5% of segments with the smallest angle, or the smallest |f|, changes the estimate by at most 0.0002. The gap is 0.0138. All shifts lie inside the null of random removals.
- **Across rho** (26, 28, 30, 33, 36, 40) the gap of my NILSS against finite differences is +4.47% (oracle not converged), +1.34%, +0.87%, -0.40%, +1.00% and -1.02%. It is smallest in size at rho = 33, as in arXiv 1806.02077. Near-tangencies between stable and unstable directions appear from rho = 33 (P below 0.1 rad up to 0.4%). By the rule fixed in advance, the correlation of the gap with the angle statistics is an association (magnitude 0.81 to 0.83). It is the same as the correlation with rho itself (-0.83), and without the rho = 26 point it falls to 0.70. I read it as a trend with rho, not as evidence for the hyperbolicity explanation.
- **Result:** the gap exists at rho = 28 and 30, where no near-tangency between the stable and unstable directions was seen. The test removes that explanation for rho = 28. It does not remove the other candidate (below).

## 5. The central result: shadowing against finite differences

d<z>/d(parameter) at rho = 28, sigma = 10, beta = 8/3:

| Method | rho | sigma | beta |
|---|---|---|---|
| Finite differences (A1, weighted mean; beta smallest step) | 1.0032 +- 0.0007 | 0.1488 +- 0.0007 | -1.658 +- 0.006 |
| Noise-regularised finite differences, eps = 0.125 (A3b) | 1.0012 +- 0.0015 | 0.1466 +- 0.0014 | not run |
| `nilss-jax` 0.1.0 (A2) | 1.0171 | 0.1338 | -1.6593 |
| My LSS, T = 320 (A3) | 1.0164 | 0.1343 | -1.6559 |
| My NILSS (A5) | 1.0170 | 0.1336 | -1.6574 |

The three shadowing codes use two formulations (LSS and NILSS). I wrote two of them and the `nilss-jax` author wrote the third. They agree within 0.5%. They all use RK4 with a trapezoid-style quadrature, so a shared numerical effect is not excluded. The `fds` check that would test this (A7b) was not done.
The published paper arXiv 1806.02077 reports the same pattern for rho with the same observable, sigma, beta and step (Delta rho = 0.5). In its sections 5.3 and 5.4 it finds shadowing-type methods converge to about 1.017 and finite differences stay lower. It states the bias does not depend on the integration step or on the finite-difference step. It attributes the bias to the lack of uniform hyperbolicity in Lorenz-63, and it says linear shadowing-based methods are "questionable" there. It also reports that a method on unstable periodic orbits converges to the shadowing value.
**What this repository does not settle:** which number is closer to the true derivative. The noise check and the reading of that paper point to the finite-difference side. Neither is a proof.
**Two candidate causes of the gap remain.** The hyperbolicity test (section 4.10) did not support the first. The second was not tested.
1. *A lack of uniform hyperbolicity*, the reading in arXiv 1806.02077. At rho = 28 the unstable and stable directions stay at least 0.12 rad apart in my samples. Near-tangencies appear only from rho = 33. The unstable direction does approach the flow direction (as it must near the equilibrium), but removing those segments does not change the estimate.
2. *A missing unstable contribution.* Ni (arXiv 2009.00595, abstract and sections 1.1 to 1.3, which I read) writes the linear response of a hyperbolic system as a shadowing contribution plus an unstable contribution, and NILSS computes only the first. He says the shadowing part alone is a reasonable approximation when the unstable dimension is much smaller than the dimension. For Lorenz-63 it is 1 out of 3. That could give a gap even for a uniformly hyperbolic system. I did not implement the unstable part, so I cannot test this. **Corrections reported to me by the owner, which I have not checked in the sources:** the claim comes from Ni's arXiv 2003.09801, which I did not read, and it carries two further conditions (fast decay of correlations, and an objective gradient and a perturbation that are not particularly aligned with the unstable direction). A5b tested neither. The owner also reports that the papers with the continuous-time version call themselves a building block, say that Lorenz systems are not hyperbolic and that the linear response might not rigorously exist, and say the algorithm can explode. The papers in the library on the unstable contribution (2002.04117, 2103.08816, 2111.07692) cover maps only. So this candidate explanation may have no meaningful test for Lorenz-63, and I did not run one.
The hyperbolicity test could have explained a gap. It could not say which side is true.

## 6. Every failed prediction

"Point miss" means the value was inside my stated range. Phase 1 had no separate predictions file. Its gates came from the plan.

| Task | Prediction | What happened |
|---|---|---|
| Phase 2 | SE of the oracle about 0.02 | 0.0021 (10 times smaller) |
| Phase 2 | control error about 1e-8 | 5.9e-8 |
| Phase 2 | compute about 3 minutes | 1.9 s |
| Phase 3 | pendulum slope about +0.01 | -0.0146 (wrong sign, gate passes) |
| Phase 3 | T* = 6 | 8 (point miss) |
| Phase 3 | T_leave64 = 4, T_leave32 = 16 | 2 and 14 (point misses) |
| Phase 4 | truncated works at T = 80 only | works at 5, 20, 40, 80 |
| Phase 4 | ensemble of naive works at T = 5 | fails (43% low, IQR 5.9) |
| Phase 4 | LSS fails at T = 5 | works (error 1.4%) |
| Phase 4 | LSS IQR / oracle about 0.3 at T = 20 | 0.012 |
| Phase 4 | all methods pass the control, LSS error about 1e-5 | LSS fails the control (4.5e-2) |
| Phase 4 | naive 0.05 s, truncated 0.05 s, LSS 0.3 s | 2.8 ms, 2.9 ms, 79 ms |
| Phase 4 | compute 15 minutes plus a few | 59 s and 13 s |
| A1 | rho slope 1.008 and description "in between" | 1.0015, "stays near the Phase 2 value" |
| A1 | rho ladder: h = 1 rung low by under 0.01 | not low (partial miss) |
| A1 | sigma not converged at the first pair | converged |
| A1 | beta converged | one pair fails |
| A1 | beta about -3 | -1.658 (point miss) |
| A1 | compute 10 minutes | 4.4 minutes |
| A1b | no variant beyond 3 SE, beta within 0.02 | beta with spin-up 10 is 5.1 SE and 0.04 off |
| A1b | compute 4 minutes | 72 s |
| A2 | compute 11 minutes | 23 s |
| A3 | LSS rho within 2 SE of `nilss-jax` | fails the strict test (relative difference 0.07%) |
| A3 | LSS converged in T for rho | not converged |
| A3b | slope at eps = 4 within 0.03 of eps = 0.125 | rho 0.077 lower, sigma 0.041 lower |
| A3b | compute 10 minutes | 27 minutes |
| A4 | verdicts at rho = 45, 50 and sigma at rho = 45 are `caution` | `unreliable` in all four ensembles |
| A4 | Lorenz-96 and Roessler errors 15% to 60% | 5.5% to 14717% |
| A4 | at least 10 of 13 oracles converge | 4 |
| A5 | controls pass at the first run | C-A5.4 failed (bug in my test) |
| A5 | A5.1a fails for at least two parameters | passes for all three |
| A5 | compute 1 to 5 minutes | 4 s |

| A5b | controls pass at the first run | C-B1 and C-B2 failed at first, and my angle function had a precision floor (repaired, no gate changed) |
| A5b | smallest unstable-stable angle above 0.3 rad | 0.123 and 0.134 rad |
| A5b | correlation of the smallest \|f\| per segment with the contribution is -0.1 to -0.5 | +0.004 |
| A5b | gap above 1.5% at rho = 28 | 1.34% |
| A5b | no association between the gap and the angle statistic | association by the rule (+0.83), but it is a trend with rho |
| A5b | compute 10 minutes | about 1.6 minutes |

In all, 38 rows. Several predictions in the audit tasks were made after I had read `nilss-jax` numbers. Hits there are weak evidence.

## 7. Compute used

CPU only. One process at a time on a machine with 4 cores. Approximate wall time of the measurements, from the files in `results/` and the handoff entries.

| Task | Wall time |
|---|---|
| Test suite (fresh run) | about 1 minute |
| Phase 2 oracle | 1.9 s |
| Phase 3 | about 70 s |
| Phase 4 (gradients, LSS, cost) | about 2 minutes |
| A1 | 4.4 minutes |
| A1b | 72 s |
| A2 | 28 s |
| A3 | 3.6 plus 2.1 minutes |
| A3b | 27 minutes |
| A4 | about 24 minutes |
| A5 | 4 s |
| A5b | about 1.6 minutes |

The total is about 72 minutes of measurement, apart from development and testing.

## 8. Prior art, named plainly

What I read is stated for each item. I did not describe any paper I did not open.

- **`nilss-jax`** (JAX, MIT, one author, first release 2026-10-07 to 08). I read its README, changelog, `docs/usage.md` and `docs/reliability.md`. I read no source. It already provides streaming multi-parameter NILSS, a finite-difference reference, a Lyapunov tool and a reliability report. Its documents already state the Lorenz-63 numbers and the sigma discrepancy that this repository reproduces. It also already contains a calibration of its report on several systems.
- **SciMLSensitivity** (Julia): `ForwardLSS`, `AdjointLSS`, `NILSS`, `NILSAS`, as reported by the planning notes. I did not open it or run it.
- **`fds`** (Python, GPL-3.0): reported by the planning notes. I did not open it or run it.
- **Lea, Allen and Haine 2000, Tellus A.** The text was read through a delegated reading. I re-checked the key numbers (0.96, 0.933, 1.121) in the source text. They show the gradient of a time average grows like exp(lambda tau) with lambda = 0.933 +- 0.004 for r from 25 to 30 and propose an ensemble adjoint. Phase 3 measured the same growth independently.
- **Eyink, Haine and Lea 2004, Nonlinearity.** Pages 1 to 19 were read through a delegated reading, and I re-checked the quoted 0.96 and 0.906. They give the ensemble adjoint and show heavy tails make the error shrink slower than the usual rate.
- **Lasagna et al., arXiv 1806.02077.** I read the abstract, sections 5.3 and 5.4 and the conclusion. It reports the rho bias described in section 5.
- **Wang, Hu and Blonigan 2014 (LSS, arXiv 1204.0159); Ni and Wang 2017 (NILSS, arXiv 1611.00880); Ni et al. (FD-NILSS, arXiv 1711.06633).** I read the parts needed to implement the methods, as logged.
- **Metz et al. 2021 (arXiv 2111.05803); Chandramoorthy, Magri and Wang 2020 (arXiv 2011.08794); Hückelheim et al. (arXiv 2305.07546); Mikhaeil et al. (arXiv 2110.07238).** I read the abstracts only. Metz et al. describe a chaos-based failure of gradients traced to the Jacobian spectrum. Chandramoorthy et al. use automatic differentiation to build a shadowing sensitivity for a time-delayed system. Hückelheim et al. list AD pitfalls, chaos among them. Mikhaeil et al. treat diverging gradients for chaotic recurrent networks.
- **Ni 2020, arXiv 2009.00595 (fast differentiation of hyperbolic chaos).** I read the abstract, sections 1.1 to 1.3, the hyperbolicity assumptions in section 2.1 and the discussion of non-hyperbolic systems. It splits the linear response of hyperbolic systems into a shadowing and an unstable contribution, and says the shadowing code for Lorenz-96 converges with a large systematic error from the missing unstable part. I did not read the proofs or the algorithm.
- **Ginelli et al., arXiv 1212.3961; Kuptsov and Parlitz, arXiv 1105.5228; Morriss, arXiv 1202.1571.** I read the algorithm sections of the first (equations checked on a page image), the hyperbolicity-test passage of the second and the tangency section of the third, which concerns a hard-disk system only.
- **Truncation and warm-up.** The planning notes say they are standard practice. I did not trace specific papers.
- **`lyapax`** (JAX): reported by the planning notes to warn about gradients through chaotic trajectories. I did not open it.

## 9. What is and is not new here

Claimed: a small, reproducible, independent comparison.
- An independent finite-difference oracle for three Lorenz-63 parameters, with step, window, dt, spin-up and noise checks.
- A replication of the `nilss-jax` Lorenz-63 numbers, and its stability against its settings.
- Two further shadowing implementations (LSS and NILSS) written from papers, agreeing with it.
- A calibration of its reliability report on 13 new cases. Only 4 oracles converged.
- A measurement of how fast plain gradients, truncated gradients and LSS fail on Lorenz-63, with failures reported.

Not claimed as new: that gradients blow up, that shadowing fixes it, the sigma gap, the rho bias, or the existence of any of the three shadowing codes.

## 10. Limits

- One chaotic system with a published use as a benchmark (Lorenz-63), one parameter point (rho = 28). Other systems appear only in the A4 calibration, where most oracles did not converge.
- One observable (z). One integrator family (RK4).
- Three shadowing codes of one family. A shared effect of that family is not excluded. `fds` and the Julia code were not run.
- I do not know which of shadowing and finite differences is closer to the true derivative. The hyperbolicity test (A5b) did not support the hyperbolicity explanation at rho = 28. The missing-unstable-contribution explanation was not tested. The angles were sampled every 0.1 time units over about 2e5 samples, so short dips could be missed.
- The noise-regularised check rests on an untested assumption.
- The `nilss-jax` code is one author's code from October 2026. Findings are about one commit.
- Predictions in the audit tasks followed my reading of its documents. They are weak evidence.
- Not done: Phases 5 to 9 of the plan (fitting task, double pendulum for the fixes, Julia port, Lorenz-96 4D-Var), and A6, A7 and A7b of `TODO.md`. The double pendulum appears only in Phase 1.
- The finite-difference oracle needs many start points. Its standard errors scale like 1/h.

## 11. Reproduce

```bash
pip install -e ".[dev]"
pytest -q                                   # 67 passed, 1 xfailed
python experiments/phase2_oracle.py         # about 2 s
python experiments/phaseA1_oracle_v2.py run rho    # about 90 s per parameter, then `report`
```

Each task has its own script in `experiments/`, listed in its `docs/PREDICTIONS-*.md`. The `nilss-jax` runs need that package installed at the pinned commit. Large raw arrays are not committed. The `run` commands regenerate them.
Last full test run on the final tree: `67 passed, 1 xfailed in 66.20s`.
Software of the runs: Python 3.13, JAX 0.12.0.1, NumPy 2.5.3, SciPy 1.18.1.

## Appendix: where each headline number is

| Number | File |
|---|---|
| lambda_1 0.9064, 0.9018; pendulum 1.3626 | `results/phase1_measurements.json` |
| Phase 2 oracle 1.0029 +- 0.0021 | `results/phase2_oracle.json` |
| rho line slope 0.9888 | `results/phase2b_wide_slope.json` |
| Phase 3 slope 0.9057, T* = 8, T_leave 14 and 2 | `results/phase3.json` |
| Phase 4 table and costs | `results/phase4.json`, `results/phase4_cost.json` |
| Oracle v2 | `results/phaseA1_oracle_v2.json` |
| Window, dt, spin-up | `results/phaseA1b_windows.json` |
| `nilss-jax` run | `results/phaseA2_stage2.json`, `results/phaseA2_report.json` |
| Settings and LSS for sigma and beta | `results/phaseA3_settings.json`, `results/phaseA3_lss.json` |
| Noise-regularised check | `results/phaseA3b_noise.json` |
| Calibration | `results/phaseA4_calibration.json` and the per-case files |
| My NILSS | `results/phaseA5_own.json`, `results/phaseA5_report.json` |
| Hyperbolicity test | `results/phaseA5b_angles.json`, `results/phaseA5b_link.json`, `results/phaseA5b_scan.json` |
