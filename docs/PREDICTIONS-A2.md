# Task A2 predictions and gates: run nilss-jax on Lorenz-63

Written and committed before nilss-jax was installed or run. Gates follow `TODO.md` (task A2) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Pin

nilss-jax is the package built from `https://github.com/BrianP2002/Guiding_Center_Optimizer`.
Pinned commit: **`ad5b635f5798d89f5d31afa4118d31b778ab86f7`** (2026-10-08, "Merge pull request #1: nilss-jax 0.1.0", 7 commits on `main`). Every A2 results file records this hash.
It is cloned outside this repository, in the session scratchpad. It is not vendored and it is not committed here.

## What I read, and what I have seen (disclosure)

Read, as `TODO.md` allows: `README.md`, `CHANGELOG.md`, `docs/usage.md`. Not read: anything in `src/`, `examples/`, `experiments/`, `legacy/`, `tests/`, and `docs/theory.md`, `docs/findings.md`, `docs/reliability.md`, `docs/hpc.md`. I did not run any of its code before this file was committed.

Numbers I have now seen, so my predictions below are **not independent** of them:
- README and `docs/usage.md`: on Lorenz-63 its d<z>/d(rho) is "near 1.02" (1.017 from 16 orbits of T = 1000), d<z>/d(sigma) = +0.133 to +0.134, d<z>/d(beta) = -1.658. The README says the sigma value is about 11% below finite differences (+0.150 +- 0.004), and says rho and beta are within 1.5% of finite differences. The leading exponent is "near 0.90" (0.89 to 0.90).
- `docs/usage.md` also gives its own finite differences over 256 copies of T = 1000: rho 1.004 +- 0.001 (h = 0.5), beta -1.647 +- 0.004 (h = 0.125), sigma 0.167, 0.154, 0.149, 0.144 (+- 0.004) for h = 1, 0.5, 0.25, 0.125.
- The A1 oracle (below) was computed **before** I saw any of the docs numbers, except the sigma figures relayed in `TODO.md`. A1's beta value (-1.658) is the same as the nilss-jax beta value to three digits. I did not tune anything to either.

The oracle values to compare against (from `results/phaseA1_oracle_v2.json`, A1):
- rho: 1.0032 +- 0.0007 (weighted mean of four converged rungs).
- sigma: 0.1488 +- 0.0007 (weighted mean of four converged rungs).
- beta: -1.658 +- 0.006 (smallest step; the ladder was **not converged** at the largest step, so I use the smallest step and say so).

## Method (fixed now)

Documented API only: `NILSS(lorenz63.rhs, lorenz63.J, params=("rho", "sigma", "beta"), dt=..., T_seg=..., nus=1)`, `run_ensemble`, `ens.summary()`, `ens.mean()`, `ens.sem()`, `ens.report()`. Parameters (sigma, rho, beta) = (10, 28, 8/3). Objective: z (its `lorenz63.J`, checked against the docs to be the mean of z).

- **Stage 1, replication of the documented example:** `dt = 0.005`, `T_seg = 0.5`, `nus = 1`, `T = 200`, `T_spinup = 20`, 8 orbits from `lorenz63.initial_condition(i)`, i = 0..7. This is the README quickstart. It checks that I can reproduce the README's stated rho, sigma and verdict. It is **not** used for the A2 gates.
- **Stage 2, the gate run:** same `dt`, `T_seg` and `nus`. `T = 1000`, `T_spinup = 20`, **32 orbits** from start points in my A1 box (x, y uniform in [-15, 15], z uniform in [5, 45], `PRNGKey(70)`). I chose them to differ from its own initial conditions. The mean and standard error come from its `ens.mean()` and `ens.sem()`, over the runs it keeps. I record how many it dropped.
- Cost: I time one orbit of T = 100 first. This probe prints only its wall time. Then I state the expected time for Stage 1 and Stage 2 before running them. The orbits go in the background, one stage at a time.

## Controls first (on the nilss-jax output, before reading any sensitivity)

- **C1:** the ensemble mean of its reported leading exponent is within 0.02 of 0.9041 (the Phase 1 value, with the G1.3 band). 
- **C2:** the ensemble mean of its reported long-time average of z is within 0.2 of 23.55 (the Phase 2 value at rho = 28; the spread of one 100-unit average over start points was 0.095).
- If C1 or C2 fails, I stop and find out why before looking at the sensitivities. These controls only check that the tool integrates the same system as my code.

## Gates

- **A2.1 Agreement per parameter.** |mean_nilss - oracle| <= 2 x sqrt(SEM_nilss^2 + SE_oracle^2), for rho, sigma and beta, using the Stage 2 ensemble and the oracle values above. I report the relative difference for each, pass or fail.
- **A2.2 The sigma claim.** Two parts, each reported: (a) the Stage 2 sigma mean is within 2 combined standard errors of +0.133 (reproduces the README claim); (b) the Stage 2 sigma mean differs from the oracle 0.1488 by more than 2 combined standard errors (confirms the claimed gap).
- Statistical errors on both sides are small. Differences of 1% to 2% can fail A2.1. That is intended: the gate is as stated in `TODO.md`.

## Predictions (not gates)

I made these knowing the numbers above.
- **P-A2.1 Stage 1 reproduces the README:** rho about 1.02 (range 0.98 to 1.06), sigma about +0.13 (range +0.10 to +0.17), the verdict is CAUTION.
- **P-A2.2 Stage 2 means:** rho **1.017** (range 1.005 to 1.03), sigma **+0.134** (range +0.12 to +0.145), beta **-1.658** (range -1.68 to -1.64).
- **P-A2.3 A2.1:** rho **fails** (the gap of about 0.014 is larger than twice the combined SE, about 0.008), sigma **fails** (gap about 0.015), beta **passes**.
- **P-A2.4 A2.2:** (a) holds, (b) holds.
- **P-A2.5 Controls:** C1 and C2 pass. Leading exponent about 0.90.
- **P-A2.6 Reliability report on Stage 2:** CAUTION. This is not a gate. A4 tests the report.
- **P-A2.7 Cost:** unknown. My guess is about 20 s per orbit of T = 1000, so about 11 minutes for 32 orbits on one core. I will replace this by a measured number after the probe, before the main runs.

## Amendment, 2026-10-26: timing probe (before any stage ran)

The probe (one orbit, T = 100, T_spinup = 20, after one compile call) took **0.08 s**. P-A2.7 (about 20 s per orbit of T = 1000) was a miss by a large factor. At this speed I expect Stage 1 in under 10 s and Stage 2 (32 orbits of T = 1000) in about 30 to 60 s. No gate or other prediction changes. Because it is cheap, A3 can use larger ensembles. A2 keeps the 32 orbits stated above.

## Results (added after the runs, 2026-10-27)

Pinned commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7`, nilss-jax 0.1.0, installed with `pip install --no-deps` from a clone outside this repo.
Commands: `python experiments/phaseA2_run.py probe | stage1 | stage2 | controls | report`. Output in `results/phaseA2_stage1.json`, `phaseA2_stage2.json`, `phaseA2_report.json` (each records the pin).
Compute: Stage 1 2 s, Stage 2 23 s, one core. `python -m pytest -q` gives `42 passed, 1 xfailed in 36.68s` (no new test in A2).

### Controls (run first, before reading any sensitivity)

| Control | Stage 1 | Stage 2 | Gate |
|---|---|---|---|
| C1 leading exponent, mean | 0.9038 | 0.9053 | within 0.02 of 0.9041: pass |
| C2 mean z | 23.552 | 23.552 | within 0.2 of 23.55: pass |

All orbits were kept (8 of 8, 32 of 32). The tool integrates the same system as my code.

### Sensitivities, Stage 2 (32 orbits, T = 1000), against the A1 oracle

| Parameter | nilss-jax mean +- SEM | oracle +- SE | difference | A2.1 (limit = 2 combined SE) |
|---|---|---|---|---|
| rho | **1.01708 +- 0.00007** | 1.00317 +- 0.00074 | +0.01391 (+1.39%) | **FAIL** (limit 0.00149) |
| sigma | **0.13375 +- 0.00008** | 0.14882 +- 0.00072 | -0.01507 (-10.13%) | **FAIL** (limit 0.00144) |
| beta | **-1.65927 +- 0.00116** | -1.65814 +- 0.00595 | -0.00113 (-0.07%) | pass (limit 0.01213) |

Stage 1 (README quickstart, 8 orbits, T = 200): rho 1.01682 +- 0.00043, sigma 0.13306 +- 0.00047, beta -1.65439 +- 0.00679. Verdict CAUTION, as the README says. Same pattern as Stage 2. Per-orbit spread in Stage 2: rho 0.0004, sigma 0.00044, beta 0.0066. Reliability verdict for Stage 2: CAUTION.

- **A2.2a** the sigma mean is within 2 combined standard errors of +0.133: **yes** (0.13375 against 0.133). The README claim is reproduced.
- **A2.2b** the sigma mean differs from the oracle by more than 2 combined standard errors: **yes** (10.1% below).

### Predictions against results

| Prediction | Result |
|---|---|
| P-A2.1 Stage 1: rho about 1.02, sigma about +0.13, verdict CAUTION | Hit (1.017, +0.133, CAUTION). |
| P-A2.2 Stage 2: rho 1.017, sigma +0.134, beta -1.658 | Hit (1.0171, 0.1338, -1.659). |
| P-A2.3 A2.1: rho fails, sigma fails, beta passes | Hit on all three. |
| P-A2.4 A2.2 (a) and (b) hold | Hit. |
| P-A2.5 Controls pass, exponent about 0.90 | Hit (0.905). |
| P-A2.6 Stage 2 verdict CAUTION | Hit. |
| P-A2.7 about 20 s per orbit, 11 minutes in all | **Miss**: 0.7 s per orbit, 23 s in all (already amended after the probe). |

I made these predictions after reading the numbers above (disclosed in the file). The hits are therefore weak evidence. They show that my reading was right. They do not test an independent guess.

### What this shows, and what it does not

- **The nilss-jax README numbers reproduce on this machine from independent start points.** rho 1.017, sigma 0.134, beta -1.659, exponent 0.905. The run-to-run spread is very small (0.0004 for rho). The statistical error of the tool is not the issue.
- **rho and sigma sit away from the finite-difference oracle by amounts far outside both error bars.** The gap is +1.4% for rho and -10% for sigma. beta agrees to 0.07%.
- **My own LSS (Phase 4) gives 1.014 to 1.016 for rho.** That is within about 0.001 of nilss-jax (1.017) and about 0.013 above the oracle. So two implementations written independently, one by me from the LSS paper and one by the nilss-jax author from the NILSS paper, agree with each other for rho, and both differ from finite differences. I did not run my LSS for sigma or beta (that is part of A3). I did not read nilss-jax source, so I cannot say how independent the two are in method. Both use shadowing and the same Lorenz equations.
- **What this does not tell me is which is right.** The disagreement is between shadowing estimators on one side and a finite-difference oracle on the other. The oracle converged in h for rho and sigma (A1). It has one averaging window (100) and one dt (0.01). The shadowing estimators have T and window effects of their own. I have not tested either side against a third method. The README's own finite differences give rho 1.004 and sigma 0.149 at the smallest h, which agree with my oracle.
- **Limits:** one system, one dt for the oracle, the oracle for beta not converged at the largest step.
