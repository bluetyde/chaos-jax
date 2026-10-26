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
