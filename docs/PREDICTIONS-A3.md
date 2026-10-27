# Task A3 predictions and gates: how much do the settings matter, and a second opinion from our own LSS

Written and committed before any A3 run. Gates follow `TODO.md` (task A3) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Pin and disclosure

nilss-jax 0.1.0, commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7`, as in A2. Every results file records it. I have read only its README, CHANGELOG and `docs/usage.md`, and no source. I have not read `docs/reliability.md` (that is for A4).
Numbers I have seen: the same as in `docs/PREDICTIONS-A2.md`. The README says the sigma gap is "stable against `T`, `nus`, `dt` and `T_seg`", and `docs/usage.md` says the values are "the same to the last digit" for dt = 0.01, 0.005, 0.0025, for T_seg = 0.25 to 1 and for nus = 1, 2. My predictions below follow those statements, so they are not independent of them.
From A2 (my own runs): nilss-jax at dt = 0.005, T_seg = 0.5, nus = 1, T = 1000, 32 orbits gives rho 1.01708 +- 0.00007, sigma 0.13375 +- 0.00008, beta -1.65927 +- 0.00116. From A1/A1b: the finite-difference oracle gives 1.0032 +- 0.0007, 0.1488 +- 0.0007, -1.658 +- 0.006, stable against window, dt and spin-up for rho and sigma. From Phase 4: my own LSS gives rho 1.014 to 1.016 (T = 40 to 80, rising with T).

## Part 1: vary the nilss-jax settings

Same 32 start points as A2 Stage 2 (`PRNGKey(70)`, box as in A1), `T_spinup = 20`. One setting changed at a time from the A2 baseline (dt 0.005, T_seg 0.5, nus 1, T 1000). The baseline is the A2 Stage 2 result (same settings, same start points; not rerun).

| Variant | dt | T_seg | nus | T |
|---|---|---|---|---|
| dt0.01 | 0.01 | 0.5 | 1 | 1000 |
| dt0.0025 | 0.0025 | 0.5 | 1 | 1000 |
| seg0.25 | 0.005 | 0.25 | 1 | 1000 |
| seg1.0 | 0.005 | 1.0 | 1 | 1000 |
| nus2 | 0.005 | 0.5 | 2 | 1000 |
| T250 | 0.005 | 0.5 | 1 | 250 |
| T4000 | 0.005 | 0.5 | 1 | 4000 |

Per variant: mean and SEM of rho, sigma, beta over the orbits it keeps (its own `ens.mean()` and `ens.sem()`), the leading exponent, mean z, how many orbits it keeps, and its report verdict.
Expected compute: about 5 minutes (A2 took 23 s for the baseline; dt0.0025 about 2x, T4000 about 4x, nus2 maybe 1.5x). One run at a time, in the background.

### Controls first (per variant, before reading any sensitivity)

C1 mean leading exponent within 0.02 of 0.9041. C2 mean z within 0.2 of 23.55. If one fails I stop and find out why.

### Gates (part 1)

- **A3.1 Stability against settings.** For every variant, |sigma - 0.13375| <= 0.004 (about a quarter of the gap to the oracle), |rho - 1.01708| <= 0.004, |beta - (-1.65927)| <= 0.02. The thresholds are fixed now. Differences are also reported in SEM units.
- **A3.2 The sigma gap stays.** For every variant the sigma mean differs from the oracle 0.1488 by more than 2 combined standard errors (oracle SE 0.0007). The README says the gap is stable. This tests it.

## Part 2: our own LSS for rho, sigma and beta

The Phase 4 LSS (`chaos_jax.fixes.lss_derivative`, trapezoid on RK4 samples, block Thomas solve). The parameter derivative of the vector field comes from `jax.jacfwd`. dt = 0.01, alpha = 10 (the Phase 4 value).
Start points: 200 from the A1 box, key `PRNGKey(90)`, 50 time units of spin-up at the reference parameters. Horizons T = 40, 80, 160, 320 (LSS in Phase 4 rose with T from 1.004 at T = 10 to 1.016 at T = 80, so I add longer ones). For each parameter and T: mean over start points with SEM.
Also: alpha in {3, 30} at T = 160, to see how much alpha matters for sigma and beta.
Expected compute: about 3 minutes.

### Controls first (part 2)

- **C3** dense KKT test of the LSS code for sigma and beta, built from the analytic d f / d sigma = (y - x, 0, 0) and d f / d beta = (0, 0, -z). This is a new test, `tests/test_fixes_1_lss_solver.py`. It must pass.
- **C4** my LSS for rho at T = 80 gives a mean within 0.005 of 1.0158 (the Phase 4 median). If not, I stop.

### Gates (part 2)

- **A3.3 Second opinion.** At T = 320, for each parameter, my LSS mean and the A2 nilss-jax mean agree within 2 combined SE, or I report the relative difference. A third number is also given: the relative difference to the oracle.
- **A3.4 Convergence in T.** For each parameter, the LSS means at T = 160 and T = 320 agree within 2 combined SE. If not, I say the LSS has not converged in T.

## Predictions (not gates)

- **P-A3.1** all variants pass A3.1 and A3.2. The README claims it, and I expect the claim to hold for dt, T_seg and nus at T = 1000. T = 250 is the least certain: Stage 1 (T = 200, 8 orbits) gave sigma 0.13306, 0.0007 below the baseline, inside the threshold.
- **P-A3.2** the nus2 variant keeps all 32 orbits and gives values within 0.001 of the baseline.
- **P-A3.3 LSS, rho:** 1.017 +- 0.002 at T = 320 (agrees with nilss-jax within 2 SE).
- **P-A3.4 LSS, sigma:** **0.135** (range 0.125 to 0.145), so it sits near the nilss-jax value 0.134 and below the oracle 0.149. Reason: for rho my LSS and nilss-jax agree to 0.001, so I expect two shadowing codes to agree for sigma too. This is a guess. The other outcome, my LSS near 0.149, would mean the sigma gap is specific to nilss-jax.
- **P-A3.5 LSS, beta:** -1.66 +- 0.01.
- **P-A3.6 convergence:** the LSS values at T = 160 and 320 agree within 2 SE for rho and beta. For sigma I am unsure.
- **P-A3.7 alpha:** changing alpha from 3 to 30 moves sigma by under 2%.
- **P-A3.8 compute:** about 8 minutes in all.
