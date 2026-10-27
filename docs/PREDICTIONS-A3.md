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

## Results (added after the runs, 2026-10-27)

Pinned commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7`. `python -m pytest -q` gives `44 passed, 1 xfailed in 39.03s` (two new test cases: the dense KKT control now runs for sigma and beta too).
Commands: `python experiments/phaseA3_settings.py run | controls | report` (3.6 minutes), `python experiments/phaseA3_lss.py run | control | report` (2.1 minutes). Output in `results/phaseA3_*.json`.

### Controls (ran first)

- C1, C2 for every nilss-jax variant: exponent 0.9049 to 0.9061 (gate: within 0.02 of 0.9041), mean z 23.547 to 23.554 (gate: within 0.2 of 23.55). All pass. All 32 orbits kept in every variant.
- C3 dense KKT test of the LSS code for sigma and beta: pass (`tests/test_fixes_1_lss_solver.py`, 4 cases pass).
- C4 my LSS for rho at T = 80: 1.01521 (gate: within 0.005 of 1.0158). Pass.

### Part 1: nilss-jax settings (32 orbits each; mean +- SEM; baseline is the A2 Stage 2 result)

| Variant | rho | sigma | beta | verdict |
|---|---|---|---|---|
| baseline (dt 0.005, T_seg 0.5, nus 1, T 1000) | 1.01708 +- 0.00007 | 0.13375 +- 0.00008 | -1.65927 +- 0.00116 | caution |
| dt 0.01 | 1.01694 +- 0.00007 | 0.13365 +- 0.00008 | -1.65871 +- 0.00098 | caution |
| dt 0.0025 | 1.01697 +- 0.00008 | 0.13366 +- 0.00006 | -1.65828 +- 0.00109 | caution |
| T_seg 0.25 | 1.01714 +- 0.00007 | 0.13375 +- 0.00008 | -1.65936 +- 0.00116 | caution |
| T_seg 1.0 | 1.01710 +- 0.00007 | 0.13375 +- 0.00008 | -1.65934 +- 0.00116 | caution |
| nus 2 | 1.01705 +- 0.00006 | 0.13362 +- 0.00007 | -1.65912 +- 0.00107 | caution |
| T 250 | 1.01779 +- 0.00021 | 0.13377 +- 0.00020 | -1.66337 +- 0.00227 | caution |
| T 4000 | 1.01700 +- 0.00002 | 0.13370 +- 0.00004 | -1.65838 +- 0.00063 | caution |

- **A3.1 (all variants within 0.004, 0.004, 0.02 of the baseline): pass.** The largest shifts are 0.0007 for rho (T 250), 0.00013 for sigma (nus 2) and 0.004 for beta (T 250).
- **A3.2 (sigma stays more than 2 combined SE from the oracle 0.1488): pass in every variant.** The gap is 20 to 21 combined SE in all of them.

### Part 2: our own LSS (200 start points; mean +- SEM; alpha 10 unless stated)

| Case | rho | sigma | beta |
|---|---|---|---|
| T = 40 | 1.01363 +- 0.00042 | 0.13206 +- 0.00025 | -1.64804 +- 0.00290 |
| T = 80 | 1.01521 +- 0.00024 | 0.13327 +- 0.00014 | -1.65436 +- 0.00185 |
| T = 160 | 1.01604 +- 0.00012 | 0.13393 +- 0.00009 | -1.65546 +- 0.00127 |
| T = 320 | 1.01637 +- 0.00006 | 0.13429 +- 0.00006 | -1.65591 +- 0.00078 |
| T = 160, alpha 3 | 1.01681 +- 0.00013 | 0.13390 +- 0.00010 | -1.65377 +- 0.00144 |
| T = 160, alpha 30 | 1.01524 +- 0.00013 | 0.13427 +- 0.00009 | -1.65988 +- 0.00120 |

| T = 320 against | rho | sigma | beta |
|---|---|---|---|
| nilss-jax (A2) | -0.07% | +0.41% | +0.20% |
| A3.3 strict test (within 2 combined SE) | **fail** (limit 0.0002) | **fail** (limit 0.0002) | **fail** (limit 0.0028) |
| oracle (A1) | +1.32% | -9.76% | +0.13% |
| A3.4 (T = 160 and T = 320 agree within 2 SE) | **fail** | **fail** | pass |

- **A3.3:** the strict criterion fails for all three, because the standard errors on both sides are tiny. The relative differences to nilss-jax are all below 0.5%. As the gate says, I report them.
- **A3.4:** my LSS has not converged in T for rho and sigma. It still rises: rho 1.0136, 1.0152, 1.0160, 1.0164 and sigma 0.1321, 0.1333, 0.1339, 0.1343 for T = 40 to 320. The increments roughly halve each time T doubles. A rough extrapolation (a geometric series) gives about 1.0167 for rho and 0.1347 for sigma. This is exploratory and I did not test the form.

### Predictions against results

| Prediction | Result |
|---|---|
| P-A3.1 all variants pass A3.1 and A3.2 | Hit. |
| P-A3.2 nus 2 keeps 32 orbits, within 0.001 of baseline | Hit (shifts 0.00003, 0.00013, 0.0002). |
| P-A3.3 LSS rho 1.017 +- 0.002 at T = 320, within 2 SE of nilss-jax | Value hit (1.0164). **Miss** on 2 SE. |
| P-A3.4 LSS sigma 0.135 (range 0.125 to 0.145), near nilss-jax | Hit: 0.1343, 0.0007 below my point value. |
| P-A3.5 LSS beta -1.66 +- 0.01 | Hit (-1.6559). |
| P-A3.6 convergence for rho and beta | **Miss for rho** (not converged). Hit for beta. I said I was unsure for sigma: it also fails. |
| P-A3.7 alpha 3 to 30 moves sigma by under 2% | Hit (0.3%). |
| P-A3.8 compute about 8 minutes | About 6 minutes. Close. |

### What this shows, and what it does not

- **The nilss-jax result does not depend on its settings** over the ranges tried (dt 0.0025 to 0.01, T_seg 0.25 to 1, nus 1 and 2, T 250 to 4000). The README claim of stability holds here, for all three parameters. This includes the sigma gap: it stays at about 21 combined SE in every variant.
- **A second, independently written shadowing code gives the same answer.** My LSS (written from the LSS paper, before I looked at nilss-jax) gives rho 1.0164, sigma 0.1343, beta -1.6559 at T = 320. nilss-jax gives 1.0171, 0.1338, -1.6593. The differences are 0.07%, 0.4% and 0.2%.
- **The sigma gap is therefore not a fault specific to nilss-jax.** Both shadowing codes give sigma near 0.134 and the finite-difference oracle gives 0.149, about 10% apart. For rho the same holds at 1.4%. For beta all three agree.
- **I do not know which is closer to the true derivative.** The oracle passed the h-ladder (A1) and the window, dt and spin-up checks (A1b). The shadowing codes agree with each other and are stable against settings. That leaves a systematic difference between shadowing sensitivities and finite differences of the average on this system, for sigma and rho. I did not find the cause. Candidates, not tested: Lorenz-63 at rho = 28 is not uniformly hyperbolic, so the linear response assumed by shadowing may differ from the slope of the finite-difference averages. Another is a bias of the finite-difference average at the scale of h that no variant here changes. Another is something both shadowing codes share, for example the same formula for the derivative of the average.
- **My LSS and nilss-jax are not fully independent.** Both implement a shadowing least-squares problem for the same equations, and I read the nilss-jax README and docs before writing the A3 predictions. I did not read its source.
- **Limits:** one system, one parameter point, 32 orbits (nilss-jax) and 200 start points (my LSS). One factor changed at a time. The extrapolation in T is a guess.
