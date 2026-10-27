# Task A4 predictions and gates: calibrate the nilss-jax reliability report

Written and committed before any A4 ensemble or oracle ran. Gates follow `TODO.md` (task A4) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Pin and disclosure

nilss-jax 0.1.0, commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7`. Every results file records it.
I read, besides the README, CHANGELOG and `docs/usage.md`: **`docs/reliability.md`** (full). I read no source.
What that file tells me, so my predictions below are **not independent** of it:
- the checks and thresholds of the report (tail index below 2 is unreliable, below 4 is caution; error scaling below 0.2 is unreliable, below 0.35 is caution; at least 8 runs; resolved exponent lambda_1 T at least 20);
- the author's own calibration, with finite differences, on Lorenz-63 at rho = 28, 35, 40, on Lorenz-96 with 8 variables at F = 8, on Roessler at c = 5.7, on guiding-center seas and on synthetic samples;
- its stated blind spots: an `ok` verdict with a 1.5% bias for rho at 28, the sigma sensitivity 11% off with only `caution`, rare outlier orbits that the tail checks miss, and weak `nus` consistency check when the runs scatter widely.
Also known to me from A1 to A3b: for Lorenz-63 at rho = 28 the finite-difference oracle and the noise-regularised check agree with each other and not with shadowing (rho 1.003 against 1.017, sigma 0.149 against 0.134), and three lines of evidence favour the finite differences. A4 uses finite differences as the reference for the true error. That choice carries the assumption stated in A3b.

## What is new here, against the author's calibration

Cases not in the author's table (different rho, other F and size, other c), my own oracle written independently, a separate verdict for each parameter, four independent ensembles per case so the stability of the verdict is seen, and the verdict compared with the error of each ensemble directly.
Overlap with the author's table: Lorenz-63 at rho = 28 (the rho sensitivity).

## Cases (fixed now): 13

All use the nilss-jax API only (`NILSS`, `run_ensemble`, `ens.report()`), one differentiated parameter per case so each verdict belongs to one parameter. Own systems written with `jax.numpy` as in the docs.

| # | System | Parameter | Objective J | Other settings |
|---|---|---|---|---|
| 1 | Lorenz-63 rho = 28 | rho | z | dt 0.005, T_seg 0.5, nus 1, T 1000, spin-up 20 |
| 2 | Lorenz-63 rho = 45 | rho | z | same |
| 3 | Lorenz-63 rho = 50 | rho | z | same |
| 4 | Lorenz-63 rho = 60 | rho | z | same |
| 5 | Lorenz-63 rho = 28 | sigma | z | same |
| 6 | Lorenz-63 rho = 45 | sigma | z | same |
| 7 | Lorenz-63 rho = 28 | beta | z | same |
| 8 | Lorenz-96, N = 8, F = 6 | F | mean of u^2 | dt 0.01, T_seg 0.5, nus 2, T 500, spin-up 50 |
| 9 | Lorenz-96, N = 8, F = 10 | F | mean of u^2 | same |
| 10 | Lorenz-96, N = 16, F = 8 | F | mean of u^2 | nus 5, otherwise same |
| 11 | Roessler a = b = 0.2, c = 5.0 | c | z | dt 0.01, T_seg 1.0, nus 1, T 1000, spin-up 100 |
| 12 | Roessler c = 7.0 | c | z | same |
| 13 | Roessler c = 9.0 | c | z | same |

`nus` is the number of exponents above 0.02 in my own Benettin spectrum (Phase 1 code, T = 500): 2 for Lorenz-96 N = 8 at F = 6, 8, 10 and 5 for N = 16 at F = 8. For Roessler the leading exponent is 0.066 (c = 5.0), 0.079 (7.0), 0.100 (9.0). I dropped c = 8, where lambda_1 is about 0.003 (not chaotic), and c = 6, where it is 0.005.
The settings for each system are the ones that the docs use for it (Lorenz-63, Lorenz-96, Roessler). I did not tune them.

## Ensembles and verdicts

Per case: **4 independent ensembles of 16 orbits**. Start points are random and differ between ensembles. `ens.report()` gives the verdict; the mean is `ens.mean()` for the case's parameter. Secondary: for ensemble 0 of each case, a second ensemble from the same start points with `nus + 1`, and `ens.report(nus_check=...)`. Its verdict is reported as a second column. It is not used for the gates.

## Oracle for each case (fixed now)

Finite-difference ensemble average, my own code (`chaos_jax.response`), no nilss-jax code. Two steps h per case: rho 0.5 and 1.0; sigma 0.5 and 1.0; beta 0.125 and 0.25; Lorenz-96 F 0.25 and 0.5; Roessler c 0.1 and 0.2. Averaging window and spin-up: Lorenz-63 and Lorenz-96 100 and 50 time units, dt 0.01; Roessler 400 and 100, dt 0.01. Start points start from N = 4000 and double (to 64000 at most) until the bootstrap SE of the slope at the smaller h is below 5% of its absolute value.
The oracle value for a case is the slope at the smaller h. The case is **oracle-converged** if the two steps agree within 2 combined SE and the SE rule is met. The gates use only oracle-converged cases. Cases that are not are reported and marked.

## Error and the tabulation

Per ensemble: relative error = |mean - oracle| / |oracle|. Table: verdict (ok, caution, unreliable) against error bin (below 5%, 5% to 25%, above 25%).
Definitions: a **miss** is an `ok` verdict with error above 10%. A **false alarm** is an `unreliable` verdict with error below 5%. A **confident miss** is an `ok` verdict with error above 25%.

## Controls first

- **C-A4.1** the Lorenz-63 rho = 28 case (case 1) must give a leading exponent within 0.02 of 0.9041 and a mean z within 0.2 of 23.55 (as in A2). It checks the tool and my systems code on a known case.
- **C-A4.2** my Lorenz-96 and Roessler vector fields: the RK4 energy-like identity is not available, so check the **fixed points**. Lorenz-96: u_i = F for all i gives zero. Roessler: the fixed point from the standard formula gives zero. This is a unit test (`tests/test_systems_a4.py`).
- **C-A4.3** the oracle code on a known answer: for Lorenz-63 rho = 28 the oracle at h = 0.5 reproduces A1 (1.003 +- 0.002). If not, I stop.

## Gates

- **A4.1 No confident miss.** Among oracle-converged cases, no ensemble with verdict `ok` has error above 25%.
- **A4.2 The gate is tested.** At least 6 ensembles among oracle-converged cases have verdict `ok`. If fewer, A4.1 is reported as weakly tested, not as passed.
- Reported, not gated: the full table, the misses (ok and error above 10%), the false alarms, the verdict stability (how many of the four ensembles of a case agree), the second verdict with `nus_check`.

## Predictions (not gates)

Made with the author's table in mind.
- **P-A4.1 verdicts by case:** cases 1 and 7: `ok` in all four ensembles. Case 5 (sigma): `caution` in all four. Cases 2, 3, 6: `caution` in most ensembles (at least 3 of 4). Case 4 (rho = 60): `caution` or `unreliable`. Cases 8 to 13: `unreliable` in all four.
- **P-A4.2 errors:** case 1 about 1.4%; case 7 below 1%; case 5 about 10%; cases 2 to 4 between 2% and 20%; Lorenz-96 and Roessler ensembles between 15% and 60% (the heavy tails make a 16-orbit mean wild).
- **P-A4.3 A4.1:** passes. The count of `ok` ensembles is 8 to 12, so A4.2 is met.
- **P-A4.4 misses:** none (no `ok` with error above 10%).
- **P-A4.5 false alarms:** at most 3 among the 24 or so `unreliable` ensembles.
- **P-A4.6 stability:** the four ensembles of a case give the same verdict in at least 11 of 13 cases.
- **P-A4.7 oracle:** at least 10 of 13 cases are oracle-converged. The Roessler cases are the likeliest not to be, because the author notes that the response is not linear over its window there.
- **P-A4.8 compute:** NILSS about 20 minutes in all, oracles about 20 minutes. One case at a time, in the background.

## Results (added after the runs, 2026-10-27)

Pinned commit `ad5b635f5798d89f5d31afa4118d31b778ab86f7`. `python -m pytest -q` gives `50 passed, 1 xfailed in 52.01s` (three new tests: the Lorenz-96 and Roessler fields).
Commands: `python experiments/phaseA4_calibration.py oracle|nilss <case>` for cases 1 to 13 (1 minute for case 1, then 23 minutes for cases 2 to 13), `controls`, `report`. Output in `results/phaseA4_*.json`. NILSS used 52 ensembles of 16 orbits (plus 13 `nus + 1` ensembles) and every orbit was kept (16 of 16).

### Controls (ran first)

- **C-A4.1** case 1: leading exponent 0.9058 (within 0.02 of 0.9041), mean z 23.549 (within 0.2 of 23.55). Pass.
- **C-A4.2** fixed points of my Lorenz-96 and Roessler fields: pass (`tests/test_systems_a4.py`).
- **C-A4.3** oracle for case 1 at h = 0.5: 1.00197 +- 0.00195 against A1 1.003 +- 0.002. Pass.

### Table (relative error = |ensemble mean - oracle| / |oracle|, in %, for the four ensembles; "conv" = oracle-converged under the rule fixed in advance)

| Case | System, parameter | Oracle (smaller h) | conv | Verdicts (4 ensembles) | Relative errors, % | nus check |
|---|---|---|---|---|---|---|
| 1 | Lorenz-63 rho = 28, rho | +1.0020 | yes | ok x4 | 1.5, 1.5, 1.5, 1.5 | ok |
| 2 | rho = 45, rho | +0.9576 | no | unreliable x4 | 5.2, 1.1, 5.0, 2.4 | unreliable |
| 3 | rho = 50, rho | +0.9698 | no | unreliable x4 | 25.1, 9.6, 2.7, 0.5 | unreliable |
| 4 | rho = 60, rho | +0.9392 | no | unreliable x4 | 12.0, 2.9, 2.2, 7.7 | unreliable |
| 5 | rho = 28, sigma | +0.1526 | yes | caution x4 | 12.4, 12.3, 12.3, 12.4 | caution |
| 6 | rho = 45, sigma | +0.4013 | no | unreliable x4 | 60, 2824, 23, 25 | unreliable |
| 7 | rho = 28, beta | -1.6454 | yes | ok x4 | 0.9, 0.8, 0.8, 1.0 | ok |
| 8 | Lorenz-96 N = 8, F = 6 | +3.403 | no | unreliable x4 | 847, 524, 245, 14717 | unreliable |
| 9 | Lorenz-96 N = 8, F = 10 | +3.679 | no | unreliable x4 | 475, 5.5, 508, 214 | unreliable |
| 10 | Lorenz-96 N = 16, F = 8 | +3.459 | yes | unreliable x4 | 2946, 504, 741, 38 | unreliable |
| 11 | Roessler c = 5.0 | +0.0163 | no | unreliable x4 | 295, 304, 375, 1316 | unreliable |
| 12 | Roessler c = 7.0 | -0.0108 | no | unreliable x4 | 480, 66, 347, 6.3 | unreliable |
| 13 | Roessler c = 9.0 | -0.0285 | no | unreliable x4 | 409, 1391, 305, 186 | unreliable |

Oracle detail (`results/phaseA4_oracle_*.json`), pair of steps h: case 2 0.9576 and 0.9641 (difference 1.8 times its limit); case 3 0.9698 and 0.9659 (1.07 times); case 4 0.9392 and 0.9945 (11 times); case 6 0.4013 and 0.4057 (1.2 times); case 8 3.40 and 3.66 (6 times); case 9 3.68 and 3.62 (1.03 times); cases 11, 12, 13 differ by 74, 147 and 9 times the limit (Roessler c = 7.0 changes sign between the two steps; case 12 also missed the SE rule at 64000 start points).

Verdict by error bin, all ensembles (counts): ok: below 5% 8, 5% to 25% 0, above 25% 0. caution: 0, 4, 0. unreliable: 6, 9, 25.

### Gates

- **A4.1 (no confident miss, oracle-converged cases only): pass.** There are 0 ensembles with verdict `ok` and error above 25%. Also none above 10%, and the largest `ok` error is 1.5%.
- **A4.2 (the gate is tested, at least 6 `ok` ensembles in converged cases): met**, with 8. All 8 come from two cases of one system (Lorenz-63 at rho = 28, parameters rho and beta). A4.1 is a weak test.
- Only **4 of 13 cases are oracle-converged** (1, 5, 7, 10).
- Misses (ok and error above 10%) in converged cases: none. False alarms (unreliable and error below 5%) in converged cases: none.
- Verdict stability: the four ensembles of a case agree in **13 of 13** cases.

### Exploratory, not gated (post hoc)

For the marginal cases 2, 3, 6, 9 I measured the NILSS error against the range spanned by the two oracle values. Case 2: ensemble means 1.008, 0.969, 1.006, 0.980, which lie 4.6%, 0.5%, 4.3% and 1.7% outside the range 0.958 to 0.964. All four are `unreliable`. If I accepted that oracle, three of the four ensembles would count as false alarms (error below 5%). I do not score them, because the oracle did not converge in h (the difference is 3.6 SE). Case 3: errors 25%, 9.6%, 2.7%, 0.1%, so the large spread between ensembles fits the verdict. Cases 6 and 9: errors in the hundreds or thousands of percent against the range, so `unreliable` fits.

### Predictions against results

| Prediction | Result |
|---|---|
| P-A4.1 cases 1, 7 ok; case 5 caution; cases 8 to 13 unreliable | Hit. |
| P-A4.1 cases 2, 3, 6 caution in at least 3 of 4; case 4 caution or unreliable | **Miss for cases 2, 3, 6**: unreliable in all four ensembles. Case 4 hit. |
| P-A4.2 errors: case 1 about 1.4%, case 7 under 1%, case 5 about 10% | Hit (1.5%, 0.8% to 1.0%, 12.3%). Case 7 is just above 1% in one ensemble. |
| P-A4.2 cases 2 to 4 between 2% and 20% | Mostly (0.5% to 25%, but the oracle there is not converged). |
| P-A4.2 Lorenz-96 and Roessler errors 15% to 60% | **Miss.** 5.5% to 14717%. The 16-orbit means are far wilder than I expected. |
| P-A4.3 A4.1 passes and 8 to 12 `ok` ensembles | Hit (8, the bottom of my range). |
| P-A4.4 no misses | Hit. |
| P-A4.5 at most 3 false alarms | Hit (0 in converged cases). |
| P-A4.6 same verdict across the four ensembles in at least 11 of 13 cases | Hit (13). |
| P-A4.7 at least 10 of 13 oracle-converged | **Miss, large**: 4 of 13. |
| P-A4.8 about 40 minutes | About 24 minutes. |

### What this shows, and what it does not

- **On the four oracle-converged cases the report behaves as hoped.** `ok` came with errors of 0.8% to 1.5%, `caution` with 12.3%, `unreliable` with 38% to 2946%. It never said `ok` when the error was large. In 13 of 13 cases, four independent ensembles agreed on the verdict.
- **This is a weak test of misses.** The only `ok` ensembles are two parameters of one system at one parameter value. Its documented blind spot (a 1.5% bias that is invisible to the report) shows here: the `ok` verdict for rho at 28 comes with 1.5% error, in the same direction in all four ensembles.
- **Nine of 13 finite-difference oracles did not converge in h.** Seven of those because the response of the average to the parameter is rough or nonlinear at the step sizes I could afford (all of Rossler, Lorenz-63 at rho = 60, Lorenz-96 F = 6) and two marginally. For those cases I have no clean "true error". The calibration of the report against the true error is therefore only as good as the four converged cases.
- **The report is conservative at rho = 45 to 60** (my exploratory reading). NILSS means are within a few percent of the finite-difference range at rho = 45 while the verdict is `unreliable`. I did not score this, because the oracle did not pass its convergence rule.
- **The `nus` check agreed with the main verdict in all 13 cases.** It adds no information here.
- **Limits:** 13 cases, three systems, one parameter each, one objective each, 16 orbits per ensemble, one T per system. The oracle is a finite difference and carries the assumption of A3b. Settings per system are the ones the docs use and I did not vary them.
