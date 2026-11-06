# H1 predictions and gates: the evaluator

Written and committed before any H1 code ran. Gates follow `PLAN.md` (stage H1) and are made exact here.
Do not edit a gate or prediction after results. Add a dated Amendment instead.

## Disclosure: what I had seen before predicting

- The published exponents in `LITERATURE.md` (Sprott 1994 Table I, Chlouverakis and Sprott 2006, Li and Sprott 2014, Kopp 2025). Predictions for those systems are **not independent** of them. The Sprott table was read from a scanned image by the H0 session, so a misread digit is possible.
- chaos-jax measurements for Lorenz-63: leading exponent 0.9064 and 0.9018 in Phase 1, 0.9040 in A5. The literature value 0.9056 is from memory.
- The Rössler 3D leading exponent (about 0.07), the Rössler 1979 exponents (about 0.11 and 0.02) and the Lorenz periodic window at rho = 160 are **from memory**. No published exponent for Rössler 1979 was found in what the H0 session read.
- I have not run any H1 code and have not read `lyapax` source. For the cross-check I will install a pinned `lyapax` and use it as a black box. I will read its documentation and docstrings only, and log that in `HANDOFF.md`.

## What H1 builds

`src/hyperchaos/` gets: (1) a fixed-step RK4 integrator in JAX (float64) that carries the state and an orthonormal tangent frame; (2) the Benettin method (QR re-orthonormalisation, sum of the log of the diagonal of R) giving the full spectrum, a standard error for each exponent from block means, and the time average of the divergence `tr J` along the same trajectory; (3) a classifier; (4) four verification checks; (5) a library of control systems; (6) tests. Systems are plain JAX functions `f(x) -> dx/dt`. Polynomial term bookkeeping and `vmap` batching over candidates are H3, not H1.

## Method (fixed now)

- Float64 everywhere. Fixed-step RK4 with `dt = 0.01` unless a system's table row says otherwise. The tangent frame uses the exact Jacobian from `jax.jacfwd` inside every RK4 stage.
- Initial condition: random in a box fixed per system, then `T_spin = 200` time units discarded.
- Spectrum run: `T = 5000` time units after spin-up, QR every 10 steps, 10 blocks of 500. Exponent = mean over blocks of the block exponent. Standard error = sample standard deviation of the block values divided by sqrt(10).
- Exponents are sorted from largest to smallest.
- **Exact bookkeeping identity:** the sum of the exponents must equal the time average of `tr J` along the trajectory. I use it as a control on every system, since it tests the QR bookkeeping and holds for any system, chaotic or not.

## Classification (fixed now)

Order of tests:
1. **diverges:** any `|x|` above 1e8 or any non-finite value, during spin-up or the run.
2. **fixed point:** over the last 500 time units of the run the range of every coordinate is below 1e-6 times (1 + the largest `|x|`).
3. Otherwise compute the spectrum. The exponent with the smallest absolute value is the **flow direction**; call its value `z_err` (it should be 0).
4. `eps = max(0.005, 3 * max SE, 3 * |z_err|)`. An exponent is **positive** if it exceeds `eps`, **negative** if it is below `-eps`, **zero** otherwise. The flow-direction exponent counts as zero.
5. Labels: **periodic** (no positive exponent, one zero), **quasi-periodic** (no positive, two or more zeros), **chaotic** (one positive), **hyperchaotic** (two or more positive). Anything else, for example no zero exponent on a non-fixed bounded orbit, is labelled **inconsistent** and reported, never hidden.
6. **conservative flag:** if `|sum of exponents| < 0.005` the system is flagged conservative. A conservative system never counts as a catalogue candidate.

## Verification checks (a chaotic or hyperchaotic label is "verified" only if all four pass)

- **V1 second integrator:** scipy `solve_ivp` with DOP853, rtol 1e-10, atol 1e-12, integrating the state and the tangent frame together with QR every 1 time unit, over `T = 300` after the same spin-up, from the same start. Every exponent within `3 * sqrt(SE_a^2 + SE_b^2)` of the RK4 run over the same 300 time units. Same class label.
- **V2 half step:** RK4 with `dt / 2`. Every exponent within `max(0.002, 3 * combined SE)` of the main run. Same label.
- **V3 converged:** the first and second half of the main run (5 blocks each). Every exponent within 3 combined SE.
- **V4 start points:** four independent random start points. Same label for all four, and every exponent within 3 combined SE of the pooled mean.

## Controls

A. **Positive controls** (expected label in brackets; published exponents from `LITERATURE.md` section A, C, D):

| Id | System | Expected |
|---|---|---|
| L | Lorenz-63 (10, 28, 8/3) | chaotic, exponents (0.906, 0, -14.573) |
| R76 | Rössler 1976 (a 0.2, b 0.2, c 5.7) | chaotic |
| SB, SC, SF, SS | Sprott 1994 cases B, C, F, S | chaotic, published exponents from Table I |
| R79 | Rössler 1979 (a 0.25, b 3, c 0.5, d 0.05) | hyperchaotic |
| LS | Li and Sprott 2014 (a 2.6, b 0.44) | hyperchaotic, published (0.0704, 0.0128, 0, -1.0832) |
| CS13 | Chlouverakis and Sprott Eq. 13 (A 3.6) | hyperchaotic, published (0.132, 0.035, 0, -1.25) |
| CS12 | Chlouverakis and Sprott Eq. 12 | hyperchaotic, published (0.141, 0.0168, 0, -0.257) |
| KP | Kopp 2025 (a 0.62, b 0.55) | hyperchaotic (published third exponent 0.0076) |

B. **Negative controls** (must never be labelled chaotic or hyperchaotic):

| Id | System | Expected |
|---|---|---|
| N-fix | damped oscillator `x' = y, y' = -x - 0.5 y` | fixed point |
| N-div | `x' = x` | diverges |
| N-lor14 | Lorenz-63 at rho = 14 | fixed point |
| N-per | Hopf normal form `r' = r(1 - r^2)` with `theta' = 1`, written in x, y | periodic |
| N-vdp | Van der Pol, mu = 1 | periodic |
| N-per4 | the Hopf system plus two decaying coordinates (4D) | periodic |
| N-qp | two decoupled Hopf systems with frequencies 1 and sqrt(2) (4D) | quasi-periodic |
| N-lor160 | Lorenz-63 at rho = 160 | periodic (from memory) |
| N-cons | Sprott 1994 case A (conservative) | not a catalogue candidate (conservative flag) |

## Gates

- **G1 bookkeeping:** for every control that has a spectrum, `|sum of exponents - mean tr J| <= 1e-3 * max(1, |mean tr J|)`.
- **G2 Lorenz:** leading exponent within 0.02 of 0.906, `|second exponent| < 0.01`, the third within 0.02 of `-13.6667 - first`.
- **G3 labels:** every control gets its expected label (N-cons: conservative flag set; N-lor160: if it is not periodic, report it as a wrong memory, not a failure of the evaluator, and then it does not count for G3).
- **G4 published values:** SB, SC, SF, SS, LS, CS12, CS13: each exponent within `max(0.01, 0.02 * |published|)` of the published value. Exponents the paper prints as 0 are checked as the flow direction (`|z_err| < 0.005`).
- **G5 verification:** every positive control passes V1 to V4, except those named in the predictions below as expected to fail; every negative control either fails verification or is not labelled chaotic (both count as a pass for this gate).
- **G6 planted bugs:** each of the three bugs below makes at least one test fail.
- **G7 cross-check:** a pinned `lyapax` run on L, LS and R79 agrees with the main run: the two largest exponents within 0.01, the others within 5% or 0.02. If `lyapax` does not install or run, I report that, the gate is "not met", and no replacement is claimed.

## Break-it (after committing, before results are reported)

Plant each bug in the working tree, run the test suite, confirm a test fails, restore from git.
1. Drop the QR re-orthonormalisation (the frame is only normalised). Expect G1 or G2 to fail.
2. Use the transposed Jacobian in the tangent equation. Expect G2 or G4 to fail.
3. Classifier: count an exponent as positive if it is above 0 (ignore `eps`). Expect G3 to fail (Lorenz second exponent, or the periodic controls).

## Predictions (not gates)

- **P-H1.1 Lorenz:** leading exponent 0.905 (range 0.895 to 0.915), SE about 0.003; `|sum - mean tr J|` below 1e-5; all of V1 to V4 pass.
- **P-H1.2 Rössler 1976:** chaotic, leading exponent between 0.06 and 0.08 (from memory).
- **P-H1.3 Sprott cases B, C, F, S** pass G4 (80%). If one fails I will check the scanned digit against the paper before blaming the evaluator.
- **P-H1.4 Rössler 1979:** hyperchaotic, leading exponent 0.11 (range 0.05 to 0.2), second 0.02 (range 0.005 to 0.06). Not independent: from memory.
- **P-H1.5 Li and Sprott 2014:** second exponent 0.0128 +- 0.005. Chance (75%) that `eps` lets it count as positive, so the label is hyperchaotic. With `eps = 0.005` and a small SE this works; if `3 * |z_err|` or `3 * SE` exceeds 0.0128 it is labelled chaotic. G3 requires "hyperchaotic" for LS, so a "chaotic" label fails G3 and I report it as a failure. I do not tune `eps` afterwards.
- **P-H1.6 CS13:** hyperchaotic, exponents within the G4 tolerance (80%). The hyperjerk has a degree-5 term; the integrator needs `dt = 0.01`, which should be enough.
- **P-H1.7 Kopp 2025:** hyperchaotic with our zero-flow exponent below 0.002 in size (the paper's 0.0076 is probably a short-run error); second exponent about 0.02 (range 0.01 to 0.04).
- **P-H1.8 Sprott A (conservative):** the conservative flag is set (95%); it also fails V3 or V4 (60%) because its phase space is mixed.
- **P-H1.9 Lorenz rho = 14** is a fixed point (95%). **rho = 160 periodic** (60%, from memory).
- **P-H1.10** all three planted bugs are caught (85%). A bug nothing catches is reported and the test is added, as in chaos-jax A5.
- **P-H1.11 compute:** the full test suite takes under 10 minutes of CPU on this machine, V1 (scipy) being the slowest part. Wall time is not a result: other agents were already using about 59% of the CPU when I wrote this.
- **P-H1.12 `lyapax`:** it installs under Python 3.14 and runs on L, LS and R79 (60%).
- **P-H1.13** at least one gate or prediction misses at the first run (90%).

## Out of scope for H1

Batched evaluation of many candidate systems, term counting, canonical forms, any search. Those are H2 to H4.

## Amendment, 2026-10-26 (before any H1 run): two clauses made exact

- "No zero exponent on a non-fixed bounded orbit" is implemented as: the exponent with the smallest absolute value has `|z_err| > 0.05`. The classifier forces that exponent to be the flow direction, so without this rule the label "inconsistent" could never occur.
- V4 uses `T = 2000` per start point (10 blocks of 200) to save time. V1 to V3 use the times stated above.

## Amendment 2, 2026-10-26 (after the first run; gates are NOT changed)

First run under the rules above (`results/h1_run1_summary.txt`): 11 tests failed, 56 passed. Diagnosis, from short probe runs (`results/` keeps only the summary, the probes were in the session scratchpad):

1. **The `eps` rule was badly built, my mistake.** It used `3 * max SE` over all exponents. The large negative exponent of a stiff system has a large standard error for a trivial reason (R79: 0.70, CS13: 0.020), so `eps` came out at 2.09 for R79 and 0.06 for CS13, and everything fell into the "zero" band. R79 was labelled quasi-periodic; LS, CS12, CS13 chaotic instead of hyperchaotic. **Rule v2** (used from now on): exponent `i` is positive if `lambda_i > max(0.005, 3 * SE_i, 3 * |z_err|)`, negative if below the negative of that, zero otherwise. Rule v1 stays in the code (`rule="v1"`) so the first run can be reproduced. The G3, G5 and V4 results below are reported under both rules, v1 as pre-registered and v2 as amended after seeing the failure.
2. **G1 failed for R79 because `dt = 0.01` is too coarse for it** (the stiff exponent is about -25). `sum - mean tr J` was 0.043, 0.0033, 6e-5 and 0.0 at `dt` = 0.01, 0.005, 0.002, 0.001. The R79 control now uses `dt = 0.002`. The gate (G1) is unchanged and applies to the new `dt`.
3. **G4 for CS12 and CS13** failed as written (the large negative exponent is off by 0.012 and 0.037, tolerance 0.01 and 0.025). The standard errors of those exponents are 0.008 and 0.020, so the differences are within 2 SE of our own estimate. The gate is not changed. A longer run (`T = 20000`) is added as an extra, reported separately, to see if the differences shrink.
4. **KP (Kopp 2025):** the probes found two exponents at zero (about 0.0003 and -0.0001), not 0.0201 and 0.0076 as published, for the paper's own initial condition (1, 1, 1, 1) and for six nearby start points. That is a result about the system as I transcribed it (the sum of exponents matches the published sum), not a reason to change a gate. G3 for KP stays "expected hyperchaotic" and is reported as failed if the label is chaotic. P-H1.7 missed.

## Amendment 3, 2026-10-26 (after the second run; gates are NOT changed)

Second run, rule v2 and `dt = 0.002` for R79, `T = 5000` (as pre-registered): 6 failed, 61 passed. The G1 and G3 failures of the first run are gone except G3 for KP. Diagnosis of what is left:

1. **G5 fails at `T = 5000` because the second exponent of LS and CS12 (about 0.013 to 0.02) is only 3 to 4 standard errors above zero at that run length.** Labels flip between runs at the threshold (LS: one of four start points is labelled chaotic; CS12: the half-step run and one start point are labelled chaotic). That is the expected behaviour of a check that works, and it shows `T = 5000` is too short for these systems. The run itself costs about 0.5 s per 5000 time units, much less than I assumed when I fixed `T`. **Amended setting "long": `T = 100000` (10 blocks of 10000), V4 at `T = 40000`.** The `T = 5000` results stay in the report as the pre-registered configuration; both are reported.
2. **V1 no longer requires the label to agree** (it still requires every exponent to agree within 3 combined SE). With `T = 300` the standard error of a second exponent near 0.015 is about 0.01, so the label is unresolvable there (LS: DOP853 0.0116 against RK4 0.035). V1 now verifies that the exponents do not depend on the integrator; it says nothing about the label. The other three checks keep the label requirement.
3. **R79 start points:** one of the four V4 start points (box width 1.0) diverged, so the basin of the Rössler 1979 attractor is small. The control's start box is narrowed to width 0.2 around (-10, -6, 0, 10). I record that this was chosen after seeing a divergence. The width-1.0 outcome (one of four diverge) is a result about the system and stays in this file.
4. **KP:** no change; reported as it is.

## Results (added after the runs, 2026-10-26)

Commands (all with `PYTHONPATH=src` and the project venv: Python 3.14.6 on Windows, jax and jaxlib 0.12.0.1, numpy 2.5.4, scipy 1.18.1, **lyapax 0.3.0**, pytest 9.1.1):
- pre-registered configuration: `python -m pytest -q tests` (`T = 5000`): first run 11 failed, 56 passed (`results/h1_run1_summary.txt`, rule v1, `dt = 0.01`); second run after Amendment 2 (rule v2, R79 `dt = 0.002`): 6 failed, 61 passed.
- amended "long" configuration: `H1_T=100000 python -m pytest -q tests`: **1 failed, 66 passed in 710.7 s** (the failure is G3 for KP). The lyapax test was added after that run started and was run on its own with `H1_T=100000`: 3 passed in 84 s.
- numbers: `python experiments/h1_report.py results/h1_report_short.json` (`H1_T=5000`) and `results/h1_report_long.json` (`H1_T=100000`).
- planted bugs: `python experiments/h1_breakit.py`: `results/h1_breakit_run1.txt` and `results/h1_breakit_run2.txt`.
- Wall times are not results: other jobs used the CPU (about 59% load when I started).

### The spectra (rule v2; long run `T = 100000`; SE from 10 blocks)

| System | Label | Exponents (SE in brackets) | Published | Verified V1 to V4 |
|---|---|---|---|---|
| L | chaotic | 0.9059, 0.0000, -14.572 (0.0004) | (0.906, 0, -14.573) | yes |
| R76 | chaotic | 0.0711, 0.0000, -5.394 (0.0005) | about 0.07 (memory) | yes |
| SB | chaotic | 0.2108, 0, -1.2108 (0.0004) | (0.210, 0, -1.210) | yes |
| SC | chaotic | 0.1624, 0, -1.1624 (0.0009) | (0.163, 0, -1.163) | yes |
| SF | chaotic | 0.1166, 0, -0.6166 (0.0005) | (0.117, 0, -0.617) | yes |
| SS | chaotic | 0.1870, 0, -1.187 (0.0007) | (0.188, 0, -1.188) | yes |
| R79 | hyperchaotic | 0.1126, 0.0207, 0.0000, -25.10 (0.0008, 0.0005, 0.0000, 0.20) | none found | yes |
| LS | hyperchaotic | 0.0708, 0.0128, -0.0001, -1.0835 (0.0012, 0.0009, 0.0001, 0.001) | (0.0704, 0.0128, 0, -1.0832) | yes |
| CS13 | hyperchaotic | 0.1319, 0.0343, 0.0000, -1.2575 (0.001, 0.0012, 0.0001, 0.0056) | (0.132, 0.035, 0, -1.25) | yes |
| CS12 | hyperchaotic | 0.1427, 0.0187, -0.0001, -0.2614 (0.0027, 0.0014, 0.0002, 0.0017) | (0.141, 0.0168, 0, -0.257) | yes |
| **KP** | **chaotic** | 0.1471, 0.0001, 0.0000, -0.7672 (0.0003, 0.0001, 0.0001, 0.0003) | (0.1385, 0.0201, 0.0076, -0.7864) | yes, as chaotic |
| N-fix, N-lor14 | fixed point | all negative | | n/a |
| N-div | diverges | | | n/a |
| N-per, N-vdp, N-per4, N-lor160 | periodic | one zero, the rest negative | | n/a |
| N-qp | quasi-periodic | (0, 0, -2, -2) | | n/a |
| N-cons (Sprott A, start near 0.05) | quasi-periodic, conservative flag | (0.0001, 0, -0.0001) | | n/a |

At `T = 5000` the same systems give the same picture with larger standard errors (`results/h1_report_short.json`). LS has second exponent 0.0175 +- 0.0039 there, and V4 and G5 fail for LS and CS12 because the label flips between runs.

### Gates

| Gate | `T = 5000` (pre-registered) | `T = 100000` (amended) |
|---|---|---|
| G1 sum equals mean divergence | pass for every control (R79 only with `dt = 0.002`; at `dt = 0.01` it failed: error 0.043) | pass (largest: L 1.0e-4, R79 4.9e-5, N-lor160 9.5e-4) |
| G2 Lorenz | pass | pass (0.9059) |
| G3 labels | **fail** under rule v1 (R79 quasi-periodic; LS, CS12, CS13 chaotic; KP chaotic). Under rule v2: **fail for KP only** | **fail for KP only** under rule v2. Under rule v1 also R79 |
| G4 published values | **fail for CS13 (fourth exponent off by -0.037) and CS12 (-0.012)** | pass (largest differences 0.0075 and 0.0044) |
| G5 verification | **fail for LS and CS12** (R79 passes with its narrowed box) | pass for every positive control; no negative control is labelled chaotic |
| G6 planted bugs | bug 3 **not caught** on the first run; classifier unit tests added; then all three caught | |
| G7 lyapax | pass (3 of 3; the two largest exponents agree within 0.01, the others within the 5% or 0.02 tolerance) | |

**Overall: G3 is not met** (KP), so H1 is not fully passed. G1, G2, G6 and G7 are met. G4 and G5 are met only at the amended length `T = 100000`. G3 is met for every system except KP only under the amended classifier rule v2.

### Predictions against results

| Prediction | Result |
|---|---|
| P-H1.1 Lorenz 0.905, SE 0.003, sum error below 1e-5, V1 to V4 pass | 0.9046 +- 0.0025 at `T = 5000`; V1 to V4 pass. **Miss:** the sum error is 1.0e-4, from the RK4 step (`dt = 0.01`). |
| P-H1.2 R76 between 0.06 and 0.08 | Hit: 0.0695 and 0.0711. |
| P-H1.3 B, C, F, S pass G4 | Hit: all four within 0.003 at `T = 5000`, within 0.001 at `T = 100000`. The scanned Table I digits are confirmed for these four. |
| P-H1.4 R79 hyperchaotic, 0.11 and 0.02 | Hit on the values (0.1126, 0.0207; fourth exponent -25.1). **Miss on method:** it needed `dt = 0.002`, rule v2, and a narrower start box. |
| P-H1.5 LS second exponent 0.0128 +- 0.005; labelled hyperchaotic (75%) | Exponent hit (0.0175 at `T = 5000`; 0.0128 +- 0.0009 at `T = 100000`, equal to the published value). Label: hyperchaotic at `T = 5000` only under v2; **v1 gave chaotic.** |
| P-H1.6 CS13 hyperchaotic and within G4 (80%) | **Miss at `T = 5000`** (fourth exponent -1.287 against -1.25, SE 0.020). Hit at `T = 100000`. |
| P-H1.7 Kopp hyperchaotic, second exponent about 0.02, zero exponent below 0.002 | **Miss.** The second exponent is 0.0001 +- 0.0001; the system is chaotic with two zero exponents. Zero exponent below 0.002: hit. |
| P-H1.8 Sprott A: conservative flag (95%); fails V3 or V4 (60%) | Flag: hit. The orbit from the start near 0.05 is quasi-periodic (all exponents near 0), so V3 and V4 were not run: no information on the second part. |
| P-H1.9 rho = 14 fixed point; rho = 160 periodic (60%) | Both hit. The memory about rho = 160 was right. |
| P-H1.10 all three bugs caught (85%) | **Miss on the first run** (bug 3). Caught after adding `tests/test_h1_classifier.py`. |
| P-H1.11 suite under 10 minutes of CPU | Pre-registered `T = 5000`: 69 s wall. Amended `T = 100000`: 711 s wall (11.8 min, CPU shared, so not comparable). CPU time was not measured. |
| P-H1.12 lyapax installs and runs on Python 3.14 | Hit (lyapax 0.3.0). |
| P-H1.13 at least one miss at the first run | Hit: 11 failures. |

### What this shows, and what it does not

- **The evaluator reproduces published exponents** to about 0.001 for seven systems at `T = 100000`, including the small second exponent of Li and Sprott 2014 (0.0128 +- 0.0009 against the published 0.0128). That also checks the H0 session's reading of the scanned Sprott Table I for B, C, F, S.
- **Short runs are not enough for weak hyperchaos.** A second exponent near 0.013 to 0.02 needs about `T = 100000` for a stable label. `T = 5000` was a poor choice on my part; the runs are cheap (about 0.5 s per 5000 time units for a 3D or 4D system at `dt = 0.01`).
- **Kopp 2025 (a = 0.62, b = 0.55) is not hyperchaotic in our runs** (second exponent 0.0001, at `T = 100000`, and also from the paper's own start (1, 1, 1, 1) at `T = 5000`). Possible causes: an error in my transcription (the sum of the exponents does match the published sum), a different coexisting attractor, a short run in the paper (its published third exponent 0.0076 suggests so), or the published claim itself. I did not find which. Until that is settled, the H0 table should not count Kopp 2025 as a known 7-term hyperchaotic flow.
- **Not tested:** Yu 2023 and PHYSCON 2015 (other 7-term systems in the H0 table), start-point dependence of the weakly hyperchaotic systems beyond four start points, non-smooth systems other than Kopp, and any claim about candidates outside the controls.
- **Independence:** the cross-check used lyapax as a black box (public names and three docstrings read, no source). It uses the same RK4 and the same QR idea, so it checks my code but not the method. The scipy DOP853 check (V1) is a different integrator, but only at `T = 300`.

## Amendment 4 (2026-10-27, after the Kopp chase)

Found by the Kopp chase (`docs/PREDICTIONS-KOPP.md`): the orbit started at the origin drifts to infinity along x3 and `classify` labelled it "chaotic" with exponents (162, 0, 0, -98), because the "diverges" test only fires above |x| = 1e8. Fix: `classify` now returns "inconsistent" when the sum of the exponents differs from the mean divergence by more than the gate G1 tolerance (`1e-3 * max(1, |mean divergence|)`). No gate was changed and no earlier label changes: the full suite at `T = 5000` fails the same five tests as before (KP G3, CS13 G4, CS12 G4, LS G5, CS12 G5; 75 passed; `results/h1_run_after_a4.txt`). Two tests added (`tests/test_h1_classifier.py`: synthetic sum mismatch; Kopp origin start). Planted bug 4 (sum check disabled) is caught: `results/h1_breakit_run3.txt`, baseline green (48 passed), all four bugs caught, tree clean after restore.

## Amendment 5 (2026-10-28, before the run; gates G1-G4 are NOT changed): expanding orbits are not attractors

Found by the stiff-hit check (`docs/PREDICTIONS-H2-STIFF.md`): 39 of the 4,673 night-run hits and 5 of the 472 main-run hits have a positive exponent sum (mean divergence above +0.005). A bounded orbit of a flow whose phase volume expands on average cannot lie on an attractor of positive volume; the cases seen have part of the state frozen and the orbit on an invariant line, and the exponents measure its unstable transverse direction. Fix: `classify` returns label "expanding" when the sum of the exponents is above `CONSERVATIVE_SUM` (0.005), placed after the sum-consistency check of Amendment 4. It is not "chaotic", so `is_chaotic` and the search reject it. Gates for this amendment in `docs/PREDICTIONS-H2-AMEND5.md`.
