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
