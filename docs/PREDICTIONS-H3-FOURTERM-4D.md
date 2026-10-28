# H3 start: four-term search in 4D, predictions and gates (written 2026-10-28, before any code or run)

## Question
Is there a 4D quadratic flow with exactly 4 nonzero terms that is numerically chaotic, or hyperchaotic? In 4D every equation needs a term (otherwise its variable is constant), so 4 terms means exactly one monomial per equation. The smallest published 4D hyperchaotic flows have 7 terms (`LITERATURE.md` section A, with the caveats written there).

## Space
Monomials per equation: 15 (the constant, 4 linear, 10 quadratic). Term-sets: 15^4 = 50,625 (rank 4: 41,273; rank 3: 8,869; rank 2: 478; rank 1: 5, where rank is that of the 4 x 5 matrix of the scaling action: time plus 4 variables). Coefficients are reduced by the same positive-scaling argument as the 3D search (`docs/PREDICTIONS-H2-FOURTERM.md`): rank 4 leaves only signs (16 patterns); each missing rank is one scale invariant scanned on a grid: step 0.25 in ln t over [-3, 3] when there is one invariant, step 0.5 when two or three. Term-sets are reduced by the 24 variable renamings (a canonical support), which cannot change whether a flow is chaotic. Sign flips are not removed (they only repeat work).

## Pipeline
Stage 0: state-only integration, dt 0.02, 1600 + 400 steps, drop if not finite, max |x| above 1e4, or not moving (range below 1e-3 in the last 400 steps): the same rule as the 3D stage 1a. Tier 1: spectrum, T = 300 (spin 100, dt 0.01), drop if diverges, fixed point, or first exponent + 3 se at or below 0.005 (the H2B middle-filter rule). Stage 2: spectrum T = 5000, classifier rule v2 with Amendment 5, label chaotic or hyperchaotic and not conservative. Confirmation: T = 20000 from two starts, both must give the same kind of label. Any hit gets the dt 0.005/0.0025 and first-integral checks before it is called anything. Code: `src/hyperchaos/enum4.py`, runner `experiments/h3_fourterm4d.py`.

## Controls (before the search)
- **C1.** In 4D, scaling a known system by random positive factors multiplies the exponents by tau and keeps the support (unit test, on Rossler 1979 or Li-Sprott coefficients).
- **C2.** The new 4D flow function reproduces the published Li-Sprott exponents at a = 2.6, b = 0.44 (0.0704, 0.0128, 0, -1.0832) within 0.01 each, T = 20000.
- **C3.** The canonical support is invariant under the 24 renamings (unit test).
- **C4.** Planted bug: wrong scaling formula; a test fails.
- **C5.** The screening tiers keep two known hyperchaotic systems: Li-Sprott (7 terms) and Rossler 1979 (9 terms) pass stage 0 and tier 1 and are labelled hyperchaotic at stage 2.

## Gates
- **GH1.** C1-C5 all pass before the search is run.
- **GH2 (report).** Systems screened, stage 0 survivors, tier 1 survivors, stage 2 hits by label, conservative candidates (with first-integral checks), wall time.

## Predictions (probability the statement is true)
- P-H1: no hyperchaotic 4-term 4D hit: 97%.
- P-H2: no chaotic (one positive exponent) 4-term 4D hit either: 80%. If one appears, I first suspect a numerical artifact (stiffness or a first integral) and say so.
- P-H3: stage 0 survivors are below 25% of the systems screened: 70% (one monomial per equation: many equations are constants or pure drift).
- P-H4: C5 passes: 85%.
- P-H5: the search takes under 30 minutes on 6 workers: 75%.
- P-H6: at least one prediction is missed: 55%.

## Not claimed
A "no 4-term 4D chaotic flow exists" statement. Covered: quadratic monomials only; invariants on the stated grids within ln t in [-3, 3]; two start points; the screening tiers' known losses; supports reduced by renaming only. Flows with more than 4 terms (5 and 6 terms in 4D) are not part of this run.

## Amendment A1 (2026-10-28, after control C5, before the search; gates and predictions above are NOT changed)

C1 (tests/test_enum4.py), C2 (max difference from the published Li-Sprott exponents 0.0022) and C3 pass. **C5 half-fails**: Li-Sprott passes stage 0 and tier 1 and is labelled hyperchaotic (0.069, 0.011, -0.0001, -1.080), but Rossler 1979 is dropped at stage 0. Diagnosis (`results/h3_c5.log` and a short run): from both near-origin starts it diverges at dt 0.01 and 0.002 (z' = 3 + xz escapes unless the start is in the basin); from its published start (-10, -6, 0, 10) it is "inconsistent" at dt 0.01 and "chaotic" at dt 0.002. So the pipeline as written has two limits: it only follows orbits reached from points near the origin, and dt 0.01 is too coarse for stiff 4D systems. P-H4 is therefore missed (predicted both pass).
Change before the search: stage 0 and tier 1 try 4 fixed starts, chosen now and not tuned to Rossler: (0.05, 0.05, 0.05, 0.05), (0.1, -0.05, 0.07, 0.03), (1, 1, 1, 1), (-1, 0.5, -0.5, 1). A system continues from the first start that passes stage 0 and tier 1. Confirmation uses that start and a second start (the first scaled by 1.1 plus 0.01 times (1, -1, 1, -1)); both must give the same label. dt stays 0.01 (the amplitude of normalised 4-term flows is about 1, where stiffness is milder than in Rossler; any hit is re-run at dt 0.005 and 0.0025). GH1 is not fully met (C5 for Rossler); the search runs anyway with that limit stated: the result covers only attractors reached from these four starts at dt 0.01. The Rossler published start (-10, -6, 0, 10) is not added, since that would tune the pipeline to the control.

## Amendment A2 (2026-10-28, after the first run was stopped, before the replacement is coded or run; gates and predictions above are NOT changed)

The first run (A1 pipeline) was stopped by the owner after about 25 minutes: 11 of 234 chunks done (4,832 systems, 1,600 passed stage 0, 946 passed tier 1, 0 reached a confirmed hit, 8 conservative candidates), projecting 6 to 7 hours. Cause: tier 1 (spectrum T = 300, keep if first exponent + 3 se > 0.005) is the H2B middle-filter rule, which H2B measured as removing almost nothing; 59% of stage-0 survivors passed it, and each then cost a T = 5000 spectrum. P-H5 (under 30 minutes) is missed.
Change: tier 1 is replaced by a batched one-vector exponent test, the 4D analogue of the 3D stage 1b: jvp-based tangent vector, dt 0.02, 1200 chunks of 25 steps (spin 200 chunks), keep if max |x| <= 1e4 and the exponent estimate > 0.01 (no standard-error allowance). It runs for each start that passed stage 0; the first such start in the A1 order that also passes it is used for stage 2 and the confirmation, unchanged. The runner now writes one result file per chunk (resumable) and stops all workers if no chunk finishes for 30 minutes.
**Lossiness is a known cost and is measured, not assumed:** the one-vector test has a known loss in 3D (the rule is the H2 stage 1b). Positive-control gates below.

### Added gates
- **GA2-1 (recall on controls).** Li-Sprott (7 terms) and the four 4D embeddings of Sprott cases B, C, D, E (the 3D flow plus `w' = -w`, 6 or 7 terms, one extra negative exponent) all pass stage 0 and the new test from at least one of the four starts at their normal coefficients. Any control lost is reported as a recall hole.
- **GA2-2 (selectivity).** On the systems of the first 20 chunks, the new test keeps at most 5% of all systems (the old tier 1 kept 20%).
- **GA2-3 (time).** Measured on those 20 chunks before the full run: projected full-run time at most 90 minutes; if it is longer the full run is not started and I report.

### Predictions
- P-A2-1: GA2-1 passes for all 5 controls: 70% (the 3D stage 1b lost about 2% of hits; the embeddings add a stiff negative exponent).
- P-A2-2: GA2-2 passes (at most 5%): 80%.
- P-A2-3: projected time at most 90 minutes: 75%; at most 30 minutes: 40%.
- P-A2-4: the full run again finds no confirmed hyperchaotic hit: 97%; no confirmed chaotic hit either: 80% (as P-H1, P-H2).
- P-A2-5: at least one of these is missed: 60%.

## Amendment A3 (2026-10-28, written while the A2 run was still going, before its final result and before any verification run; nothing above is changed)

The finished chunks of the A2 run (195 of 234 at the time of writing) contain 15 "confirmed hits". All 15 are one family, `x' = -a x`, `y' = b z`, `z' = b y`, `w' = -w^2` (4 terms, rank 3, a and b on the invariant grid), found from the start (-1, 0.5, -0.5, 1) with exponents (b, 0, -a, -b) or its reordering. They are not chaotic: (y, z) is a linear saddle and x, w decay. The cause is a defect in my start design: that start has y = -z, and the second start of the confirmation (the first scaled by 1.1 plus 0.01 times (1, -1, 1, -1)) keeps y = -z. The pair (y, z) then stays exactly on the stable manifold of the saddle (the symmetric Runge-Kutta steps preserve y + z = 0 exactly), the orbit decays to the origin, and the Lyapunov routine reports the saddle's rates along it. The classifier's fixed-point test does not fire because w decays like 1/t, so its range over the last block (about 2e-5) is above its 1e-6 threshold. So both "independent" starts sat on the same invariant plane and the confirmation was not independent.
Consequences: (1) these 15 are false positives; (2) the confirmation rule needs a generic second start; (3) the search itself is not invalidated for missing true positives by this (a true chaotic attractor does not need y = -z), but start C wasted coverage for systems where an antisymmetric start is special.
Verification added after the run (`experiments/h3_verify_hits.py`): every hit is re-run from two generic starts, (0.31, -0.72, 0.46, 0.83) and (-0.4, 0.2, 0.9, -0.6), at dt 0.01 and 0.005, T = 20000. A hit counts as verified only if all four runs give chaotic or hyperchaotic, not conservative, a late-block range above 1e-3 (the orbit does not settle), and an exponent sum below -0.005.
- **P-A3-1:** none of the hits verify (0 verified): 85%. One or more verify: 15%. If any verifies, it gets the dt 0.0025 and first-integral checks as well before being called anything.
- **P-A3-2:** the hits all lie on supports of this one saddle form or similar linear-plus-decay forms: 75%.
- **Miss already visible:** P-H5 (under 30 minutes) is missed again; GA2-3 (projected at most 90 minutes) is on course to be missed: the sample of 20 chunks took 184 s, but chunks 130 to 200 took about 15 to 95 s each as the later supports have more stage-0 survivors.
