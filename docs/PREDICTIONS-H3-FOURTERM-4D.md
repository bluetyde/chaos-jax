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
