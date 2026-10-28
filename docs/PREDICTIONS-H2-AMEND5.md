# H1 Amendment 5 (expanding orbits): predictions and gates (written 2026-10-28, before the code change)

Rule: `classify` labels an orbit "expanding" if sum(exponents) > 0.005 (after the sum-consistency check). Applied by re-scoring the stored hits; the hit files are not edited, a rescored file is written beside them.

## Gates
- **GA1 (exactness).** Re-scored from stored exponents: every hit removed has an exponent sum above 0.005, and every hit kept has a sum at or below 0.005.
- **GA2 (no valid system lost).** All published controls in `systems.py` keep their expected labels (the controls test file passes unchanged), and the Sprott 1994 and Rössler-class hyperchaotic controls are not labelled "expanding".
- **GA3 (tests and a planted bug).** A new classifier test (sum +0.5 labelled "expanding", sum -0.5 not) passes; the full fast suite is no worse than before (same count of failures as on main); a planted bug that disables the new rule is caught by a test.
- **GA4 (end to end).** Re-confirm every removed hit with `search3d.confirm` (current code, dt 0.01, T = 20000): all return not-confirmed.

## Predictions
- P-A1: GA1 passes and the night run loses exactly 39 hits (4,673 to 4,634): 80%. Between 30 and 45: 95%.
- P-A2: the main run loses exactly 5 (472 to 467): 80%.
- P-A3: GA2 passes with no control changed: 90%.
- P-A4: GA4 passes (all removed hits fail confirmation in the new code): 95%. This is close to a tautology for hits with sum > 0.005 and is a wiring check, not independent evidence.
- P-A5: the planted bug is caught: 90%.
- P-A6: at least one prediction is missed: 55%.

## Not claimed
That the remaining 4,634 are all valid. The 22 hits with sum in (-0.01, 0] and any hit with a frozen coordinate but a negative sum are not examined by this amendment.

## Results (2026-10-28, written after the runs; no gate or prediction above was changed)

Files: `experiments/h2_rescore_amend5.py`, `results/h2_amend5_rescored.{log,json}`, `results/h1_controls_amend5.log`, `results/h1_breakit_amend5_first.log`, `results/h1_breakit_amend5.log`. float64, Windows, CPU.

- **GA1 passes.** Night run: 39 removed (7 with k = 5, 32 with k = 6), 4,634 kept; smallest removed sum +0.0058, largest kept sum -0.0053, none misplaced. Main run: 5 removed (all k = 6), 467 kept; smallest removed sum +0.068, largest kept -0.013.
- **GA2 passes.** `tests/test_h1_controls.py` and `tests/test_h1_classifier.py` at `H1_T=5000`: 5 failed, 72 passed on the amendment branch; 5 failed, 71 passed on main. The five failures are the same tests on both (KP, CS13, CS12 published values, LS and CS12 verify): the known T = 5000 failures, none new. The extra pass is the new test.
- **GA3: the planted bug was NOT caught on the first run.** The fast suite selects tests by name (`g1`-`g4`), and my new test was not named with one, so the disabled rule passed. After renaming the test to `test_g3_amendment5_expanding_orbit_is_not_chaotic`, bugs 2, 3, 4 and 5 are all caught (49 baseline tests pass). Bug 1 reports "BUG TEXT NOT FOUND": the text it expects no longer matches `spectrum.py`, so that planted bug is not being run. That is an old defect in the harness, not touched here.
- **GA4 passes.** All 44 removed hits (39 + 5) fail `search3d.confirm` under the current code (wiring check only, as predicted).

**Predictions.** P-A1 hit (exactly 39). P-A2 hit (exactly 5). P-A3 hit. P-A4 hit. P-A5 **missed on the first attempt** (caught only after the rename). P-A6 hit.

**Effect on the counts.** Night run 1: 4,673 raw, **4,634** after Amendment 5 (3,538 with k = 6, 1,096 with k = 5). Main run: 472 raw, **467**. Earlier documents keep their original numbers and say so.

**Still open.** The 22 night-run hits with sum between -0.01 and -0.005 are weakly dissipative and kept; I have not examined them or looked for frozen coordinates among the kept hits.
