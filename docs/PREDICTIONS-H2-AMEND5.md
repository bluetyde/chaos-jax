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
