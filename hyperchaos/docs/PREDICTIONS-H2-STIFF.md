# H2 stiff-hit check: predictions and gates (written 2026-10-28, before the run)

Question: the 13 night-run-1 hits with a first exponent above 10 (all found at dt 0.01, confirmed at T = 20000 from two starts), are they real at a smaller step? Benkouider (docs/PREDICTIONS-BENKOUIDER.md) showed RK4 at dt 0.01 can diverge or mislead on stiff systems.

Method: for each of the 13, rerun the confirmation (both starts START and START2, T = 20000, classifier rule v2 with the sum check) at dt 0.01 (must reproduce the stored result), 0.005 and 0.0025. Script `experiments/h2_stiff_check.py`.

Gates
- **GS1 (control).** At dt 0.01 all 13 reproduce "chaotic" from both starts with a first exponent within 5% of the stored value. If not, the harness is wrong and nothing else counts.
- **GS2 (the question).** A hit is *stiffness-safe* if it is chaotic at dt 0.0025 from both starts with a first exponent within 10% of its dt 0.01 value.

Predictions
- P-S1: GS1 passes for all 13: 90%.
- P-S2: at least 10 of 13 are stiffness-safe: 45%. All 13: 15%. Fewer than 5: 20%. (A first exponent of 10 to 40 against exponents of order 1 for ordinary systems suggests fast dynamics that dt 0.01 under-resolves, so I expect a real share to fail.)
- P-S3: the failures, if any, are all hits with first exponent above 15: 55%.
- P-S4: at least one prediction is missed: 70%.

## Results (2026-10-28, written after the run; no gate or prediction above was changed)

Files: `experiments/h2_stiff_check.py`, `results/h2_night1_stiff.{log,json}`, `results/h2_night1_positive_sum.json`. float64, Windows, CPU, about 17 minutes.

**Gates.** GS1 passes: all 13 reproduce "chaotic" at dt 0.01 from the first start with a first exponent within 5% of the stored value. GS2: **10 of 13 are stiffness-safe**; hits 2, 3 and 10 fail (hit 2: first exponent 15.7 at dt 0.01, 27 to 34 at smaller dt; hit 3: diverges from the second start at dt 0.0025; hit 10: becomes a fixed point from the first start at dt 0.005).
**Predictions.** P-S1 hit. P-S2 hit (at least 10 safe, 45%). P-S3 **missed**: the failures have first exponents 15.7, 11.8 and 15.2, so they are not all above 15. P-S4 hit.

**The larger finding, not predicted: the "stiffness" is a symptom, not the cause.** 12 of the 13 have a positive mean divergence (+5 to +42; only hit 7 is dissipative, -0.10). The exponent sum equals the divergence (the classifier's sum check passes), so the sum of exponents is positive: phase volume expands on average. A bounded orbit with a positive-volume attractor cannot do that, so these are not strange attractors. Two inspected cases (hits 1 and 6) show part of the state frozen (two of three coordinates have zero range over the last block) while one coordinate moves: the orbit has settled onto an invariant line or curve, and the Benettin exponents measure the unstable transverse direction of that non-attracting set. This is the "drift on an invariant subspace" failure that Amendment 4 of H1 meant to catch, in a form the sum check does not catch (the sum still matches the divergence).

**Scope in the whole night run.** 39 of 4,673 hits (0.8%) have a positive exponent sum (36 of the 91 with first exponent above 1); another 22 have a sum within 0.01 below zero. The earlier 1e7 run has 5 of 472. See `results/h2_night1_positive_sum.json`. Not yet checked: whether the 22 near-zero cases are conservative flows or the same defect, and whether the other hits with a negative sum have a frozen coordinate.

**Proposed amendment (not applied to the classifier, owner's call):** reject any orbit whose exponent sum is not clearly negative (sum above -0.01 in absolute terms, or a frozen-coordinate test on the last block). It is a pure filter on stored data, loses no valid dissipative chaotic system by the argument above (an attractor of positive volume needs a negative sum), and would turn the 39 into rejections. It would also change the earlier published counts (472 to about 467, 4,673 to about 4,634).
