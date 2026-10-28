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
